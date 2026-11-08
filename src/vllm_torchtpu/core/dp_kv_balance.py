# SPDX-License-Identifier: Apache-2.0
"""Opt-in KV-aware choice of the data-parallel engine (TPU_DP_KV_BALANCE=1).

vLLM's data-parallel balancer (``DPLBAsyncMPClient.get_core_engine_for_request``
in each API server) sends a new request to the engine with the fewest waiting
plus running requests. Data-parallel engines of an expert-parallel model step
in lockstep: every MoE layer exchanges tokens across them, so each attention
layer waits for the engine with the most KV tokens to read. With mixed context
lengths, equal request counts leave unequal KV.

With the knob on, a request goes to the engine holding the fewest KV tokens
among the engines with a free sequence slot (fewer than ``max_num_seqs``
requests, counting this client's unfinished requests and the coordinator's
waiting plus running counts). When no engine has a free slot, vLLM's own choice
applies. KV tokens are counted per client, exactly: the prompt and every
generated token of its unfinished requests. Requests pinned to an engine
(``X-data-parallel-rank``) keep it and are counted. Only the placement of a
request changes, not what any engine computes for it.
"""

from collections import Counter

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

_MARKER = "_tpu_dp_kv_balance"


def pick_engine(
    loads: list[float], kv_tokens: list[int], cap: int, start: int
) -> int | None:
    """Index of the engine with the fewest KV tokens among engines below ``cap``.

    Engines are scanned from ``start`` (wrapping), so ties go to the first one
    scanned. None when every engine is at ``cap``.
    """
    best = None
    for i in range(len(loads)):
        idx = (start + i) % len(loads)
        if loads[idx] < cap and (best is None or kv_tokens[idx] < kv_tokens[best]):
            best = idx
    return best


class _KVState:
    """This client's unfinished requests: engine and KV tokens of each."""

    def __init__(self):
        self.engine_tokens: Counter = Counter()
        self.requests: dict[str, list] = {}

    def add(self, request_id: str, engine, tokens: int) -> None:
        self.engine_tokens[engine] += tokens
        self.requests[request_id] = [engine, tokens]

    def grow(self, request_id: str, tokens: int) -> None:
        entry = self.requests.get(request_id)
        if entry is not None:
            entry[1] += tokens
            self.engine_tokens[entry[0]] += tokens

    def finish(self, request_id: str) -> None:
        entry = self.requests.pop(request_id, None)
        if entry is not None:
            self.engine_tokens[entry[0]] -= entry[1]


def _prompt_tokens(request) -> int:
    if request.prompt_token_ids is not None:
        return len(request.prompt_token_ids)
    embeds = getattr(request, "prompt_embeds", None)
    return 0 if embeds is None else int(embeds.shape[0])


def patch_dp_kv_balance() -> None:
    """Install the KV-aware choice on vLLM's data-parallel balancer."""
    if not envs.TPU_DP_KV_BALANCE:
        return
    from vllm.v1.engine.core_client import DPLBAsyncMPClient

    cls = DPLBAsyncMPClient
    if cls.__dict__.get(_MARKER, False):
        return
    original_choose = cls.get_core_engine_for_request
    original_outputs = cls.__dict__["process_engine_outputs"].__func__

    def get_core_engine_for_request(self, request):
        state = self.__dict__.get("_tpu_kv_state")
        if state is None:
            state = self._tpu_kv_state = _KVState()
        engine = None
        if request.data_parallel_rank is None and request.pooling_params is None:
            counts = self.lb_engines
            loads = [
                max(self.client_count * self.engine_inflight[e], c[0] + c[1])
                for e, c in zip(self.core_engines, counts)
            ]
            kv_tokens = [state.engine_tokens[e] for e in self.core_engines]
            cap = self.vllm_config.scheduler_config.max_num_seqs
            idx = pick_engine(loads, kv_tokens, cap, self.eng_start_index)
            if idx is not None:
                # The bookkeeping of vLLM's own choice, for the chosen engine.
                engine = self.core_engines[idx]
                counts[idx][0] += self.client_count
                self.eng_start_index = (self.eng_start_index + 1) % len(counts)
                self.reqs_in_flight[request.request_id] = engine
                self.engine_inflight[engine] += 1
        if engine is None:
            engine = original_choose(self, request)
        state.add(request.request_id, engine, _prompt_tokens(request))
        return engine

    async def process_engine_outputs(self, outputs):
        state = self.__dict__.get("_tpu_kv_state")
        if state is not None:
            for out in outputs.outputs:
                if out.new_token_ids:
                    state.grow(out.request_id, len(out.new_token_ids))
            for request_id in outputs.finished_requests or ():
                state.finish(request_id)
        await original_outputs(self, outputs)

    cls.get_core_engine_for_request = get_core_engine_for_request
    cls.process_engine_outputs = staticmethod(process_engine_outputs)
    setattr(cls, _MARKER, True)
    logger.info(
        "Data-parallel balancer: requests go to the engine with fewest KV tokens"
    )
