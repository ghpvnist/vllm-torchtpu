# SPDX-License-Identifier: Apache-2.0
"""KV-aware data-parallel engine choice (TPU_DP_KV_BALANCE); no engines run."""

import asyncio
from collections import Counter
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
from vllm.v1.engine.core_client import DPLBAsyncMPClient

from vllm_torchtpu import envs as tpu_envs
from vllm_torchtpu.core import dp_kv_balance
from vllm_torchtpu.core.dp_kv_balance import patch_dp_kv_balance, pick_engine

pytestmark = pytest.mark.cpu_test

ENGINES = [b"\x00\x00", b"\x01\x00"]


@pytest.fixture
def patched(monkeypatch):
    for name in ("get_core_engine_for_request", "process_engine_outputs"):
        monkeypatch.setattr(DPLBAsyncMPClient, name, DPLBAsyncMPClient.__dict__[name])
    monkeypatch.setattr(DPLBAsyncMPClient, dp_kv_balance._MARKER, False, raising=False)
    with patch.object(tpu_envs, "TPU_DP_KV_BALANCE", True):
        patch_dp_kv_balance()
        patch_dp_kv_balance()  # idempotent
        yield DPLBAsyncMPClient


def client(max_num_seqs=4, engines=ENGINES, client_count=1):
    c = object.__new__(DPLBAsyncMPClient)
    c.core_engines = list(engines)
    c.lb_engines = [[0, 0, 0.0] for _ in engines]
    c.engine_inflight = Counter()
    c.reqs_in_flight = {}
    c.eng_start_index = 0
    c.client_count = client_count
    c.vllm_config = NS(scheduler_config=NS(max_num_seqs=max_num_seqs))
    return c


def request(rid, tokens, rank=None):
    return NS(
        request_id=rid,
        prompt_token_ids=[0] * tokens,
        prompt_embeds=None,
        data_parallel_rank=rank,
        pooling_params=None,
    )


def outputs(new=(), finished=()):
    return NS(
        outputs=[NS(request_id=r, new_token_ids=[1] * n) for r, n in new],
        finished_requests=set(finished),
    )


def test_pick_engine():
    assert pick_engine([1, 1, 1], [30, 10, 20], cap=4, start=0) == 1
    assert pick_engine([1, 4, 1], [30, 10, 20], cap=4, start=0) == 2  # engine 1 full
    assert pick_engine([4, 4], [1, 2], cap=4, start=0) is None
    assert pick_engine([0, 0, 0], [5, 5, 5], cap=4, start=2) == 2  # ties: first scanned


def test_knob_off_leaves_vllm_untouched():
    before = DPLBAsyncMPClient.__dict__["get_core_engine_for_request"]
    with patch.object(tpu_envs, "TPU_DP_KV_BALANCE", False):
        patch_dp_kv_balance()
    assert DPLBAsyncMPClient.__dict__["get_core_engine_for_request"] is before


def test_requests_go_to_the_engine_with_fewest_kv_tokens(patched):
    c = client()
    picks = [
        c.get_core_engine_for_request(request(f"r{i}", n))
        for i, n in enumerate([100_000, 20_000, 30_000, 40_000])
    ]
    # 100K on engine 0; then engine 1 takes requests until it holds more KV.
    assert picks == [ENGINES[0], ENGINES[1], ENGINES[1], ENGINES[1]]
    assert c.engine_inflight == Counter({ENGINES[0]: 1, ENGINES[1]: 3})
    assert c.reqs_in_flight["r3"] == ENGINES[1]
    assert [row[0] for row in c.lb_engines] == [1, 3]  # vLLM's local waiting bump
    state = c._tpu_kv_state
    assert state.engine_tokens == Counter({ENGINES[0]: 100_000, ENGINES[1]: 90_000})


def test_full_engines_are_skipped_and_vllm_decides_when_all_are_full(patched):
    c = client(max_num_seqs=2)
    for i in range(2):
        assert c.get_core_engine_for_request(request(f"long{i}", 100_000)) in ENGINES
    # Each engine holds one 100K request; short requests fill the slots.
    c.get_core_engine_for_request(request("a", 1_000))
    c.get_core_engine_for_request(request("b", 1_000))
    assert c.engine_inflight == Counter({ENGINES[0]: 2, ENGINES[1]: 2})
    # Both engines at max_num_seqs: vLLM's own choice (fewest waiting + running).
    c.lb_engines = [[0, 2, 0.4], [1, 2, 0.4]]
    assert c.get_core_engine_for_request(request("c", 50_000)) == ENGINES[0]
    assert c._tpu_kv_state.requests["c"][0] == ENGINES[0]


def test_coordinator_counts_also_mark_an_engine_full(patched):
    c = client(max_num_seqs=3)
    c.lb_engines = [[1, 2, 0.9], [0, 0, 0.0]]  # another client's requests on engine 0
    c.get_core_engine_for_request(request("x", 10))
    assert c.reqs_in_flight["x"] == ENGINES[1]


def test_pinned_requests_keep_their_engine_and_count(patched):
    c = client()
    assert c.get_core_engine_for_request(request("p", 80_000, rank=0)) == ENGINES[0]
    assert c.get_core_engine_for_request(request("q", 10_000)) == ENGINES[1]
    assert c._tpu_kv_state.engine_tokens[ENGINES[0]] == 80_000


def test_generated_tokens_count_and_finished_requests_release(patched):
    c = client()
    c.get_core_engine_for_request(request("a", 50_000))  # engine 0
    c.get_core_engine_for_request(request("b", 60_000))  # engine 1
    asyncio.run(
        DPLBAsyncMPClient.process_engine_outputs(c, outputs(new=[("a", 4), ("b", 1)]))
    )
    state = c._tpu_kv_state
    assert state.engine_tokens == Counter({ENGINES[0]: 50_004, ENGINES[1]: 60_001})
    asyncio.run(DPLBAsyncMPClient.process_engine_outputs(c, outputs(finished=["b"])))
    assert state.engine_tokens[ENGINES[1]] == 0 and "b" not in state.requests
    # vLLM's own bookkeeping still runs: the finished request leaves its count.
    assert c.engine_inflight[ENGINES[1]] == 0 and "b" not in c.reqs_in_flight
    assert c.get_core_engine_for_request(request("c", 1)) == ENGINES[1]
