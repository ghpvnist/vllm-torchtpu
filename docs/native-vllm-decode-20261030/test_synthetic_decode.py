# SPDX-License-Identifier: Apache-2.0

import inspect
import os
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import synthetic_decode_connector as module
from vllm.distributed.kv_transfer.kv_connector.v1.decode_bench_connector import (
    DecodeBenchConnectorMetadata,
    DecodeBenchConnectorScheduler,
)

C = module.SyntheticTPUDecodeConnector


class TestSynthetic(unittest.TestCase):
    def test_logging_enabled(self):
        self.assertTrue(module.logger.isEnabledFor(20))

    def test_concrete(self):
        self.assertFalse(inspect.isabstract(C))

    def test_requires_dummy(self):
        cfg = NS(
            load_config=NS(load_format="auto"),
            kv_transfer_config=NS(
                kv_connector_extra_config={"synthetic_decode_benchmark": True}
            ),
        )
        with self.assertRaises(ValueError):
            C(cfg, None, None)

    def test_requires_optin(self):
        cfg = NS(
            load_config=NS(load_format="dummy"),
            kv_transfer_config=NS(kv_connector_extra_config={}),
        )
        with self.assertRaises(ValueError):
            C(cfg, None, None)

    def test_long_prompt_and_all_groups(self):
        s = DecodeBenchConnectorScheduler(NS(cache_config=NS(block_size=2176)))
        req = NS(request_id="r", num_tokens=131072)
        self.assertEqual(s.get_num_new_matched_tokens(req, 0), (131071, False))
        self.assertEqual(s.get_num_new_matched_tokens(req, 130560), (511, False))
        groups = tuple(list(range(70)) for _ in range(4))
        s.update_state_after_alloc(req, NS(get_block_ids=lambda: groups), 131071)
        self.assertEqual(s.get_num_new_matched_tokens(req, 0), (0, False))
        meta = s.build_connector_meta(None)
        self.assertEqual([len(g) for g in meta.reqs_to_fill["r"][0]], [61] * 4)
        self.assertFalse(s.build_connector_meta(None).reqs_to_fill)
        s.request_finished(req)
        self.assertEqual(s.get_num_new_matched_tokens(req, 0), (131071, False))

    def test_pool_reset_once(self):
        raws = [object(), object()]
        c = C.__new__(C)
        c._runner = NS(kv_cache_raw_tensors=raws)
        c._pool_initialized = False
        c._connector_metadata = DecodeBenchConnectorMetadata(
            reqs_to_fill={"a": (([],), 131071)}
        )
        zeroed = []
        sync = object()
        with (
            patch.dict(
                sys.modules, {"vllm_torchtpu.utils": NS(synchronize_tensors=sync)}
            ),
            patch.object(
                module, "_zero_pool_in_place", lambda r, s: zeroed.append((r, s))
            ),
        ):
            c.start_load_kv(None)
            c._connector_metadata = DecodeBenchConnectorMetadata(
                reqs_to_fill={"b": (([],), 131071)}
            )
            c.start_load_kv(None)
        self.assertEqual(zeroed, [(raws, sync)])

    def test_pool_rezero_idle(self):
        raws = [object()]
        c = C.__new__(C)
        c._runner = NS(kv_cache_raw_tensors=raws, input_batch=NS(req_ids=["a", None]))
        c._pool_initialized = True
        zeroed = []

        def fill(req_ids, batch):
            c._runner.input_batch.req_ids = batch
            c._connector_metadata = DecodeBenchConnectorMetadata(
                reqs_to_fill={r: (([],), 131071) for r in req_ids}
            )
            c.start_load_kv(None)

        with (
            patch.dict(
                sys.modules, {"vllm_torchtpu.utils": NS(synchronize_tensors=None)}
            ),
            patch.object(module, "_zero_pool_in_place", lambda r, s: zeroed.append(r)),
        ):
            fill(["a"], ["a", None])  # knob off: the first zero is the only one
            with patch.dict(os.environ, {"SYNTHETIC_DECODE_REZERO_IDLE": "1"}):
                fill(["b"], ["a", "b"])  # joins a live request: no zero
                fill(["c", "d"], ["c", None, "d"])  # idle engine, new run: zero
        self.assertEqual(zeroed, [raws])

    def test_engine_kv_log(self):
        out = NS(
            num_scheduled_tokens={"n": 1, "a": 4, "b": 4},
            scheduled_new_reqs=[NS(req_id="n", num_computed_tokens=16383)],
            scheduled_cached_reqs=NS(
                req_ids=["a", "b"], num_computed_tokens=[131000, 50000]
            ),
        )
        cfg = NS(parallel_config=NS(data_parallel_rank=3))
        with self.assertLogs(module.logger, level="INFO") as logs:
            module._log_engine_kv(cfg, out)
        line = logs.output[-1]
        self.assertIn("SYNTHETIC_DECODE_KV dp=3 ", line)
        self.assertIn(" reqs=3 kv_tokens=197392 max_tokens=131004", line)


if __name__ == "__main__":
    unittest.main()
