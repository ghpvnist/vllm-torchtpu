# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the megakernel advisor trace calibration engine."""

from __future__ import annotations

import importlib
import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from vllm_torchtpu.megakernel import advisor, errors

if TYPE_CHECKING:
    from typing import Protocol

    class _MutableXStatProto(Protocol):
        """Mutable XStat protobuf message protocol for test fixture construction."""

        metadata_id: int
        int64_value: int
        double_value: float
        str_value: str

    class _StatContainerProto(Protocol):
        """Repeated XStat container protocol."""

        def add(self) -> _MutableXStatProto:
            """Append and return a new XStat entry."""

    class _MutableXEventProto(Protocol):
        """Mutable XEvent protobuf message protocol for test fixture construction."""

        metadata_id: int
        offset_ps: int
        duration_ps: int
        stats: _StatContainerProto

    class _EventContainerProto(Protocol):
        """Repeated XEvent container protocol."""

        def add(self) -> _MutableXEventProto:
            """Append and return a new XEvent entry."""

    class _MutableXLineProto(Protocol):
        """Mutable XLine protobuf message protocol for test fixture construction."""

        id: int
        name: str
        timestamp_ns: int
        events: _EventContainerProto


_PERFETTO_FIXTURE = (
    Path(__file__).resolve().parent / "data" / "advisor" / "synthetic_perfetto.json.gz"
)
_PS_PER_US = 1_000_000
_EXPECTED_SINGLE_GAPS = 9
_EXPECTED_MULTI_GAPS = 18
_EXPECTED_GROUP_SIZE = 8
_EXPECTED_FALLBACK_TP = 4
_EXPECTED_AR_BYTES = 32768
_EXPECTED_A2A_BYTES = 65536
_EXPECTED_STEADY_STEPS = 3
_EXPECTED_MIXED_SINGLE_GAPS = 12
_EXPECTED_MIXED_MULTI_GAPS = 24
_EXPECTED_MIXED_STEADY_STEPS = 4
_EXPECTED_BATCH = 16
_EXPECTED_TP = 8


def _add_int_stat(ev: _MutableXEventProto, meta_id: int, value: int) -> None:
    """Append an int64 XStat entry to `ev`."""
    st = ev.stats.add()
    st.metadata_id = meta_id
    st.int64_value = value


def _add_str_stat(ev: _MutableXEventProto, meta_id: int, value: str) -> None:
    """Append a string XStat entry to `ev`."""
    st = ev.stats.add()
    st.metadata_id = meta_id
    st.str_value = value


def _append_steady_step_events(
    step_line: _MutableXLineProto,
    ops_line: _MutableXLineProto,
    step_spec: tuple[int, int, int, str, int, int],
    time_shift_us: int,
) -> None:
    """Append one steady-state step and its 4 compute ops to `step_line`/`ops_line`."""
    step_num, base_us, dur_us, phase_str, act_tok, elig_tok = step_spec
    t0 = time_shift_us + base_us
    st_ev = step_line.events.add()
    st_ev.metadata_id = 1
    st_ev.offset_ps = t0 * _PS_PER_US
    st_ev.duration_ps = dur_us * _PS_PER_US
    _add_int_stat(st_ev, 1, step_num)
    _add_str_stat(st_ev, 2, phase_str)
    _add_int_stat(st_ev, 3, act_tok)
    _add_int_stat(st_ev, 4, elig_tok)

    for meta_id, rel_us, op_dur_us, coll_bytes in (
        (2, 5.0, 10, 0),
        (3, 17.0, 12, _EXPECTED_AR_BYTES),
        (4, 31.5, 15, 0),
        (5, 50.0, 18, _EXPECTED_A2A_BYTES),
    ):
        op_ev = ops_line.events.add()
        op_ev.metadata_id = meta_id
        op_ev.offset_ps = int((t0 + rel_us) * _PS_PER_US)
        op_ev.duration_ps = op_dur_us * _PS_PER_US
        if coll_bytes > 0:
            _add_int_stat(op_ev, 5, _EXPECTED_GROUP_SIZE)
            _add_int_stat(op_ev, 6, coll_bytes)


def _build_synthetic_xspace_bytes(time_shift_us: int = 0) -> bytes:
    """Build a deterministic XSpace proto with device step and XLA op lines."""
    xplane_pb2 = importlib.import_module("vllm_torchtpu.tools.xplane_pb2")
    space = xplane_pb2.XSpace()
    space.hostnames.append("tpu-worker-0")

    meta_plane = space.planes.add()
    meta_plane.id = 99
    meta_plane.name = "/host:metadata"

    cpu_plane = space.planes.add()
    cpu_plane.id = 98
    cpu_plane.name = "/host:CPU"

    dev_plane = space.planes.add()
    dev_plane.id = 1
    dev_plane.name = "/device:TPU:0"
    for ev_id, ev_name in (
        (1, "Step"),
        (2, "qkv_proj"),
        (3, "all-reduce.1"),
        (4, "attn_core"),
        (5, "all-to-all.2"),
    ):
        dev_plane.event_metadata[ev_id].id = ev_id
        dev_plane.event_metadata[ev_id].name = ev_name

    for st_id, st_name in (
        (1, "step_num"),
        (2, "phase"),
        (3, "active_tokens"),
        (4, "fused_eligible_tokens"),
        (5, "group_size"),
        (6, "bytes"),
    ):
        dev_plane.stat_metadata[st_id].id = st_id
        dev_plane.stat_metadata[st_id].name = st_name

    step_line = dev_plane.lines.add()
    step_line.id = 10
    step_line.name = "Steps"
    ops_line = dev_plane.lines.add()
    ops_line.id = 1
    ops_line.name = "XLA Ops"

    w_step = step_line.events.add()
    w_step.metadata_id = 1
    w_step.offset_ps = time_shift_us * _PS_PER_US
    w_step.duration_ps = 140 * _PS_PER_US
    _add_int_stat(w_step, 1, 0)
    _add_str_stat(w_step, 2, "decode")

    for rel_us, meta_id in ((10, 2), (55, 3)):
        w_op = ops_line.events.add()
        w_op.metadata_id = meta_id
        w_op.offset_ps = (time_shift_us + rel_us) * _PS_PER_US
        w_op.duration_ps = 20 * _PS_PER_US

    for spec in (
        (1, 200, 98, "decode", 16, 16),
        (2, 400, 100, "verify", 16, 16),
        (3, 600, 100, "decode", 16, 16),
        (4, 800, 102, "prefill", 16, 0),
    ):
        _append_steady_step_events(step_line, ops_line, spec, time_shift_us)
    return bytes(space.SerializeToString())


def test_calibrate_perfetto_extracts_launch_gap_and_collectives(
    tmp_path: Path,
) -> None:
    """Verify Perfetto trace calibration extracts steady-state gaps and collectives."""
    assert _PERFETTO_FIXTURE.is_file()
    out_json = tmp_path / "perfetto_measurements.json"

    rc = advisor.main(
        [
            "calibrate",
            "--trace",
            str(_PERFETTO_FIXTURE),
            "--chip",
            "tpu7x",
            "--model",
            "Qwen/Qwen3-30B-A3B-FP8",
            "--batch",
            str(_EXPECTED_BATCH),
            "--tp",
            str(_EXPECTED_TP),
            "--out",
            str(out_json),
        ]
    )
    assert rc == 0
    assert out_json.is_file()

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["schema"] == "advisor.measurements.v1"
    assert payload["source"]["kind"] == "perfetto"
    assert payload["source"]["batch"] == _EXPECTED_BATCH
    assert payload["source"]["tp"] == _EXPECTED_TP

    lg = payload["launch_gap_us"]
    assert lg["p50"] == pytest.approx(2.5)
    assert lg["p90"] == pytest.approx(3.5)
    assert lg["n"] == _EXPECTED_SINGLE_GAPS

    coll_by_op = {item["op"]: item for item in payload["collectives"]}
    assert set(coll_by_op) == {"all-reduce", "all-to-all"}
    assert coll_by_op["all-reduce"]["group_size"] == _EXPECTED_GROUP_SIZE
    assert coll_by_op["all-reduce"]["bytes"] == _EXPECTED_AR_BYTES
    assert coll_by_op["all-reduce"]["latency_us"] == pytest.approx(12.0)
    assert coll_by_op["all-reduce"]["hop_latency_us"] == pytest.approx(4.0)
    assert coll_by_op["all-reduce"]["n"] == _EXPECTED_STEADY_STEPS
    assert coll_by_op["all-to-all"]["latency_us"] == pytest.approx(18.0)
    assert coll_by_op["all-to-all"]["n"] == _EXPECTED_STEADY_STEPS

    cov = payload["step_coverage"]
    assert cov["fused_eligible_fraction"] == pytest.approx(0.75)
    assert cov["basis"] == "trace"

    st = payload["step_time_us"]
    assert st["p50"] == pytest.approx(100.0)
    assert st["n"] == _EXPECTED_STEADY_STEPS

    profile = advisor.resolve_model_profile("Qwen/Qwen3-30B-A3B-FP8")
    chip = advisor.resolve_chip_spec("tpu7x")
    report = advisor.advise(
        profile,
        advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8),
        chip,
        measurements=out_json,
    )
    assert report.confidence == "validated_end_to_end"
    assert report.metrics["launch_gap_us"]["tag"] == "MEASURED"
    assert report.metrics["launch_gap_us"]["value"] == pytest.approx(2.5)
    assert report.metrics["collective_latency_us"]["tag"] == "MEASURED"
    assert report.metrics["collective_latency_us"]["value"] == pytest.approx(4.0)
    assert report.metrics["step_coverage"]["tag"] == "MEASURED"
    assert report.metrics["step_coverage"]["value"] == pytest.approx(0.75)


def test_calibrate_xplane_roundtrip_with_in_repo_parser(tmp_path: Path) -> None:
    """Verify single-file and multi-rank XPlane calibration via xplane_pb2."""
    single_pb = tmp_path / "single_rank.xplane.pb"
    single_pb.write_bytes(_build_synthetic_xspace_bytes(time_shift_us=0))

    out_single = tmp_path / "xplane_single_measurements.json"
    payload = advisor.calibrate_trace(
        single_pb,
        out_single,
        chip="tpu7x",
        model="Qwen/Qwen3-30B-A3B-FP8",
    )
    assert out_single.is_file()
    assert payload["schema"] == "advisor.measurements.v1"
    assert payload["source"]["kind"] == "xplane"
    assert payload["launch_gap_us"]["p50"] == pytest.approx(2.5)
    assert payload["launch_gap_us"]["p90"] == pytest.approx(3.5)
    assert payload["launch_gap_us"]["n"] == _EXPECTED_SINGLE_GAPS
    assert payload["step_coverage"]["fused_eligible_fraction"] == pytest.approx(0.75)
    assert payload["step_time_us"]["p50"] == pytest.approx(100.0)

    session_dir = tmp_path / "decode_trace" / "plugins" / "profile" / "2026_11_04"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "rank0_worker0.xplane.pb").write_bytes(
        _build_synthetic_xspace_bytes(time_shift_us=0)
    )
    (session_dir / "rank1_worker1.xplane.pb").write_bytes(
        _build_synthetic_xspace_bytes(time_shift_us=7)
    )

    out_multi = tmp_path / "xplane_multi_measurements.json"
    rc = advisor.main(
        [
            "calibrate",
            "--trace",
            str(tmp_path / "decode_trace"),
            "--out",
            str(out_multi),
        ]
    )
    assert rc == 0
    multi_payload = json.loads(out_multi.read_text(encoding="utf-8"))
    assert multi_payload["launch_gap_us"]["p50"] == pytest.approx(2.5)
    assert multi_payload["launch_gap_us"]["p90"] == pytest.approx(3.5)
    assert multi_payload["launch_gap_us"]["n"] == _EXPECTED_MULTI_GAPS

    profile = advisor.resolve_model_profile("Qwen/Qwen3-30B-A3B-FP8")
    chip = advisor.resolve_chip_spec("tpu7x")
    report = advisor.advise(
        profile,
        advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8),
        chip,
        measurements=out_multi,
    )
    assert report.confidence == "validated_end_to_end"
    assert report.metrics["launch_gap_us"]["tag"] == "MEASURED"
    assert report.metrics["collective_latency_us"]["tag"] == "MEASURED"


def test_calibrate_float_stats_and_empty_gaps_preserve_assumed_prior(
    tmp_path: Path,
) -> None:
    """Verify float-valued token stats parse cleanly and n=0 gaps stay ASSUMED."""
    trace_file = tmp_path / "single_op_float_stats.json"
    trace_file.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "ph": "X",
                        "name": "Step",
                        "cat": "step",
                        "pid": 1,
                        "tid": 1,
                        "ts": 100.0,
                        "dur": 50.0,
                        "args": {
                            "step_num": 1.0,
                            "phase": "decode",
                            "active_tokens": 16.0,
                            "fused_eligible_tokens": 12.0,
                        },
                    },
                    {
                        "ph": "X",
                        "name": "all-reduce.1",
                        "cat": "op",
                        "pid": 1,
                        "tid": 2,
                        "ts": 110.0,
                        "dur": 9.0,
                        "args": {"group_size": 8.0, "bytes": 32768.0},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    out_json = tmp_path / "empty_gap_measurements.json"
    payload = advisor.calibrate_trace(trace_file, out_json)
    assert payload["launch_gap_us"]["n"] == 0
    assert "p50" not in payload["launch_gap_us"]
    assert payload["step_coverage"]["fused_eligible_fraction"] == pytest.approx(0.75)

    for model_id in ("qwen3.5-397b", "kimi-k3"):
        prof = advisor.resolve_model_profile(model_id)
        rep = advisor.advise(
            prof,
            advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8),
            advisor.resolve_chip_spec("tpu7x"),
            measurements=out_json,
        )
        assert rep.metrics["launch_gap_us"]["tag"] == "ASSUMED"
        assert rep.metrics["collective_latency_us"]["tag"] == "MEASURED"
        assert rep.metrics["step_coverage"]["tag"] == "MEASURED"


def test_calibrate_rejects_multi_session_and_unannotated_collectives(
    tmp_path: Path,
) -> None:
    """Verify multi-session error guard and collective group_size fallback to tp."""
    phased_root = tmp_path / "phased_run"
    for phase_dir in ("prefill_only", "decode_only"):
        sdir = phased_root / phase_dir / "plugins" / "profile" / "2026_11_04"
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / "rank0_worker0.xplane.pb").write_bytes(
            _build_synthetic_xspace_bytes(time_shift_us=0)
        )
    with pytest.raises(
        errors.MegakernelConfigError, match="Multiple XPlane sessions found"
    ):
        advisor.calibrate_trace(phased_root)

    with pytest.raises(errors.MegakernelConfigError, match="Missing trace path"):
        advisor.calibrate_trace(tmp_path / "nonexistent.json")

    bad_ext = tmp_path / "trace.txt"
    bad_ext.write_text("plain text", encoding="utf-8")
    with pytest.raises(errors.MegakernelConfigError, match="Unsupported trace format"):
        advisor.calibrate_trace(bad_ext)

    empty_dir = tmp_path / "empty_xplane_dir"
    empty_dir.mkdir()
    with pytest.raises(errors.MegakernelConfigError, match=r"No \.xplane\.pb files"):
        advisor.calibrate_trace(empty_dir)

    no_grp_trace = tmp_path / "unannotated_collective.json"
    no_grp_trace.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "ph": "X",
                        "name": "all-reduce-start.1",
                        "cat": "op",
                        "pid": 1,
                        "tid": 2,
                        "ts": 100.0,
                        "dur": 9.0,
                        "args": {"bytes": 16384},
                    },
                    {
                        "ph": "X",
                        "name": "all-reduce-done.1",
                        "cat": "op",
                        "pid": 1,
                        "tid": 2,
                        "ts": 112.0,
                        "dur": 1.0,
                        "args": {"group_size": 4, "bytes": 16384},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    skipped_payload = advisor.calibrate_trace(no_grp_trace)
    assert skipped_payload["collectives"] == []

    tp_payload = advisor.calibrate_trace(no_grp_trace, tp=_EXPECTED_FALLBACK_TP)
    assert len(tp_payload["collectives"]) == 1
    assert tp_payload["collectives"][0]["group_size"] == _EXPECTED_FALLBACK_TP
    assert tp_payload["collectives"][0]["hop_latency_us"] == pytest.approx(4.5)


def test_steady_state_steps_drops_compilation_warmup_outlier() -> None:
    """Verifies 2-step trace drops step 0 if it is a severe compilation outlier."""
    from vllm_torchtpu.megakernel.advisor import calibrate

    step0 = calibrate._StepWindow(start_us=0.0, end_us=30000.0, dur_us=30000.0)
    step1 = calibrate._StepWindow(start_us=30000.0, end_us=35000.0, dur_us=5000.0)
    filtered = calibrate._steady_state_steps([step0, step1])
    assert len(filtered) == 1
    assert filtered[0].start_us == 30000.0


def test_steady_state_steps_keeps_normal_two_step_trace() -> None:
    """Verifies 2-step trace preserves step 0 if within normal variation."""
    from vllm_torchtpu.megakernel.advisor import calibrate

    step0 = calibrate._StepWindow(start_us=0.0, end_us=5200.0, dur_us=5200.0)
    step1 = calibrate._StepWindow(start_us=5200.0, end_us=10200.0, dur_us=5000.0)
    filtered = calibrate._steady_state_steps([step0, step1])
    assert len(filtered) == 2
    assert filtered[0].start_us == 0.0


def test_calibrate_trace_filters_by_requested_phase() -> None:
    """Verifies calibrate_trace isolates step metrics by requested phase."""
    from vllm_torchtpu.megakernel.advisor import calibrate

    perfetto_events = [
        {
            "ph": "X",
            "name": "Step 0",
            "cat": "ModelStep",
            "ts": 1000,
            "dur": 2000,
            "pid": 1,
            "tid": 1,
            "args": {"step_num": 0, "phase": "decode"},
        },
        {
            "ph": "X",
            "name": "Step 1",
            "cat": "ModelStep",
            "ts": 5000,
            "dur": 2000,
            "pid": 1,
            "tid": 1,
            "args": {"step_num": 1, "phase": "decode"},
        },
        {
            "ph": "X",
            "name": "Step 2",
            "cat": "ModelStep",
            "ts": 8000,
            "dur": 2100,
            "pid": 1,
            "tid": 1,
            "args": {"step_num": 2, "phase": "decode"},
        },
        {
            "ph": "X",
            "name": "Step 3",
            "cat": "ModelStep",
            "ts": 12000,
            "dur": 8000,
            "pid": 1,
            "tid": 1,
            "args": {"step_num": 3, "phase": "prefill"},
        },
    ]
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(perfetto_events, f)
        trace_file = f.name

    try:
        cal_decode = calibrate.calibrate_trace(trace_file, phase="decode")
        assert "step_time_us" in cal_decode
        assert cal_decode["step_time_us"]["n"] == 2
        assert cal_decode["step_time_us"]["p50"] == pytest.approx(2050.0)
        assert cal_decode["source"]["phase"] == "decode"

        cal_prefill = calibrate.calibrate_trace(trace_file, phase="prefill")
        assert "step_time_us" in cal_prefill
        assert cal_prefill["step_time_us"]["n"] == 1
        assert cal_prefill["step_time_us"]["p50"] == pytest.approx(8000.0)
        assert cal_prefill["source"]["phase"] == "prefill"

        cal_mixed = calibrate.calibrate_trace(trace_file, phase="mixed")
        assert "step_time_us" in cal_mixed
        assert cal_mixed["step_time_us"]["n"] == 3
        assert cal_mixed["source"]["phase"] == "mixed"

        with pytest.raises(errors.MegakernelConfigError, match="found phases:"):
            calibrate.calibrate_trace(trace_file, phase="saturation")
    finally:
        os.unlink(trace_file)


_CONSERVED_NUM_LAYERS = 60
_CONSERVED_OPS_PER_STEP = 9_000
_CONSERVED_PERF_OPS = 240
_CONSERVED_STEP_IDLE_US = 104.68
_CONSERVED_STEP_DUR_US = 1_004.68
_CONSERVED_BOUNDARY_DIVISOR = 120.0


def _populate_conserved_step(
    lines: tuple[_MutableXLineProto, ...],
    step_idx: int,
    step_dur_ps: int,
    per_op_stride_ps: int,
    per_op_busy_ps: int,
) -> None:
    """Append one steady-state step and 9,000 HLO ops across device streams."""
    host_step_line, mod_line, dev_mod_line, ops_line, async_line = lines
    base_ps = step_idx * 2_000 * _PS_PER_US
    st_ev = host_step_line.events.add()
    st_ev.metadata_id = 1
    st_ev.offset_ps = base_ps
    st_ev.duration_ps = step_dur_ps
    _add_int_stat(st_ev, 1, step_idx)
    _add_str_stat(st_ev, 2, "decode")

    for wrap_line in (mod_line, dev_mod_line):
        wrap_ev = wrap_line.events.add()
        wrap_ev.metadata_id = 1
        wrap_ev.offset_ps = base_ps
        wrap_ev.duration_ps = step_dur_ps

    for op_idx in range(_CONSERVED_OPS_PER_STEP):
        op_start_ps = base_ps + op_idx * per_op_stride_ps
        op_ev = ops_line.events.add()
        op_ev.metadata_id = 2
        op_ev.offset_ps = op_start_ps
        op_ev.duration_ps = per_op_busy_ps
        if op_idx % 1_500 == 0:
            ov_ev = async_line.events.add()
            ov_ev.metadata_id = 3
            ov_ev.offset_ps = op_start_ps
            ov_ev.duration_ps = per_op_busy_ps


def _build_conserved_xspace_bytes() -> bytes:
    """Build a 60-layer XPlane trace with host steps and XLA Modules wrappers."""
    xplane_pb2 = importlib.import_module("vllm_torchtpu.tools.xplane_pb2")
    space = xplane_pb2.XSpace()
    space.hostnames.append("tpu-worker-0")

    cpu_plane = space.planes.add()
    cpu_plane.id = 98
    cpu_plane.name = "/host:CPU"
    cpu_plane.event_metadata[1].id = 1
    cpu_plane.event_metadata[1].name = "Step"
    cpu_plane.stat_metadata[1].id = 1
    cpu_plane.stat_metadata[1].name = "step_num"
    cpu_plane.stat_metadata[2].id = 2
    cpu_plane.stat_metadata[2].name = "phase"
    host_step_line = cpu_plane.lines.add()
    host_step_line.id = 10
    host_step_line.name = "Steps"

    mod_plane = space.planes.add()
    mod_plane.id = 97
    mod_plane.name = "XLA Modules"
    mod_plane.event_metadata[1].id = 1
    mod_plane.event_metadata[1].name = "jit_model_step"
    mod_line = mod_plane.lines.add()
    mod_line.id = 1
    mod_line.name = "XLA Modules"

    dev_plane = space.planes.add()
    dev_plane.id = 1
    dev_plane.name = "/device:TPU:0"
    for meta_id, ev_name in (
        (1, "jit_execute"),
        (2, "fusion_op"),
        (3, "async_collective_overlap"),
    ):
        dev_plane.event_metadata[meta_id].id = meta_id
        dev_plane.event_metadata[meta_id].name = ev_name

    dev_mod_line = dev_plane.lines.add()
    dev_mod_line.id = 1
    dev_mod_line.name = "XLA Modules"
    ops_line = dev_plane.lines.add()
    ops_line.id = 2
    ops_line.name = "XLA Ops"
    async_line = dev_plane.lines.add()
    async_line.id = 3
    async_line.name = "Async Collectives"

    busy_total_us = _CONSERVED_STEP_DUR_US - _CONSERVED_STEP_IDLE_US
    per_op_busy_ps = round(busy_total_us * _PS_PER_US / _CONSERVED_OPS_PER_STEP)
    per_op_stride_ps = int(
        _CONSERVED_STEP_DUR_US * _PS_PER_US // _CONSERVED_OPS_PER_STEP
    )
    step_dur_ps = round(_CONSERVED_STEP_DUR_US * _PS_PER_US)
    lines = (host_step_line, mod_line, dev_mod_line, ops_line, async_line)
    for step_idx in range(3):
        _populate_conserved_step(
            lines, step_idx, step_dur_ps, per_op_stride_ps, per_op_busy_ps
        )
    return bytes(space.SerializeToString())


def _build_conserved_perfetto_events() -> list[dict[str, object]]:
    """Build Perfetto trace events with host steps and module wrappers."""
    events: list[dict[str, object]] = []
    busy_total_us = _CONSERVED_STEP_DUR_US - _CONSERVED_STEP_IDLE_US
    per_op_busy_us = busy_total_us / _CONSERVED_PERF_OPS
    per_op_stride_us = _CONSERVED_STEP_DUR_US / _CONSERVED_PERF_OPS
    for step_idx in range(3):
        base_us = step_idx * 2_000.0
        events.extend(
            [
                {
                    "ph": "X",
                    "name": "Step",
                    "cat": "step",
                    "pid": 0,
                    "tid": 1,
                    "ts": base_us,
                    "dur": _CONSERVED_STEP_DUR_US,
                    "args": {"step_num": step_idx, "phase": "decode"},
                },
                {
                    "ph": "X",
                    "name": "xla_module_step",
                    "cat": "xla_modules",
                    "pid": 1,
                    "tid": 1,
                    "ts": base_us,
                    "dur": _CONSERVED_STEP_DUR_US,
                },
            ]
        )
        events.extend(
            [
                {
                    "ph": "X",
                    "name": f"hlo_{op_idx}",
                    "cat": "op",
                    "pid": 1,
                    "tid": 2,
                    "ts": base_us + op_idx * per_op_stride_us,
                    "dur": per_op_busy_us,
                }
                for op_idx in range(_CONSERVED_PERF_OPS)
            ]
        )
    return events


def test_calibrate_xplane_conserves_step_idle_sum(tmp_path: Path) -> None:
    """Verify XPlane and Perfetto calibration conserve total step idle gap."""
    pb_file = tmp_path / "compiled_60l.xplane.pb"
    pb_file.write_bytes(_build_conserved_xspace_bytes())
    out_xplane = tmp_path / "xplane_conserved.json"

    payload = advisor.calibrate_trace(
        pb_file,
        out_xplane,
        chip="tpu7x",
        model="qwen3.5-397b",
        phase="decode",
    )
    lg = payload["launch_gap_us"]
    assert lg["unit"] == "us_per_sublayer_boundary"
    assert lg["num_layers_norm"] == _CONSERVED_NUM_LAYERS
    assert lg["step_idle_gap_us"] == pytest.approx(_CONSERVED_STEP_IDLE_US, abs=1e-3)
    assert lg["p50"] == pytest.approx(
        _CONSERVED_STEP_IDLE_US / _CONSERVED_BOUNDARY_DIVISOR, abs=1e-6
    )

    rep = advisor.advise(
        "qwen3.5-397b",
        advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8),
        "tpu7x",
        measurements=out_xplane,
    )
    per_op_est = next(s for s in rep.scopes if s.scope == "per_op")
    total_per_op_bubble_us = per_op_est.ops_per_step * float(
        rep.metrics["launch_gap_us"]["value"]
    )
    assert total_per_op_bubble_us == pytest.approx(_CONSERVED_STEP_IDLE_US, abs=1e-6)

    perf_path = tmp_path / "compiled_60l_perfetto.json"
    perf_path.write_text(
        json.dumps({"traceEvents": _build_conserved_perfetto_events()}),
        encoding="utf-8",
    )
    out_perf = tmp_path / "perfetto_conserved.json"
    rc = advisor.main(
        [
            "calibrate",
            "--trace",
            str(perf_path),
            "--chip",
            "tpu7x",
            "--num-layers",
            str(_CONSERVED_NUM_LAYERS),
            "--out",
            str(out_perf),
        ]
    )
    assert rc == 0
    perf_payload = json.loads(out_perf.read_text(encoding="utf-8"))
    perf_rep = advisor.advise(
        "qwen3.5-397b",
        advisor.WorkloadSpec(phase="decode", batch=16, tp=8, ep=8),
        "tpu7x",
        measurements=out_perf,
    )
    perf_per_op = next(s for s in perf_rep.scopes if s.scope == "per_op")
    assert perf_per_op.ops_per_step * float(
        perf_rep.metrics["launch_gap_us"]["value"]
    ) == pytest.approx(_CONSERVED_STEP_IDLE_US, abs=1e-6)
    assert perf_payload["launch_gap_us"]["num_layers_norm"] == _CONSERVED_NUM_LAYERS
