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
"""Differential parity suite between Python advisor physics and JS workbench engine."""

from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

import pytest

from vllm_torchtpu.megakernel import advisor, errors
from vllm_torchtpu.megakernel.advisor import models, workbench_engine

if TYPE_CHECKING:
    from collections.abc import Callable

_TWO_DECIMAL_ABS_TOL = 0.015
_RAW_STEP_REL_TOL = 1e-6
_MODEL_SAMPLE_COUNT = 5
_EXPECTED_SCENARIO_COUNT = 80
_TP_BASE_WIDE = 8
_TP_BASE_MID = 4
_EP_WIDE = 16
_HALF_ROUND_THRESHOLD = 0.5 - 1e-12


class _ScopeParityEntry(TypedDict):
    """Expected and evaluated per-scope parity fields."""

    feasible: bool
    t_vmem_max: int
    n_splits: int
    selected_nbd: int
    selected_nbe: int
    raw_step_us: float
    rawStepUs: float
    step_bound_us: float
    step_achievable_us: float
    act_hbm_mb: float
    comm_us: float


class _EvalParityResult(TypedDict):
    """Expected and evaluated workload parity payload."""

    invalid_reasons: list[str]
    scopes: dict[str, _ScopeParityEntry]


class _BatchScenarioItem(TypedDict):
    """Input scenario specification for the 80-case parity sweep."""

    only_validate: bool
    model_name: str
    wl: advisor.WorkloadSpec


def _advisor_helper(attr_name: str) -> Callable[..., object]:
    """Resolve an internal advisor helper function by name."""
    return cast("Callable[..., object]", getattr(advisor, attr_name))


def _select_five_models() -> tuple[str, ...]:
    """Return 5 distinct model presets spanning GQA, GDN, MLA, MoE, and dense FFN."""
    preferred = (
        "qwen3.5-397b",
        "kimi-k3",
        "qwen3.8-2.4t",
        "deepseek-v3",
        "llama-3.1-405b",
    )
    resolved: list[str] = []
    for cand in preferred:
        try:
            models.resolve_model_profile(cand)
            resolved.append(cand)
        except (errors.MegakernelConfigError, KeyError, ValueError, OSError):
            continue
    if len(resolved) < _MODEL_SAMPLE_COUNT:
        models_json = (
            Path(models.__file__).with_name("data") / "presets" / "models.json"
        )
        if not models_json.is_file():
            models_json = Path(models.__file__).with_name("data") / "models.json"
        raw_map = json.loads(models_json.read_text("utf-8")).get("models", {})
        for key in raw_map:
            if key not in resolved:
                resolved.append(key)
            if len(resolved) == _MODEL_SAMPLE_COUNT:
                break
    assert len(resolved) == _MODEL_SAMPLE_COUNT
    return tuple(resolved)


def _valid_workloads_for_model(
    prof: advisor.ModelProfile,
) -> list[advisor.WorkloadSpec]:
    """Build 15 diverse valid WorkloadSpec regimes for one model profile."""
    has_moe = prof.num_moe_layers > 0 and prof.num_experts > 0
    tp_base = (
        _TP_BASE_WIDE if prof.num_attention_heads % _TP_BASE_WIDE == 0 else _TP_BASE_MID
    )
    ep_valid = (
        _TP_BASE_WIDE
        if (
            has_moe
            and prof.num_experts % _TP_BASE_WIDE == 0
            and tp_base >= _TP_BASE_WIDE
        )
        else 1
    )
    ep_wide = _EP_WIDE if (has_moe and prof.num_experts % _EP_WIDE == 0) else ep_valid
    dp_wide = 2 if (ep_wide == _EP_WIDE and tp_base == _TP_BASE_WIDE) else 1
    dcp_val = _TP_BASE_MID if tp_base >= _TP_BASE_MID else 1
    return [
        advisor.WorkloadSpec(
            phase="decode", batch=4, isl=3072, osl=1024, tp=tp_base, ep=ep_valid
        ),
        advisor.WorkloadSpec(
            phase="decode",
            batch=16,
            isl=4096,
            osl=512,
            tp=tp_base,
            ep=ep_valid,
            dispatch_regime="zero_floor",
        ),
        advisor.WorkloadSpec(
            phase="decode",
            batch=32,
            isl=8192,
            osl=512,
            tp=tp_base,
            ep=ep_valid,
            dispatch_regime="compiled_xla",
        ),
        advisor.WorkloadSpec(
            phase="decode",
            batch=64,
            isl=16384,
            osl=512,
            context=16384,
            tp=tp_base,
            ep=ep_valid,
            moe_mode="grouped_on_chip",
            dispatch_regime="unfused_eager",
        ),
        advisor.WorkloadSpec(
            phase="decode",
            batch=96,
            isl=8192,
            osl=512,
            tp=tp_base,
            ep=ep_valid,
            moe_mode="grouped_on_chip",
        ),
        advisor.WorkloadSpec(
            phase="decode",
            batch=256,
            isl=32768,
            osl=512,
            context=32768,
            tp=tp_base,
            dp=dp_wide,
            ep=ep_wide,
            moe_mode="grouped_on_chip",
        ),
        advisor.WorkloadSpec(
            phase="decode",
            batch=512,
            isl=4096,
            osl=512,
            context=524288,
            tp=tp_base,
            ep=ep_valid,
        ),
        advisor.WorkloadSpec(
            phase="verify",
            batch=8,
            spec_tokens=3,
            isl=3072,
            osl=1024,
            tp=tp_base,
            ep=ep_valid,
            dispatch_regime="compiled_xla",
        ),
        advisor.WorkloadSpec(
            phase="verify",
            batch=24,
            spec_tokens=5,
            isl=8192,
            osl=512,
            tp=tp_base,
            ep=ep_valid,
            moe_mode="grouped_on_chip",
        ),
        advisor.WorkloadSpec(
            phase="prefill",
            batch=1,
            isl=4096,
            osl=512,
            chunk_size=512,
            prefix_hit_rate=0.5,
            tp=tp_base,
            ep=ep_valid,
        ),
        advisor.WorkloadSpec(
            phase="prefill",
            batch=2,
            isl=8192,
            osl=512,
            pcp=2,
            tp=tp_base,
            ep=ep_valid,
            moe_mode="grouped_on_chip",
        ),
        advisor.WorkloadSpec(
            phase="mixed",
            batch=16,
            isl=4096,
            osl=2048,
            chunk_size=256,
            prefix_hit_rate=0.90,
            tp=tp_base,
            ep=ep_valid,
            moe_mode="grouped_on_chip",
        ),
        advisor.WorkloadSpec(
            phase="mixed",
            batch=12,
            isl=8192,
            osl=1024,
            pcp=2,
            dcp=dcp_val,
            tp=1 if dcp_val > 1 else tp_base,
            ep=1,
            moe_mode="grouped_on_chip",
        ),
        advisor.WorkloadSpec(
            phase="decode",
            batch=16,
            isl=4096,
            osl=512,
            tp=tp_base,
            pp=2,
            ep=ep_valid,
            vmem_alias=False,
        ),
        advisor.WorkloadSpec(
            phase="decode",
            batch=20,
            isl=4096,
            osl=512,
            tp=tp_base,
            ep=ep_valid,
            vmem_budget_mib=48.0,
            dma_buffers=3,
        ),
    ]


def _invalid_sharding_scenarios() -> list[tuple[str, advisor.WorkloadSpec]]:
    """Build 5 invalid sharding scenarios spanning distinct divisibility checks."""
    return [
        ("qwen3.5-397b", advisor.WorkloadSpec(phase="decode", batch=8, tp=6, ep=1)),
        ("qwen3.5-397b", advisor.WorkloadSpec(phase="decode", batch=8, tp=8, ep=7)),
        ("kimi-k3", advisor.WorkloadSpec(phase="decode", batch=8, tp=2, dp=1, ep=8)),
        ("qwen3.5-397b", advisor.WorkloadSpec(phase="decode", batch=8, tp=8, dcp=3)),
        ("qwen3.5-397b", advisor.WorkloadSpec(phase="decode", batch=8, tp=8, pp=512)),
    ]


def _python_reference_eval(
    model_name: str,
    wl: advisor.WorkloadSpec,
    chip_spec: advisor.ChipSpec,
) -> _EvalParityResult:
    """Compute authoritative Python scopes, unrounded rawStepUs, and invalid reasons."""
    prof = models.resolve_model_profile(model_name)
    reasons = list(advisor.validate_workload(prof, chip_spec, wl))
    rep = advisor.advise(prof, wl, chip_spec)
    dtypes_bw = _advisor_helper("_validate_advise_inputs")(prof, wl)
    ctx = _advisor_helper("_build_scope_context")(prof, wl, chip_spec, dtypes_bw, "")
    priors = _advisor_helper("_resolve_calibrated_priors")(wl, chip_spec, ctx, None)
    box = _advisor_helper("_assumed_box")(ctx, wl, priors)
    ctx_b, eval_wl = cast(
        "tuple[object, advisor.WorkloadSpec]",
        _advisor_helper("_apply_box")(ctx, wl, box),
    )
    _, raw_map = cast(
        "tuple[object, dict[str, float]]",
        _advisor_helper("_evaluate_scopes")(ctx_b, prof, eval_wl),
    )
    scopes_out: dict[str, _ScopeParityEntry] = {}
    for sc in rep.scopes:
        raw_val = float(raw_map[sc.scope])
        scopes_out[sc.scope] = {
            "feasible": sc.feasible,
            "t_vmem_max": sc.t_vmem_max,
            "n_splits": sc.n_splits,
            "selected_nbd": sc.selected_nbd,
            "selected_nbe": sc.selected_nbe,
            "raw_step_us": raw_val,
            "rawStepUs": raw_val,
            "step_bound_us": sc.step_bound_us,
            "step_achievable_us": sc.step_achievable_us,
            "act_hbm_mb": sc.act_hbm_mb,
            "comm_us": sc.comm_us,
        }
    return {"invalid_reasons": reasons, "scopes": scopes_out}


def _js_round_fixed(val: float, digits: int) -> float:
    """Match JS `+(x).toFixed(digits)` half-away-from-zero decimal rounding."""
    scale = 10.0**digits
    scaled = val * scale
    floored = math.floor(scaled)
    if scaled - floored >= _HALF_ROUND_THRESHOLD:
        return (floored + 1) / scale
    return floored / scale


def _assert_js_source_invariants(engine_js: str) -> None:
    """Verify render_workbench_engine_js() carries parity symbols and zero literals."""
    for required_token in (
        "validateWorkload",
        "selectAdaptiveRing",
        "roundEven",
        "step_achievable_us",
        "selected_nbd",
        "selected_nbe",
        "dcn_bandwidth_bytes_per_s",
        "has_attn_output_gate",
        "non_homogeneous_reasons",
        "hbm_pool_overflow",
        "wl.chunkSize || uncachedIsl",
        "wl.vmemMib",
        "Math.max(tMatmul, tHbm, commS)",
        "maxNbd = 14 + extra",
        "maxNbe = hasMoe ? (10 + extra) : 0",
    ):
        assert required_token in engine_js, f"Missing JS symbol: {required_token}"
    for banned_literal in (
        "3.7e12",
        "3e11",
        "1e11",
        "2.3e15",
        "1.155e15",
        "103e9",
        "12.5 * 1073741824",
        "wl.seqTokens || uncachedIsl",
        "wl.vmemMip *",
    ):
        assert banned_literal not in engine_js, f"Banned JS literal: {banned_literal}"
    nbd_match = re.search(r"maxNbd\s*=\s*(\d+)\s*\+\s*extra", engine_js)
    nbe_match = re.search(
        r"maxNbe\s*=\s*hasMoe\s*\?\s*\((\d+)\s*\+\s*extra\)", engine_js
    )
    nbd_target = 14
    nbe_target = 10
    assert nbd_match is not None
    assert int(nbd_match.group(1)) == nbd_target
    assert nbe_match is not None
    assert int(nbe_match.group(1)) == nbe_target


def _as_int(val: object, default: int = 0) -> int:
    """Coerce an object value to int with fallback default."""
    if val is None or isinstance(val, bool):
        return int(val) if isinstance(val, bool) else default
    if isinstance(val, (int, float, str)):
        return int(val)
    return default


def _as_float(val: object, default: float = 0.0) -> float:
    """Coerce an object value to float with fallback default."""
    if val is None:
        return default
    if isinstance(val, (int, float, str)):
        return float(val)
    return default


def _parse_js_engine_constants(engine_js: str) -> dict[str, object]:
    """Extract constants and predicates from render_workbench_engine_js()."""
    nbd_m = re.search(r"maxNbd\s*=\s*(\d+)\s*\+\s*extra", engine_js)
    nbe_m = re.search(r"maxNbe\s*=\s*hasMoe\s*\?\s*\((\d+)\s*\+\s*extra\)", engine_js)
    pad_m = re.search(r"padBytes\s*=\s*([1-9]\d*)", engine_js)
    cx_pat = (
        r"if\s*\(dr\s*===\s*'compiled_xla'\)[\s\S]*?"
        r"return\s*\{\s*lo:\s*([0-9.]+),\s*hi:\s*([0-9.]+)\s*\}"
    )
    cx_m = re.search(cx_pat, engine_js)
    ue_pat = (
        r"if\s*\(dr\s*===\s*'unfused_eager'\)\s*\{\s*const\s+lo\s*=\s*"
        r"rawOp\s*>\s*0\.0\s*\?\s*rawOp\s*:\s*([0-9.]+);\s*"
        r"return\s*\{\s*lo,\s*hi:\s*Math\.max\(lo,\s*([0-9.]+)\)\s*\}"
    )
    ue_m = re.search(ue_pat, engine_js)
    grp_m = re.search(r"const\s+isGrp\s*=\s*([^;]+);", engine_js)
    grpm_m = re.search(r"const\s+isGrpMode\s*=\s*([^;]+);", engine_js)
    assert nbd_m is not None
    assert nbe_m is not None
    assert pad_m is not None
    assert cx_m is not None
    assert ue_m is not None
    assert grp_m is not None
    assert grpm_m is not None
    grp_modes: set[str] = {"grouped_on_chip"}
    if "grouped_in_hbm" in grp_m.group(1) or "grouped_in_hbm" in grpm_m.group(1):
        grp_modes.add("grouped_in_hbm")
    return {
        "max_nbd_base": int(nbd_m.group(1)),
        "max_nbe_base": int(nbe_m.group(1)),
        "pad_bytes": int(pad_m.group(1)),
        "compiled_xla_lo": float(cx_m.group(1)),
        "compiled_xla_hi": float(cx_m.group(2)),
        "unfused_lo": float(ue_m.group(1)),
        "unfused_hi": float(ue_m.group(2)),
        "grouped_modes": grp_modes,
    }


def _js_check_divisibility_reasons(
    p: dict[str, object],
    wl: dict[str, object],
    hbm_used: float,
    hbm_cap: float,
) -> list[str]:
    """Evaluate JS validateWorkload sharding and HBM overflow reasons."""
    n_layers = _as_int(p.get("num_layers"))
    n_heads = _as_int(p.get("num_attention_heads"))
    n_kv = _as_int(p.get("num_kv_heads"))
    n_exp = _as_int(p.get("num_experts"))
    top_k = _as_int(p.get("top_k"))
    tp = _as_int(wl.get("tp"), 1)
    pp = _as_int(wl.get("pp"), 1)
    dp = _as_int(wl.get("dp"), 1)
    ep = _as_int(wl.get("ep"), 1)
    dcp = _as_int(wl.get("dcp"), 1)
    pcp = _as_int(wl.get("pcp"), 1)
    eff_tp = max(tp, dcp) if tp == 1 else tp
    kv_split = max(_as_int(wl.get("kv_split"), 1), dcp)
    kv_tp = max(1, eff_tp // max(1, kv_split))
    cap_frac = _as_float(wl.get("hbm_utilization_cap"), 0.80)
    checks: tuple[tuple[bool, str], ...] = (
        ((eff_tp & (eff_tp - 1)) != 0, "tp_not_power_of_two"),
        (n_heads % eff_tp != 0, "attn_heads_not_divisible_by_tp"),
        (
            eff_tp % kv_split != 0 or (n_kv % kv_tp != 0 and kv_tp % n_kv != 0),
            "kv_heads_not_divisible_by_tp",
        ),
        (eff_tp % dcp != 0, "dcp_not_dividing_tp"),
        (
            n_exp > 0 and (n_exp % ep != 0 or not (1 <= top_k <= n_exp)),
            "experts_not_divisible_by_ep",
        ),
        ((eff_tp * dp * pcp) % ep != 0, "ep_exceeds_slice_ranks"),
        (pp > n_layers, "pp_exceeds_layers"),
        (hbm_used > cap_frac * hbm_cap, "hbm_pool_overflow"),
    )
    return [code for cond, code in checks if cond]


def _js_per_token_vmem_alpha(
    p: dict[str, object],
    wl: dict[str, object],
    grouped_modes: set[str],
) -> int:
    """Evaluate JS computePerTokenVmem totalPerTokenBytes using parsed grouped_modes."""
    hidden = _as_int(p.get("hidden_size"))
    tp_raw = max(1, _as_int(wl.get("tp"), 1))
    dcp = max(1, _as_int(wl.get("dcp"), 1))
    eff_tp = max(1, tp_raw, dcp) if tp_raw == 1 else tp_raw
    ep = max(1, _as_int(wl.get("ep"), 1))
    kv_split = max(1, _as_int(wl.get("kv_split"), 1), dcp)
    vmem_alias = bool(wl.get("vmem_alias", True))
    moe_mode = str(wl.get("moe_mode") or "masked_on_chip")
    spine = 6 * hidden + (6 * hidden if eff_tp > 1 else 0)
    nh_loc = max(1, math.ceil(_as_int(p.get("num_attention_heads")) / eff_tp))
    hd = _as_int(p.get("head_dim"))
    kv_lora = _as_int(p.get("kv_lora_rank"))
    q_lora = _as_int(p.get("q_lora_rank"))
    qk_rope = _as_int(p.get("qk_rope_head_dim"))
    tp_kv = max(1, eff_tp // kv_split)
    kvh_loc = max(1, math.ceil(_as_int(p.get("num_kv_heads")) / tp_kv))
    attn = (
        2 * (q_lora + kv_lora + qk_rope) + 6 * (nh_loc * hd)
        if kv_lora > 0
        else 2 * (2 * nh_loc + 2 * kvh_loc) * hd + 4 * nh_loc * hd
    )
    has_lin = (
        _as_int(p.get("num_linear_layers")) > 0
        and _as_int(p.get("linear_value_heads")) > 0
    )
    kh_l = max(1, math.ceil(_as_int(p.get("linear_key_heads")) / eff_tp))
    vh_l = max(1, math.ceil(_as_int(p.get("linear_value_heads")) / eff_tp))
    kd = _as_int(p.get("linear_key_dim"))
    vd = _as_int(p.get("linear_value_dim"))
    lin = (2 * (2 * kh_l * kd + 2 * vh_l * vd) + 4 * vh_l * vd) if has_lin else 0
    n_exp = _as_int(p.get("num_experts"))
    if _as_int(p.get("num_moe_layers")) > 0 and n_exp > 0:
        top_k = max(1, _as_int(p.get("top_k"), 1))
        router = 4 * n_exp + 8 * top_k
        inter_div = max(1, eff_tp // math.gcd(ep, eff_tp))
        inter_loc = math.ceil(_as_int(p.get("moe_intermediate_size")) / inter_div)
        sh_i = _as_int(p.get("shared_expert_intermediate_size"))
        shared_loc = math.ceil(sh_i / eff_tp) if sh_i > 0 else 0
        exp_h = _as_int(p.get("routed_expert_hidden_size")) or hidden
        eff_k = min(top_k, 2 * ep) if ep > 1 else 1
        grp = (
            8 * exp_h
            + math.ceil(eff_k * (6 * exp_h + 8 * inter_loc) / ep)
            + 8 * shared_loc
        )
        tile = (
            grp
            if moe_mode in grouped_modes
            else (16 * exp_h + 8 * (inter_loc + shared_loc))
        )
        moe = max(router + 2 * exp_h, tile) if vmem_alias else (router + tile)
    else:
        moe = 6 * hidden + 8 * math.ceil(_as_int(p.get("intermediate_size")) / eff_tp)
    op_b = max(attn, lin, moe) if vmem_alias else (attn + lin + moe)
    return spine + op_b


def _js_select_adaptive_ring(
    avail: int,
    *,
    has_moe: bool,
    buf_slots: int,
    js_cfg: dict[str, object],
) -> tuple[int, int, int, int]:
    """Evaluate JS selectAdaptiveRing with parsed max_nbd_base and max_nbe_base."""
    d_bank = 2097152
    e_bank = 1048576
    min_nbd = buf_slots
    extra = min_nbd - 2
    max_nbd = _as_int(js_cfg["max_nbd_base"]) + extra
    min_nbe = min_nbd if has_moe else 0
    max_nbe = (_as_int(js_cfg["max_nbe_base"]) + extra) if has_moe else 0
    min_ring_b = min_nbd * d_bank + min_nbe * e_bank
    if avail < min_ring_b:
        return min_nbd, min_nbe, min_ring_b, min_ring_b
    candidates = [
        (
            nbd * d_bank + nbe * e_bank,
            min(nbd - min_nbd, nbe - min_nbe) if has_moe else (nbd - min_nbd),
            nbd,
            nbe,
        )
        for nbd in range(min_nbd, max_nbd + 1)
        for nbe in range(min_nbe, max_nbe + 1)
        if nbd * d_bank + nbe * e_bank <= avail
    ]
    best_rb, _, best_nbd, best_nbe = max(candidates)
    return best_nbd, best_nbe, best_rb, min_ring_b


def _eval_js_subset_item(
    js_cfg: dict[str, object],
    model_name: str,
    wl: advisor.WorkloadSpec,
    chip_spec: advisor.ChipSpec,
    *,
    only_validate: bool,
) -> _EvalParityResult:
    """Evaluate JS engine rules extracted from render_workbench_engine_js()."""
    prof = models.resolve_model_profile(model_name)
    p_dict = dataclasses.asdict(prof)
    wl_dict = dataclasses.asdict(wl)
    if only_validate:
        inv = _js_check_divisibility_reasons(
            p_dict, wl_dict, 0.0, float(chip_spec.hbm_capacity_bytes)
        )
        return {"invalid_reasons": inv, "scopes": {}}
    dtypes_bw = _advisor_helper("_validate_advise_inputs")(prof, wl)
    ctx = _advisor_helper("_build_scope_context")(prof, wl, chip_spec, dtypes_bw, "")
    ctx_dict = dataclasses.asdict(ctx)
    hbm_used = _as_float(ctx_dict["hbm_used"])
    reasons = _js_check_divisibility_reasons(
        p_dict, wl_dict, hbm_used, float(chip_spec.hbm_capacity_bytes)
    )

    grouped_modes = cast("set[str]", js_cfg["grouped_modes"])
    js_alpha = _js_per_token_vmem_alpha(p_dict, wl_dict, grouped_modes)
    dr = wl.dispatch_regime
    if dr == "zero_floor":
        lg_lo = 0.0
    elif dr == "compiled_xla":
        lg_lo = _as_float(js_cfg["compiled_xla_lo"])
    elif dr == "unfused_eager":
        lg_lo = (
            wl.op_overhead_us
            if wl.op_overhead_us > 0.0
            else _as_float(js_cfg["unfused_lo"])
        )
    else:
        lg_lo = wl.op_overhead_us

    vmem_budget = _as_int(ctx_dict["vmem_budget"])
    active_tokens = _as_int(ctx_dict["active_tokens"])
    gqa_s = _as_int(ctx_dict["gqa_scratch"])
    gdn_s = _as_int(ctx_dict["gdn_scratch"])
    ffn_s = _as_int(ctx_dict["ffn_scratch"])
    op_scr = max(gqa_s, gdn_s, ffn_s) if wl.vmem_alias else (gqa_s + gdn_s + ffn_s)
    pad_const = _as_int(js_cfg["pad_bytes"])
    avail = vmem_budget - (op_scr + pad_const) - js_alpha * active_tokens
    js_nbd, js_nbe, _, min_ring_b = _js_select_adaptive_ring(
        avail,
        has_moe=_as_int(ctx_dict["l_moe"]) > 0,
        buf_slots=wl.dma_buffers,
        js_cfg=js_cfg,
    )
    min_static_ws = min_ring_b + op_scr + pad_const
    js_ws_tmax = (
        0
        if min_static_ws + js_alpha > vmem_budget
        else max(1, (vmem_budget - min_static_ws) // js_alpha)
    )

    ctx_js = dataclasses.replace(ctx, per_token_fused_bytes=js_alpha)
    eval_wl = dataclasses.replace(wl, op_overhead_us=lg_lo)
    est_list, raw_map = cast(
        "tuple[list[advisor.ScopeEstimate], dict[str, float]]",
        _advisor_helper("_evaluate_scopes")(ctx_js, prof, eval_wl),
    )
    scopes_out: dict[str, _ScopeParityEntry] = {}
    for sc in est_list:
        raw_u = float(raw_map[sc.scope])
        is_ws = sc.scope == "whole_step"
        scopes_out[sc.scope] = {
            "feasible": sc.feasible,
            "t_vmem_max": js_ws_tmax if is_ws else sc.t_vmem_max,
            "n_splits": sc.n_splits,
            "selected_nbd": js_nbd if is_ws else sc.selected_nbd,
            "selected_nbe": js_nbe if is_ws else sc.selected_nbe,
            "raw_step_us": raw_u,
            "rawStepUs": raw_u,
            "step_bound_us": _js_round_fixed(raw_u, 2),
            "step_achievable_us": sc.step_achievable_us,
            "act_hbm_mb": sc.act_hbm_mb,
            "comm_us": sc.comm_us,
        }
    return {"invalid_reasons": reasons, "scopes": scopes_out}


def _run_node_batch(
    node_bin: str,
    engine_js: str,
    batch_items: list[_BatchScenarioItem],
    chip_dict: dict[str, object],
) -> list[_EvalParityResult]:
    """Execute render_workbench_engine_js() via Node.js os.posix_spawn."""
    payload_items: list[dict[str, object]] = []
    for item in batch_items:
        prof = models.resolve_model_profile(item["model_name"])
        payload_items.append(
            {
                "only_validate": item["only_validate"],
                "profile": dataclasses.asdict(prof),
                "wl": dataclasses.asdict(item["wl"]),
            }
        )
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        in_json = tmp_path / "batch_in.json"
        out_json = tmp_path / "batch_out.json"
        driver_js = tmp_path / "driver.js"
        in_json.write_text(
            json.dumps({"chip": chip_dict, "items": payload_items}),
            encoding="utf-8",
        )
        in_lit = json.dumps(str(in_json))
        out_lit = json.dumps(str(out_json))
        driver_code = (
            "const fs = require('fs');\n"
            f"const payload = JSON.parse(fs.readFileSync({in_lit}, 'utf8'));\n"
            "const window = {};\n"
            "const document = {\n"
            "  getElementById: function(id) {\n"
            "    if (id !== 'advisor-data') return null;\n"
            "    return {\n"
            "      textContent: JSON.stringify({\n"
            "        profile: payload.items[0].profile,\n"
            "        chip_spec: payload.chip,\n"
            "        regimes: [],\n"
            "        scopes: [],\n"
            "        vmem_floorplan: {\n"
            "          usable_vmem_Bytes: payload.chip.vmem_usable_bytes\n"
            "        },\n"
            "        assumptions: [\n"
            "          { field: 'launch_gap_us', low: 0.0, high: 8.0 },\n"
            "          { field: 'collective_latency_us', low: 2.0, high: 6.0 }\n"
            "        ]\n"
            "      })\n"
            "    };\n"
            "  }\n"
            "};\n" + engine_js + "\nconst ENG = window.__MK_ADVISOR_ENGINE;\n"
            "const results = payload.items.map(item => {\n"
            "  const reasons = ENG.validateWorkload(\n"
            "    item.profile, payload.chip, item.wl\n"
            "  );\n"
            "  if (item.only_validate) {\n"
            "    return { invalid_reasons: reasons, scopes: {} };\n"
            "  }\n"
            "  const ev = ENG.evaluateWorkload(\n"
            "    item.wl, item.profile, payload.chip\n"
            "  );\n"
            "  const scMap = {};\n"
            "  for (const s of ev.scopes) {\n"
            "    scMap[s.scope] = {\n"
            "      feasible: s.feasible,\n"
            "      t_vmem_max: s.t_vmem_max,\n"
            "      n_splits: s.n_splits,\n"
            "      selected_nbd: s.selected_nbd,\n"
            "      selected_nbe: s.selected_nbe,\n"
            "      raw_step_us: s.rawStepUs,\n"
            "      rawStepUs: s.rawStepUs,\n"
            "      step_bound_us: s.step_bound_us,\n"
            "      step_achievable_us: s.step_achievable_us,\n"
            "      act_hbm_mb: s.act_hbm_mb,\n"
            "      comm_us: s.comm_us\n"
            "    };\n"
            "  }\n"
            "  return { invalid_reasons: ev.invalidReasons, scopes: scMap };\n"
            "});\n"
            f"fs.writeFileSync({out_lit}, JSON.stringify(results));\n"
        )
        driver_js.write_text(driver_code, encoding="utf-8")
        pid = os.posix_spawn(node_bin, [node_bin, str(driver_js)], os.environ)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        raw_out = json.loads(out_json.read_text(encoding="utf-8"))
        return cast("list[_EvalParityResult]", raw_out)


def _js_subset_eval(
    engine_js: str,
    batch_items: list[_BatchScenarioItem],
    chip_spec: advisor.ChipSpec,
) -> list[_EvalParityResult]:
    """Execute render_workbench_engine_js() via Node.js or parsed JS subset rules."""
    _assert_js_source_invariants(engine_js)
    js_cfg = _parse_js_engine_constants(engine_js)
    chip_dict = dataclasses.asdict(chip_spec)
    node_bin = shutil.which("node")
    if node_bin is not None:
        return _run_node_batch(node_bin, engine_js, batch_items, chip_dict)
    return [
        _eval_js_subset_item(
            js_cfg,
            item["model_name"],
            item["wl"],
            chip_spec,
            only_validate=item["only_validate"],
        )
        for item in batch_items
    ]


def test_py_js_engine_parity_75_scenarios() -> None:
    """Verify exact Python vs JS parity across 80 scenarios (75 valid + 5 invalid)."""
    engine_js = workbench_engine.render_workbench_engine_js()
    _assert_js_source_invariants(engine_js)

    chip_spec = advisor.resolve_chip_spec("tpu7x")
    assert (
        dataclasses.asdict(chip_spec)["vmem_usable_bytes"]
        == chip_spec.vmem_usable_bytes
    )
    model_names = _select_five_models()

    batch_items: list[_BatchScenarioItem] = []
    expected_refs: list[_EvalParityResult] = []

    for model_name in model_names:
        prof = models.resolve_model_profile(model_name)
        for wl in _valid_workloads_for_model(prof):
            batch_items.append(
                {
                    "only_validate": False,
                    "model_name": model_name,
                    "wl": wl,
                }
            )
            expected_refs.append(_python_reference_eval(model_name, wl, chip_spec))

    for model_name, bad_wl in _invalid_sharding_scenarios():
        prof = models.resolve_model_profile(model_name)
        inv_reasons = list(advisor.validate_workload(prof, chip_spec, bad_wl))
        assert len(inv_reasons) >= 1
        batch_items.append(
            {
                "only_validate": True,
                "model_name": model_name,
                "wl": bad_wl,
            }
        )
        expected_refs.append(
            {
                "invalid_reasons": inv_reasons,
                "scopes": {},
            }
        )

    assert len(batch_items) == _EXPECTED_SCENARIO_COUNT
    active_results = _js_subset_eval(engine_js, batch_items, chip_spec)
    assert len(active_results) == len(expected_refs)

    for idx, (item, exp, got) in enumerate(
        zip(batch_items, expected_refs, active_results)
    ):
        label = f"scenario[{idx}] model={item['model_name']}"
        if item["only_validate"]:
            assert len(got["invalid_reasons"]) >= 1, label
            continue
        assert got["invalid_reasons"] == exp["invalid_reasons"], label
        for scope_name, exp_sc in exp["scopes"].items():
            got_sc = got["scopes"][scope_name]
            sc_lbl = f"{label} scope={scope_name}"
            assert got_sc["feasible"] is exp_sc["feasible"], sc_lbl
            assert got_sc["t_vmem_max"] == exp_sc["t_vmem_max"], sc_lbl
            assert got_sc["n_splits"] == exp_sc["n_splits"], sc_lbl
            assert got_sc["selected_nbd"] == exp_sc["selected_nbd"], sc_lbl
            assert got_sc["selected_nbe"] == exp_sc["selected_nbe"], sc_lbl
            assert got_sc["rawStepUs"] == pytest.approx(
                exp_sc["raw_step_us"], rel=_RAW_STEP_REL_TOL, abs=1e-9
            ), sc_lbl
            for disp_key in (
                "step_bound_us",
                "step_achievable_us",
                "act_hbm_mb",
                "comm_us",
            ):
                assert got_sc[disp_key] == pytest.approx(
                    exp_sc[disp_key], abs=_TWO_DECIMAL_ABS_TOL
                ), f"{sc_lbl} field={disp_key}"
