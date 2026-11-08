# SPDX-License-Identifier: Apache-2.0
"""Per pass of a multi-pass arm (BENCH_PASSES): device step, per-engine KV, imbalance.

    pass_report.py ARM_DIR [--log LOG ...] [--marker-log LOG ...] [--json out.json]

ARM_DIR holds arm-*.json, the server logs (pjrt_client and SYNTHETIC_DECODE_KV
lines) and the client's <stage>.p<k>.summary.json files (any depth). The window
of a pass is its all-decoding window (last first token + pad .. first last
token - pad) minus the client's profile window. Device step = mean spacing of
the once-per-step sampler program on each core of the --marker-log logs (default
all; on 4 hosts host 0's), median over cores. KV per engine = mean kv_tokens of
its SYNTHETIC_DECODE_KV lines in the window; imbalance = mean over steps of the
largest engine's KV minus the engines' mean (latest line per engine at each
core-0 step). Also the median step and the count of steps above 1.3x it (host
stalls inflate the mean). --series DIR writes one per-step CSV per pass.
"""

import argparse
import bisect
import collections
import glob
import gzip
import json
import os
import re
import statistics
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEV = re.compile(
    r"^I(\d\d)(\d\d) (\d\d):(\d\d):(\d\d)\.(\d{6})\s+\d+ tpu_pjrt_client\.cc:\d+\]\s+"
    r"\(([^)]*)\)ExecutablesComplete\(([^)]*)\)"
)
KV = re.compile(r"SYNTHETIC_DECODE_KV dp=(\d+) t=([\d.]+) reqs=(\d+) kv_tokens=(\d+)")
MARKERS = (
    "spec_bonus_and_target_tokens",
    "spec_gather_bonus_and_target_logits",
    "candidate_rejection_output",
    "candidate_verify",
    "sample_from_logits_argmax",
)


def parse_log(paths, tz, year=2026, marker_paths=None):
    sign = 1 if tz[0] == "+" else -1
    zone = timezone(timedelta(hours=int(tz[1:3]), minutes=int(tz[3:5])) * sign)
    marks = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )  # core -> marker -> [unix]
    kv = collections.defaultdict(list)  # dp -> [(t, reqs, kv, max)]
    marker_paths = set(paths if marker_paths is None else marker_paths)
    for path in paths:
        want_marks = path in marker_paths
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", errors="ignore") as f:
            for line in f:
                if "SYNTHETIC_DECODE_KV" in line:
                    m = KV.search(line)
                    if m:
                        t, reqs, tokens = m.group(2, 3, 4)
                        kv[int(m.group(1))].append((float(t), int(reqs), int(tokens)))
                    continue
                if not want_marks or "ExecutablesComplete(" not in line:
                    continue
                m = DEV.match(line)
                if not m:
                    continue
                mo, dd, hh, mi, ss, us, core, name = m.groups()
                mk = next((x for x in MARKERS if x in name), None)
                if mk is None:
                    continue
                ts = datetime(
                    year,
                    int(mo),
                    int(dd),
                    int(hh),
                    int(mi),
                    int(ss),
                    int(us),
                    tzinfo=zone,
                ).timestamp()
                marks[core][mk].append(ts)
    for d in kv.values():
        d.sort()
    return marks, kv


def windows_of(stamp):
    """Window [w0, w1] minus the profile window -> list of (a, b) intervals."""
    w0, w1, prof = stamp
    if prof is None:
        return [(w0, w1)]
    p0, p1 = prof["start_call"][0] - 0.2, prof["stop_call"][-1] + 1.0
    out = []
    if p0 > w0:
        out.append((w0, min(p0, w1)))
    if p1 < w1:
        out.append((max(p1, w0), w1))
    return [(a, b) for a, b in out if b - a > 1.0]


def step_of(times, ivals):
    """Mean marker spacing over the intervals (pooled), and the number of steps."""
    num = den = 0
    for a, b in ivals:
        t = [x for x in times if a <= x <= b]
        if len(t) >= 3:
            num += t[-1] - t[0]
            den += len(t) - 1
    return (1000 * num / den, den) if den else (None, 0)


def step_times(marks_t, ivals):
    """Marker-to-marker step durations (ms) with both markers inside one interval."""
    t = [x for x in marks_t if any(a <= x <= b for a, b in ivals)]
    return [
        1000 * (b - a)
        for a, b in zip(t, t[1:])
        if any(lo <= a and b <= hi for lo, hi in ivals)
    ]


def write_series(path, marks_t, kv, engines, ivals):
    """Per core-0 step: time, ms since the previous step, each engine's reqs and KV."""
    ts = {dp: [p[0] for p in kv.get(dp, [])] for dp in range(engines)}
    with open(path, "w") as f:
        f.write(
            "t,step_ms," + ",".join(f"reqs{dp},kv{dp}" for dp in range(engines)) + "\n"
        )
        prev = None
        for t in marks_t:
            inside = any(a <= t <= b for a, b in ivals)
            if inside and prev is not None and any(a <= prev <= b for a, b in ivals):
                cols = []
                for dp in range(engines):
                    i = bisect.bisect_right(ts[dp], t) - 1
                    cols += (
                        [str(kv[dp][i][1]), str(kv[dp][i][2])] if i >= 0 else ["", ""]
                    )
                f.write(f"{t:.4f},{1000 * (t - prev):.3f}," + ",".join(cols) + "\n")
            prev = t


def pass_rows(arm_dir, logs, pad, marker_logs=None, series_dir=None):
    meta = json.loads(Path(sorted(glob.glob(f"{arm_dir}/arm-*.json"))[0]).read_text())
    tz = meta.get("tz_offset", "+0800")
    marks, kv = parse_log(logs, tz, marker_paths=marker_logs)
    rows = []
    for sp in sorted(
        glob.glob(f"{arm_dir}/**/*.p[0-9]*.summary.json", recursive=True),
        key=lambda p: int(re.search(r"\.p(\d+)\.summary\.json$", p).group(1)),
    ):
        s = json.loads(Path(sp).read_text())
        reqs = s["requests"]
        if not all("first" in r for r in reqs):
            rows.append(
                dict(
                    index=s["pass"]["index"],
                    token=s["pass"]["token"],
                    error="incomplete",
                )
            )
            continue
        off = statistics.median(r["wall_start"] - r["start"] for r in reqs)
        w0 = max(r["first"] for r in reqs) + pad + off
        w1 = min(r["last"] for r in reqs) - pad + off
        ivals = windows_of((w0, w1, s.get("profile_window")))
        steps = {}
        for core, by in marks.items():
            mk = max(by, key=lambda k: sum(1 for x in by[k] if w0 <= x <= w1))
            st, n = step_of(by[mk], ivals)
            if st:
                steps[core] = (st, n, mk)
        engines = s["engines"]
        kv_mean, req_mean, imb = {}, {}, []
        for dp in range(engines):
            pts = [p for p in kv.get(dp, []) if any(a <= p[0] <= b for a, b in ivals)]
            if pts:
                kv_mean[dp] = statistics.mean(p[2] for p in pts)
                req_mean[dp] = statistics.mean(p[1] for p in pts)
        core0 = sorted(steps)[0] if steps else None
        if core0 and len(kv_mean) == engines:
            mk = steps[core0][2]
            ts = {dp: [p[0] for p in kv[dp]] for dp in range(engines)}
            for t in (
                x for x in marks[core0][mk] if any(a <= x <= b for a, b in ivals)
            ):
                vals = []
                for dp in range(engines):
                    i = bisect.bisect_right(ts[dp], t) - 1
                    if i < 0:
                        break
                    vals.append(kv[dp][i][2])
                if len(vals) == engines:
                    imb.append(max(vals) - sum(vals) / engines)
        if series_dir and core0:
            os.makedirs(series_dir, exist_ok=True)
            write_series(
                f"{series_dir}/p{s['pass']['index']}-{s['pass']['token']}.csv",
                marks[core0][steps[core0][2]],
                kv,
                engines,
                ivals,
            )
        med = slow = None
        if core0:
            d = step_times(marks[core0][steps[core0][2]], ivals)
            if d:
                med = statistics.median(d)
                slow = sum(1 for x in d if x > 1.3 * med)
        st = s.get("steady") or {}
        rows.append(
            dict(
                index=s["pass"]["index"],
                token=s["pass"]["token"],
                seed=s["pass"]["seed"],
                exact=s.get("all_exact_output"),
                profiled=bool(s.get("profile_window")),
                step_ms=statistics.median(v[0] for v in steps.values())
                if steps
                else None,
                step_ms_cores={c: round(v[0], 3) for c, v in sorted(steps.items())},
                steps=max((v[1] for v in steps.values()), default=0),
                tpot_ms=st.get("mean_request_tpot_ms"),
                window_s=sum(b - a for a, b in ivals),
                kv_engine={dp: round(v) for dp, v in sorted(kv_mean.items())},
                reqs_engine={dp: round(v, 2) for dp, v in sorted(req_mean.items())},
                kv_imbalance=statistics.mean(imb) if imb else None,
                step_median_ms=med,
                slow_steps=slow,
                planned_kv=s["pass"].get("planned_engine_kv"),
                length_mean=s["pass"].get("length_mean"),
            )
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm_dir")
    ap.add_argument("--log", action="append", default=None)
    ap.add_argument(
        "--marker-log",
        action="append",
        default=None,
        help="logs whose pjrt lines give the step (default: all; 4 hosts: host 0's)",
    )
    ap.add_argument("--pad", type=float, default=2.0)
    ap.add_argument("--json", default=None)
    ap.add_argument(
        "--series", default=None, help="directory for per-step CSVs (one per pass)"
    )
    a = ap.parse_args()
    logs = a.log or sorted(glob.glob(f"{a.arm_dir}/server-*.log.gz"))
    rows = pass_rows(a.arm_dir, logs, a.pad, a.marker_log, a.series)
    for r in rows:
        if "error" in r:
            print(r)
            continue
        kvs = " ".join(f"{v / 1e6:.3f}" for v in r["kv_engine"].values())
        step = r["step_ms"] or float("nan")
        tpot = r["tpot_ms"] or float("nan")
        imb = (r["kv_imbalance"] or 0) / 1e3
        reqs = list(r["reqs_engine"].values())
        print(
            f"p{r['index']:<2d} {r['token']:9s} seed={r['seed']} step={step:7.3f} ms "
            f"(n={r['steps']:4d}) tpot={tpot:7.3f} kvM=[{kvs}] imb={imb:7.1f}K "
            f"reqs={reqs} prof={int(r['profiled'])} exact={r['exact']} "
            f"median={r['step_median_ms'] or float('nan'):.3f} slow={r['slow_steps']}"
        )
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
