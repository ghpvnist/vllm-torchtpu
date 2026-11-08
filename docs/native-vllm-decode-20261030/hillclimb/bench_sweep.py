# SPDX-License-Identifier: Apache-2.0
"""Fixed-length P/D streaming client for any engine layout.

Same request construction, arrival pacing and steady-window statistics as
../benchmark.py, generalized to the layouts of layout.sh: requests are spread
evenly over the engines, each engine is addressed through its host's API
server (and X-data-parallel-rank for hosts that run several engines).

  LAYOUT=tp32dcp8 GLOBAL_SEQS=96 python bench_sweep.py --stage tp32dcp8-s96-k3 \
      --input 131072 --output 1024 --arrival-interval 0.25

Reads NATIVE_HOST_IPS, NATIVE_BENCH_DATASET, NATIVE_BENCH_OUTPUT_DIR like benchmark.py.
Also records the spec-decode and step counters from /metrics before and after the
run, so the MTP acceptance per draft position can be derived.

Mixed context lengths and placement (mixed_lengths.py), all optional:
  BENCH_LENGTHS   per-request context length spec (default fixed:<--input>), e.g.
                  uniform:16384:131072 or lognormal:24576:1.0:2048:131072
  BENCH_ASSIGN    even (default, engine-major blocks) | count | kv | sorted | greedy |
                  server (no X-data-parallel-rank header: the host's balancer picks) |
                  hostkv (as server, hosts chosen by fewest KV tokens)
  BENCH_SEED      length draw seed (default 0)
  BENCH_REQUESTS  requests to send (default GLOBAL_SEQS, which sizes the server)
  BENCH_CAP       most requests per engine for the greedy and hostkv placements
                  (default BENCH_REQUESTS / engines; above it when the server has
                  spare slots)
  BENCH_PASSES    comma list of <D>.<assign>[.g] runs in this process, one after
                  another (lengths from BENCH_LEN_<D>), e.g. U.count,U.kv with
                  BENCH_LEN_U=uniform:16384:131072; a pass starts when every engine
                  is idle; .g = greedy, token ids kept, BENCH_GREEDY_OUTPUT (64) tokens
  BENCH_PROFILE_PASSES  pass indices that get a BENCH_PROFILE_S (2) second
                  /start_profile .. /stop_profile on every API host 3 s after all
                  requests decode
Pass k writes <stage>.p<k>.summary.json (pass 0 also <stage>.summary.json) and runs
under benchmarks/<layout>-p<k>-<id>/.
"""

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import time
import uuid
from pathlib import Path

import aiohttp
from mixed_lengths import ASSIGN_MODES, assign, draw_lengths, engine_kv

ROOT = Path(os.environ.get("NATIVE_BENCH_OUTPUT_DIR", "./benchmark-results"))
ROOT.mkdir(parents=True, exist_ok=True)
HOSTS = os.environ["NATIVE_HOST_IPS"].split(",")
LAYOUT = os.environ.get("LAYOUT", "dp8tp4")
GLOBAL_SEQS = int(os.environ.get("GLOBAL_SEQS", "96"))

# Sampling sent with every request: greedy unless BENCH_TEMPERATURE is set.
SAMPLING = {"temperature": float(os.environ.get("BENCH_TEMPERATURE", "0") or 0)}
if os.environ.get("BENCH_TOP_P"):
    SAMPLING["top_p"] = float(os.environ["BENCH_TOP_P"])
if os.environ.get("BENCH_TOP_K"):
    SAMPLING["top_k"] = int(os.environ["BENCH_TOP_K"])

# BENCH_SAVE_IDS=1 keeps every request's token ids (request-*.json) and adds their
# digest to the summary (greedy identity checks).
SAVE_IDS = os.environ.get("BENCH_SAVE_IDS") == "1"
GREEDY_OUTPUT = int(os.environ.get("BENCH_GREEDY_OUTPUT") or 64)

# Request lengths and placement (module docstring).
BENCH_LENGTHS = os.environ.get("BENCH_LENGTHS", "")
BENCH_ASSIGN = os.environ.get("BENCH_ASSIGN", "even")
BENCH_SEED = int(os.environ.get("BENCH_SEED", "0") or 0)
REQUESTS = int(os.environ.get("BENCH_REQUESTS") or GLOBAL_SEQS)
BENCH_CAP = int(os.environ.get("BENCH_CAP") or 0)
BENCH_PASSES = [p for p in os.environ.get("BENCH_PASSES", "").split(",") if p]
PROFILE_PASSES = {
    int(x) for x in os.environ.get("BENCH_PROFILE_PASSES", "").split(",") if x
}
PROFILE_S = float(os.environ.get("BENCH_PROFILE_S", "2") or 2)

# engine -> (host index, local data-parallel rank or None)
ENGINE_MAP = {
    "dp8tp4": [(i // 2, i % 2) for i in range(8)],
    "dp16tp4": [(i // 2, i % 2) for i in range(16)],
    "dp32tp4": [(i // 2, i % 2) for i in range(32)],
    "dp4tp8dcp2": [(i, 0) for i in range(4)],
    "dp4tp8": [(i, 0) for i in range(4)],
    "tp32dcp8": [(0, None)],
    "tp32dcp4": [(0, None)],
    "dp2tp16dcp4": [
        (0, None),
        (0, None),
    ],  # one API server on host 0 fronts both engines
    "tp8": [(0, None)],
    "tp8xdcp2": [(0, None)],
    "dp2xtp4": [(0, 0), (0, 1)],
}
METRIC_KEYS = [
    "spec_decode",
    "prefix_cache",
    "request_success",
    "num_preemptions",
    "num_requests_running",
    "iteration",
    "generation_tokens",
]


def percentile(v, q):
    v = sorted(v)
    return v[int((len(v) - 1) * q)] if v else None


def pass_plan():
    """[(index, label, length spec, placement, greedy)] of BENCH_PASSES or one run."""
    if not BENCH_PASSES:
        return [(0, "", BENCH_LENGTHS, BENCH_ASSIGN, False)]
    plan = []
    for k, token in enumerate(BENCH_PASSES):
        dist, mode, *flag = token.split(".")
        spec = os.environ.get(f"BENCH_LEN_{dist}")
        assert spec, f"pass {token}: BENCH_LEN_{dist} is not set"
        assert flag in ([], ["g"]), f"pass {token}: unknown flag"
        plan.append((k, token, spec, mode, flag == ["g"]))
    return plan


async def run(a):
    dataset = json.loads(Path(os.environ["NATIVE_BENCH_DATASET"]).read_text())
    engines = ENGINE_MAP[LAYOUT]
    per_engine = REQUESTS // len(engines)
    assert per_engine * len(engines) == REQUESTS, (REQUESTS, len(engines))
    plan = pass_plan()
    for _, token, spec, mode, _ in plan:
        assert mode in ASSIGN_MODES, (token, mode)
        draw_lengths(spec or f"fixed:{a.input}", 1, BENCH_SEED)
    api_hosts = sorted({HOSTS[h] for h, _ in engines})
    timeout = aiohttp.ClientTimeout(total=7200, sock_read=7200)
    ok = True
    async with aiohttp.ClientSession(
        read_bufsize=16 << 20,
        timeout=timeout,
        # No connection cap: every request of the arm must be in flight at once.
        connector=aiohttp.TCPConnector(limit=0),
    ) as client:
        for k, token, spec, mode, greedy in plan:
            if k:
                await wait_idle(client, api_hosts)
            ok &= await run_pass(
                a,
                client,
                dataset,
                engines,
                per_engine,
                api_hosts,
                k,
                token,
                spec,
                mode,
                greedy,
            )
    assert ok, "Invalid output count or request error"


async def scrape(client, host):
    async with client.get(f"http://{host}:8000/metrics") as response:
        response.raise_for_status()
        return await response.text()


async def wait_idle(client, api_hosts, limit_s=300):
    """Wait until no request runs or waits on any engine, then 5 s more."""
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < limit_s:
        busy = 0.0
        for host in api_hosts:
            for line in (await scrape(client, host)).splitlines():
                if line.startswith(
                    ("vllm:num_requests_running", "vllm:num_requests_waiting")
                ):
                    busy += float(line.rsplit(" ", 1)[1])
        if busy == 0:
            break
        await asyncio.sleep(1)
    await asyncio.sleep(5)


async def run_pass(
    a, client, dataset, engines, per_engine, api_hosts, k, token, spec, mode, greedy
):
    spec = spec or f"fixed:{a.input}"
    output = GREEDY_OUTPUT if greedy else a.output
    sampling = {"temperature": 0.0} if greedy else SAMPLING
    save_ids = SAVE_IDS or greedy
    lengths = draw_lengths(spec, REQUESTS, BENCH_SEED)
    cap = BENCH_CAP if mode in ("greedy", "hostkv") and BENCH_CAP else per_engine
    placement = assign(lengths, len(engines), mode, cap)
    hosts = sorted({h for h, _ in engines})
    if mode == "hostkv":
        local = len(engines) // len(hosts)
        host_of = [hosts[g] for g in assign(lengths, len(hosts), "greedy", cap * local)]
    else:
        host_of = [
            engines[e if e is not None else i % len(engines)][0]
            for i, e in enumerate(placement)
        ]
    run_id = f"{LAYOUT}-p{k}-" + uuid.uuid4().hex[:12]
    root = ROOT / "benchmarks" / run_id
    root.mkdir(parents=True)
    pass_info = {
        "index": k,
        "token": token,
        "lengths": spec,
        "assign": mode,
        "cap": cap,
        "seed": BENCH_SEED,
        "length_mean": statistics.mean(lengths),
        "length_min": min(lengths),
        "length_max": max(lengths),
        "planned_engine_kv": engine_kv(lengths, placement, len(engines)),
        "planned_host_kv": engine_kv(
            lengths, [hosts.index(h) for h in host_of], len(hosts)
        ),
    }
    (root / "parameters.json").write_text(
        json.dumps(
            {
                **vars(a),
                "layout": LAYOUT,
                "global_seqs": REQUESTS,
                "server_seqs": GLOBAL_SEQS,
                "pass": pass_info,
            },
            indent=2,
        )
    )

    async def metrics():
        out = {}
        for host in api_hosts:
            try:
                text = await scrape(client, host)
            except Exception as e:  # metrics are best effort
                out[host] = {"error": repr(e)}
                continue
            rows = {}
            for line in text.splitlines():
                if line.startswith("#") or not any(k in line for k in METRIC_KEYS):
                    continue
                try:
                    name, value = line.rsplit(" ", 1)
                    rows[name] = float(value)
                except ValueError:
                    continue
            out[host] = rows
        return out

    before = await metrics()
    ready = asyncio.Event()
    all_decoding = asyncio.Event()
    first_tokens = [0]
    all_events = [[] for _ in lengths]

    async def one(i):
        engine = placement[i]
        host_idx = host_of[i]
        # Without an engine the host's balancer picks one.
        local_rank = None if engine is None else engines[engine][1]
        # 128 prompts x 262144 tokens: index by request; beyond 128 take the
        # second half of a prompt so every request still has distinct content.
        # Beyond 256, later rounds take evenly spaced windows inside the prompt.
        item = dataset[i % len(dataset)]
        suffix = item["suffix_ids"]
        body_len = lengths[i] - len(suffix)
        offset = (i // len(dataset)) * body_len
        rounds = -(-REQUESTS // len(dataset))
        if rounds > 2:
            span = len(item["token_ids"]) - body_len
            offset = (i // len(dataset)) * (span // (rounds - 1))
        prompt = item["token_ids"][offset : offset + body_len] + suffix
        assert len(prompt) == lengths[i], (len(prompt), lengths[i])
        await ready.wait()
        await asyncio.sleep((i % (2 * per_engine)) * a.arrival_interval)
        start = time.perf_counter()
        row = {
            "index": i,
            "engine": engine,
            "host": host_idx,
            "local_dp": local_rank,
            "prompt_len": lengths[i],
            "start": start,
            "wall_start": time.time(),
        }
        events = all_events[i]
        payload = {
            "model": "qwen38-native-dummy",
            "prompt": prompt,
            "max_tokens": output,
            "min_tokens": output,
            "ignore_eos": True,
            **sampling,
            "stream": True,
            "stream_options": {"include_usage": True},
            "return_token_ids": True,
        }
        headers = {"X-Request-Id": f"{run_id}-{i}"}
        if local_rank is not None:
            headers["X-data-parallel-rank"] = str(local_rank)
        try:
            async with client.post(
                f"http://{HOSTS[host_idx]}:8000/v1/completions",
                json=payload,
                headers=headers,
            ) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"HTTP {response.status}: " + (await response.text())[:1500]
                    )
                async for raw in response.content:
                    line = raw.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    value = line[5:].strip()
                    if value == "[DONE]":
                        break
                    d = json.loads(value)
                    if "error" in d:
                        raise RuntimeError(str(d["error"]))
                    if d.get("usage"):
                        row["usage"] = d["usage"]
                    for c in d.get("choices", []):
                        ids = c.get("token_ids") or []
                        if c.get("finish_reason") in ["error", "abort"]:
                            raise RuntimeError(str(c))
                        if ids:
                            if not events:
                                first_tokens[0] += 1
                                if first_tokens[0] == len(lengths):
                                    all_decoding.set()
                            events.append(
                                {"time": time.perf_counter(), "tokens": len(ids)}
                            )
                            if save_ids:
                                row.setdefault("ids", []).extend(ids)
                        if c.get("finish_reason"):
                            row["finish_reason"] = c["finish_reason"]
        except Exception as e:
            row["error"] = repr(e)
        row["end"] = time.perf_counter()
        row["tokens"] = sum(e["tokens"] for e in events)
        if events:
            row["first"] = events[0]["time"]
            row["last"] = events[-1]["time"]
            row["ttft_s"] = row["first"] - start
            if row["tokens"] > events[0]["tokens"]:
                row["tpot_ms"] = (
                    (row["last"] - row["first"])
                    * 1000
                    / (row["tokens"] - events[0]["tokens"])
                )
        (root / f"request-{i}.json").write_text(
            json.dumps({"summary": row, "events": events})
        )
        print(
            json.dumps(
                {
                    "completed": i,
                    **{k: v for k, v in row.items() if k not in ("usage", "ids")},
                }
            ),
            flush=True,
        )
        return row

    async def profile():
        await all_decoding.wait()
        await asyncio.sleep(3)
        window = {"start_call": [time.time()]}
        codes = await asyncio.gather(
            *[profile_call(client, host, "start_profile") for host in api_hosts]
        )
        window["start_call"].append(time.time())
        await asyncio.sleep(PROFILE_S)
        window["stop_call"] = [time.time()]
        codes += await asyncio.gather(
            *[profile_call(client, host, "stop_profile") for host in api_hosts]
        )
        window["stop_call"].append(time.time())
        window["codes"] = codes
        return window

    tasks = [asyncio.create_task(one(i)) for i in range(len(lengths))]
    prof = asyncio.create_task(profile()) if k in PROFILE_PASSES else None
    started = time.perf_counter()
    ready.set()
    rows = await asyncio.gather(*tasks)
    profile_window = None
    if prof is not None:
        if all_decoding.is_set():
            profile_window = await prof
        else:
            prof.cancel()
    await asyncio.sleep(2)
    after = await metrics()
    result = {
        "run_id": run_id,
        "layout": LAYOUT,
        "global_seqs": REQUESTS,
        "server_seqs": GLOBAL_SEQS,
        "engines": len(engines),
        "parameters": vars(a),
        "pass": pass_info,
        "profile_window": profile_window,
        "sampling": sampling,
        "requests": rows,
        "metrics_before": before,
        "metrics_after": after,
        "elapsed_seconds": time.perf_counter() - started,
        "wall_end": time.time(),
        "all_exact_output": all(
            r["tokens"] == output and "error" not in r for r in rows
        ),
    }
    if save_ids:
        ids = json.dumps([r.get("ids", []) for r in rows]).encode()
        result["ids_sha256"] = hashlib.sha256(ids).hexdigest()[:16]
        result["requests"] = [
            {k: v for k, v in r.items() if k != "ids"} for r in result["requests"]
        ]
    tpots = [r["tpot_ms"] for r in rows if "tpot_ms" in r]
    result["full_request_mean_tpot_ms"] = statistics.mean(tpots) if tpots else None
    ttfts = [r["ttft_s"] for r in rows if "ttft_s" in r]
    result["ttft_s"] = (
        {"min": min(ttfts), "max": max(ttfts), "mean": statistics.mean(ttfts)}
        if ttfts
        else None
    )
    cached = [
        r.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens")
        for r in rows
    ]
    result["cached_tokens_hist"] = {str(c): cached.count(c) for c in set(cached)}
    if all("first" in r for r in rows):
        lo = max(r["first"] for r in rows) + 2
        hi = min(r["last"] for r in rows) - 2
        if hi > lo:
            counts = [
                sum(e["tokens"] for e in events if lo < e["time"] <= hi)
                for events in all_events
            ]
            total = sum(counts)
            tp = [1000 * (hi - lo) / n for n in counts if n]
            result["steady"] = {
                "seconds": hi - lo,
                "tokens": total,
                "tokens_per_second": total / (hi - lo),
                "amortized_tpot_ms": len(rows) * 1000 * (hi - lo) / total,
                "mean_request_tpot_ms": statistics.mean(tp),
                "p95_request_tpot_ms": percentile(tp, 0.95),
                "all_requests_progressed": all(counts),
                "min_request_tokens": min(counts),
                "max_request_tokens": max(counts),
            }
    # spec-decode acceptance from the counters, when the server exposes them
    accept = {}
    for host, rows_after in after.items():
        if not isinstance(rows_after, dict) or "error" in rows_after:
            continue
        for name, v in rows_after.items():
            if "spec_decode" in name:
                accept[f"{host} {name}"] = v - before.get(host, {}).get(name, 0.0)
    result["spec_decode_counter_deltas"] = accept
    (root / "summary.json").write_text(json.dumps(result, indent=2))
    (ROOT / f"{a.stage}.p{k}.summary.json").write_text(json.dumps(result, indent=2))
    if k == 0:
        (ROOT / (a.stage + ".summary.json")).write_text(json.dumps(result, indent=2))
    print(
        "SUMMARY",
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ["requests", "metrics_before", "metrics_after"]
            }
        ),
        flush=True,
    )
    return result["all_exact_output"]


async def profile_call(client, host, path):
    try:
        async with client.post(f"http://{host}:8000/{path}") as response:
            return response.status
    except Exception as e:  # the profile is best effort
        return repr(e)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True)
    p.add_argument("--input", type=int, default=131072)
    p.add_argument("--output", type=int, default=1024)
    p.add_argument("--arrival-interval", type=float, default=0.25)
    asyncio.run(run(p.parse_args()))
