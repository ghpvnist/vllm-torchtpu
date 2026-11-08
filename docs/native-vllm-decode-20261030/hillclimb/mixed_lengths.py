# SPDX-License-Identifier: Apache-2.0
"""Per-request context lengths and engine placement for the decode benchmark.

Lengths (``draw_lengths``), one spec string, seeded so every pass and every pool
draws the same requests:
  fixed:N                      every request N tokens
  uniform:LO:HI                integers uniform in [LO, HI]
  lognormal:MEDIAN:SIGMA:LO:HI exp(N(ln MEDIAN, SIGMA)) rounded, clipped to [LO, HI]

Placements (``assign``), requests in arrival order, ``cap`` requests per engine:
  even    engine-major blocks (request i -> engine i // cap), the original spread
  count   round robin by arrival, what a count-based balancer does
  kv      offline: longest first to the engine with the fewest KV tokens among
          engines below ``cap`` (equal counts, near-equal KV: the oracle bound)
  sorted  longest requests on the lowest engines, equal counts (worst case)
  greedy  online: each arrival to the engine with the fewest KV tokens among
          engines below ``cap``
  server  no placement (None): the API server's balancer picks the engine
  hostkv  no engine placement (None); the client sends each request to the host
          with the fewest KV tokens among hosts below cap x local engines (online,
          as a KV-aware router would), and that host's balancer picks the engine
"""

import math
import random

ASSIGN_MODES = ("even", "count", "kv", "sorted", "greedy", "server", "hostkv")


def draw_lengths(spec: str, n: int, seed: int = 0) -> list[int]:
    kind, *args = spec.split(":")
    rng = random.Random(f"{spec}/{seed}")
    if kind == "fixed" and len(args) == 1:
        return [int(args[0])] * n
    if kind == "uniform" and len(args) == 2:
        lo, hi = int(args[0]), int(args[1])
        return [rng.randint(lo, hi) for _ in range(n)]
    if kind == "lognormal" and len(args) == 4:
        median, sigma, lo, hi = (
            float(args[0]),
            float(args[1]),
            int(args[2]),
            int(args[3]),
        )
        return [
            min(hi, max(lo, round(median * math.exp(rng.gauss(0.0, sigma)))))
            for _ in range(n)
        ]
    raise ValueError(f"bad length spec {spec!r}")


def assign(lengths: list[int], engines: int, mode: str, cap: int) -> list[int | None]:
    n = len(lengths)
    if n > engines * cap:
        raise ValueError(f"{n} requests exceed {engines} engines x {cap}")
    if mode in ("server", "hostkv"):
        return [None] * n
    if mode == "even":
        return [i // cap for i in range(n)]
    if mode == "count":
        return [i % engines for i in range(n)]
    if mode == "sorted":
        order = sorted(range(n), key=lambda i: -lengths[i])
        out = [0] * n
        for rank, i in enumerate(order):
            out[i] = rank // cap
        return out
    if mode in ("kv", "greedy"):
        order = (
            range(n)
            if mode == "greedy"
            else sorted(range(n), key=lambda i: -lengths[i])
        )
        kv, cnt, out = [0] * engines, [0] * engines, [0] * n
        for i in order:
            e = min(
                (x for x in range(engines) if cnt[x] < cap), key=lambda x: (kv[x], x)
            )
            kv[e] += lengths[i]
            cnt[e] += 1
            out[i] = e
        return out
    raise ValueError(f"bad placement {mode!r}; one of {ASSIGN_MODES}")


def engine_kv(
    lengths: list[int], placement: list[int | None], engines: int
) -> list[int]:
    kv = [0] * engines
    for length, e in zip(lengths, placement):
        if e is not None:
            kv[e] += length
    return kv
