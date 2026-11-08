# SPDX-License-Identifier: Apache-2.0

import collections
import unittest

from mixed_lengths import assign, draw_lengths, engine_kv


class TestMixedLengths(unittest.TestCase):
    def test_draws_are_seeded_and_bounded(self):
        u = draw_lengths("uniform:16384:131072", 500, seed=0)
        self.assertEqual(u, draw_lengths("uniform:16384:131072", 500, seed=0))
        self.assertNotEqual(u, draw_lengths("uniform:16384:131072", 500, seed=1))
        self.assertTrue(all(16384 <= x <= 131072 for x in u))
        t = draw_lengths("lognormal:24576:1.0:2048:131072", 2000)
        self.assertTrue(all(2048 <= x <= 131072 for x in t))
        self.assertLess(sorted(t)[1000], sum(t) / len(t))  # long tail: median < mean
        self.assertEqual(draw_lengths("fixed:131072", 3), [131072] * 3)
        with self.assertRaises(ValueError):
            draw_lengths("uniform:1", 3)

    def test_equal_count_placements(self):
        lengths = draw_lengths("uniform:16384:131072", 48, seed=4)
        for mode in ("even", "count", "kv", "sorted", "greedy"):
            p = assign(lengths, 2, mode, 24)
            self.assertEqual(sorted(collections.Counter(p).values()), [24, 24], mode)
        self.assertEqual(assign(lengths, 2, "even", 24), [i // 24 for i in range(48)])
        self.assertEqual(assign(lengths, 2, "count", 24), [i % 2 for i in range(48)])
        self.assertEqual(assign(lengths, 2, "server", 24), [None] * 48)

    def test_kv_oracle_balances_and_sorted_is_worst(self):
        lengths = draw_lengths("lognormal:24576:1.0:2048:131072", 208, seed=0)

        def spread(mode):
            kv = engine_kv(lengths, assign(lengths, 8, mode, 26), 8)
            return max(kv) - sum(kv) / 8

        self.assertLess(spread("kv"), 0.01 * sum(lengths) / 8)
        self.assertGreater(spread("count"), spread("kv"))
        self.assertGreater(spread("sorted"), spread("count"))

    def test_greedy_uses_spare_slots(self):
        lengths = draw_lengths("uniform:16384:131072", 192, seed=2)

        def spread(mode, cap):
            kv = engine_kv(lengths, assign(lengths, 8, mode, cap), 8)
            return max(kv) - sum(kv) / 8

        placement = assign(lengths, 8, "greedy", 26)
        self.assertLessEqual(max(collections.Counter(placement).values()), 26)
        self.assertLess(spread("greedy", 26), 0.5 * spread("count", 24))

    def test_too_many_requests(self):
        with self.assertRaises(ValueError):
            assign([1] * 5, 2, "kv", 2)


if __name__ == "__main__":
    unittest.main()
