"""Boundary, geometry and integration checks for the new physical-case workflow."""
import math
import unittest

import numpy as np

from fire_evac.post_safestep_physics import (
    fed_rate_per_s, integrate_rate, rectangle_union_contains,
    build_graph, solid_rectangles, case_text,
)


class DoseUnits(unittest.TestCase):
    def test_normal_air_zero_and_ppm_not_accepted_as_fraction(self):
        self.assertEqual(float(fed_rate_per_s(0, 0.0004, 0.209)), 0)
        with self.assertRaises(ValueError):
            fed_rate_per_s(1000, 0.04, 0.18)

    def test_oxygen_threshold_and_minute_to_second(self):
        self.assertEqual(float(fed_rate_per_s(0, 0, 0.20)), 0)
        # Published minute-based hypoxia formula, with 18 volume-percent oxygen.
        expected_per_minute = math.exp(-8.13 + .54 * (20.9 - 18))
        self.assertAlmostEqual(float(fed_rate_per_s(0, 0, .18)) * 60, expected_per_minute, places=14)

    def test_known_co_and_co2_multiplier(self):
        base = float(fed_rate_per_s(.001, 0, .209))
        self.assertAlmostEqual(base * 60, 2.7641667e-5 * 1000 ** 1.036, places=14)
        exposed = float(fed_rate_per_s(.001, .05, .209))
        self.assertAlmostEqual(exposed / base, math.exp(.1903 * 5 + 2.0004) / 7.1, places=13)

    def test_piecewise_integral_respects_exact_time_and_does_not_extrapolate(self):
        # 0.01 per second through t=10, then linearly to 0.03 at t=20.
        t = np.array([0, 10, 20.])
        r = np.array([.01, .01, .03])
        self.assertAlmostEqual(integrate_rate(t, r, 5, 15), .125)
        with self.assertRaises(ValueError):
            integrate_rate(t, r, 10, 21)


class GeometryChecks(unittest.TestCase):
    def test_union_and_blocked_floor_are_complementary(self):
        layout = {"domain_xyz": [4, 4, 3], "open_rectangles_xy": [[0, 4, 1, 3]],
                  "exit_centers_xy": [[0, 2], [4, 2]], "start_centers_xy": [[2, 2]]}
        blocked = solid_rectangles(layout)
        for x in np.arange(.25, 4, .5):
            for y in np.arange(.25, 4, .5):
                self.assertNotEqual(rectangle_union_contains(x, y, layout['open_rectangles_xy']),
                                    rectangle_union_contains(x, y, blocked))
        graph = build_graph(layout)
        self.assertTrue(graph['all_starts_reach_both_exits'])
        self.assertNotIn([2, 0], graph['coordinates_xy'])

    def test_mesh_input_has_unchanged_one_square_metre_burner(self):
        layout = {"domain_xyz": [4, 4, 3], "open_rectangles_xy": [[0, 4, 0, 4]],
                  "exit_centers_xy": [[0, 2], [4, 2]], "start_centers_xy": [[1, 2]],
                  "fire_centers_xy": {"F2": [2, 2]}, "extra_window_xy_plane": [2, 0]}
        for spacing in (.5, .25, .125):
            text = case_text('check', layout, 'F2', 'V1', spacing, 10, [[1, 2]])
            self.assertIn("XB=1.5,2.5,1.5,2.5,0,0", text)
            self.assertIn("TAU_Q=-150", text)
            self.assertIn("TIME_AVERAGED=.FALSE.", text)
            self.assertNotIn('HCN_YIELD=0.01', text)


if __name__ == '__main__':
    unittest.main()
