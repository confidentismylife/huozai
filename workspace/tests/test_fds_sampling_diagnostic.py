"""Catch dose attribution, moving-node timing, and unsupported-time mistakes."""
import math
import unittest

import numpy as np

from fire_evac import fds_sampling_diagnostic as sampling


def field(co0=1e-6, co1=1e-5, co2=.0, o2=.209):
    data = {'Time': np.array([0., 1., 2., 3.])}
    for node, co in enumerate((co0, co1)):
        for species, value in [('CO', co), ('CO2', co2), ('O2', o2)]:
            data[f'N{node:03d}_{species}'] = np.full(4, value)
    return data


class SamplingDiagnostics(unittest.TestCase):
    def test_route_changes_node_at_exact_step_time(self):
        # Each of two graph moves lasts 1.25 seconds, not a rounded output row.
        route = {'departure_s': 0., 'route': [0, 1, 2]}
        result = sampling.route_breakdown(field(), route)
        expected = 1.25 * 2.7641667e-5 / 60 * (1 + 10 ** 1.036)
        self.assertAlmostEqual(result['FED'], expected, places=15)
        self.assertEqual([s['begin_s'] for s in result['segments']], [0., 1.25])
        self.assertEqual([s['end_s'] for s in result['segments']], [1.25, 2.5])

    def test_additive_components_reconstruct_co_co2_interaction_and_hypoxia(self):
        route = {'departure_s': .2, 'route': [0, 1]}
        result = sampling.route_breakdown(field(co0=.001, co2=.05, o2=.18), route)
        co = 2.7641667e-5 * 1000 ** 1.036 / 60
        multiplier = math.exp(.1903 * 5 + 2.0004) / 7.1
        hypoxia = math.exp(-8.13 + .54 * (20.9 - 18)) / 60
        self.assertAlmostEqual(result['components']['CO_base'], co * 1.25, places=14)
        self.assertAlmostEqual(result['components']['CO2_increment'], co * (multiplier - 1) * 1.25, places=14)
        self.assertAlmostEqual(result['components']['O2_deficit'], hypoxia * 1.25, places=14)
        self.assertAlmostEqual(sum(result['components'].values()), result['FED'], places=14)

    def test_gas_first_integral_matches_independent_power_law_antiderivative(self):
        data = field()
        data['N000_CO'] = data['Time'] * 4e-6
        route = {'departure_s': 0., 'route': [0, 1]}
        dose = sampling.gas_first_dose(data, route, max_step_s=.002)
        # Over 0..1.25 seconds, CO rises linearly from zero to 5 ppm.
        exact = 2.7641667e-5 / 60 * 5 ** 1.036 * 1.25 / 2.036
        self.assertAlmostEqual(dose / exact, 1., places=6)

    def test_route_outside_native_support_is_rejected(self):
        with self.assertRaises(ValueError):
            sampling.route_breakdown(field(), {'departure_s': 2., 'route': [0, 1]})
        with self.assertRaises(ValueError):
            sampling.gas_first_dose(field(), {'departure_s': -.1, 'route': [0, 1]})

    def test_invalid_fraction_is_not_hidden_by_clipping(self):
        data = field(co0=10.)
        with self.assertRaises(ValueError):
            sampling.route_breakdown(data, {'departure_s': 0., 'route': [0, 1]})


if __name__ == '__main__':
    unittest.main()
