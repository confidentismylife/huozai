import unittest

import numpy as np

from fire_evac import fds_cell_stencil as stencil


class CellStencil(unittest.TestCase):
    def test_eight_support_centers_bracket_nominal_vertex_at_both_spacings(self):
        for dx in (.25, .125):
            points = stencil.support_centers([21., 4., 1.5], dx)
            self.assertEqual(len(points), 8)
            self.assertEqual(set(p[0] for p in points), {21-dx/2, 21+dx/2})
            self.assertEqual(set(p[1] for p in points), {4-dx/2, 4+dx/2})
            self.assertEqual(set(p[2] for p in points), {1.5-dx/2, 1.5+dx/2})

    def test_wrong_vertex_alignment_is_rejected(self):
        with self.assertRaises(ValueError):
            stencil.support_centers([.5, .5, .5], .2)

    def test_uniform_gases_and_twenty_forty_step_reconstruct_known_values(self):
        data = {'Time': np.array([0., .1])}
        for corner in range(8):
            for label, value in [('CO', .001), ('CO2', .04), ('O2', .18), ('T', 20 if corner < 4 else 40)]:
                data[f'C005_{corner}_{label}'] = np.full(2, value)
        result = stencil.reconstruct(data, [5])
        np.testing.assert_allclose(result['S005_CO'], [.001, .001], rtol=0, atol=1e-17)
        np.testing.assert_allclose(result['S005_T'], [30., 30.], rtol=0, atol=1e-14)

    def test_missing_or_invalid_support_is_never_filled(self):
        data = {'Time': np.array([0., .1])}
        with self.assertRaises(ValueError):
            stencil.reconstruct(data, [5])
        for corner in range(8):
            for label, value in [('CO', .001), ('CO2', .04), ('O2', .18), ('T', 30)]:
                data[f'C005_{corner}_{label}'] = np.full(2, value)
        data['C005_3_CO'][1] = 10.
        with self.assertRaises(ValueError):
            stencil.reconstruct(data, [5])


if __name__ == '__main__':
    unittest.main()
