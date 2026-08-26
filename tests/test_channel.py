import unittest

import numpy as np

from ota_sim.channel import OTALink, free_space_path_loss_db, propagation_delay_s


class FreeSpacePathLossTests(unittest.TestCase):
    def test_doubling_distance_adds_6db(self):
        loss_1km = free_space_path_loss_db(1000, 433e6)
        loss_2km = free_space_path_loss_db(2000, 433e6)
        self.assertAlmostEqual(loss_2km - loss_1km, 20 * np.log10(2), places=6)

    def test_doubling_frequency_adds_6db(self):
        loss_433 = free_space_path_loss_db(1000, 433e6)
        loss_866 = free_space_path_loss_db(1000, 866e6)
        self.assertAlmostEqual(loss_866 - loss_433, 20 * np.log10(2), places=6)

    def test_zero_distance_does_not_explode(self):
        # floored at 1.0m internally - should be finite, not -inf/nan.
        loss = free_space_path_loss_db(0, 433e6)
        self.assertTrue(np.isfinite(loss))


class PropagationDelayTests(unittest.TestCase):
    def test_scales_linearly_with_distance(self):
        self.assertAlmostEqual(propagation_delay_s(299792458), 1.0, places=6)
        self.assertAlmostEqual(propagation_delay_s(0), 0.0, places=6)

    def test_negative_distance_clamped_to_zero(self):
        self.assertEqual(propagation_delay_s(-100), 0.0)


class OTALinkDropTests(unittest.TestCase):
    def test_drop_prob_one_always_drops(self):
        link = OTALink(distance_m=1000, drop_prob=1.0, seed=1)
        samples = np.ones(100, dtype=complex)
        received, delay_s = link.transmit(samples)
        self.assertIsNone(received)
        self.assertEqual(delay_s, 0.0)

    def test_drop_prob_zero_never_drops(self):
        link = OTALink(distance_m=1000, drop_prob=0.0, seed=1)
        samples = np.ones(50, dtype=complex)
        for _ in range(20):
            received, _ = link.transmit(samples)
            self.assertIsNotNone(received)


class OTALinkFidelityTests(unittest.TestCase):
    def test_clean_high_snr_no_fading_reproduces_samples_closely(self):
        link = OTALink(distance_m=1000, reference_distance_m=1000, base_snr_db=60,
                        fading=False, drop_prob=0.0, seed=2)
        samples = np.array([1 + 0j, -1 + 0j, 1 + 0j, -1 + 0j] * 50, dtype=complex)
        received, delay_s = link.transmit(samples)
        self.assertIsNotNone(received)
        max_err = np.max(np.abs(received - samples))
        self.assertLess(max_err, 0.05)
        self.assertGreater(delay_s, 0.0)

    def test_farther_distance_increases_noise_relative_to_signal(self):
        # Same seed, same everything except distance - the farther link's
        # received samples should deviate more from the clean originals.
        samples = np.array([1 + 0j, -1 + 0j] * 200, dtype=complex)

        near = OTALink(distance_m=100, reference_distance_m=100, base_snr_db=20,
                        fading=False, drop_prob=0.0, seed=7)
        far = OTALink(distance_m=100_000, reference_distance_m=100, base_snr_db=20,
                       fading=False, drop_prob=0.0, seed=7)

        near_rx, _ = near.transmit(samples)
        far_rx, _ = far.transmit(samples)

        near_err = np.mean(np.abs(near_rx - samples) ** 2)
        far_err = np.mean(np.abs(far_rx - samples) ** 2)
        self.assertGreater(far_err, near_err)


class OTALinkFadingTests(unittest.TestCase):
    def test_fading_magnitude_is_unbiased_on_average(self):
        # Over many trials, fading shouldn't systematically inflate or
        # deflate received power - E[fade magnitude] == 1 by construction.
        link = OTALink(distance_m=1000, reference_distance_m=1000, base_snr_db=60,
                        fading=True, drop_prob=0.0, seed=3)
        samples = np.ones(1000, dtype=complex)
        ratios = []
        for _ in range(300):
            received, _ = link.transmit(samples)
            ratios.append(np.mean(np.abs(received)) / np.mean(np.abs(samples)))
        self.assertAlmostEqual(np.mean(ratios), 1.0, delta=0.15)

    def test_fading_disabled_leaves_magnitude_unchanged(self):
        link = OTALink(distance_m=1000, reference_distance_m=1000, base_snr_db=60,
                        fading=False, drop_prob=0.0, seed=4)
        samples = np.ones(500, dtype=complex)
        received, _ = link.transmit(samples)
        self.assertAlmostEqual(np.mean(np.abs(received)), 1.0, delta=0.02)


class OTALinkDelayTests(unittest.TestCase):
    def test_delay_within_configured_bounds(self):
        link = OTALink(distance_m=1000, drop_prob=0.0, latency_s=0.05,
                        jitter_s=0.02, seed=5)
        samples = np.ones(10, dtype=complex)
        for _ in range(20):
            _, delay_s = link.transmit(samples)
            self.assertGreaterEqual(delay_s, 0.05)
            self.assertLessEqual(delay_s, 0.05 + 0.02 + 0.001)  # + tiny propagation term


class OTALinkNominalSnrTests(unittest.TestCase):
    def test_nominal_snr_matches_base_at_reference_distance(self):
        link = OTALink(distance_m=1000, reference_distance_m=1000, base_snr_db=17.5)
        self.assertAlmostEqual(link.nominal_snr_db(), 17.5, places=6)

    def test_nominal_snr_degrades_farther_than_reference(self):
        link = OTALink(distance_m=2000, reference_distance_m=1000, base_snr_db=20)
        self.assertLess(link.nominal_snr_db(), 20)

    def test_nominal_snr_improves_closer_than_reference(self):
        link = OTALink(distance_m=500, reference_distance_m=1000, base_snr_db=20)
        self.assertGreater(link.nominal_snr_db(), 20)


if __name__ == '__main__':
    unittest.main()
