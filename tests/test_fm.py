import unittest

import numpy as np

from ota_sim.fm import (CHANNEL_PRESETS, PORT_PRESETS, FMLink, doppler_hz,
                        measure_sinad)

FS = 48000
PORTS = sorted(PORT_PRESETS)


def tone(freq_hz, seconds=1.0, amplitude=1.0):
    t = np.arange(int(seconds * FS)) / FS
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


def clean_link(port, **kwargs):
    """No key-up loss, squelch open, no noise to speak of - for measuring
    the radio chain's own transfer behavior."""
    params = dict(base_cnr_db=60, tx_keyup_ms=0, squelch_cnr_db=None, seed=1)
    params.update(kwargs)
    return FMLink(port, **params)


def received_tone_db(link, freq_hz, amplitude=0.3):
    """Received amplitude of a test tone relative to the input, in dB,
    measured after filter settling."""
    out, _ = link.transmit(tone(freq_hz, seconds=1.0, amplitude=amplitude))
    segment = out[int(0.2 * FS):FS]
    return 20 * np.log10(np.sqrt(2 * np.mean(segment ** 2)) / amplitude)


def sinad_at(port, cnr_db, channel='narrow', seed=1):
    link = FMLink(port, channel=channel, base_cnr_db=cnr_db, tx_keyup_ms=0,
                  squelch_cnr_db=None, seed=seed)
    out, _ = link.transmit(tone(1000, seconds=2.0))
    return measure_sinad(out[int(0.1 * FS):2 * FS], FS, 1000), link.last_info


class MeasureSinadTests(unittest.TestCase):
    def test_known_noise_level(self):
        rng = np.random.default_rng(0)
        signal = tone(1000, seconds=2.0)  # power 0.5
        noise = rng.normal(0, np.sqrt(0.5 / 100), size=len(signal))  # 20 dB below
        sinad = measure_sinad(signal + noise, FS, 1000)
        expected = 10 * np.log10((0.5 + 0.005) / 0.005)
        self.assertAlmostEqual(sinad, expected, delta=0.3)

    def test_ignores_dc_and_phase(self):
        audio = 0.2 + np.cos(2 * np.pi * 1000 * np.arange(FS) / FS + 1.0)
        self.assertGreater(measure_sinad(audio, FS, 1000), 100)


class ConstructionTests(unittest.TestCase):
    def test_port_is_required(self):
        with self.assertRaises(TypeError):
            FMLink()

    def test_unknown_presets_rejected(self):
        with self.assertRaises(ValueError):
            FMLink('speaker')
        with self.assertRaises(ValueError):
            FMLink('mic', channel='medium')
        with self.assertRaises(ValueError):
            FMLink('mic', fading='awful')

    def test_custom_preset_dicts(self):
        link = FMLink(dict(PORT_PRESETS['data']), channel={'max_deviation_hz': 3000.0,
                                                           'if_bw_hz': 12000.0})
        self.assertEqual(link.max_deviation_hz, 3000.0)

    def test_sample_rate_must_cover_if_bandwidth(self):
        with self.assertRaises(ValueError):
            FMLink('mic', channel='wide', sample_rate=8000)

    def test_same_seed_is_deterministic(self):
        audio = tone(1000, seconds=0.5)
        a, _ = FMLink('mic', base_cnr_db=8, fading='rayleigh', speed_kmh=30, seed=7).transmit(audio)
        b, _ = FMLink('mic', base_cnr_db=8, fading='rayleigh', speed_kmh=30, seed=7).transmit(audio)
        np.testing.assert_array_equal(a, b)


class ThresholdTests(unittest.TestCase):
    """The FM threshold must EMERGE from the limiter/discriminator - these
    pin the behavior measured when the model was built."""

    def test_sinad_tracks_cnr_above_threshold(self):
        # Narrowband FM: roughly dB-for-dB above threshold (no big FM
        # improvement at modulation index < 1).
        for port in PORTS:
            with self.subTest(port=port):
                hi, _ = sinad_at(port, 20)
                lo, _ = sinad_at(port, 12)
                self.assertGreater(hi - lo, 6.5)
                self.assertLess(hi - lo, 9.5)

    def test_sinad_monotonic_in_cnr(self):
        for port in PORTS:
            with self.subTest(port=port):
                values = [sinad_at(port, c)[0] for c in (20, 15, 12, 10, 8, 6, 4)]
                self.assertEqual(values, sorted(values, reverse=True))

    def test_mic_path_threshold_knee(self):
        # Below ~8 dB CNR the mic path's SINAD falls off much faster than
        # the ~1 dB/dB it follows above threshold.
        s20, s12 = sinad_at('mic', 20)[0], sinad_at('mic', 12)[0]
        s8, s4 = sinad_at('mic', 8)[0], sinad_at('mic', 4)[0]
        self.assertGreater((s8 - s4) / 4, 1.5 * (s20 - s12) / 8)

    def test_mic_path_twelve_db_sinad_region(self):
        # De-emphasis + voice band put 12 dB SINAD a little below 10 dB CNR.
        sinad, _ = sinad_at('mic', 10)
        self.assertGreater(sinad, 14)
        self.assertLess(sinad, 19)

    def test_phase_slips_appear_only_below_threshold(self):
        for port in PORTS:
            with self.subTest(port=port):
                for cnr in (20, 12, 10):
                    self.assertEqual(sinad_at(port, cnr)[1]['phase_slips'], 0, f"at {cnr} dB")
                slips_per_s = sinad_at(port, 4)[1]['phase_slips'] / 2
                self.assertGreater(slips_per_s, 50)

    def test_wide_channel_beats_narrow(self):
        for port in PORTS:
            with self.subTest(port=port):
                self.assertGreater(sinad_at(port, 12, channel='wide')[0],
                                   sinad_at(port, 12, channel='narrow')[0] + 3)


class FrequencyResponseTests(unittest.TestCase):
    def test_unity_gain_at_reference(self):
        for port in PORTS:
            with self.subTest(port=port):
                self.assertAlmostEqual(received_tone_db(clean_link(port), 1000), 0.0, delta=0.5)

    def test_mic_voice_band(self):
        link = clean_link('mic')
        for f in (500, 700, 1500, 2000):
            self.assertAlmostEqual(received_tone_db(link, f), 0.0, delta=1.0, msg=f"{f} Hz")
        for f in (300, 3000):  # TX and RX Butterworth corners stack: ~-6 dB
            self.assertTrue(-7.5 < received_tone_db(link, f) < -4.5, f"{f} Hz")
        self.assertLess(received_tone_db(link, 100), -30)
        self.assertLess(received_tone_db(link, 5000), -25)

    def test_data_port_flat(self):
        link = clean_link('data')
        for f in (50, 100, 300, 1000, 2000, 3000):
            self.assertAlmostEqual(received_tone_db(link, f), 0.0, delta=1.0, msg=f"{f} Hz")
        self.assertLess(received_tone_db(link, 6000), -10)

    def test_mic_preemphasis_is_6db_per_octave(self):
        link = clean_link('mic')

        def peak_dev(f):
            return np.max(np.abs(link.tx_deviation_hz(tone(f, amplitude=0.3))[int(0.2 * FS):]))

        for lo, hi in ((500, 1000), (1000, 2000)):
            ratio_db = 20 * np.log10(peak_dev(hi) / peak_dev(lo))
            self.assertTrue(4.5 < ratio_db < 6.5, f"{lo}->{hi} Hz: {ratio_db:.1f} dB")

    def test_data_port_has_no_emphasis(self):
        link = clean_link('data')
        devs = [np.max(np.abs(link.tx_deviation_hz(tone(f, amplitude=0.3))[int(0.2 * FS):]))
                for f in (500, 1000, 2000)]
        self.assertLess(20 * np.log10(max(devs) / min(devs)), 0.5)


class LimiterTests(unittest.TestCase):
    def peak_fraction(self, port, gain):
        link = FMLink(port, audio_deviation_hz=gain * CHANNEL_PRESETS['narrow']['max_deviation_hz'])
        dev = link.tx_deviation_hz(tone(1000, amplitude=0.9))[int(0.2 * FS):]
        return np.max(np.abs(dev)) / link.max_deviation_hz

    def test_below_limit_is_linear(self):
        for port in PORTS:
            with self.subTest(port=port):
                self.assertAlmostEqual(self.peak_fraction(port, 1), 0.9, delta=0.02)

    def test_overdrive_saturates(self):
        # Splatter filtering after the hard clip overshoots (the mic path's
        # 3 kHz filter leaves mostly the clipped square's 4/pi fundamental),
        # but driving harder must not keep raising deviation.
        limits = {'mic': 1.3, 'data': 1.12}
        for port in PORTS:
            with self.subTest(port=port):
                p4, p8 = self.peak_fraction(port, 4), self.peak_fraction(port, 8)
                self.assertLess(p8 / p4, 1.05)
                self.assertTrue(1.0 <= p8 <= limits[port], p8)

    def test_input_clipped_at_sound_card_full_scale(self):
        link = clean_link('data')
        a = link.tx_deviation_hz(tone(1000, amplitude=1.0))
        b = link.tx_deviation_hz(tone(1000, amplitude=3.0))
        self.assertLess(np.max(np.abs(b)), 1.3 * np.max(np.abs(a)))


class FadingTests(unittest.TestCase):
    def test_rayleigh_statistics(self):
        link = FMLink('data', fading='rayleigh', speed_kmh=50, freq_hz=440e6, seed=1)
        fd = doppler_hz(50, 440e6)
        seconds = 200
        g = link.fading_gain(np.arange(seconds * 2000) / 2000)
        power = np.abs(g) ** 2
        self.assertAlmostEqual(np.mean(power), 1.0, delta=0.05)
        self.assertTrue(0.08 < np.mean(power < 0.1) < 0.11)  # P(fade > 10 dB) = 1 - e^-0.1
        r = np.abs(g) / np.sqrt(np.mean(power))
        crossings = np.sum((r[1:] >= 1) & (r[:-1] < 1)) / seconds
        expected = np.sqrt(2 * np.pi) * fd * np.exp(-1)  # level-crossing rate at rms
        self.assertAlmostEqual(crossings, expected, delta=0.15 * expected)

    def test_doppler(self):
        self.assertAlmostEqual(doppler_hz(36, 300e6), 10 * 300e6 / 299_792_458.0)

    def test_stationary_and_none_are_static(self):
        t = np.arange(10000) / 1000
        g = FMLink('data', fading='rayleigh', speed_kmh=0, seed=3).fading_gain(t)
        self.assertTrue(np.allclose(g, g[0]))
        np.testing.assert_array_equal(FMLink('data').fading_gain(t), np.ones(len(t)))

    def test_strong_rician_barely_fades(self):
        g = FMLink('data', fading='rician', rician_k_db=20, speed_kmh=50,
                   seed=1).fading_gain(np.arange(100000) / 2000)
        self.assertLess(np.max(np.abs(20 * np.log10(np.abs(g)))), 3.0)

    def test_mobile_fading_causes_clicks_within_a_burst(self):
        audio = tone(1000, seconds=2.0)
        static = FMLink('mic', base_cnr_db=15, tx_keyup_ms=0, squelch_cnr_db=None, seed=1)
        static.transmit(audio)
        mobile = FMLink('mic', base_cnr_db=15, tx_keyup_ms=0, squelch_cnr_db=None,
                        fading='rayleigh', speed_kmh=50, freq_hz=440e6, seed=1)
        mobile.transmit(audio)
        self.assertEqual(static.last_info['phase_slips'], 0)
        self.assertGreater(mobile.last_info['phase_slips'], 10)
        self.assertLess(mobile.last_info['min_fade_db'], -15)


class TimingAndSquelchTests(unittest.TestCase):
    def test_output_length_includes_squelch_tail(self):
        out, delay_s = FMLink('mic', squelch_tail_ms=150, seed=1).transmit(tone(1000, 0.5))
        self.assertEqual(len(out), int(0.5 * FS) + int(0.15 * FS))
        self.assertGreater(delay_s, 0)

    def test_keyup_and_squelch_open_delay(self):
        link = FMLink('data', base_cnr_db=40, tx_keyup_ms=100, squelch_open_ms=20, seed=1)
        out, _ = link.transmit(tone(1000, amplitude=0.5))
        first_ms = np.nonzero(out)[0][0] / FS * 1000
        self.assertGreaterEqual(first_ms, 120)
        self.assertLess(first_ms, 130)

    def test_squelch_tail_passes_noise_then_closes(self):
        link = FMLink('data', base_cnr_db=40, tx_keyup_ms=0, squelch_tail_ms=100, seed=1)
        out, _ = link.transmit(tone(1000, amplitude=0.5))
        tail = out[FS + int(0.02 * FS):]  # after the burst's last filtered audio
        self.assertGreater(np.max(np.abs(tail)), 0.05)  # squelch-tail noise is audible

    def test_squelch_stays_closed_below_threshold(self):
        out, _ = FMLink('data', base_cnr_db=3, squelch_cnr_db=6, tx_keyup_ms=0,
                        seed=1).transmit(tone(1000))
        self.assertEqual(np.max(np.abs(out)), 0.0)

    def test_open_squelch_passes_keyup_noise(self):
        out, _ = FMLink('data', base_cnr_db=20, tx_keyup_ms=100, squelch_cnr_db=None,
                        seed=1).transmit(tone(1000))
        self.assertGreater(np.std(out[:int(0.08 * FS)]), 0.1)

    def test_clock_offset_resamples(self):
        ppm = 1000
        link = clean_link('data', clock_offset_ppm=ppm, squelch_tail_ms=0)
        out, _ = link.transmit(tone(1000, seconds=1.0, amplitude=0.5))
        self.assertEqual(len(out), round(FS * (1 + ppm * 1e-6)))
        segment = out[int(0.2 * FS):FS]
        self.assertGreater(measure_sinad(segment, FS, 1000 / (1 + ppm * 1e-6)), 30)
        self.assertLess(measure_sinad(segment, FS, 1000), 10)

    def test_carrier_offset_removed_by_ac_coupling(self):
        for port in PORTS:
            with self.subTest(port=port):
                base = clean_link(port, base_cnr_db=30)
                shifted = clean_link(port, base_cnr_db=30, freq_offset_hz=2000)
                a, _ = base.transmit(tone(1000, amplitude=0.5))
                b, _ = shifted.transmit(tone(1000, amplitude=0.5))
                seg = slice(int(0.2 * FS), FS)
                self.assertLess(abs(np.mean(b[seg])), 1e-3)
                self.assertAlmostEqual(measure_sinad(b[seg], FS, 1000),
                                       measure_sinad(a[seg], FS, 1000), delta=1.0)

    def test_path_loss_lowers_cnr(self):
        near = FMLink('mic', distance_m=1000, reference_distance_m=1000, base_cnr_db=20)
        far = FMLink('mic', distance_m=2000, reference_distance_m=1000, base_cnr_db=20)
        self.assertAlmostEqual(near.nominal_cnr_db() - far.nominal_cnr_db(), 20 * np.log10(2), places=6)


if __name__ == '__main__':
    unittest.main()
