import unittest

import numpy as np

from ota_sim import FMLink, OTALink
from ota_sim.control import describe, handle_command
from ota_sim.fm import doppler_hz


def fm():
    return FMLink('mic', base_cnr_db=15, tx_keyup_ms=0, seed=1)


def iq():
    return OTALink(distance_m=1000, reference_distance_m=1000, base_snr_db=20, seed=1)


class ShowAndHelpTests(unittest.TestCase):
    def test_show_lists_every_setting_for_the_channel_type(self):
        fm_text = '\n'.join(describe(fm()))
        for name in ('cnr', 'speed', 'squelch', 'ppm', 'deviation', 'fading'):
            self.assertIn(name, fm_text)
        self.assertNotIn('drop', fm_text)
        iq_text = '\n'.join(describe(iq()))
        self.assertIn('drop', iq_text)
        self.assertNotIn('squelch', iq_text)

    def test_help_and_unknown(self):
        self.assertIn('/cnr', '\n'.join(handle_command(fm(), '/help')))
        self.assertIn('unknown command /bogus', handle_command(fm(), '/bogus')[0])
        self.assertIn('only for the IQ channel', handle_command(fm(), '/drop 0.1')[0])
        self.assertEqual(handle_command(fm(), '   '), [])


class FMSettingTests(unittest.TestCase):
    def test_cnr_change_reaches_the_next_burst(self):
        link = fm()
        handle_command(link, '/cnr 9')
        link.transmit(np.zeros(4800))
        self.assertEqual(link.last_info['cnr_db'], 9.0)

    def test_snr_and_cnr_are_aliases(self):
        link = fm()
        self.assertIn('cnr: 15 dB -> 11 dB', handle_command(link, '/snr 11')[0])
        self.assertEqual(link.base_cnr_db, 11.0)

    def test_unit_scaling(self):
        link = fm()
        handle_command(link, '/distance 2.5')
        handle_command(link, '/freq 440')
        handle_command(link, '/latency 120')
        self.assertEqual(link.distance_m, 2500.0)
        self.assertEqual(link.freq_hz, 440e6)
        self.assertAlmostEqual(link.latency_s, 0.12)

    def test_fading_with_speed_updates_doppler_live(self):
        link = fm()
        handle_command(link, '/freq 440')
        reply = handle_command(link, '/fading rayleigh 80')
        self.assertEqual(reply, ['fading: none -> rayleigh at 80 km/h'])
        self.assertEqual(link.fading, 'rayleigh')
        self.assertAlmostEqual(link._fd_hz, doppler_hz(80, 440e6))
        handle_command(link, '/speed 5')
        self.assertAlmostEqual(link._fd_hz, doppler_hz(5, 440e6))

    def test_squelch_off_and_back(self):
        link = fm()
        handle_command(link, '/squelch off')
        self.assertIsNone(link.squelch_cnr_db)
        handle_command(link, '/squelch 8')
        self.assertEqual(link.squelch_cnr_db, 8.0)

    def test_bad_values_are_rejected_without_changing_anything(self):
        link = fm()
        for line in ('/cnr ten', '/keyup -5', '/speed -1', '/deviation 0', '/fading sideways'):
            with self.subTest(line=line):
                reply = handle_command(link, line)
                self.assertTrue(reply and not reply[0].startswith(('cnr:', 'keyup:', 'speed:')))
        self.assertEqual((link.base_cnr_db, link.tx_keyup_ms, link.speed_kmh, link.fading),
                         (15, 0, 0.0, 'none'))

    def test_no_argument_reports_current_value(self):
        self.assertIn('current value 15 dB', handle_command(fm(), '/cnr')[0])


class IQSettingTests(unittest.TestCase):
    def test_iq_settings(self):
        link = iq()
        handle_command(link, '/snr 12')
        handle_command(link, '/fading off')
        handle_command(link, '/drop 0.25')
        self.assertEqual((link.base_snr_db, link.fading, link.drop_prob), (12.0, False, 0.25))
        self.assertIn('at most 1', handle_command(link, '/drop 2')[0])
        self.assertEqual(handle_command(link, '/fading rayleigh'), ['usage: /fading on|off'])


if __name__ == '__main__':
    unittest.main()
