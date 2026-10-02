import asyncio
import unittest

import numpy as np

from ota_sim import FMLink, OTALink
from ota_sim.simulator import (OTASimulator, SimClient, decode_peer_id, encode_message,
                               encode_peer_id, read_message)

FS = 8000  # small sample rate keeps FM processing fast; still covers a narrow IF


def clean_iq_link():
    return OTALink(distance_m=1000, reference_distance_m=1000, base_snr_db=60, fading=False,
                   drop_prob=0.0, latency_s=0.01, jitter_s=0.0, seed=1)


def clean_fm_link():
    return FMLink('data', channel={'max_deviation_hz': 1000.0, 'if_bw_hz': 3000.0},
                  sample_rate=FS, base_cnr_db=40, tx_keyup_ms=0, squelch_cnr_db=None,
                  squelch_tail_ms=0, latency_s=0.0, jitter_s=0.0, seed=1)


def pcm(seconds):
    return np.zeros(int(seconds * FS), dtype='<i2').tobytes()


async def _setup(channel):
    """Simulator plus two connected clients, each recording (loop time,
    payload) for every burst it receives."""
    sim = await OTASimulator(channel, log=lambda msg: None).start('127.0.0.1', 0)
    loop = asyncio.get_running_loop()
    inbox = {'A': [], 'B': []}
    clients = {}
    for name in ('A', 'B'):
        client = SimClient(name, lambda p, n=name: inbox[n].append((loop.time(), p)))
        clients[name] = await client.connect(*sim.address)
    await asyncio.sleep(0.05)  # let both hellos register
    return sim, clients, inbox


async def _wait_for(predicate, timeout_s=5.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return False


def _teardown(sim, clients):
    for client in clients.values():
        client.close()
    sim.close()


class WireFormatTests(unittest.TestCase):
    def test_peer_id_round_trip_and_truncation(self):
        self.assertEqual(decode_peer_id(encode_peer_id('W1AW')), 'W1AW')
        self.assertEqual(len(encode_peer_id('X' * 40)), 16)

    def test_message_round_trip(self):
        async def scenario():
            reader = asyncio.StreamReader()
            reader.feed_data(encode_message(b'hello') + encode_message(b''))
            reader.feed_eof()
            self.assertEqual(await read_message(reader), b'hello')
            self.assertEqual(await read_message(reader), b'')
            with self.assertRaises(asyncio.IncompleteReadError):
                await read_message(reader)
        asyncio.run(scenario())

    def test_rejects_absurd_length(self):
        async def scenario():
            reader = asyncio.StreamReader()
            reader.feed_data(b'\xff\xff\xff\xff')
            with self.assertRaises(ValueError):
                await read_message(reader)
        asyncio.run(scenario())

    def test_rejects_unknown_channel_type(self):
        with self.assertRaises(TypeError):
            OTASimulator(object())


class IQRelayTests(unittest.TestCase):
    def test_relays_large_burst_both_ways(self):
        # 1 MB - far past the old single-UDP-datagram ceiling.
        async def scenario():
            sim, clients, inbox = await _setup(clean_iq_link())
            samples = (np.ones(131072) * (1 + 1j)).astype(np.complex64)
            clients['A'].send(samples.tobytes())
            clients['B'].send(samples[:10].tobytes())
            ok = await _wait_for(lambda: inbox['A'] and inbox['B'])
            _teardown(sim, clients)
            self.assertTrue(ok)
            received = np.frombuffer(inbox['B'][0][1], dtype=np.complex64)
            self.assertEqual(len(received), len(samples))
            self.assertLess(np.max(np.abs(received - samples)), 0.05)
            self.assertEqual(len(np.frombuffer(inbox['A'][0][1], dtype=np.complex64)), 10)
        asyncio.run(scenario())

    def test_iq_bursts_never_collide(self):
        async def scenario():
            sim, clients, inbox = await _setup(clean_iq_link())
            burst = np.ones(100, dtype=np.complex64).tobytes()
            clients['A'].send(burst)
            clients['B'].send(burst)
            ok = await _wait_for(lambda: inbox['A'] and inbox['B'])
            _teardown(sim, clients)
            self.assertTrue(ok)
        asyncio.run(scenario())


class FMRelayTests(unittest.TestCase):
    def test_delivered_after_airtime(self):
        async def scenario():
            sim, clients, inbox = await _setup(clean_fm_link())
            loop = asyncio.get_running_loop()
            sent_at = loop.time()
            clients['A'].send(pcm(0.4))
            ok = await _wait_for(lambda: inbox['B'])
            _teardown(sim, clients)
            self.assertTrue(ok)
            arrived_at, payload = inbox['B'][0]
            self.assertGreaterEqual(arrived_at - sent_at, 0.4)
            self.assertEqual(len(payload), len(pcm(0.4)))
        asyncio.run(scenario())

    def test_one_transmitter_queues_bursts_back_to_back(self):
        async def scenario():
            sim, clients, inbox = await _setup(clean_fm_link())
            loop = asyncio.get_running_loop()
            sent_at = loop.time()
            for _ in range(3):
                clients['A'].send(pcm(0.2))
            ok = await _wait_for(lambda: len(inbox['B']) == 3)
            _teardown(sim, clients)
            self.assertTrue(ok)
            arrivals = [t - sent_at for t, _ in inbox['B']]
            for i, t in enumerate(arrivals):
                self.assertGreaterEqual(t, 0.2 * (i + 1) - 0.01, arrivals)
        asyncio.run(scenario())

    def test_half_duplex_collision_loses_both(self):
        async def scenario():
            sim, clients, inbox = await _setup(clean_fm_link())
            clients['A'].send(pcm(0.3))
            clients['B'].send(pcm(0.3))
            await asyncio.sleep(0.8)
            _teardown(sim, clients)
            self.assertEqual(inbox['A'], [])
            self.assertEqual(inbox['B'], [])
        asyncio.run(scenario())

    def test_reply_after_burst_ends_gets_through(self):
        async def scenario():
            sim, clients, inbox = await _setup(clean_fm_link())
            clients['A'].send(pcm(0.2))
            self.assertTrue(await _wait_for(lambda: inbox['B']))
            clients['B'].send(pcm(0.2))
            ok = await _wait_for(lambda: inbox['A'])
            _teardown(sim, clients)
            self.assertTrue(ok)
        asyncio.run(scenario())


class PeerHandlingTests(unittest.TestCase):
    def test_third_peer_is_disconnected(self):
        async def scenario():
            sim, clients, inbox = await _setup(clean_iq_link())
            third = await SimClient('C', lambda p: None).connect(*sim.address)
            ok = await _wait_for(lambda: not third.connected or third._reader_task.done())
            third.close()
            _teardown(sim, clients)
            self.assertTrue(ok)
        asyncio.run(scenario())

    def test_reconnect_replaces_old_connection(self):
        async def scenario():
            sim, clients, inbox = await _setup(clean_iq_link())
            clients['B'].close()
            fresh = []
            clients['B'] = await SimClient('B', lambda p: fresh.append(p)).connect(*sim.address)
            await asyncio.sleep(0.05)
            clients['A'].send(np.ones(10, dtype=np.complex64).tobytes())
            ok = await _wait_for(lambda: fresh)
            _teardown(sim, clients)
            self.assertTrue(ok)
        asyncio.run(scenario())

    def test_burst_with_no_peer_is_dropped(self):
        async def scenario():
            sim = await OTASimulator(clean_iq_link(), log=lambda m: None).start('127.0.0.1', 0)
            got = []
            alone = await SimClient('A', got.append).connect(*sim.address)
            alone.send(np.ones(10, dtype=np.complex64).tobytes())
            await asyncio.sleep(0.1)
            alone.close()
            sim.close()
            self.assertEqual(got, [])
        asyncio.run(scenario())

    def test_send_before_connect_raises(self):
        with self.assertRaises(ConnectionError):
            SimClient('A', lambda p: None).send(b'x')


if __name__ == '__main__':
    unittest.main()
