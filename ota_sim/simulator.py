"""
Network-facing OTA relay service: an asyncio TCP server that sits between
two peers and relays whatever bursts they send each other through a
simulated channel - either an OTALink (complex baseband IQ, see channel.py)
or an FMLink (FM transceiver audio, see fm.py). The two peers can be two
separate processes anywhere reachable over the network - same machine,
different machines, or the simulator on a third machine entirely.

Wire format (deliberately tiny and protocol-agnostic - this module never
looks inside a burst beyond decoding its samples):
    every message, both directions:  4-byte big-endian length + body
    first message from a peer:       its peer_id (UTF-8, at most 16 bytes)
    every later message, both ways:  one burst of samples -
        OTALink channel: raw complex64 IQ samples
        FMLink channel:  mono 16-bit little-endian PCM audio at the
                         FMLink's sample_rate

TCP rather than UDP because bursts don't fit in datagrams: one second of
48 kHz 16-bit audio is 96 KB, over UDP's 64 KB ceiling, and a multi-
datagram burst would overflow a typical 208 KB socket receive buffer
unless paced. TCP is ordered, lossless and flow-controlled; channel loss
is the simulator's job, not the transport's. SimClient is the client half,
for peers to use instead of handling the framing themselves.

This simulator supports exactly two distinct peer_ids at a time; a third,
unexpected peer_id is logged and disconnected rather than silently
mis-routed. A peer reconnecting with the same peer_id replaces its old
connection.

Airtime: with an FMLink, each burst occupies the air for its real
duration. A peer's bursts go out one after another (a radio can only
transmit one thing at a time), the far end receives a burst only once
it has fully aired, and the link is half-duplex - a burst is lost if the
receiving peer was itself transmitting at any point while it aired. IQ
bursts have no sample rate, so they take no airtime and never collide.

    python3 -m ota_sim.simulator --distance-km 5 --base-snr-db 15
    python3 -m ota_sim.simulator --channel fm --port mic --base-snr-db 12
"""
import argparse
import asyncio
import concurrent.futures
import struct
import sys
import threading

import numpy as np

from . import control
from .channel import OTALink
from .fm import CHANNEL_PRESETS, PORT_PRESETS, FMLink

PEER_ID_LEN = 16
_LENGTH = struct.Struct('>I')
MAX_MESSAGE_BYTES = 256 * 1024 * 1024  # sanity cap against a corrupt length prefix
_INTERVAL_HISTORY_S = 120.0


def encode_peer_id(peer_id):
    """peer_id string -> fixed 16-byte, null-padded/truncated UTF-8 field."""
    raw = peer_id.encode('utf-8')[:PEER_ID_LEN]
    return raw.ljust(PEER_ID_LEN, b'\x00')


def decode_peer_id(raw):
    """Inverse of encode_peer_id()."""
    return raw[:PEER_ID_LEN].rstrip(b'\x00').decode('utf-8', errors='replace')


def encode_message(body):
    """Length-prefix one message body for the wire."""
    return _LENGTH.pack(len(body)) + body


async def read_message(reader):
    """Read one length-prefixed message body. Raises
    asyncio.IncompleteReadError at a clean or mid-message EOF."""
    (length,) = _LENGTH.unpack(await reader.readexactly(_LENGTH.size))
    if length > MAX_MESSAGE_BYTES:
        raise ValueError(f"message length {length} exceeds {MAX_MESSAGE_BYTES} - corrupt stream?")
    return await reader.readexactly(length)


# -- channel adapters: wire bytes <-> channel model -------------------------

class _IQAdapter:
    def __init__(self, link):
        self.link = link

    def process(self, payload, t0):
        samples = np.frombuffer(payload[:len(payload) - len(payload) % 8], dtype=np.complex64)
        noisy, delay_s = self.link.transmit(samples)
        if noisy is None:
            return None, 0.0, f"{len(samples)} samples, DROPPED (fading/collision)"
        note = f"{len(samples)} samples, snr={self.link.nominal_snr_db():.1f}dB"
        return noisy.astype(np.complex64).tobytes(), delay_s, note

    def airtime_s(self, payload):
        return 0.0  # IQ bursts carry no sample rate

    def describe(self):
        link = self.link
        return (f"IQ channel: distance={link.distance_m / 1000:.2f}km, freq={link.freq_hz / 1e6:.1f}MHz, "
                f"nominal_snr={link.nominal_snr_db():.1f}dB, drop_prob={link.drop_prob}, "
                f"latency={link.latency_s * 1000:.0f}ms+{link.jitter_s * 1000:.0f}ms jitter")


class _FMAdapter:
    def __init__(self, link):
        self.link = link

    def process(self, payload, t0):
        audio = np.frombuffer(payload[:len(payload) - len(payload) % 2], dtype='<i2') / 32768.0
        out, delay_s = self.link.transmit(audio, t0=t0)
        pcm = np.clip(np.round(out * 32767.0), -32768, 32767).astype('<i2').tobytes()
        info = self.link.last_info
        note = (f"{self.airtime_s(payload):.2f}s audio, cnr={info['cnr_db']:.1f}dB, "
                f"{info['phase_slips']} clicks, squelch open {100 * info['squelch_open_fraction']:.0f}%, "
                f"deepest fade {info['min_fade_db']:.1f}dB")
        return pcm, delay_s, note

    def airtime_s(self, payload):
        return (len(payload) // 2) / self.link.sample_rate

    def describe(self):
        link = self.link
        moving = f" at {link.speed_kmh:g}km/h" if link.fading != 'none' else ''
        return (f"FM channel: {link.port_name} port, {link.channel_name} channel, "
                f"{link.sample_rate}Hz 16-bit PCM, distance={link.distance_m / 1000:.2f}km, "
                f"freq={link.freq_hz / 1e6:.1f}MHz, nominal_cnr={link.nominal_cnr_db():.1f}dB, "
                f"fading={link.fading}{moving}, "
                f"latency={link.latency_s * 1000:.0f}ms+{link.jitter_s * 1000:.0f}ms jitter")


def _make_adapter(channel):
    if isinstance(channel, FMLink):
        return _FMAdapter(channel)
    if isinstance(channel, OTALink):
        return _IQAdapter(channel)
    raise TypeError(f"channel must be an OTALink or FMLink, not {type(channel).__name__}")


# -- the relay -----------------------------------------------------------------

class OTASimulator:
    """TCP relay between exactly two peers through an OTALink or FMLink.
    Use start() to listen, close() to stop; .address is the bound
    (host, port) once started."""

    def __init__(self, channel, log=None):
        self.channel = channel
        self._adapter = _make_adapter(channel)
        self._log = log or (lambda msg: print(msg, file=sys.stderr))
        self._writers = {}           # peer_id -> StreamWriter
        self._busy_until = {}        # peer_id -> loop time its TX queue frees up
        self._tx_intervals = {}      # peer_id -> [(start, end)] recent on-air time
        self._server = None
        self._tasks = set()
        self._t_start = None
        # Channel processing is CPU work (an FM burst is ~50ms/s of audio);
        # one worker keeps it off the event loop AND keeps the channel's RNG
        # use in a deterministic order.
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.address = None

    async def start(self, host='0.0.0.0', port=9500):
        self._t_start = asyncio.get_running_loop().time()
        self._server = await asyncio.start_server(self._handle_peer, host, port)
        self.address = self._server.sockets[0].getsockname()[:2]
        return self

    def close(self):
        if self._server is not None:
            self._server.close()
        for writer in self._writers.values():
            writer.close()
        for task in self._tasks:
            task.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def describe(self):
        return self._adapter.describe()

    async def _handle_peer(self, reader, writer):
        peer_id = None
        try:
            peer_id = decode_peer_id(await read_message(reader))
            if peer_id not in self._writers and len(self._writers) >= 2:
                self._log(f"[ota_sim] unexpected peer_id {peer_id!r} from "
                          f"{writer.get_extra_info('peername')} - already tracking "
                          f"{list(self._writers)}, disconnecting it")
                peer_id = None
                return
            old = self._writers.get(peer_id)
            self._writers[peer_id] = writer
            if old is not None:
                old.close()
            self._log(f"[ota_sim] {peer_id!r} connected from {writer.get_extra_info('peername')}")
            while True:
                payload = await read_message(reader)
                if not payload:
                    continue
                dest_id = next((p for p in self._writers if p != peer_id), None)
                if dest_id is None:
                    self._log(f"[ota_sim] {peer_id!r} sent {len(payload)}B but no peer "
                              f"connected yet - dropping")
                    continue
                task = asyncio.ensure_future(self._relay(payload, peer_id, dest_id))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        except ValueError as exc:
            self._log(f"[ota_sim] {peer_id!r}: {exc} - disconnecting")
        finally:
            if peer_id is not None and self._writers.get(peer_id) is writer:
                del self._writers[peer_id]
                self._log(f"[ota_sim] {peer_id!r} disconnected")
            writer.close()

    async def _relay(self, payload, src_id, dest_id):
        loop = asyncio.get_running_loop()
        now = loop.time()
        # Reserve this transmitter's air time BEFORE awaiting anything, so
        # back-to-back bursts from one peer queue up in arrival order.
        airtime_s = self._adapter.airtime_s(payload)
        start = max(now, self._busy_until.get(src_id, now))
        end = start + airtime_s
        self._busy_until[src_id] = end
        intervals = self._tx_intervals.setdefault(src_id, [])
        intervals.append((start, end))
        intervals[:] = [(s, e) for s, e in intervals if e > now - _INTERVAL_HISTORY_S]

        out, delay_s, note = await loop.run_in_executor(
            self._executor, self._adapter.process, payload, start - self._t_start)
        if out is None:
            self._log(f"[ota_sim] {src_id!r} -> {dest_id!r}: {note}")
            return
        await asyncio.sleep(max(0.0, end + delay_s - loop.time()))

        # Half-duplex: dest couldn't hear this while its own radio was keyed.
        if any(s < end and start < e for s, e in self._tx_intervals.get(dest_id, [])):
            self._log(f"[ota_sim] {src_id!r} -> {dest_id!r}: LOST - {dest_id!r} was "
                      f"transmitting while it aired ({note})")
            return
        writer = self._writers.get(dest_id)
        if writer is None:
            return  # dest disconnected between scheduling and delivery
        writer.write(encode_message(out))
        self._log(f"[ota_sim] {src_id!r} -> {dest_id!r}: {note} "
                  f"(delay={delay_s * 1000:.1f}ms after {airtime_s:.2f}s airtime)")


class SimClient:
    """Client half of the wire format, for a peer process: connect(),
    then send() bursts; every burst the simulator delivers to this peer
    is passed to on_burst(payload_bytes) from the event loop."""

    def __init__(self, peer_id, on_burst):
        self.peer_id = peer_id
        self._on_burst = on_burst
        self._writer = None
        self._reader_task = None

    async def connect(self, host, port):
        reader, self._writer = await asyncio.open_connection(host, port)
        self._writer.write(encode_message(encode_peer_id(self.peer_id)))
        await self._writer.drain()
        self._reader_task = asyncio.ensure_future(self._read_loop(reader))
        return self

    async def _read_loop(self, reader):
        try:
            while True:
                self._on_burst(await read_message(reader))
        except (asyncio.IncompleteReadError, ConnectionError):
            pass

    @property
    def connected(self):
        return self._writer is not None and not self._writer.is_closing()

    def send(self, payload):
        """Queue one burst for sending (non-blocking)."""
        if not self.connected:
            raise ConnectionError('not connected to the simulator')
        self._writer.write(encode_message(payload))

    def close(self):
        if self._reader_task is not None:
            self._reader_task.cancel()
        if self._writer is not None:
            self._writer.close()


def _start_command_reader(loop, on_line):
    """Read typed lines on a daemon thread and hand each to on_line on the
    event loop. A daemon thread rather than run_in_executor(input): a
    pending input() in an executor thread would keep asyncio.run() from
    exiting on Ctrl+C until Enter was pressed."""
    try:
        import readline  # noqa: F401 - gives input() history and line editing
    except ImportError:
        pass

    def reader():
        while True:
            try:
                line = input()
            except (EOFError, OSError):
                return
            loop.call_soon_threadsafe(on_line, line)

    threading.Thread(target=reader, daemon=True, name='ota_sim-commands').start()


async def run(simulator, listen_host='0.0.0.0', listen_port=9500, commands=True):
    """commands: read live /commands from stdin (see ota_sim.control). With
    no stdin (e.g. started in the background) the reader just ends."""
    await simulator.start(listen_host, listen_port)
    simulator._log(f"[ota_sim] listening on {listen_host}:{simulator.address[1]} (TCP) - "
                   f"{simulator.describe()}")
    if commands:
        simulator._log("[ota_sim] type /help for live settings (/show, /cnr, /fading, ...)")

        def on_line(line):
            for message in control.handle_command(simulator.channel, line):
                simulator._log(f"[ota_sim] {message}")

        _start_command_reader(asyncio.get_running_loop(), on_line)
    try:
        await asyncio.Event().wait()
    finally:
        simulator.close()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--listen-host', default='0.0.0.0',
                        help='Address to listen on (default: 0.0.0.0 - all interfaces)')
    parser.add_argument('--listen-port', type=int, default=9500,
                        help='TCP port to listen on (default: 9500)')
    parser.add_argument('--channel', choices=['iq', 'fm'], default='iq',
                        help="'iq': complex64 IQ through OTALink (default). 'fm': 16-bit PCM audio "
                             "through an FM transceiver pair (FMLink) - needs --port")
    parser.add_argument('--distance-km', type=float, default=1.0,
                        help='Simulated distance between the two peers, in km (default: 1.0 - '
                             'matches --reference-distance-km, so the bare defaults give a clean '
                             '~20dB link; raising this without also raising --base-snr-db adds '
                             'real path-loss dB and weakens the link)')
    parser.add_argument('--freq-mhz', type=float, default=433.0,
                        help='Carrier frequency in MHz, for path loss and Doppler (default: 433.0)')
    parser.add_argument('--reference-distance-km', type=float, default=1.0,
                        help='Distance at which --base-snr-db applies (default: 1.0)')
    parser.add_argument('--base-snr-db', type=float, default=20.0,
                        help='SNR at --reference-distance-km (default: 20.0). With --channel fm '
                             'this is the CNR in the receiver IF bandwidth (~10 is the FM threshold)')
    parser.add_argument('--latency-ms', type=float, default=50.0,
                        help='Base one-way latency in ms (default: 50.0)')
    parser.add_argument('--jitter-ms', type=float, default=20.0,
                        help='Extra random latency, uniform(0, jitter_ms) (default: 20.0)')
    parser.add_argument('--seed', type=int, default=None,
                        help='RNG seed for reproducible runs (default: unseeded)')

    iq = parser.add_argument_group('--channel iq only')
    iq.add_argument('--no-fading', action='store_true',
                    help='Disable Rayleigh flat fading (path loss + AWGN only)')
    iq.add_argument('--drop-prob', type=float, default=0.02,
                    help='Probability a given burst is lost entirely (default: 0.02)')

    fm = parser.add_argument_group('--channel fm only')
    fm.add_argument('--port', choices=sorted(PORT_PRESETS),
                    help="radio audio connection: 'mic' (voice band, emphasis) or 'data' (flat data port)")
    fm.add_argument('--fm-channel', choices=sorted(CHANNEL_PRESETS), default='narrow',
                    help='12.5 kHz (narrow) or 25 kHz (wide) channel (default: narrow)')
    fm.add_argument('--sample-rate', type=int, default=48000,
                    help='PCM sample rate both peers must use (default: 48000)')
    fm.add_argument('--fading', choices=['none', 'rayleigh', 'rician'], default='none')
    fm.add_argument('--speed-kmh', type=float, default=0.0, help='mobile speed, for Doppler fading')
    fm.add_argument('--rician-k-db', type=float, default=6.0)
    fm.add_argument('--deviation-hz', type=float, default=None,
                    help='deviation for a full-scale 1 kHz tone (default: 60%% of max deviation)')
    fm.add_argument('--freq-offset-hz', type=float, default=0.0)
    fm.add_argument('--clock-offset-ppm', type=float, default=0.0)
    fm.add_argument('--tx-keyup-ms', type=float, default=100.0)
    fm.add_argument('--squelch-cnr-db', type=float, default=6.0)
    fm.add_argument('--no-squelch', action='store_true', help='squelch fully open')
    fm.add_argument('--squelch-open-ms', type=float, default=20.0)
    fm.add_argument('--squelch-tail-ms', type=float, default=100.0)
    args = parser.parse_args()

    common = dict(distance_m=args.distance_km * 1000, freq_hz=args.freq_mhz * 1e6,
                  reference_distance_m=args.reference_distance_km * 1000,
                  latency_s=args.latency_ms / 1000, jitter_s=args.jitter_ms / 1000, seed=args.seed)
    if args.channel == 'fm':
        if args.port is None:
            parser.error("--channel fm needs --port mic or --port data")
        channel = FMLink(port=args.port, channel=args.fm_channel, base_cnr_db=args.base_snr_db,
                         sample_rate=args.sample_rate, audio_deviation_hz=args.deviation_hz,
                         fading=args.fading, speed_kmh=args.speed_kmh,
                         rician_k_db=args.rician_k_db, freq_offset_hz=args.freq_offset_hz,
                         clock_offset_ppm=args.clock_offset_ppm, tx_keyup_ms=args.tx_keyup_ms,
                         squelch_cnr_db=None if args.no_squelch else args.squelch_cnr_db,
                         squelch_open_ms=args.squelch_open_ms,
                         squelch_tail_ms=args.squelch_tail_ms, **common)
    else:
        channel = OTALink(base_snr_db=args.base_snr_db, fading=not args.no_fading,
                          drop_prob=args.drop_prob, **common)
    try:
        asyncio.run(run(OTASimulator(channel), args.listen_host, args.listen_port))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
