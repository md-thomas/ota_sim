"""
Network-facing OTA relay service: an asyncio UDP process that sits between
two peers and relays whatever they send each other through a
SimulatedOTAChannel (see channel.py), so the two peers can be two separate
processes anywhere reachable over the network - same machine, different
machines, or the simulator on a third machine entirely.

Wire format (deliberately tiny and protocol-agnostic - this module never
looks at what's inside the payload):
    bytes[0:16]  peer_id, UTF-8, null-padded/truncated to exactly 16 bytes
    bytes[16:]   opaque payload (e.g. complex64 IQ samples)

A peer's real (host, port) is learned dynamically from the first datagram
it sends carrying its peer_id (a UDP "rendezvous" - no static peer address
configuration needed on the simulator, so it works regardless of which
machine each peer ends up running on). This simulator supports exactly two
distinct peer_ids at a time; a third, unexpected peer_id is logged and
dropped rather than silently mis-routed.

    python3 -m ota_sim.simulator --distance-km 5 --base-snr-db 15
"""
import argparse
import asyncio
import sys

import numpy as np

from .channel import OTALink

PEER_ID_LEN = 16


def encode_peer_id(peer_id):
    """Public wire-format helper: peer_id string -> fixed 16-byte,
    null-padded/truncated UTF-8 field. Any client of a running simulator
    (e.g. a station process) needs this to tag its outgoing datagrams."""
    raw = peer_id.encode('utf-8')[:PEER_ID_LEN]
    return raw.ljust(PEER_ID_LEN, b'\x00')


def decode_peer_id(raw):
    """Inverse of encode_peer_id()."""
    return raw[:PEER_ID_LEN].rstrip(b'\x00').decode('utf-8', errors='replace')


class OTASimulator(asyncio.DatagramProtocol):
    """asyncio UDP protocol that relays datagrams between exactly two
    peer_ids through the given OTALink channel model."""

    def __init__(self, channel: OTALink):
        self.channel = channel
        self.transport = None
        self._peer_addrs = {}  # peer_id -> (host, port), learned dynamically

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        if len(data) < PEER_ID_LEN:
            return
        peer_id = decode_peer_id(data[:PEER_ID_LEN])
        payload = data[PEER_ID_LEN:]

        if peer_id not in self._peer_addrs and len(self._peer_addrs) >= 2:
            print(f"[ota_sim] unexpected peer_id {peer_id!r} from {addr} - "
                  f"already tracking {list(self._peer_addrs)}, ignoring", file=sys.stderr)
            return
        is_new = peer_id not in self._peer_addrs
        self._peer_addrs[peer_id] = addr
        if is_new:
            print(f"[ota_sim] {peer_id!r} connected from {addr}", file=sys.stderr)

        if not payload:
            return  # a zero-length datagram is purely a "hello"/registration -
                     # nothing to relay, just needed to learn this peer's address.

        dest_id = next((p for p in self._peer_addrs if p != peer_id), None)
        if dest_id is None:
            print(f"[ota_sim] {peer_id!r} sent {len(payload)}B but no peer "
                  f"seen yet - dropping", file=sys.stderr)
            return

        asyncio.ensure_future(self._relay(payload, peer_id, dest_id))

    async def _relay(self, payload, src_id, dest_id):
        samples = np.frombuffer(payload, dtype=np.complex64)
        noisy, delay_s = self.channel.transmit(samples)
        if noisy is None:
            print(f"[ota_sim] {src_id!r} -> {dest_id!r}: DROPPED "
                  f"({len(samples)} samples, fading/collision)", file=sys.stderr)
            return

        await asyncio.sleep(delay_s)

        dest_addr = self._peer_addrs.get(dest_id)
        if dest_addr is None:
            return  # dest disconnected/never seen between schedule and now
        out = encode_peer_id(dest_id) + noisy.astype(np.complex64).tobytes()
        self.transport.sendto(out, dest_addr)
        print(f"[ota_sim] {src_id!r} -> {dest_id!r}: {len(samples)} samples "
              f"(delay={delay_s * 1000:.1f}ms, snr={self.channel.nominal_snr_db():.1f}dB)",
              file=sys.stderr)


async def run(channel: OTALink, listen_host='0.0.0.0', listen_port=9500):
    loop = asyncio.get_running_loop()
    transport, protocol = await loop.create_datagram_endpoint(
        lambda: OTASimulator(channel),
        local_addr=(listen_host, listen_port),
    )
    print(f"[ota_sim] listening on {listen_host}:{listen_port} "
          f"(distance={channel.distance_m / 1000:.2f}km, freq={channel.freq_hz / 1e6:.1f}MHz, "
          f"nominal_snr={channel.nominal_snr_db():.1f}dB, drop_prob={channel.drop_prob}, "
          f"latency={channel.latency_s * 1000:.0f}ms+{channel.jitter_s * 1000:.0f}ms jitter)",
          file=sys.stderr)
    try:
        await asyncio.Event().wait()
    finally:
        transport.close()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--listen-host', default='0.0.0.0',
                         help='Address to listen on (default: 0.0.0.0 - all interfaces)')
    parser.add_argument('--listen-port', type=int, default=9500,
                         help='UDP port to listen on (default: 9500)')
    parser.add_argument('--distance-km', type=float, default=1.0,
                         help='Simulated distance between the two peers, in km (default: 1.0 - '
                              'matches --reference-distance-km, so the bare defaults give a clean '
                              '~20dB link; raising this without also raising --base-snr-db adds '
                              'real path-loss dB and weakens the link)')
    parser.add_argument('--freq-mhz', type=float, default=433.0,
                         help='Carrier frequency in MHz, for path-loss calculation (default: 433.0)')
    parser.add_argument('--reference-distance-km', type=float, default=1.0,
                         help='Distance at which --base-snr-db applies (default: 1.0)')
    parser.add_argument('--base-snr-db', type=float, default=20.0,
                         help='SNR at --reference-distance-km (default: 20.0)')
    parser.add_argument('--no-fading', action='store_true',
                         help='Disable Rayleigh flat fading (path loss + AWGN only)')
    parser.add_argument('--drop-prob', type=float, default=0.02,
                         help='Probability a given burst is lost entirely (default: 0.02)')
    parser.add_argument('--latency-ms', type=float, default=50.0,
                         help='Base one-way latency in ms (default: 50.0)')
    parser.add_argument('--jitter-ms', type=float, default=20.0,
                         help='Extra random latency, uniform(0, jitter_ms) (default: 20.0)')
    parser.add_argument('--seed', type=int, default=None,
                         help='RNG seed for reproducible runs (default: unseeded)')
    args = parser.parse_args()

    channel = OTALink(
        distance_m=args.distance_km * 1000,
        freq_hz=args.freq_mhz * 1e6,
        reference_distance_m=args.reference_distance_km * 1000,
        base_snr_db=args.base_snr_db,
        fading=not args.no_fading,
        drop_prob=args.drop_prob,
        latency_s=args.latency_ms / 1000,
        jitter_s=args.jitter_ms / 1000,
        seed=args.seed,
    )
    try:
        asyncio.run(run(channel, args.listen_host, args.listen_port))
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
