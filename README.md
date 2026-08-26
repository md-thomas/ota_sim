# ota_sim - Simulated Over-The-Air RF Channel

`ota_sim` is a small, standalone, protocol-agnostic simulator for a
point-to-point RF link between two peers talking in complex baseband IQ
samples - the same representation a real SDR's ADC/DAC produces. It models:

- **Path loss** - free-space path loss (Friis) as a function of simulated
  distance and carrier frequency.
- **Fading** - flat Rayleigh fading applied per burst, on top of the
  path-loss-derived SNR.
- **AWGN** - additive white Gaussian noise sized to the link's resulting
  SNR.
- **Latency** - speed-of-light propagation delay plus a configurable base
  latency and random jitter.
- **Drop** - a configurable probability that a given burst is lost
  entirely (a deep fade or collision the receiver never detects).

It has no notion of frames, headers, modulation, or any particular
link-layer protocol - it only knows "a burst of IQ samples left one peer"
and "what a receiver some distance away, through this channel, would
actually see." That makes it reusable by any project that needs to test
two simulated radios talking to each other, not just one specific one.

## Two ways to use it

**As a library** (`ota_sim.channel.OTALink`) - call `.transmit(samples)`
directly in your own process/test if you don't need real network
separation between the two ends:

```python
from ota_sim import OTALink
import numpy as np

link = OTALink(distance_m=5000, freq_hz=433e6, base_snr_db=15, seed=1)
tx_samples = np.array([1+0j, -1+0j, 1+0j], dtype=complex)
rx_samples, delay_s = link.transmit(tx_samples)
# rx_samples is None if this burst was dropped entirely
```

**As a network service** (`ota_sim.simulator`) - run it as its own
process, and have two independent peer processes (anywhere reachable over
the network - same machine, different machines, or the simulator on a
third machine) send/receive UDP datagrams through it:

```bash
python3 -m ota_sim.simulator --base-snr-db 15 --latency-ms 50
```

`--distance-km` defaults to 1.0, matching `--reference-distance-km`'s
default of 1.0, so the bare defaults (and the command above) give a clean
link at exactly `--base-snr-db`. Raising `--distance-km` above
`--reference-distance-km` costs real path-loss dB on top - e.g.
`--distance-km 5` against the 1km reference costs ~14dB, so pair it with a
correspondingly higher `--base-snr-db` if you want a comparably strong
link at that distance.

### Wire format

Datagrams sent to (and received from) the simulator are:

```
bytes[0:16]   peer_id, UTF-8, null-padded/truncated to exactly 16 bytes
bytes[16:]    opaque payload (e.g. raw complex64 IQ samples)
```

The simulator learns each peer's real `(host, port)` from the first
datagram it sees carrying that peer_id - no static peer-address
configuration is needed, so it doesn't matter which machine each peer ends
up running on. It supports exactly two distinct peer_ids at a time; a
third, unexpected peer_id is logged and dropped rather than silently
mis-routed.

## Configuration

All of `OTALink`'s parameters are exposed as CLI flags on
`ota_sim.simulator` - see `python3 -m ota_sim.simulator -h`. Key ones:

- `--distance-km` / `--freq-mhz` - drives the path-loss calculation.
- `--base-snr-db` / `--reference-distance-km` - the SNR budget the link
  has at `reference-distance-km`; moving the simulated distance further
  degrades it via path loss, closer improves it. This is a simulator
  convenience, not a calibrated link-budget tool (no real transmit
  power/antenna gain/noise-figure modeling).
- `--no-fading` - disable Rayleigh fading (path loss + AWGN only).
- `--drop-prob` - probability of a full-burst loss.
- `--latency-ms` / `--jitter-ms` - the tunable delay knobs.
- `--seed` - reproducible runs.

## Testing

```bash
python3 -m unittest discover -s tests -v
```
