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

## FM radio link model (`ota_sim.fm`)

`OTALink` adds noise to complex baseband IQ, which is what an SDR sees.
`FMLink` models what a pair of **VHF/UHF FM transceivers** does to *audio*
pushed into one radio's audio input and taken from the other's audio
output. That's the interface an audio-interfaced modem actually has. It
simulates the radio chain itself:

```
TX radio:  audio in -> full-scale clip -> TX high-pass [+ pre-emphasis]
           -> deviation gain -> deviation limiter -> splatter filter -> FM modulator
Channel:   path loss -> CNR in IF bandwidth -> time-varying fading (Doppler)
           -> TX/RX frequency offset -> AWGN
RX radio:  IF filter -> limiter/discriminator -> [de-emphasis] -> RX audio band-pass
           -> squelch -> sound-card clock offset -> full-scale clip
```

FM's characteristic behavior comes out of that chain rather than being
added on separately:

- **The threshold** (SINAD collapsing below roughly 8-10 dB CNR).
- **Threshold clicks** (2π phase slips in the discriminator).
- **Mobile picket-fencing** (from fading that varies *within* a burst).
- **Audio-path phase distortion** (all filters are minimum-phase, like
  analog ones).

Narrowband FM gives almost no FM SNR improvement: above threshold, SINAD
tracks CNR roughly dB for dB.

```python
from ota_sim import FMLink, measure_sinad
import numpy as np

link = FMLink(port='mic', channel='narrow', base_cnr_db=10, seed=1)
audio = np.sin(2 * np.pi * 1000 * np.arange(48000) / 48000)   # [-1, 1] full scale, 48 kHz
rx_audio, delay_s = link.transmit(audio)   # burst + squelch tail
print(link.last_info)   # cnr_db, phase_slips, squelch_open_fraction, min_fade_db
```

`port` is required, and neither value is the default:

- `'mic'`: the microphone/speaker path, 300-3000 Hz with 6 dB/octave
  pre- and de-emphasis.
- `'data'`: the 6-pin mini-DIN data port, roughly flat from 10 Hz to
  5 kHz with no emphasis.

`channel` is `'narrow'` (12.5 kHz, ±2.5 kHz deviation) or `'wide'`
(25 kHz, ±5 kHz). Either one can also be a dict with the same keys as
`PORT_PRESETS`/`CHANNEL_PRESETS`, for a specific radio.

Other knobs:

- `audio_deviation_hz`: the radio's mic/data gain.
- `fading` (`none`/`rayleigh`/`rician`) with `speed_kmh` and `freq_hz`
  for the Doppler rate.
- `freq_offset_hz`.
- `clock_offset_ppm`.
- `tx_keyup_ms`: audio lost at the start of every burst.
- `squelch_cnr_db` (`None` = open), `squelch_open_ms`, and
  `squelch_tail_ms`.

Unlike `OTALink`, nothing is dropped at random: a lost burst is the
result of fading or the squelch.

Offline WAV tool for listening to and inspecting the output (mono 16-bit PCM):

```bash
python3 -m ota_sim.fmtool --port mic --cnr-db 10 in.wav out.wav
python3 -m ota_sim.fmtool --port data --tone 1000 --cnr-db 8 \
    --fading rayleigh --speed-kmh 50 --freq-mhz 440 out.wav   # also reports SINAD
```

`FMLink` is not wired into the network relay (`ota_sim.simulator`) yet.
At 48 kHz, audio bursts are far larger than one UDP datagram can carry, so
that needs chunked transport first.

## Testing

```bash
python3 -m unittest discover -s tests -v
```
