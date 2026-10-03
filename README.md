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
third machine) connect to it over TCP and exchange bursts through it:

```bash
python3 -m ota_sim.simulator --base-snr-db 15 --latency-ms 50
python3 -m ota_sim.simulator --channel fm --port mic --base-snr-db 12   # FM radio audio
```

`--distance-km` defaults to 1.0, matching `--reference-distance-km`'s
default of 1.0, so the bare defaults (and the command above) give a clean
link at exactly `--base-snr-db`. Raising `--distance-km` above
`--reference-distance-km` costs real path-loss dB on top - e.g.
`--distance-km 5` against the 1km reference costs ~14dB, so pair it with a
correspondingly higher `--base-snr-db` if you want a comparably strong
link at that distance.

### Wire format

TCP, port 9500 by default. Every message in both directions is a 4-byte
big-endian length followed by that many bytes of body:

```
first message from a peer:   its peer_id, UTF-8, at most 16 bytes
every later message:         one burst of samples
    --channel iq (default):  raw complex64 IQ samples
    --channel fm:            mono 16-bit little-endian PCM at --sample-rate (48 kHz)
```

`ota_sim.simulator.SimClient` implements the client side (`connect()`,
`send()`, plus an `on_burst` callback), so peers don't have to handle the
framing themselves. Peers need no static address configuration, so it
doesn't matter which machine each one runs on. The simulator supports
exactly two distinct peer_ids at a time. A third, unexpected peer_id is
logged and disconnected rather than silently mis-routed, and a peer that
reconnects with the same peer_id replaces its old connection.

It used to be UDP with one burst per datagram. Bursts outgrew that: one
second of 48 kHz audio is 96 KB, over UDP's 64 KB datagram limit, and
sending a burst as several datagrams would overflow a typical 208 KB
socket receive buffer. TCP has no size cap and is ordered, lossless and
flow-controlled. Burst loss is the simulated channel's job, not the
transport's.

**Airtime (FM only).** Each burst occupies the air for its real
duration:

- A peer's bursts air one after another, because a radio transmits one
  thing at a time.
- The far end receives a burst once it has fully aired, plus latency.
- The link is half-duplex: a burst is lost if the receiving peer was
  transmitting at any point while it aired.

IQ bursts have no sample rate, so they take no airtime and never collide.

## Configuration

All of `OTALink`'s and `FMLink`'s parameters are exposed as CLI flags on
`ota_sim.simulator`. See `python3 -m ota_sim.simulator -h`; the FM ones
are grouped under `--channel fm`. Key ones:

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

### Changing settings while it runs

Type commands into the running simulator's window. Each change applies
from the next burst; one already in flight finishes with the old values.

```
/show                         current settings (and the resulting nominal SNR/CNR)
/cnr 10        (or /snr 10)   SNR/CNR at the reference distance
/fading rayleigh 80           FM: none|rayleigh|rician [km/h];  IQ: /fading on|off
/speed 5                      FM: mobile speed (Doppler rate)
/distance 3   /freq 440       km, MHz
/ppm 50   /offset 500         FM: sound-card clock offset, carrier error (Hz)
/squelch off   /squelch 6     FM: squelch threshold (dB) or off
/keyup 200   /squelch-open 30   /squelch-tail 100   (ms)
/deviation 1500   /rician-k 6
/drop 0.1                     IQ: probability a burst is lost
/latency 50   /jitter 20      ms
/help                         the full list
```

Settings the two stations must agree on (FM port, channel width, sample
rate) can't be changed live. `ota_sim.control.handle_command(channel,
line)` is the same thing as a function, for driving it from other code
(a GUI, tests).

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

To use it over the network, run the relay with `--channel fm` (see
[Wire format](#wire-format)).

## Testing

```bash
python3 -m unittest discover -s tests -v
```
