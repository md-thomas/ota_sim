"""
Simulated narrowband FM voice/data radio link: what a pair of VHF/UHF FM
transceivers does to AUDIO pushed into one radio's audio input and taken
from the other's audio output.

Unlike channel.py's OTALink (which adds AWGN directly to complex baseband
IQ), this simulates the actual radio chain and lets FM's characteristic
behavior emerge from it rather than bolting effects on separately:

    TX radio:  audio in -> sound-card full-scale clip -> TX audio high-pass
               [+ pre-emphasis] -> deviation gain -> deviation limiter
               (hard clip) -> splatter low-pass -> FM modulator
    Channel:   path loss -> carrier-to-noise ratio in the IF bandwidth ->
               time-varying fading (Doppler) -> TX/RX frequency offset ->
               AWGN
    RX radio:  IF channel filter -> limiter/discriminator -> [de-emphasis]
               -> RX audio band-pass -> squelch -> sound-card clock offset
               -> full-scale clip

The FM threshold (audio SINAD collapsing below ~10 dB CNR) and threshold
"clicks" (2*pi phase slips in the discriminator) are NOT modeled as
separate effects - they fall out of running a real limiter/discriminator
on a noisy, filtered carrier. Likewise mobile flutter/picket-fencing falls
out of time-varying fading inside a burst, and the audio passband's phase
distortion falls out of using minimum-phase (analog-like) filters rather
than linear-phase ones.

Narrowband FM gives almost no FM SNR improvement (modulation index well
under 1 for a 3 kHz audio bandwidth): above threshold, audio SINAD sits
only a couple of dB above CNR. That is real, not a model artifact.

Samples are real-valued audio floats in [-1, 1] (sound-card full scale) at
a fixed sample rate (48 kHz default). The RF side is simulated as complex
baseband at the same rate - +/-24 kHz comfortably covers even a 25 kHz
channel's IF bandwidth, so no resampling stage is needed.

numpy-only, same as channel.py: all filters are designed here from a
target magnitude response (Butterworth-shaped band edges, 6 dB/octave
emphasis) via the cepstral minimum-phase method and applied with FFT
convolution.
"""
import numpy as np

from .channel import free_space_path_loss_db, propagation_delay_s

SPEED_OF_LIGHT_M_S = 299_792_458.0

# RF channel presets (TIA-603 style): peak deviation and the receiver's IF
# channel-filter bandwidth (two-sided, i.e. total width centered on the
# carrier).
CHANNEL_PRESETS = {
    'narrow': {'max_deviation_hz': 2500.0, 'if_bw_hz': 11000.0},  # 12.5 kHz channel
    'wide':   {'max_deviation_hz': 5000.0, 'if_bw_hz': 16000.0},  # 25 kHz channel
}

# Audio-port presets. Each band edge is (corner_hz, butterworth_order).
# 'mic': the microphone/speaker audio path - voice-band filtering, pre/de-
# emphasis, and a steep RX high-pass (that's what keeps CTCSS tones out of
# the speaker). 'data': the 6-pin mini-DIN "9600" data port - taps the
# modulator/discriminator directly, so no emphasis, ~flat from a few tens
# of Hz to ~5 kHz.
PORT_PRESETS = {
    'mic': {
        'tx_highpass': (300.0, 3), 'tx_lowpass': (3000.0, 4),
        'rx_highpass': (300.0, 6), 'rx_lowpass': (3000.0, 4),
        'emphasis': True,
    },
    'data': {
        'tx_highpass': (10.0, 1), 'tx_lowpass': (6000.0, 2),
        'rx_highpass': (10.0, 1), 'rx_lowpass': (5000.0, 2),
        'emphasis': False,
    },
}

# +6 dB/octave pre-emphasis (750 us time constant -> 212 Hz corner), with an
# upper corner well above the voice band to keep it bounded. Normalized to
# 0 dB at EMPHASIS_REF_HZ, the conventional FM reference frequency, so a
# 1 kHz tone sees the same deviation with or without emphasis.
EMPHASIS_LOW_CORNER_HZ = 1 / (2 * np.pi * 750e-6)
EMPHASIS_HIGH_CORNER_HZ = 6000.0
EMPHASIS_REF_HZ = 1000.0

# Standard FM test-tone level: a full-scale (amplitude 1.0) tone at
# EMPHASIS_REF_HZ deviates the carrier by this fraction of max deviation,
# leaving limiter headroom for peaks above that.
DEFAULT_AUDIO_DEVIATION_FRACTION = 0.6

_AUDIO_FILTER_TAPS = 8192
_IF_FILTER_TAPS = 512
_FADING_SINUSOIDS = 64
# transmit() evaluates fading on a coarse grid and interpolates: the process
# only changes at the Doppler rate, and per-audio-sample evaluation would
# need a (samples x sinusoids) array - ~250 MB for a 5 s burst.
_FADING_POINTS_PER_DOPPLER_CYCLE = 50
_FADING_MAX_STEP_S = 0.001
_SQUELCH_BLOCK_S = 0.001
_SQUELCH_AVG_BLOCKS = 5
_SQUELCH_HYSTERESIS_DB = 2.0


# -- filter design / application ------------------------------------------

def _butter_lowpass_mag(f, corner_hz, order):
    return 1.0 / np.sqrt(1.0 + (np.abs(f) / corner_hz) ** (2 * order))


def _butter_highpass_mag(f, corner_hz, order):
    f = np.abs(f)
    with np.errstate(divide='ignore'):
        ratio = np.where(f > 0, corner_hz / np.maximum(f, 1e-12), np.inf)
    return 1.0 / np.sqrt(1.0 + ratio ** (2 * order))


def _preemphasis_mag(f):
    def raw(x):
        x = np.abs(x)
        return (np.sqrt(1 + (x / EMPHASIS_LOW_CORNER_HZ) ** 2) /
                np.sqrt(1 + (x / EMPHASIS_HIGH_CORNER_HZ) ** 2))
    return raw(f) / raw(EMPHASIS_REF_HZ)


def min_phase_fir(magnitude_fn, sample_rate, num_taps):
    """Minimum-phase FIR taps approximating magnitude_fn(freq_hz) - the
    same phase behavior an analog (or IIR) filter with that magnitude
    response has, which is what a real radio's audio and IF filters are.
    A linear-phase FIR would hide exactly the group-delay distortion a
    modem's equalizer has to deal with. Cepstral (homomorphic) method:
    fold the real cepstrum of log|H| onto positive quefrency."""
    n_fft = 1 << int(np.ceil(np.log2(num_taps * 16)))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    log_mag = np.log(np.maximum(magnitude_fn(freqs), 1e-6))
    cepstrum = np.fft.irfft(log_mag, n_fft)
    fold = np.zeros(n_fft)
    fold[0] = 1.0
    fold[1:n_fft // 2] = 2.0
    fold[n_fft // 2] = 1.0
    taps = np.fft.irfft(np.exp(np.fft.rfft(cepstrum * fold)), n_fft)[:num_taps]
    # Taper the last 10% so truncation doesn't add ripple.
    taper_len = max(num_taps // 10, 1)
    taps[-taper_len:] *= 0.5 * (1 + np.cos(np.linspace(0, np.pi, taper_len)))
    if magnitude_fn(np.array([0.0]))[0] < 1e-3:
        # Truncating a slow high-pass's long tail breaks its DC null; restore
        # it exactly (an AC-coupled audio path must fully block DC - that's
        # what removes an FM carrier-offset's DC shift).
        spread = np.hanning(num_taps)
        taps -= taps.sum() * spread / spread.sum()
    return taps


def apply_fir(x, taps):
    """Causal FIR filtering via FFT convolution; output is len(x) (the
    filter's delay is real and kept - nothing is compensated)."""
    n = len(x)
    if n == 0:
        return x.copy()
    n_fft = 1 << int(np.ceil(np.log2(n + len(taps) - 1)))
    if np.iscomplexobj(x):
        y = np.fft.ifft(np.fft.fft(x, n_fft) * np.fft.fft(taps, n_fft))
    else:
        y = np.fft.irfft(np.fft.rfft(x, n_fft) * np.fft.rfft(taps, n_fft), n_fft)
    return y[:n]


def measure_sinad(audio, sample_rate, tone_hz):
    """SINAD (dB) of a received test tone: total power over the power left
    after least-squares removing the tone's fundamental - the standard
    (signal+noise+distortion)/(noise+distortion) radio measurement."""
    audio = np.asarray(audio, dtype=float)
    t = np.arange(len(audio)) / sample_rate
    basis = np.column_stack([np.sin(2 * np.pi * tone_hz * t),
                             np.cos(2 * np.pi * tone_hz * t),
                             np.ones_like(t)])
    coeffs, *_ = np.linalg.lstsq(basis, audio, rcond=None)
    residual = audio - basis @ coeffs
    total = np.mean((audio - coeffs[2]) ** 2)
    noise = max(np.mean(residual ** 2), 1e-20)
    return 10 * np.log10(total / noise)


def doppler_hz(speed_kmh, freq_hz):
    """Maximum Doppler shift for a station moving at speed_kmh."""
    return (speed_kmh / 3.6) * freq_hz / SPEED_OF_LIGHT_M_S


def _resolve_preset(value, presets, kind):
    if isinstance(value, dict):
        return dict(value)
    if value not in presets:
        raise ValueError(f"unknown {kind} {value!r} (expected one of {sorted(presets)} or a dict)")
    return dict(presets[value])


# -- the link ----------------------------------------------------------------

class FMLink:
    """One simulated FM radio-to-radio link carrying audio. Configure once
    with the radios and the RF scenario, then call transmit() per burst
    (one PTT keying) of TX audio.

    port is required, deliberately - 'mic' and 'data' behave very
    differently and neither is the "normal" one. Either may also be a dict
    with PORT_PRESETS' keys for a custom radio; likewise channel with
    CHANNEL_PRESETS' keys.

    base_cnr_db is the carrier-to-noise ratio in the receiver's IF
    bandwidth at reference_distance_m (same path-loss convenience as
    OTALink - not a calibrated link budget). ~10 dB CNR is roughly the FM
    threshold / 12 dB SINAD point.

    fading: 'none' (static, unit gain), 'rayleigh', or 'rician' (with
    rician_k_db). Fading varies continuously within and across bursts at
    the Doppler rate set by speed_kmh and freq_hz; speed_kmh=0 gives a
    fixed random gain for the life of the link.

    Radio timing: tx_keyup_ms of the start of each burst is lost while the
    transmitter keys up (no carrier); squelch_cnr_db is the noise-squelch
    opening threshold (None = squelch disabled/open, the receiver passes
    discriminator noise when there's no signal); squelch_open_ms is how
    long the squelch takes to open once signal is present;
    squelch_tail_ms is how long it stays open after the signal goes away
    (the squelch "tail" noise burst after every transmission). The
    returned audio is always len(input) + squelch_tail_ms worth of
    samples - the receiver keeps listening that long after the TX unkeys.

    clock_offset_ppm: RX sound card's sample clock relative to TX's
    (positive = RX runs fast, so the output has more samples per second
    of audio). freq_offset_hz: TX/RX carrier frequency error - an FM
    discriminator turns it into a DC offset, which the audio high-pass
    removes, unless it's large enough to push the signal onto the IF
    filter's skirt.
    """

    def __init__(self, port, channel='narrow', distance_m=1000.0, freq_hz=146e6,
                 reference_distance_m=1000.0, base_cnr_db=20.0, sample_rate=48000,
                 audio_deviation_hz=None, fading='none', speed_kmh=0.0, rician_k_db=6.0,
                 freq_offset_hz=0.0, clock_offset_ppm=0.0, tx_keyup_ms=100.0,
                 squelch_cnr_db=6.0, squelch_open_ms=20.0, squelch_tail_ms=100.0,
                 latency_s=0.05, jitter_s=0.02, seed=None):
        if fading not in ('none', 'rayleigh', 'rician'):
            raise ValueError(f"fading must be 'none', 'rayleigh' or 'rician', not {fading!r}")
        self.port = _resolve_preset(port, PORT_PRESETS, 'port')
        self.channel = _resolve_preset(channel, CHANNEL_PRESETS, 'channel')
        self.port_name = port if isinstance(port, str) else 'custom'
        self.channel_name = channel if isinstance(channel, str) else 'custom'
        self.distance_m = distance_m
        self.freq_hz = freq_hz
        self.reference_distance_m = reference_distance_m
        self.base_cnr_db = base_cnr_db
        self.sample_rate = sample_rate
        self.max_deviation_hz = self.channel['max_deviation_hz']
        self.if_bw_hz = self.channel['if_bw_hz']
        if self.if_bw_hz >= sample_rate:
            raise ValueError(f"sample_rate {sample_rate} too low for a {self.if_bw_hz:.0f} Hz IF bandwidth")
        self.audio_deviation_hz = (audio_deviation_hz if audio_deviation_hz is not None
                                   else DEFAULT_AUDIO_DEVIATION_FRACTION * self.max_deviation_hz)
        self.fading = fading
        self.speed_kmh = speed_kmh
        self.rician_k_db = rician_k_db
        self.freq_offset_hz = freq_offset_hz
        self.clock_offset_ppm = clock_offset_ppm
        self.tx_keyup_ms = tx_keyup_ms
        self.squelch_cnr_db = squelch_cnr_db
        self.squelch_open_ms = squelch_open_ms
        self.squelch_tail_ms = squelch_tail_ms
        self.latency_s = latency_s
        self.jitter_s = jitter_s
        self._rng = np.random.default_rng(seed)
        self._clock_s = 0.0
        self.last_info = None

        self._design_filters()
        self._init_fading()

    # -- setup --------------------------------------------------------------
    def _design_filters(self):
        fs, p = self.sample_rate, self.port
        emph = p['emphasis']

        def tx_pre(f):
            mag = _butter_highpass_mag(f, *p['tx_highpass'])
            return mag * _preemphasis_mag(f) if emph else mag

        def rx_audio(f):
            mag = _butter_highpass_mag(f, *p['rx_highpass']) * _butter_lowpass_mag(f, *p['rx_lowpass'])
            return mag / _preemphasis_mag(f) if emph else mag

        self._h_tx_pre = min_phase_fir(tx_pre, fs, _AUDIO_FILTER_TAPS)
        self._h_tx_splatter = min_phase_fir(lambda f: _butter_lowpass_mag(f, *p['tx_lowpass']),
                                            fs, _AUDIO_FILTER_TAPS)
        self._h_rx_audio = min_phase_fir(rx_audio, fs, _AUDIO_FILTER_TAPS)
        # A steep (8th-order) crystal/ceramic IF channel filter.
        self._h_if = min_phase_fir(lambda f: _butter_lowpass_mag(f, self.if_bw_hz / 2, 8),
                                   fs, _IF_FILTER_TAPS)

    def _init_fading(self):
        m = _FADING_SINUSOIDS
        self._sos_cos_alpha = np.cos(self._rng.uniform(0, 2 * np.pi, m))
        self._sos_phase = self._rng.uniform(0, 2 * np.pi, m)
        self._los_cos_alpha = np.cos(self._rng.uniform(0, 2 * np.pi))
        self._los_phase = self._rng.uniform(0, 2 * np.pi)

    @property
    def _fd_hz(self):
        # Computed on demand, not cached, so speed_kmh/freq_hz can be changed
        # on a live link (ota_sim.control) and take effect on the next burst.
        return doppler_hz(self.speed_kmh, self.freq_hz)

    # -- public helpers -----------------------------------------------------
    def nominal_cnr_db(self):
        """CNR at the configured distance, before fading."""
        return self.base_cnr_db - (free_space_path_loss_db(self.distance_m, self.freq_hz) -
                                   free_space_path_loss_db(self.reference_distance_m, self.freq_hz))

    def fading_gain(self, t):
        """Complex channel gain at times t (seconds), E[|g|^2] == 1. Sum-of-
        sinusoids Rayleigh process (random arrival angles -> Doppler
        shifts fd*cos(alpha)); Rician adds a fixed-power line-of-sight
        component."""
        t = np.asarray(t, dtype=float)
        if self.fading == 'none':
            return np.ones(t.shape, dtype=complex)
        w = 2 * np.pi * self._fd_hz
        scatter = np.exp(1j * (np.outer(t, w * self._sos_cos_alpha) + self._sos_phase))
        g = scatter.sum(axis=1) / np.sqrt(_FADING_SINUSOIDS)
        if self.fading == 'rician':
            k = 10 ** (self.rician_k_db / 10)
            los = np.exp(1j * (w * self._los_cos_alpha * t + self._los_phase))
            g = np.sqrt(k / (k + 1)) * los + np.sqrt(1 / (k + 1)) * g
        return g

    def _fading_gain_interp(self, t):
        """fading_gain(t) for a dense, uniformly spaced t, evaluated on a
        coarse grid and linearly interpolated."""
        if self.fading == 'none' or len(t) < 2:
            return self.fading_gain(t)
        step = _FADING_MAX_STEP_S
        if self._fd_hz > 0:
            step = min(step, 1 / (_FADING_POINTS_PER_DOPPLER_CYCLE * self._fd_hz))
        coarse_t = np.arange(t[0], t[-1] + step, step)
        coarse = self.fading_gain(coarse_t)
        return np.interp(t, coarse_t, coarse.real) + 1j * np.interp(t, coarse_t, coarse.imag)

    def tx_deviation_hz(self, audio):
        """Instantaneous carrier deviation (Hz) the TX radio produces for
        this audio: full-scale clip, TX filtering/emphasis, deviation gain,
        hard limiter at max deviation, splatter filter."""
        x = np.clip(np.asarray(audio, dtype=float), -1.0, 1.0)
        dev = apply_fir(x, self._h_tx_pre) * self.audio_deviation_hz
        dev = np.clip(dev, -self.max_deviation_hz, self.max_deviation_hz)
        return apply_fir(dev, self._h_tx_splatter)

    # -- the channel --------------------------------------------------------
    def transmit(self, audio, t0=None):
        """Push one PTT burst of TX audio (floats, [-1, 1] full scale, at
        sample_rate) through TX radio, RF channel, and RX radio.

        Returns (rx_audio, delay_s): rx_audio covers the burst plus the
        squelch tail (see class docstring), delay_s is propagation +
        latency + jitter (airtime is the caller's to account for).
        t0 is the burst's start time on the fading process's clock
        (seconds); by default bursts follow each other back to back.
        Diagnostics for the burst land in self.last_info."""
        fs = self.sample_rate
        audio = np.asarray(audio, dtype=float)
        n_burst = len(audio)
        n_tail = int(round(self.squelch_tail_ms / 1000 * fs))
        n_total = n_burst + n_tail
        t0 = self._clock_s if t0 is None else t0
        t = t0 + np.arange(n_total) / fs

        # TX radio + carrier (unit power; noise floor is fixed below, so
        # fading genuinely changes CNR rather than scaling noise with it).
        dev = self.tx_deviation_hz(audio) + self.freq_offset_hz
        phase = 2 * np.pi * np.cumsum(dev) / fs + self._rng.uniform(0, 2 * np.pi)
        carrier = np.zeros(n_total, dtype=complex)
        carrier[:n_burst] = np.exp(1j * phase)
        keyup = min(int(round(self.tx_keyup_ms / 1000 * fs)), n_burst)
        carrier[:keyup] = 0.0
        gain = self._fading_gain_interp(t)
        signal = carrier * gain

        # AWGN sized so CNR holds in the IF bandwidth.
        cnr_lin = 10 ** (self.nominal_cnr_db() / 10)
        sigma = np.sqrt(fs / (cnr_lin * self.if_bw_hz) / 2)
        noise = sigma * (self._rng.normal(size=n_total) + 1j * self._rng.normal(size=n_total))

        # RX radio.
        rx_if = apply_fir(signal + noise, self._h_if)
        disc = np.zeros(n_total)
        disc[1:] = np.angle(rx_if[1:] * np.conj(rx_if[:-1])) * fs / (2 * np.pi)
        out = apply_fir(disc / self.audio_deviation_hz, self._h_rx_audio)
        noise_power_if = 2 * sigma ** 2 * np.sum(self._h_if ** 2)
        gate = self._squelch_gate(rx_if, noise_power_if, n_total)
        out = out * gate
        if self.clock_offset_ppm:
            ratio = 1 + self.clock_offset_ppm * 1e-6
            n_out = int(round(n_total * ratio))
            out = np.interp(np.arange(n_out) / ratio, np.arange(n_total), out)
        out = np.clip(out, -1.0, 1.0)

        # Phase slips (threshold clicks): whole turns the received phase
        # gains/loses against the noiseless one, counted only while the
        # carrier is up - with no carrier there's no phase to slip from.
        ref_if = apply_fir(signal, self._h_if)[keyup:n_burst]
        phase_err = np.unwrap(np.angle(rx_if[keyup:n_burst] * np.conj(ref_if)))
        slip_turns = np.round(phase_err / (2 * np.pi))
        self.last_info = {
            'cnr_db': self.nominal_cnr_db(),
            'phase_slips': int(np.sum(np.abs(np.diff(slip_turns)))),
            'squelch_open_fraction': float(np.mean(gate)) if n_total else 0.0,
            'min_fade_db': (float(20 * np.log10(max(np.min(np.abs(gain[:n_burst])), 1e-12)))
                            if n_burst else 0.0),
        }
        self._clock_s = t0 + n_total / fs

        delay_s = (propagation_delay_s(self.distance_m) + self.latency_s +
                   self._rng.uniform(0, self.jitter_s))
        return out, delay_s

    def _squelch_gate(self, rx_if, noise_power_if, n_total):
        """Noise squelch: per-1ms-block CNR estimate from IF power (the
        receiver knows its own noise floor), opening after squelch_open_ms
        of continuous signal, closing squelch_tail_ms after the signal
        drops below threshold minus hysteresis. 1.0 = open, 0.0 = muted."""
        if self.squelch_cnr_db is None:
            return np.ones(n_total)
        block = max(int(round(_SQUELCH_BLOCK_S * self.sample_rate)), 1)
        n_blocks = -(-n_total // block)
        padded = np.zeros(n_blocks * block, dtype=complex)
        padded[:n_total] = rx_if
        power = np.mean(np.abs(padded.reshape(n_blocks, block)) ** 2, axis=1)
        power = np.convolve(power, np.ones(_SQUELCH_AVG_BLOCKS) / _SQUELCH_AVG_BLOCKS)[:n_blocks]
        est_cnr_db = 10 * np.log10(np.maximum(power / noise_power_if - 1, 1e-6))
        open_thresh = self.squelch_cnr_db
        close_thresh = self.squelch_cnr_db - _SQUELCH_HYSTERESIS_DB
        open_blocks = int(round(self.squelch_open_ms / 1000 / _SQUELCH_BLOCK_S))
        tail_blocks = int(round(self.squelch_tail_ms / 1000 / _SQUELCH_BLOCK_S))

        gate = np.zeros(n_blocks)
        is_open, run = False, 0
        for i, cnr in enumerate(est_cnr_db):
            if not is_open:
                run = run + 1 if cnr >= open_thresh else 0
                if run > open_blocks:
                    is_open, run = True, 0
            else:
                run = run + 1 if cnr < close_thresh else 0
                if run > tail_blocks:
                    is_open, run = False, 0
            gate[i] = 1.0 if is_open else 0.0
        return np.repeat(gate, block)[:n_total]
