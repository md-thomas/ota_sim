"""
Simulated point-to-point RF channel: path loss, flat fading, AWGN, and
propagation/processing delay applied to a burst of complex baseband IQ
samples - the same representation a real SDR's ADC/DAC would produce.

Deliberately protocol-agnostic: this module has no notion of frames,
headers, or modulation schemes. It only knows "here is a burst of complex
samples that left a transmitter" and "here is what a receiver some
distance away would actually see" - so it's reusable by any project moving
IQ samples between two simulated endpoints, not tied to any particular
link-layer protocol.
"""
import numpy as np

SPEED_OF_LIGHT_M_S = 299_792_458.0


def free_space_path_loss_db(distance_m, freq_hz):
    """Friis free-space path loss in dB:
    FSPL = 20*log10(d) + 20*log10(f) + 20*log10(4*pi/c).
    distance_m is floored at 1.0 to avoid log(0) at zero range - a
    simulated "very close" link just saturates at the 1m loss figure
    rather than producing -inf."""
    d = max(distance_m, 1.0)
    return (20 * np.log10(d) + 20 * np.log10(freq_hz) +
            20 * np.log10(4 * np.pi / SPEED_OF_LIGHT_M_S))


def propagation_delay_s(distance_m):
    """Speed-of-light propagation delay for distance_m. At the distances
    a simulator like this is typically configured for (sub-1000km), this
    is on the order of microseconds to low milliseconds - usually dwarfed
    by the configured latency_s/jitter_s, but included for completeness
    and so a deliberately huge distance behaves correctly."""
    return max(distance_m, 0.0) / SPEED_OF_LIGHT_M_S


def _snr_db_to_noise_voltage(samples, snr_db):
    """Target SNR (dB), measured against samples' own average power, ->
    the AWGN noise_voltage (std dev per I/Q rail) that achieves it.
    Standard textbook AWGN relation - not tied to any particular protocol,
    intentionally reimplemented here rather than imported from elsewhere
    so this package has no dependency outside numpy."""
    signal_power = np.mean(np.abs(samples) ** 2) if len(samples) else 0.0
    noise_power = signal_power / (10 ** (snr_db / 10))
    return float(np.sqrt(noise_power / 2))


def _add_awgn(samples, noise_voltage, rng):
    """Add complex AWGN (independent real/imaginary noise, each
    N(0, noise_voltage)) to samples using the given numpy Generator."""
    if noise_voltage <= 0 or len(samples) == 0:
        return samples.copy()
    noise = (rng.normal(0, noise_voltage, size=len(samples)) +
              1j * rng.normal(0, noise_voltage, size=len(samples)))
    return samples + noise


class OTALink:
    """One simulated point-to-point RF link. Configure once with the
    physical scenario (distance, frequency, reference SNR budget, fading/
    drop behavior, latency), then call transmit() per burst of TX samples.

    base_snr_db is the SNR you'd see at reference_distance_m - moving the
    simulated stations further apart than that degrades SNR via free-space
    path loss; closer improves it. This is a simulator convenience, not a
    calibrated link-budget tool (no real transmit power/antenna gain/noise
    figure modeling) - it exists to make "further apart or more RF-obstructed
    -> worse link" behave correctly and adjustably, not to predict a real
    radio's actual range.
    """

    def __init__(self, distance_m, freq_hz=433e6, reference_distance_m=1000.0,
                 base_snr_db=20.0, fading=True, drop_prob=0.02,
                 latency_s=0.05, jitter_s=0.02, seed=None):
        self.distance_m = distance_m
        self.freq_hz = freq_hz
        self.reference_distance_m = reference_distance_m
        self.base_snr_db = base_snr_db
        self.fading = fading
        self.drop_prob = drop_prob
        self.latency_s = latency_s
        self.jitter_s = jitter_s
        self._rng = np.random.default_rng(seed)

    def _path_loss_delta_db(self):
        return (free_space_path_loss_db(self.distance_m, self.freq_hz) -
                free_space_path_loss_db(self.reference_distance_m, self.freq_hz))

    def nominal_snr_db(self):
        """The SNR this link delivers at its configured distance, before
        any per-transmission fading - i.e. base_snr_db adjusted for the
        path-loss difference between distance_m and reference_distance_m."""
        return self.base_snr_db - self._path_loss_delta_db()

    def transmit(self, samples):
        """Push one burst of TX samples through the simulated channel.

        Returns (received_samples, delay_s), or (None, 0.0) if the burst
        is lost entirely (deep fade / collision - a receiver never even
        detects it arrived).

        samples should be a numpy array of complex baseband IQ samples
        (e.g. dtype=complex64 or complex128) - typically post pulse-shaping,
        matching what a real transmitter's DAC would actually output.
        """
        samples = np.asarray(samples, dtype=complex)

        if self._rng.random() < self.drop_prob:
            return None, 0.0

        # Compute the noise floor ONCE, from the ORIGINAL (unfaded) samples'
        # power at this link's nominal (path-loss-adjusted) SNR. This has to
        # happen before fading is applied: if noise were instead sized
        # relative to the *faded* signal's power, a deep fade would just
        # shrink both signal and noise together and cancel itself out,
        # instead of actually degrading the received SNR the way real
        # fading does. Fixing the noise floor first, then fading the signal
        # on top of it, is what makes a fade a fade.
        noise_voltage = _snr_db_to_noise_voltage(samples, self.nominal_snr_db())

        faded = samples
        if self.fading and len(samples):
            # Flat (single complex gain for the whole burst) Rayleigh
            # fading - narrowband signals at these symbol rates don't
            # need a tapped-delay multipath model. Magnitude only:
            # scale=1/sqrt(pi/2) makes E[magnitude] == 1, so fading
            # perturbs SNR around the path-loss baseline rather than
            # consistently biasing it up or down. Deliberately NOT
            # applying a random carrier phase rotation here, even though
            # a real fading channel would impose one: none of
            # openwave_modem.py's demodulators (BPSK/QPSK/16-QAM) do any
            # carrier phase recovery (no Costas loop, no differential
            # encoding - that's what the archived pi/4-QPSK GNU Radio
            # chain existed to solve). Measured directly: a pure phase
            # rotation with zero noise already corrupts 100% of a QPSK
            # burst past +-45 degrees (+-90 for BPSK) - so a uniform
            # random phase would mostly be testing "does this modem have
            # phase recovery" (answer: no, by design, today) rather than
            # "how much SNR margin does this link have", which is what
            # this simulator is actually for. See tasklist.txt for the
            # follow-up once/if phase-ambiguity handling exists for the
            # non-pi/4 modes.
            fade_mag = self._rng.rayleigh(scale=1 / np.sqrt(np.pi / 2))
            faded = samples * fade_mag

        noisy = _add_awgn(faded, noise_voltage, self._rng)

        delay_s = (propagation_delay_s(self.distance_m) + self.latency_s +
                   self._rng.uniform(0, self.jitter_s))
        return noisy, delay_s
