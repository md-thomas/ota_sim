"""
Live settings for a running simulator's channel: typed /commands that change
an OTALink's or FMLink's parameters between bursts, without restarting.

handle_command(channel, line) is the whole interface - it takes one typed
line and returns the lines to show the user, and has no I/O of its own, so
the simulator's stdin loop (see simulator.py) and anything else (a GUI,
tests) can drive the same settings. A change applies from the next burst
on; a burst already being processed finishes with the old values.

Not adjustable live: anything both stations must match (FM port, channel
width, sample rate) and anything baked into the radio filters at startup.
"""
from .channel import OTALink
from .fm import FMLink


class _Setting:
    """One adjustable parameter: the command name, which channel types have
    it, the attribute it sets, display/parse scaling, and its valid range."""

    def __init__(self, name, applies_to, attr, unit, help_text, scale=1.0,
                 minimum=None, maximum=None, allow_off=False):
        self.name = name
        self.applies_to = applies_to  # tuple of channel classes
        self.attr = attr
        self.unit = unit
        self.help_text = help_text
        self.scale = scale  # attribute value = typed value * scale
        self.minimum = minimum
        self.maximum = maximum
        self.allow_off = allow_off  # 'off' stores None (e.g. squelch)

    def show(self, channel):
        value = getattr(channel, self.attr)
        if value is None:
            return 'off'
        return f"{value / self.scale:g}{(' ' + self.unit) if self.unit else ''}"

    def apply(self, channel, text):
        if self.allow_off and text.lower() == 'off':
            setattr(channel, self.attr, None)
            return
        try:
            value = float(text)
        except ValueError:
            raise ValueError(f"/{self.name}: {text!r} is not a number") from None
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"/{self.name}: must be at least {self.minimum:g}")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"/{self.name}: must be at most {self.maximum:g}")
        setattr(channel, self.attr, value * self.scale)


_BOTH = (OTALink, FMLink)
_SETTINGS = [
    _Setting('snr', (OTALink,), 'base_snr_db', 'dB', 'SNR at the reference distance'),
    _Setting('cnr', (FMLink,), 'base_cnr_db', 'dB',
             'carrier-to-noise ratio in the IF bandwidth at the reference distance '
             '(~10 is the FM threshold, ~6 the squelch)'),
    _Setting('distance', _BOTH, 'distance_m', 'km', 'distance between the stations '
             '(path loss vs. the reference distance)', scale=1000.0, minimum=0.0),
    _Setting('freq', _BOTH, 'freq_hz', 'MHz', 'carrier frequency (path loss; Doppler for FM)',
             scale=1e6, minimum=0.001),
    _Setting('latency', _BOTH, 'latency_s', 'ms', 'base one-way latency', scale=1e-3, minimum=0.0),
    _Setting('jitter', _BOTH, 'jitter_s', 'ms', 'extra random latency, 0..this', scale=1e-3,
             minimum=0.0),
    _Setting('drop', (OTALink,), 'drop_prob', '', 'probability a burst is lost outright',
             minimum=0.0, maximum=1.0),
    _Setting('speed', (FMLink,), 'speed_kmh', 'km/h', 'mobile speed (Doppler fading rate)',
             minimum=0.0),
    _Setting('rician-k', (FMLink,), 'rician_k_db', 'dB', 'Rician K factor (with /fading rician)'),
    _Setting('ppm', (FMLink,), 'clock_offset_ppm', 'ppm', 'sound-card clock offset between the radios'),
    _Setting('offset', (FMLink,), 'freq_offset_hz', 'Hz', 'TX/RX carrier frequency error'),
    _Setting('keyup', (FMLink,), 'tx_keyup_ms', 'ms', 'audio lost at the start of each burst '
             'while the transmitter keys up', minimum=0.0),
    _Setting('squelch', (FMLink,), 'squelch_cnr_db', 'dB', 'squelch opening threshold, or off',
             allow_off=True),
    _Setting('squelch-open', (FMLink,), 'squelch_open_ms', 'ms', 'squelch opening delay', minimum=0.0),
    _Setting('squelch-tail', (FMLink,), 'squelch_tail_ms', 'ms', 'squelch tail after each burst',
             minimum=0.0),
    _Setting('deviation', (FMLink,), 'audio_deviation_hz', 'Hz',
             'deviation for a full-scale 1 kHz tone (the radio\'s audio gain)', minimum=1.0),
]
_BY_NAME = {s.name: s for s in _SETTINGS}


def _settings_for(channel):
    return [s for s in _SETTINGS if isinstance(channel, s.applies_to)]


def _fading_text(channel):
    if isinstance(channel, OTALink):
        return 'on' if channel.fading else 'off'
    if channel.fading == 'none':
        return 'none'
    extra = f", K={channel.rician_k_db:g} dB" if channel.fading == 'rician' else ''
    return f"{channel.fading} at {channel.speed_kmh:g} km/h{extra}"


def describe(channel):
    """Current settings, one per line, including the derived nominal SNR/CNR."""
    if isinstance(channel, FMLink):
        lines = [f"FM channel: {channel.port_name} port, {channel.channel_name} channel, "
                 f"nominal CNR {channel.nominal_cnr_db():.1f} dB"]
    else:
        lines = [f"IQ channel: nominal SNR {channel.nominal_snr_db():.1f} dB"]
    lines.append(f"  fading        {_fading_text(channel)}")
    for setting in _settings_for(channel):
        lines.append(f"  {setting.name:<13} {setting.show(channel)}")
    return lines


def help_lines(channel):
    rows = [('/show', 'current settings'),
            ('/fading on|off', 'Rayleigh flat fading per burst') if isinstance(channel, OTALink)
            else ('/fading none|rayleigh|rician [KM/H]', 'fading model, optionally with speed')]
    for setting in _settings_for(channel):
        arg = 'DB|off' if setting.allow_off else (setting.unit.upper().replace('/', '') or 'VALUE')
        rows.append((f"/{setting.name} {arg}", setting.help_text))
    rows.append(('/help', 'this list'))
    width = max(len(command) for command, _ in rows if len(command) < 24) + 2
    return (["Live commands (apply from the next burst):"] +
            [f"  {command:<{width}}{text}" if len(command) < 24 else f"  {command}\n  {'':<{width}}{text}"
             for command, text in rows])


def handle_command(channel, line):
    """Apply one typed command to channel (an OTALink or FMLink); returns
    the lines to show. Raises nothing - errors come back as a message."""
    parts = line.strip().split()
    if not parts:
        return []
    command = parts[0].lstrip('/').lower()
    args = parts[1:]
    if command in ('snr', 'cnr'):  # either name works on either channel type
        command = 'cnr' if isinstance(channel, FMLink) else 'snr'
    try:
        if command in ('show', 'status'):
            return describe(channel)
        if command in ('help', '?'):
            return help_lines(channel)
        if command == 'fading':
            return _set_fading(channel, args)
        setting = _BY_NAME.get(command)
        if setting is None or not isinstance(channel, setting.applies_to):
            other = (f" (only for the {'IQ' if setting.applies_to == (OTALink,) else 'FM'} channel)"
                     if setting else '')
            return [f"unknown command /{command}{other} - try /help"]
        if len(args) != 1:
            return [f"/{command}: current value {setting.show(channel)} - usage: /{command} "
                    f"{'VALUE|off' if setting.allow_off else 'VALUE'}"]
        before = setting.show(channel)
        setting.apply(channel, args[0])
        return [f"{command}: {before} -> {setting.show(channel)}" + _nominal_note(channel, command)]
    except ValueError as exc:
        return [str(exc)]


def _nominal_note(channel, command):
    if command not in ('snr', 'cnr', 'distance', 'freq'):
        return ''
    if isinstance(channel, FMLink):
        return f" (nominal CNR now {channel.nominal_cnr_db():.1f} dB)"
    return f" (nominal SNR now {channel.nominal_snr_db():.1f} dB)"


def _set_fading(channel, args):
    if not args:
        return [f"fading: {_fading_text(channel)}"]
    before = _fading_text(channel)
    if isinstance(channel, OTALink):
        if args[0].lower() not in ('on', 'off') or len(args) > 1:
            return ["usage: /fading on|off"]
        channel.fading = args[0].lower() == 'on'
    else:
        mode = args[0].lower()
        if mode not in ('none', 'rayleigh', 'rician') or len(args) > 2:
            return ["usage: /fading none|rayleigh|rician [KM/H]"]
        if len(args) == 2:
            _BY_NAME['speed'].apply(channel, args[1])
        channel.fading = mode
    return [f"fading: {before} -> {_fading_text(channel)}"]
