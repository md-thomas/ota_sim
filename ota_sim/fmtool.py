"""
Push audio through a simulated FM radio link (ota_sim.fm.FMLink) offline,
WAV in -> WAV out, to listen to / inspect what the far radio's audio
output would actually contain. Or generate a test tone instead of an
input file and report its received SINAD.

    python3 -m ota_sim.fmtool --port mic --cnr-db 10 in.wav out.wav
    python3 -m ota_sim.fmtool --port data --tone 1000 --seconds 3 --cnr-db 8 \\
        --fading rayleigh --speed-kmh 50 --freq-mhz 440 out.wav

WAV files are mono 16-bit PCM. The input's sample rate is used as the
simulation rate (must comfortably exceed the IF bandwidth - 48000 is the
expected sound-card rate).
"""
import argparse
import sys
import wave

import numpy as np

from .fm import CHANNEL_PRESETS, PORT_PRESETS, FMLink, measure_sinad


def read_wav(path):
    with wave.open(path, 'rb') as w:
        if w.getsampwidth() != 2:
            raise SystemExit(f"{path}: only 16-bit PCM WAV is supported")
        rate, channels = w.getframerate(), w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype='<i2').astype(float) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels)[:, 0]  # first channel only
    return data, rate


def write_wav(path, audio, rate):
    pcm = np.clip(np.round(np.asarray(audio) * 32767.0), -32768, 32767).astype('<i2')
    with wave.open(path, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', nargs='+', metavar='WAV',
                        help='IN.wav OUT.wav, or just OUT.wav with --tone')
    parser.add_argument('--port', required=True, choices=sorted(PORT_PRESETS),
                        help="radio audio connection: 'mic' (voice band, emphasis) or "
                             "'data' (flat 6-pin data port)")
    parser.add_argument('--channel', default='narrow', choices=sorted(CHANNEL_PRESETS),
                        help='12.5 kHz (narrow, +-2.5 kHz) or 25 kHz (wide, +-5 kHz) (default: narrow)')
    parser.add_argument('--cnr-db', type=float, default=20.0,
                        help='carrier-to-noise ratio in the IF bandwidth (default: 20; ~10 is the FM threshold)')
    parser.add_argument('--freq-mhz', type=float, default=146.0,
                        help='carrier frequency, for Doppler (default: 146)')
    parser.add_argument('--fading', default='none', choices=['none', 'rayleigh', 'rician'])
    parser.add_argument('--speed-kmh', type=float, default=0.0, help='mobile speed for Doppler fading')
    parser.add_argument('--rician-k-db', type=float, default=6.0)
    parser.add_argument('--deviation-hz', type=float, default=None,
                        help='deviation for a full-scale 1 kHz tone (radio\'s mic/data gain; '
                             'default: 60%% of max deviation)')
    parser.add_argument('--freq-offset-hz', type=float, default=0.0)
    parser.add_argument('--clock-offset-ppm', type=float, default=0.0)
    parser.add_argument('--tx-keyup-ms', type=float, default=100.0)
    parser.add_argument('--squelch-cnr-db', type=float, default=6.0)
    parser.add_argument('--no-squelch', action='store_true', help='squelch fully open')
    parser.add_argument('--squelch-open-ms', type=float, default=20.0)
    parser.add_argument('--squelch-tail-ms', type=float, default=100.0)
    parser.add_argument('--tone', type=float, default=None, metavar='HZ',
                        help='generate a full-scale test tone instead of reading an input WAV, '
                             'and report its received SINAD')
    parser.add_argument('--tone-level', type=float, default=1.0, help='test tone amplitude (default: 1.0)')
    parser.add_argument('--seconds', type=float, default=2.0, help='test tone length (default: 2)')
    parser.add_argument('--rate', type=int, default=48000, help='sample rate for --tone (default: 48000)')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    if args.tone is not None:
        if len(args.files) != 1:
            parser.error('with --tone, give only OUT.wav')
        rate = args.rate
        t = np.arange(int(args.seconds * rate)) / rate
        audio = args.tone_level * np.sin(2 * np.pi * args.tone * t)
        out_path = args.files[0]
    else:
        if len(args.files) != 2:
            parser.error('give IN.wav OUT.wav (or use --tone OUT.wav)')
        audio, rate = read_wav(args.files[0])
        out_path = args.files[1]

    link = FMLink(port=args.port, channel=args.channel, base_cnr_db=args.cnr_db,
                  freq_hz=args.freq_mhz * 1e6, sample_rate=rate,
                  audio_deviation_hz=args.deviation_hz, fading=args.fading,
                  speed_kmh=args.speed_kmh, rician_k_db=args.rician_k_db,
                  freq_offset_hz=args.freq_offset_hz, clock_offset_ppm=args.clock_offset_ppm,
                  tx_keyup_ms=args.tx_keyup_ms,
                  squelch_cnr_db=None if args.no_squelch else args.squelch_cnr_db,
                  squelch_open_ms=args.squelch_open_ms, squelch_tail_ms=args.squelch_tail_ms,
                  seed=args.seed)
    out, _ = link.transmit(audio)
    write_wav(out_path, out, rate)

    info = link.last_info
    print(f"[fmtool] {len(audio) / rate:.2f}s in -> {len(out) / rate:.2f}s out ({out_path}); "
          f"CNR {info['cnr_db']:.1f} dB, {info['phase_slips']} phase slips (clicks), "
          f"squelch open {100 * info['squelch_open_fraction']:.0f}%, "
          f"deepest fade {info['min_fade_db']:.1f} dB", file=sys.stderr)
    if args.tone is not None:
        # Skip key-up, squelch opening and filter settling; stop before the tail.
        start = int((args.tx_keyup_ms + args.squelch_open_ms + 100) / 1000 * rate)
        segment = out[start:len(audio)]
        if len(segment) < rate // 10:
            print("[fmtool] tone too short to measure SINAD after key-up/settling", file=sys.stderr)
        else:
            print(f"[fmtool] SINAD at {args.tone:g} Hz: {measure_sinad(segment, rate, args.tone):.1f} dB "
                  f"(includes any squelch dropouts/fades)", file=sys.stderr)


if __name__ == '__main__':
    main()
