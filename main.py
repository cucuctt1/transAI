import argparse
import sys


def parse_args():
    p = argparse.ArgumentParser(
        description="Japanese -> English realtime translator"
    )
    p.add_argument("--client", action="store_true", help="run the lightweight client UI (no ASR/CUDA)")
    p.add_argument("--server-mode", action="store_true", help="run the caption server UI")
    p.add_argument("--server-headless", action="store_true", help="run a headless caption server (no GUI)")
    p.add_argument("--port", type=int, default=8765, help="server port (default 8765)")
    p.add_argument("--pairing-code", default="", help="server pairing code (default: none)")
    p.add_argument("--cli", action="store_true", help="run the terminal ASR pipeline (no GUI)")

    # ASR / CLI options
    p.add_argument("--buffer", type=float, default=5.0, help="Whisper window length in seconds (default 5)")
    p.add_argument("--audio-buffer", type=float, default=7.0, help="rolling audio history length in seconds (default 7)")
    p.add_argument("--step", type=float, default=1, help="snapshot interval in seconds (default 1)")
    p.add_argument("--compute-type", default="int8", choices=["float16", "int8_float16", "int8"], help="CTranslate2 compute type")
    p.add_argument("--vad", dest="vad", action="store_true", default=False, help="enable Silero VAD (default on)")
    p.add_argument("--no-vad", dest="vad", action="store_false", help="disable VAD")
    p.add_argument("--min-logprob", type=float, default=-1.0)
    p.add_argument("--max-compression-ratio", type=float, default=2.4)
    p.add_argument("--max-temperature", type=float, default=1.0)
    p.add_argument("--min-words", type=int, default=1)
    p.add_argument("--silence-threshold", type=float, default=0.001)
    p.add_argument("--rollback", type=float, default=3.0)
    p.add_argument("--fixer-interval", type=float, default=0.5)
    p.add_argument("--fixer-context", type=float, default=7.0)
    p.add_argument("--fixer", dest="fixer", action="store_true", help="enable the background fixer (off by default)")
    p.add_argument("--no-fixer", dest="fixer", action="store_false", help="disable the background fixer (default)")
    p.set_defaults(fixer=False)
    p.add_argument("--warning-latency", type=float, default=1.0)
    p.add_argument("--backlog-latency", type=float, default=2.0)
    p.add_argument("--latency", dest="latency", action="store_true", default=True)
    p.add_argument("--no-latency", dest="latency", action="store_false")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug-logs", action="store_true")
    return p.parse_args()


def run_cli(args):
    """The original terminal ASR pipeline (preserved)."""
    import threading
    import time

    import numpy as np
    import torch

    from audio_capture import AudioCapture
    from translator import Translator
    from transcript import (
        BackgroundFixer,
        Config,
        LatestQueue,
        LatestWinsScheduler,
        LatencyMonitor,
        Snapshot,
        TranscriptRenderer,
        TranscriptState,
        tokenize,
    )

    MODEL_ID = "kotoba-tech/kotoba-whisper-bilingual-v1.0-faster"

    def fmt_elapsed(seconds):
        s = int(seconds)
        return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"

    cfg = Config(
        whisper_window=args.buffer,
        whisper_step=args.step,
        audio_buffer=args.audio_buffer,
        rollback_window=args.rollback,
        fixer_interval=args.fixer_interval,
        fixer_context=args.fixer_context,
        enable_background_fixer=args.fixer,
        enable_latency_monitor=args.latency,
        warning_latency=args.warning_latency,
        backlog_latency=args.backlog_latency,
    )

    print("=" * 50)
    print(" Japanese -> English Realtime Translator")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available.")
        print("[ERROR] This application requires an NVIDIA GPU with CUDA and will not fall back to CPU.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print("CUDA: available")
    print(f"Model: {MODEL_ID}")

    print("Loading model (this happens once)...", flush=True)
    try:
        translator = Translator(
            MODEL_ID,
            compute_type=args.compute_type,
            vad=args.vad,
            min_words=args.min_words,
            min_logprob=args.min_logprob,
            max_compression_ratio=args.max_compression_ratio,
            max_temperature=args.max_temperature,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to load model on CUDA: {exc}")
        sys.exit(1)

    capture = AudioCapture(buffer_seconds=cfg.audio_buffer)
    capture.start()
    print(f"[Audio] Using output device: {capture.device_name}")
    print("Loopback: enabled")
    print(f"Buffer: {cfg.whisper_window:g}s window / {cfg.whisper_step:g}s step / {cfg.rollback_window:g}s rollback")
    print()
    print("Listening...", flush=True)
    print()

    stop = threading.Event()
    state = TranscriptState(rollback_window=cfg.rollback_window)
    renderer = TranscriptRenderer(timestamps=True)
    whisper_q = LatestQueue(maxsize=cfg.max_whisper_queue)
    scheduler = LatestWinsScheduler(max_pending=cfg.max_whisper_pending)
    monitor = LatencyMonitor(warning=cfg.warning_latency, backlog=cfg.backlog_latency)
    fixer = BackgroundFixer(interval=cfg.fixer_interval, context_seconds=cfg.fixer_context)
    snapshot_id = 0

    def snapshot_loop():
        nonlocal snapshot_id
        while not stop.is_set():
            if capture.available_seconds() >= cfg.whisper_window:
                break
            time.sleep(0.05)
        next_t = time.monotonic()
        while not stop.is_set():
            next_t += cfg.whisper_step
            got = capture.latest(cfg.whisper_window)
            if got is not None:
                audio, audio_end = got
                rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
                if rms >= args.silence_threshold:
                    scheduler.submit(Snapshot(audio=audio, audio_end=audio_end, id=snapshot_id))
                    snapshot_id += 1
            delay = next_t - time.monotonic()
            if delay > 0:
                if stop.wait(delay):
                    break
            else:
                next_t = time.monotonic()

    def whisper_loop():
        while not stop.is_set():
            job = scheduler.take()
            if job is None:
                stop.wait(0.02)
                continue
            if capture.live_position() - job.audio_end > cfg.backlog_latency:
                scheduler.skipped += 1
                continue
            t0 = time.monotonic()
            try:
                res = translator.translate(job.audio)
            except Exception as exc:
                print(f"[ERROR] inference failed: {exc}", file=sys.stderr, flush=True)
                res = None
            dt = time.monotonic() - t0
            monitor.update(live=capture.live_position(), transcribed=job.audio_end,
                           whisper_dur=dt, skipped=scheduler.skipped, pending=scheduler.pending_count)
            if res is None:
                continue
            if args.debug:
                for s in res.segments:
                    print(f"      logp={s.avg_logprob:.3f} comp={s.compression_ratio:.2f} nsp={s.no_speech_prob:.3f} temp={s.temperature} {s.text!r}", file=sys.stderr, flush=True)
            if args.debug_logs:
                print(f"[WHISPER] window={job.audio_end - cfg.whisper_window:.1f}-{job.audio_end:.1f}s inference={dt:.2f}s live_latency={capture.live_position() - job.audio_end:.2f}s", file=sys.stderr, flush=True)
            whisper_q.put((res.text, job.audio_end, dt))

    def fixer_loop():
        while not stop.is_set():
            stop.wait(cfg.fixer_interval)
            version, finalized, unstable, needs_fix = state.snapshot()
            if not unstable or not needs_fix:
                continue
            ctx = tokenize(finalized)[-int(cfg.fixer_context * cfg.words_per_second):]
            corrected = fixer.fix_unstable_transcript(" ".join(ctx), unstable)
            if corrected != unstable:
                state.apply_fix(version, tokenize(corrected))

    threading.Thread(target=snapshot_loop, name="snapshot", daemon=True).start()
    threading.Thread(target=whisper_loop, name="whisper", daemon=True).start()
    # if cfg.enable_background_fixer:
    #     threading.Thread(target=fixer_loop, name="fixer", daemon=True).start()

    last_status_t = 0.0
    try:
        while True:
            item = whisper_q.get(timeout=0.2)
            monitor.update(live=capture.live_position())
            if item is not None:
                text, audio_end, wdt = item
                version, finalized, unstable, needs_fix = state.apply_hypothesis(text)
                monitor.update(duplicates=state.noop_count)
                renderer.render(finalized, unstable, audio_end)
            if cfg.enable_latency_monitor and time.monotonic() - last_status_t >= 2.0:
                print(monitor.status_line(), file=sys.stderr, flush=True)
                last_status_t = time.monotonic()
    except KeyboardInterrupt:
        pass

    print("\nStopping...", flush=True)
    stop.set()
    capture.stop()
    print("Stopped.", flush=True)
    sys.exit(0)


def main():
    args = parse_args()
    if args.client:
        from ui import run_client
        run_client()
    elif args.server_mode:
        _run_with_model("server", args)
    elif args.server_headless:
        run_headless_server(args)
    elif args.cli:
        run_cli(args)
    else:
        # Default: thin client UI. It never loads the model; "local" mode
        # auto-spawns a headless server (model runs in that separate process).
        from ui import run_client
        run_client()


def _run_with_model(mode, args):
    """Load the model BEFORE importing PyQt5.

    Initializing CTranslate2/CUDA after Qt has been imported crashes the process
    on Windows (native segfault, no traceback), so the ASR stack must come first.
    """
    from asr_engine import load_model
    try:
        translator, gpu_name = load_model(
            compute_type=args.compute_type,
            vad=args.vad,
            min_words=args.min_words,
            min_logprob=args.min_logprob,
            max_compression_ratio=args.max_compression_ratio,
            max_temperature=args.max_temperature,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to load model: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)

    # Import the Qt UI only after CUDA/model initialization has succeeded.
    from ui import run_server
    run_server(translator, gpu_name)


def run_headless_server(args):
    """Headless caption server: receive streamed audio, run the model, return captions."""
    import time

    from asr_engine import ASREngine, load_model
    from server import AudioCaptionServer, NetworkAudioSource
    from transcript import Config

    try:
        translator, gpu_name = load_model(
            compute_type=args.compute_type,
            vad=args.vad,
            min_words=args.min_words,
            min_logprob=args.min_logprob,
            max_compression_ratio=args.max_compression_ratio,
            max_temperature=args.max_temperature,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to load model: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)

    cfg = Config(
        whisper_window=args.buffer,
        whisper_step=args.step,
        audio_buffer=args.audio_buffer,
        rollback_window=args.rollback,
        fixer_interval=args.fixer_interval,
        fixer_context=args.fixer_context,
        enable_background_fixer=args.fixer,
        warning_latency=args.warning_latency,
        backlog_latency=args.backlog_latency,
    )

    audio_source = NetworkAudioSource(buffer_seconds=cfg.audio_buffer)
    server = AudioCaptionServer(audio_source, port=args.port, pairing_code=args.pairing_code,
                                log_cb=lambda line: print(line, flush=True))
    server.start()
    engine = ASREngine(cfg, translator, gpu_name, audio_source, caption_cb=server.broadcast,
                       log_cb=lambda line: print(line, flush=True),
                       status_cb=lambda line: print(line, flush=True))
    engine.start()
    print(f"[SERVER] headless server running on port {args.port} (fixer: {'on' if args.fixer else 'off'})", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        server.stop()
    print("Stopped.", flush=True)


if __name__ == "__main__":
    main()
