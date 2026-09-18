import argparse
import sys
from model_storage import configure

configure()


def parse_args():
    from transcript import Config
    defaults = Config()
    p = argparse.ArgumentParser(
        description="Japanese -> English realtime translator"
    )
    p.add_argument("--client", action="store_true", help="run the lightweight client UI (no ASR/CUDA)")
    p.add_argument("--server-mode", action="store_true", help="run the caption server UI")
    p.add_argument("--server-headless", action="store_true", help="run a headless caption server (no GUI)")
    p.add_argument("--port", type=int, default=8765, help="server port (default 8765)")
    p.add_argument("--pairing-code", default="", help="server pairing code (default: none)")
    p.add_argument("--cli", action="store_true", help="run the terminal ASR pipeline (no GUI)")
    p.add_argument("--parent-control", action="store_true", help=argparse.SUPPRESS)

    # ASR / CLI options
    p.add_argument("--backend", choices=["faster-whisper", "moonshine"], default="faster-whisper",
                   help="speech backend; moonshine uses Japanese ASR + OPUS-MT English translation")
    p.add_argument("--buffer", type=float, default=defaults.whisper_window, help="translation context in seconds (default 10)")
    p.add_argument("--audio-buffer", type=float, default=defaults.audio_buffer, help="audio recovery history in seconds (default 20)")
    p.add_argument("--step", type=float, default=defaults.whisper_step, help="minimum preview/display interval in seconds (default 2)")
    p.add_argument("--caption-timeout", type=float, default=defaults.caption_timeout,
                   help="clear captions after this many seconds without audio activity (default 5)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--phrase-only", dest="phrase_only", action="store_true", help="output completed phrases without live previews")
    mode.add_argument("--live", dest="phrase_only", action="store_false", help="live previews (default)")
    mode.add_argument("--commit-only", action="store_true",
                      help="publish each finalized phrase once, with committed history and Japanese boundary checks")
    p.set_defaults(phrase_only=False)
    p.add_argument("--phrase-pause", type=float, default=1.0, help="quiet seconds ending a phrase (default 1)")
    p.add_argument("--commit-pause", type=float, default=defaults.commit_pause,
                   help="quiet seconds triggering commit-only finalization (default 0.45)")
    p.add_argument("--max-phrase-seconds", type=float, default=30.0, help="safety split for speech without pauses (default 30)")
    p.add_argument("--compute-type", default="int8_float16", choices=["float16", "int8_float16", "int8"], help="CTranslate2 compute type (default int8_float16; CPU uses int8)")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="inference device (default cuda)")
    p.add_argument("--vad", dest="vad", action="store_true", default=True, help="enable Silero VAD (default on)")
    p.add_argument("--no-vad", dest="vad", action="store_false", help="disable VAD")
    p.add_argument("--min-logprob", type=float, default=-1.0)
    p.add_argument("--max-compression-ratio", type=float, default=2.4)
    p.add_argument("--max-temperature", type=float, default=1.0)
    p.add_argument("--min-words", type=int, default=1)
    p.add_argument("--beam-size", type=int, default=5,
                   help="maximum phrase-mode beam (default 5); live mode always uses 1")
    p.add_argument("--translation-prompt", default="",
                   help="English names/terms to preserve, e.g. 'OpenAI, Akihabara, Kaito'")
    p.add_argument("--silence-threshold", type=float, default=0.001)
    p.add_argument("--rollback", type=float, default=3.0)
    p.add_argument("--fixer-interval", type=float, default=0.5)
    p.add_argument("--fixer-context", type=float, default=7.0)
    p.add_argument("--fixer", dest="fixer", action="store_true", help="enable deterministic caption cleanup")
    p.add_argument("--no-fixer", dest="fixer", action="store_false", help="disable deterministic caption cleanup")
    p.add_argument("--llm-fixer", action="store_true", help="async local Qwen Q4 caption correction")
    p.add_argument("--fixer-language", choices=['en', 'ja'], default='en',
                   help="Moonshine: ja stabilizes source before MT; add --llm-fixer for Japanese Qwen editing")
    p.add_argument("--fixer-cuda", action="store_true",
                   help="enable the Qwen fixer with CUDA offloading (requires CUDA llama-cpp-python)")
    p.add_argument("--fixer-model", default="", help="optional local GGUF path for the LLM fixer")
    p.set_defaults(fixer=False, llm_fixer=False, phrase_only=False)
    p.add_argument("--warning-latency", type=float, default=1.0)
    p.add_argument("--backlog-latency", type=float, default=2.0)
    p.add_argument("--latency", dest="latency", action="store_true", default=True)
    p.add_argument("--no-latency", dest="latency", action="store_false")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug-logs", action="store_true")
    args = p.parse_args()
    if args.fixer_language == 'ja' and args.backend != 'moonshine':
        p.error('--fixer-language ja requires --backend moonshine')
    return args


def pipeline_config(args):
    from transcript import Config
    return Config(
        whisper_window=args.buffer, whisper_step=args.step,
        audio_buffer=args.audio_buffer, rollback_window=args.rollback,
        fixer_interval=args.fixer_interval, fixer_context=args.fixer_context,
        enable_background_fixer=args.fixer, enable_latency_monitor=args.latency,
        warning_latency=args.warning_latency, backlog_latency=args.backlog_latency,
        phrase_only=args.phrase_only, endpoint_silence=args.phrase_pause,
        max_phrase_seconds=args.max_phrase_seconds,
        enable_llm_fixer=args.llm_fixer or args.fixer_cuda, fixer_model=args.fixer_model,
        fixer_cuda=args.fixer_cuda,
        fixer_language=args.fixer_language,
        caption_timeout=args.caption_timeout,
        commit_only=args.commit_only,
        commit_pause=args.commit_pause,
    )


def run_cli(args):
    """Use the same scheduler, correction and deduplication as the overlay."""
    import time
    from asr_engine import ASREngine, load_model
    from audio_capture import AudioCapture

    cfg = pipeline_config(args)
    translator, device = load_model(
        compute_type=args.compute_type, vad=args.vad, min_words=args.min_words,
        min_logprob=args.min_logprob, max_compression_ratio=args.max_compression_ratio,
        max_temperature=args.max_temperature, device=args.device,
        beam_size=args.beam_size, translation_prompt=args.translation_prompt,
        backend=args.backend,
    )
    capture = AudioCapture(buffer_seconds=cfg.audio_buffer)
    import shutdown
    if shutdown.requested.is_set():
        return
    previous_width = 0

    def show_caption(text):
        nonlocal previous_width
        if cfg.phrase_only:
            if text:
                print(text, flush=True)
            return
        if sys.stdout.isatty():
            print("\r" + text.ljust(previous_width), end="", flush=True)
            previous_width = len(text)
        elif text:
            print(text, flush=True)

    engine = ASREngine(
        cfg, translator, device, capture, caption_cb=show_caption,
        log_cb=lambda line: print(line, file=sys.stderr, flush=True),
        status_cb=lambda line: print(line, file=sys.stderr, flush=True),
        silence_threshold=args.silence_threshold, debug=args.debug or args.debug_logs,
    )
    engine.start()
    try:
        import shutdown
        while engine.is_running() and not shutdown.requested.wait(0.2):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        print("\nStopped.", flush=True)


def _main():
    args = parse_args()
    if args.parent_control:
        from shutdown import watch_parent
        watch_parent()
    if args.client:
        from ui import run_client
        run_client(server_args=[arg for arg in sys.argv[1:] if arg != "--client"])
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
        run_client(server_args=sys.argv[1:])


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
            device=args.device,
            beam_size=args.beam_size,
            translation_prompt=args.translation_prompt,
            backend=args.backend,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to load model: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)

    # Import the Qt UI only after CUDA/model initialization has succeeded.
    import shutdown
    if shutdown.requested.is_set():
        return
    from ui import run_server
    run_server(translator, gpu_name, cfg=pipeline_config(args),
               silence_threshold=args.silence_threshold, debug=args.debug or args.debug_logs)


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
            device=args.device,
            beam_size=args.beam_size,
            translation_prompt=args.translation_prompt,
            backend=args.backend,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to load model: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)

    cfg = pipeline_config(args)

    import shutdown
    if shutdown.requested.is_set():
        return
    audio_source = NetworkAudioSource(buffer_seconds=cfg.audio_buffer)
    server = AudioCaptionServer(audio_source, port=args.port, pairing_code=args.pairing_code,
                                log_cb=lambda line: print(line, flush=True))
    server.start()
    engine = ASREngine(cfg, translator, gpu_name, audio_source, caption_cb=server.broadcast,
                       log_cb=lambda line: print(line, flush=True),
                       status_cb=lambda line: print(line, flush=True),
                       silence_threshold=args.silence_threshold, debug=args.debug or args.debug_logs)
    engine.start()
    print(f"[SERVER] headless server running on port {args.port} (fixer: {'on' if args.fixer else 'off'})", flush=True)
    try:
        import shutdown
        while engine.is_running() and not shutdown.requested.wait(0.2):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        engine.stop()
    print("Stopped.", flush=True)


def main():
    import shutdown
    with shutdown.signals():
        _main()


if __name__ == "__main__":
    main()
