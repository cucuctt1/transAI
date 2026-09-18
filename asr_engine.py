"""Reusable ASR engine: audio capture -> Whisper -> transcript stabilization -> captions.

Shared by the CLI and server UI. Pulls fresh audio on demand, produces cheap
previews and a quality pass at pauses, and coalesces caption updates.

IMPORTANT: torch + the model MUST be loaded via ``load_model()`` BEFORE importing
PyQt5. Initializing CTranslate2/CUDA after Qt has been imported crashes the process
on Windows (a native segfault with no traceback).
"""

import threading
import time
from types import SimpleNamespace

from transcript import BackgroundFixer, LatencyMonitor
from streaming import CaptionStabilizer, StreamPolicy, PhrasePolicy
from caption_output import CaptionOutput

MODEL_ID = "kotoba-tech/kotoba-whisper-bilingual-v1.0-faster"


def load_model(
    compute_type="int8_float16",
    vad=True,
    min_words=1,
    min_logprob=-1.0,
    max_compression_ratio=2.4,
    max_temperature=1.0,
    device="cuda",
    beam_size=5,
    translation_prompt="",
    log_cb=None,
    backend="faster-whisper",
):
    """Load torch + the Whisper model. Call on the main thread, before PyQt5."""
    log = log_cb or (lambda line: print(line, flush=True))
    if backend not in {"faster-whisper", "moonshine"}:
        raise ValueError(f"Unknown speech backend: {backend}")

    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available (pass --device cpu to run on CPU)")
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "CPU"

    log(f"Device: {gpu_name}")
    log("Loading model (this happens once)...")

    if backend == "moonshine":
        from moonshine_translator import MoonshineTranslator
        translator = MoonshineTranslator(device=device, beam_size=beam_size, log_cb=log)
        log("Moonshine Japanese ASR + OPUS-MT English translation loaded. "
            "Precision=float32; scheduler energy gate active; Whisper VAD/filters, "
            "compute-type and translation-prompt do not apply.")
        return translator, gpu_name

    from translator import Translator

    translator = Translator(
        MODEL_ID,
        compute_type=compute_type,
        vad=vad,
        min_words=min_words,
        min_logprob=min_logprob,
        max_compression_ratio=max_compression_ratio,
        max_temperature=max_temperature,
        device=device,
        beam_size=beam_size,
        translation_prompt=translation_prompt,
    )
    log(f"Model loaded: compute={translator.compute_type}, VAD={'on' if translator.vad else 'off'}.")
    return translator, gpu_name


class ASREngine:
    """One inference owner, fresh audio on demand, no stale work queue."""

    def __init__(self, cfg, translator, gpu_name, audio_source, caption_cb=None,
                 status_cb=None, log_cb=None, silence_threshold=0.001, debug=False):
        self.cfg = cfg
        self.translator = translator
        self.gpu_name = gpu_name
        self.audio_source = audio_source
        self.caption_cb = caption_cb or (lambda text: None)
        self.status_cb = status_cb or (lambda line: None)
        self.log_cb = log_cb or (lambda line: None)
        self.silence_threshold = silence_threshold
        self.debug = debug
        self._stop = threading.Event()
        self._thread = None
        self._inflight_start = 0.0
        self.inferences = 0
        self.final_passes = 0
        self.silence_skipped = 0

    def start(self):
        if self.is_running():
            raise RuntimeError("Translation engine is already running")
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="asr-engine", daemon=False)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join()
        # Native GPU inference cannot be interrupted, but capture can be.
        self.audio_source.stop()

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def _run(self):
        capture = self.audio_source
        if self.cfg.fixer_language == 'ja' and not hasattr(self.translator, 'transcribe_japanese'):
            self.log_cb('[ERROR] Japanese fixing requires the Moonshine backend')
            return
        policy = StreamPolicy(self.cfg.whisper_window, self.cfg.whisper_step,
                              self.silence_threshold, self.cfg.endpoint_silence)
        if self.cfg.phrase_only:
            policy = PhrasePolicy(self.silence_threshold, self.cfg.endpoint_silence,
                                  self.cfg.max_phrase_seconds)
        captions = CaptionStabilizer(None if self.cfg.phrase_only else self.cfg.caption_max_words)
        fixer = BackgroundFixer(self.cfg.fixer_interval, self.cfg.fixer_context)
        monitor = LatencyMonitor(self.cfg.warning_latency, self.cfg.backlog_latency)
        last_status = time.monotonic()
        llm = None
        previous_raw = ''
        previous_utterance_id = None
        if self.cfg.enable_llm_fixer:
            from llm_fixer import AsyncFixer
            llm = AsyncFixer(self.cfg.fixer_model, self.cfg.fixer_interval, self.log_cb,
                             cuda=self.cfg.fixer_cuda)

        from phrase_commit import CommitOutput, CommitPolicy, finalization_state
        if self.cfg.commit_only:
            policy = CommitPolicy(self.silence_threshold, self.cfg.commit_pause,
                                  self.cfg.max_phrase_seconds)
        output_type = CommitOutput if self.cfg.commit_only else CaptionOutput
        output = output_type(self.caption_cb, self.cfg.whisper_step,
                             self.cfg.caption_timeout, None if self.cfg.commit_only else llm,
                             self.silence_threshold)
        commit_source = ''
        finalizer_version = 0
        japanese_first = self.cfg.fixer_language == 'ja'
        if japanese_first:
            output.fixer = None  # Qwen runs before MT, never twice on the English result
        from japanese_fixer import JapaneseContext
        japanese_context = JapaneseContext()
        japanese_cache = None

        def translate_japanese(job, result):
            nonlocal finalizer_version, japanese_cache
            source = japanese_context.prepare(result.source_text, job)
            key = (source, tuple(item['phrase_id'] for item in japanese_context.history))
            corrected = source
            if japanese_cache is not None and japanese_cache[0] == key:
                corrected = japanese_cache[1]
            elif source and llm is not None and not llm.closed:
                finalizer_version += 1
                state = dict(language='ja', current=source, history=list(japanese_context.history),
                             phrase_id=job.utterance, start=job.start, end=job.end)
                llm.submit(finalizer_version, state, source)
                while not self._stop.wait(.05):
                    answer = llm.take(finalizer_version, max_age=float('inf'))
                    if isinstance(answer, str):
                        corrected = answer
                        break
                    if llm.closed or output.expired:
                        break
                japanese_cache = (key, corrected)
            result.source_text = corrected
            result.text = (self.translator.translate_source(corrected,
                           policy.decode_beam(self.translator.beam_size, job.final))
                           if corrected and not self._stop.is_set() and not output.expired else '')
            if self.debug:
                self.log_cb(f'[JA-FIX] original={source!r} corrected={corrected!r} english={result.text!r}')
            return result

        def commit_phrase(job, result):
            nonlocal commit_source, finalizer_version
            if job.cached_text is None:
                commit_source = getattr(result, 'source_text', '')
            text = result.text.strip()
            forced = policy.forced_split or job.cached_text is not None
            state = finalization_state(job, text, commit_source, output.history(), forced)
            if policy.should_wait(job, text, commit_source):
                if self.debug:
                    self.log_cb(f'[COMMIT] phrase={job.utterance} continuation-wait (once only)')
                policy.accept_result(job, text, complete=False)
                return
            # After the pause policy accepts a boundary, the LLM only edits.
            # It cannot repeatedly veto an endpoint or grow the audio window.
            state['audio_boundary_accepted'] = True
            state['audio_paused'] = (policy.last_voice is not None and
                                     job.end - policy.last_voice >= policy.pause - 1e-6)
            if japanese_first:
                result.source_text = commit_source
                try:
                    result = translate_japanese(job, result)
                except Exception as exc:
                    self.log_cb(f'[ERROR] Japanese correction/translation failed: {exc!r}')
                    policy.accept_result(job, '', complete=True)
                    commit_source = ''
                    return
                text, commit_source = result.text, result.source_text
            decision = {'action': 'commit', 'new_text': text}
            if text and llm is not None and not llm.closed and not japanese_first:
                finalizer_version += 1
                llm.submit(finalizer_version, state, text)
                while not self._stop.wait(0.05):
                    answer = llm.take(finalizer_version, max_age=float('inf'))
                    if answer is not None:
                        decision = answer
                        break
                    if llm.closed or output.expired:
                        break
                if self._stop.is_set():
                    return
            if decision['action'] == 'wait':
                decision = {'action': 'commit', 'new_text': text}
            # The application, not the LLM, owns timestamps and boundary cuts.
            final_text = decision.get('new_text') or text
            if self.debug:
                reason = ('duration-limit' if policy.forced_split else
                          'pause-fallback' if job.cached_text is not None else 'audio-pause')
                self.log_cb(f'[COMMIT] phrase={job.utterance} reason={reason} '
                            f'audio={job.start:.2f}-{job.end:.2f}s source={commit_source!r} '
                            f'raw={text!r} final={final_text!r}')
            accepted = policy.accept_result(job, final_text, complete=True)
            if accepted and output.offer_commit(job, final_text):
                # No latest-wins queue in this mode: publish this phrase before
                # decoding the next. Capture remains active in its bounded ring.
                while output.busy() and not self._stop.wait(0.05):
                    pass
                if japanese_first and output.ledger.last_id == job.utterance:
                    japanese_context.commit(job, commit_source)
            elif text:
                self.log_cb('[COMMIT] phrase expired before publication; not added to history')
            commit_source = ''

        def observe_audio():
            end = capture.live_position()
            if end > output.seen:
                seconds = min(self.cfg.audio_buffer, end, end - output.seen)
                if seconds > 0:
                    got = capture.latest(seconds)
                    if got:
                        output.observe(*got, time.monotonic())

        def watchdog():
            last_warning = 0.0
            while not self._stop.wait(0.05):
                now = time.monotonic()
                # This timer stays responsive even during native ASR inference.
                observe_audio()
                if self._stop.is_set():
                    break
                output.tick(now)
                start = self._inflight_start
                if start and now - start > 30 and now - last_warning >= 5:
                    self.log_cb("[ASR] inference exceeds 30s; check device/compute type")
                    last_warning = now

        watcher = threading.Thread(target=watchdog, name="asr-watchdog", daemon=True)
        try:
            capture.start()
            self.log_cb(f"[ASR] mode={'commit-only' if self.cfg.commit_only else 'phrase-only' if self.cfg.phrase_only else 'live'} "
                        f"context={self.cfg.whisper_window:g}s step={self.cfg.whisper_step:g}s "
                        f"fixer={'on' if self.cfg.enable_background_fixer else 'off'} "
                        f"llm_fixer={'on' if self.cfg.enable_llm_fixer else 'off'}")
            self.log_cb(f"[ASR] source: {getattr(capture, 'device_name', '') or 'streamed audio'}")
            watcher.start()
            # No snapshot producer: the worker requests audio only when idle.
            while not self._stop.wait(0.05):
                now = time.monotonic()
                available = capture.available_seconds()
                if available >= 0.1 and capture.live_position() > policy.seen + 1e-6:
                    got = capture.latest(policy.snapshot_seconds(capture.live_position(), available))
                    if got:
                        audio, end = got
                        output.observe(audio, end, time.monotonic())
                        if self.cfg.phrase_only and end - len(audio) / 16000 > policy.seen + 0.02:
                            self.log_cb("[ASR] phrase audio history exhausted; some audio was lost. "
                                        "Increase --audio-buffer or reduce --beam-size.")
                        previous_end = policy.last_decode
                        previous_utterance = policy.utterance
                        job = policy.next(audio, end)
                        if job is not None:
                            if self.cfg.phrase_only and policy.forced_split:
                                self.log_cb("[ASR] phrase safety limit reached; splitting without confirmed sentence completion")
                            beam = policy.decode_beam(self.translator.beam_size, job.final)
                            if (previous_end > 0 and previous_utterance == job.utterance
                                    and job.start > previous_end + 0.02):
                                self.log_cb(f"[ASR] audio history exhausted: "
                                            f"{job.start - previous_end:.2f}s no longer available; "
                                            "increase --audio-buffer or use a faster device")
                            self._inflight_start = time.monotonic()
                            try:
                                result = (SimpleNamespace(text=job.cached_text)
                                          if job.cached_text is not None else
                                          self.translator.transcribe_japanese(job.audio) if japanese_first else
                                          self.translator.translate(job.audio, beam_size=beam))
                                if (japanese_first and not self.cfg.commit_only and result is not None
                                        and job.cached_text is None):
                                    result.source_text = getattr(result, 'source_text', result.text)
                                    result = translate_japanese(job, result)
                            except Exception as exc:
                                self.log_cb(f"[ERROR] inference failed: {exc!r}")
                                result = None
                            finally:
                                dt = time.monotonic() - self._inflight_start
                                self._inflight_start = 0.0
                            if job.cached_text is None:
                                policy.record_duration(dt)
                                self.inferences += 1
                                self.final_passes += int(job.final)
                            monitor.update(live=capture.live_position(), transcribed=job.end,
                                           whisper_dur=dt, pending=0)
                            if self._stop.is_set():
                                break
                            if result is None and self.cfg.phrase_only:
                                policy.accept_result(job, "")
                            if result is not None:
                                if self.cfg.commit_only:
                                    commit_phrase(job, result)
                                    continue
                                text = result.text
                                if self.cfg.enable_background_fixer:
                                    text = fixer.fix_unstable_transcript("", text)
                                publish = (not self.cfg.phrase_only or policy.accept_result(job, text))
                                update = captions.update(text, job) if publish else None
                                if update is not None:
                                    context = (('Overlapping hypothesis: ' if previous_utterance_id == job.utterance
                                                else 'Earlier completed phrase (not overlapping): ') + previous_raw
                                               if previous_raw else '')
                                    previous_raw = update
                                    previous_utterance_id = job.utterance
                                    output.offer(update, context)
                                monitor.update(duplicates=captions.duplicates)
                                if self.debug:
                                    self.log_cb(f"[DECODE] {job.start:.2f}-{job.end:.2f}s "
                                                f"beam={beam} final={job.final} inference={dt:.3f}s "
                                                f"lag={capture.live_position() - job.end:.3f}s text={text!r}")
                        elif policy.start is None:
                            self.silence_skipped += 1
                            # Audio inspected and found idle is processed, not backlog.
                            monitor.update(live=end, transcribed=end)
                now = time.monotonic()
                if self.cfg.enable_latency_monitor and now - last_status >= 2.0:
                    self.status_cb(monitor.status_line())
                    last_status = now
        except Exception as exc:
            self.log_cb(f"[ERROR] translation engine stopped: {exc!r}")
        finally:
            self._stop.set()
            if watcher.is_alive():
                watcher.join()
            if llm is not None:
                llm.stop(wait=True)
            capture.stop()
