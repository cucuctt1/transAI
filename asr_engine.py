"""Reusable ASR engine: audio capture -> Whisper -> transcript stabilization -> captions.

Runs the same latest-job-wins scheduling, rollback, and background-fixer logic as the
CLI, but routes the stabilized caption to a callback instead of the terminal.

IMPORTANT: torch + the model MUST be loaded via ``load_model()`` BEFORE importing
PyQt5. Initializing CTranslate2/CUDA after Qt has been imported crashes the process
on Windows (a native segfault with no traceback).
"""

import threading
import time

import numpy as np

from transcript import (
    BackgroundFixer,
    LatestQueue,
    LatestWinsScheduler,
    LatencyMonitor,
    Snapshot,
    TranscriptState,
    tokenize,
)

MODEL_ID = "kotoba-tech/kotoba-whisper-bilingual-v1.0-faster"


def load_model(
    compute_type="int8_float16",
    vad=True,
    min_words=1,
    min_logprob=-1.0,
    max_compression_ratio=2.4,
    max_temperature=1.0,
    log_cb=None,
):
    """Load torch + the Whisper model. Call on the main thread, before PyQt5."""
    log = log_cb or (lambda line: print(line, flush=True))
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    gpu_name = torch.cuda.get_device_name(0)
    log(f"GPU: {gpu_name}")
    log("Loading model (this happens once)...")

    from translator import Translator

    translator = Translator(
        MODEL_ID,
        compute_type=compute_type,
        vad=vad,
        min_words=min_words,
        min_logprob=min_logprob,
        max_compression_ratio=max_compression_ratio,
        max_temperature=max_temperature,
    )
    log("Model loaded.")
    return translator, gpu_name


class ASREngine:
    def __init__(self, cfg, translator, gpu_name, audio_source, caption_cb=None, status_cb=None,
                 log_cb=None, silence_threshold=0.001):
        self.cfg = cfg
        self.translator = translator
        self.gpu_name = gpu_name
        self.audio_source = audio_source
        self.caption_cb = caption_cb or (lambda text: None)
        self.status_cb = status_cb or (lambda line: None)
        self.log_cb = log_cb or (lambda line: None)
        self.silence_threshold = silence_threshold

        self._stop = threading.Event()
        self._thread = None
        # pipeline counters (diagnostics)
        self.snapshots_submitted = 0
        self.inferences = 0
        self.rms_skipped = 0
        self.not_enough_audio = 0
        self.jobs_taken = 0
        self.stale_skipped = 0
        self.first_inference_logged = False
        self._inflight_start = 0.0
        self._hang_reported = False

    def start(self):
        self._thread = threading.Thread(target=self._run, name="asr-engine", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def is_running(self):
        return self._thread is not None and self._thread.is_alive()

    def _run(self):
        capture = self.audio_source
        try:
            capture.start()
        except Exception as exc:
            self.log_cb(f"[ERROR] audio source failed: {exc}")
            return
        device = getattr(capture, "device_name", "") or "streamed audio"
        self.log_cb(f"[ASR] audio source ready: {device}")

        state = TranscriptState(rollback_window=self.cfg.rollback_window)
        whisper_q = LatestQueue(maxsize=self.cfg.max_whisper_queue)
        scheduler = LatestWinsScheduler(max_pending=self.cfg.max_whisper_pending)
        monitor = LatencyMonitor(warning=self.cfg.warning_latency, backlog=self.cfg.backlog_latency)
        fixer = BackgroundFixer(interval=self.cfg.fixer_interval, context_seconds=self.cfg.fixer_context)
        snapshot_id = 0

        def snapshot_loop():
            nonlocal snapshot_id
            while not self._stop.is_set():
                if capture.available_seconds() >= self.cfg.whisper_window:
                    break
                time.sleep(0.05)
            next_t = time.monotonic()
            while not self._stop.is_set():
                next_t += self.cfg.whisper_step
                got = capture.latest(self.cfg.whisper_window)
                if got is None:
                    self.not_enough_audio += 1
                else:
                    audio, audio_end = got
                    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
                    if rms >= self.silence_threshold:
                        scheduler.submit(Snapshot(audio=audio, audio_end=audio_end, id=snapshot_id))
                        snapshot_id += 1
                        self.snapshots_submitted += 1
                    else:
                        self.rms_skipped += 1
                delay = next_t - time.monotonic()
                if delay > 0:
                    if self._stop.wait(delay):
                        break
                else:
                    next_t = time.monotonic()

        def whisper_loop():
            try:
                while not self._stop.is_set():
                    job = scheduler.take()
                    if job is None:
                        self._stop.wait(0.02)
                        continue
                    self.jobs_taken += 1
                    if capture.live_position() - job.audio_end > self.cfg.backlog_latency:
                        scheduler.skipped += 1
                        self.stale_skipped += 1
                        continue
                    if not self.first_inference_logged:
                        self.first_inference_logged = True
                        a = job.audio
                        self.log_cb(
                            f"[ASR] first inference: samples={len(a)} dtype={a.dtype} "
                            f"rms={float(np.sqrt(np.mean(a.astype(np.float64) ** 2))):.5f} "
                            f"nan={bool(np.isnan(a).any())} maxabs={float(np.max(np.abs(a))):.4f}"
                        )
                    t0 = time.monotonic()
                    self._inflight_start = t0
                    self._hang_reported = False
                    try:
                        res = self.translator.translate(job.audio)
                    except Exception as exc:
                        self.log_cb(f"[ERROR] inference failed: {exc!r}")
                        res = None
                    finally:
                        self._inflight_start = 0.0
                    dt = time.monotonic() - t0
                    self.inferences += 1
                    monitor.update(live=capture.live_position(), transcribed=job.audio_end,
                                   whisper_dur=dt, skipped=scheduler.skipped, pending=scheduler.pending_count)
                    if res is None:
                        continue
                    whisper_q.put((res.text, job.audio_end, dt))
            except Exception as exc:
                self.log_cb(f"[ERROR] whisper thread died: {exc!r}")

        def fixer_loop():
            while not self._stop.is_set():
                self._stop.wait(self.cfg.fixer_interval)
                version, finalized, unstable, needs_fix = state.snapshot()
                if not unstable or not needs_fix:
                    continue
                ctx = tokenize(finalized)[-int(self.cfg.fixer_context * self.cfg.words_per_second):]
                corrected = fixer.fix_unstable_transcript(" ".join(ctx), unstable)
                if corrected != unstable:
                    state.apply_fix(version, tokenize(corrected))

        def watchdog_loop():
            while not self._stop.is_set():
                self._stop.wait(5.0)
                start = self._inflight_start
                if start and not self._hang_reported and time.monotonic() - start > 30.0:
                    self._hang_reported = True
                    self.log_cb(
                        f"[ERROR] inference has been running for {time.monotonic() - start:.0f}s "
                        f"- CTranslate2/CUDA appears hung on this device"
                    )

        threading.Thread(target=snapshot_loop, name="snapshot", daemon=True).start()
        threading.Thread(target=whisper_loop, name="whisper", daemon=True).start()
        threading.Thread(target=watchdog_loop, name="watchdog", daemon=True).start()
        if self.cfg.enable_background_fixer:
            threading.Thread(target=fixer_loop, name="fixer", daemon=True).start()

        last_status_t = 0.0
        last_pipe_t = 0.0
        while not self._stop.is_set():
            item = whisper_q.get(timeout=0.2)
            monitor.update(live=capture.live_position())
            if item is not None:
                text, audio_end, wdt = item
                state.apply_hypothesis(text)
                monitor.update(duplicates=state.noop_count)
                self.caption_cb(state.latest_caption())
            now = time.monotonic()
            if self.cfg.enable_latency_monitor and now - last_status_t >= 2.0:
                self.status_cb(monitor.status_line())
                last_status_t = now
            if now - last_pipe_t >= 2.0:
                live = capture.live_position()
                avail = capture.available_seconds()
                pipe_state = "ok" if avail >= self.cfg.whisper_window else "WAITING_FOR_AUDIO"
                self.log_cb(
                    f"[PIPE] captured={live:.1f}s buffered={avail:.1f}s window={self.cfg.whisper_window:.1f}s "
                    f"snapshots={self.snapshots_submitted} jobs={self.jobs_taken} inferences={self.inferences} "
                    f"rms_skipped={self.rms_skipped} stale_skipped={self.stale_skipped} "
                    f"not_enough_audio={self.not_enough_audio} -> {pipe_state}"
                )
                last_pipe_t = now

        capture.stop()
