import argparse
import sys
import threading
import time

import torch

from audio_capture import AudioCapture
from translator import Translator
from text_dedup import Deduplicator

MODEL_ID = "kotoba-tech/kotoba-whisper-bilingual-v1.0-faster"


def fmt_elapsed(seconds):
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def parse_args():
    p = argparse.ArgumentParser(
        description="Japanese -> English realtime translator (system audio)"
    )
    p.add_argument("--buffer", type=float, default=5.0, help="rolling buffer length in seconds (default 5)")
    p.add_argument("--step", type=float, default=1.0, help="inference interval in seconds (default 1)")
    p.add_argument(
        "--compute-type",
        default="int8_float16",
        choices=["float16", "int8_float16", "int8"],
        help="CTranslate2 compute type (default int8_float16)",
    )
    return p.parse_args()


def main():
    args = parse_args()

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
        translator = Translator(MODEL_ID, compute_type=args.compute_type)
    except Exception as exc:
        print(f"[ERROR] Failed to load model on CUDA: {exc}")
        sys.exit(1)

    capture = AudioCapture(buffer_seconds=args.buffer)
    capture.start()
    print(f"[Audio] Using output device: {capture.device_name}")
    print("Loopback: enabled")
    print(f"Compute: {args.compute_type}")
    print(f"Buffer: {args.buffer:g} seconds")
    print(f"Step: {args.step:g} seconds")
    print()
    print("Listening...", flush=True)
    print()

    stop = threading.Event()
    dedup = Deduplicator()
    start_time = time.monotonic()

    def worker():
        while not stop.is_set():
            if capture.available_seconds() >= args.buffer:
                break
            time.sleep(0.05)

        next_t = time.monotonic()
        while not stop.is_set():
            audio = capture.latest(args.buffer)
            if audio is None:
                stop.wait(0.05)
                next_t = time.monotonic() + args.step
                continue
            try:
                text = translator.translate(audio)
            except Exception as exc:
                print(f"[ERROR] inference failed: {exc}")
                text = ""
            if text:
                new = dedup.feed(text)
                if new:
                    elapsed = time.monotonic() - start_time
                    print(f"[{fmt_elapsed(elapsed)}] {new}", flush=True)
            next_t += args.step
            delay = next_t - time.monotonic()
            if delay > 0:
                if stop.wait(delay):
                    break
            else:
                next_t = time.monotonic() + args.step

    worker_thread = threading.Thread(target=worker, name="inference-worker", daemon=True)
    worker_thread.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    print("\nStopping...", flush=True)
    stop.set()
    capture.stop()
    worker_thread.join(timeout=2.0)
    print("Stopped.", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
