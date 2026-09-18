"""Deterministic tests for latest-job-wins scheduling and latency boundedness."""

import io

from transcript import LatestWinsScheduler, Snapshot, TranscriptRenderer


def simulate_grab_fresh(duration, interval, inference_dur, window=5.0):
    """Model the whisper loop: grab the freshest window, infer, wait `interval` only
    when inference is faster than the interval. Returns (processed, skipped, final_latency)."""
    clock = float(window)
    processed = 0
    transcribed_to = 0.0
    sched = LatestWinsScheduler()
    while clock < duration:
        sched.submit(Snapshot(audio=None, audio_end=clock, id=processed))
        job = sched.take()
        assert job is not None
        assert sched.pending_count == 0
        finish = clock + inference_dur
        transcribed_to = job.audio_end
        processed += 1
        clock = max(finish, clock + interval)
    return processed, sched.skipped, (clock - transcribed_to)


def test_scheduler_latest_wins():
    s = LatestWinsScheduler()
    for i in range(5):
        s.submit(Snapshot(audio=None, audio_end=float(i), id=i))
        assert s.pending_count <= 1, "pending must never exceed 1"
    job = s.take()
    assert job is not None and job.id == 4, "newest snapshot must win"
    assert s.skipped == 4, "4 older snapshots must be counted as skipped"
    assert s.pending_count == 0


def test_latency_bounded():
    # Test 1 - faster than interval: ~every snapshot, low latency
    processed, skipped, lat = simulate_grab_fresh(60.0, 1.0, 0.5)
    assert processed >= 55, processed
    assert skipped == 0, skipped
    assert lat < 1.5, lat

    # Test 2 - slightly slower than interval: latency bounded (~inference time)
    processed, skipped, lat = simulate_grab_fresh(60.0, 1.0, 1.2)
    assert lat < 3.0, f"latency must stay bounded, got {lat}"

    # Test 3 - much slower: latency still bounded, does not grow with duration
    _, _, lat_short = simulate_grab_fresh(30.0, 1.0, 3.0)
    _, _, lat_long = simulate_grab_fresh(300.0, 1.0, 3.0)
    assert lat_long < 5.0, f"latency must not grow unboundedly, got {lat_long}"
    assert abs(lat_short - lat_long) < 2.0, (lat_short, lat_long)


def test_timestamp_is_audio_time():
    # A result whose audio_end is 90.0s must be stamped 00:01:30, not completion time.
    r = TranscriptRenderer(stream=io.StringIO(), timestamps=True)
    r.render("hello", "", 90.0)  # audio position
    out = r._stream.getvalue()
    assert "[00:01:30]" in out, out


def simulate_split(duration, interval, inference_dur, backlog=2.0):
    """Model the split architecture: an independent producer makes a snapshot every
    `interval`; the consumer runs inference on the newest pending snapshot and skips
    stale ones. Returns (processed, skipped, max_pending, final_latency)."""
    sched = LatestWinsScheduler()
    clock = 0.0
    next_snap = 0.0
    busy_until = 0.0
    sid = 0
    processed = 0
    max_pending = 0
    transcribed_to = 0.0
    while clock < duration:
        if clock >= next_snap:  # producer (independent)
            next_snap += interval
            sched.submit(Snapshot(audio=None, audio_end=clock, id=sid))
            sid += 1
        if clock >= busy_until:  # consumer
            job = sched.take()
            if job is not None:
                if clock - job.audio_end > backlog:
                    sched.skipped += 1
                else:
                    processed += 1
                    busy_until = clock + inference_dur
                    transcribed_to = job.audio_end
        max_pending = max(max_pending, sched.pending_count)
        clock += 0.001
    return processed, sched.skipped, max_pending, (clock - transcribed_to)


def test_split_cadence():
    # Test 1 - faster than interval: ~every snapshot, no backlog, ~1s cadence.
    processed, skipped, max_pending, lat = simulate_split(60.0, 1.0, 0.5)
    assert processed >= 55, processed
    assert skipped == 0, skipped
    assert max_pending <= 1
    assert lat < 1.5, lat

    # Test 2 - slightly slower: pending never exceeds 1, old windows skipped, bounded.
    processed, skipped, max_pending, lat = simulate_split(60.0, 1.0, 1.2)
    assert max_pending <= 1
    assert skipped > 0, skipped
    assert lat < 3.0, f"latency must stay bounded, got {lat}"

    # Test 3 - much slower: only newest pending survives, latency bounded.
    _, skipped, max_pending, lat = simulate_split(60.0, 1.0, 3.0)
    assert max_pending <= 1
    assert skipped > 0, skipped
    assert lat < 5.0, f"latency must not grow unboundedly, got {lat}"


if __name__ == "__main__":
    test_scheduler_latest_wins()
    test_latency_bounded()
    test_split_cadence()
    test_timestamp_is_audio_time()
    print("All scheduler tests passed.")
