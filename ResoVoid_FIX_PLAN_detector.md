# Fix Plan: Resonance Suppressor Has No Audible Effect

> STATUS (2026-09-10): **RESOLVED — implemented in `src/dsp/detector.rs`.**
> `process_frame` now computes a per-frame `spectral_median()` (half-window 6,
> 13 bands, allocation-free) and smooths *toward the median*, not the band's own
> level. Unit tests (`spectral_median_is_robust_to_outlier`, detector hold tests)
> + full `cargo test` (54 passed) green. Remaining work is ear-testing on
> sustained resonant material + Delta-mode spot-check, not code.

Companion to `PROJECT_CONTEXT.md`. This file is scoped to one specific,
user-confirmed bug: **ResoVoid produces no audible change to the source
material**, even with Depth/Selectivity pushed to extremes. This document
diagnoses the root cause and lays out a concrete implementation plan.

## Symptom

User report: "I tried to use the plugin, and I couldn't notice any
changes on how the samples sound when I used ResoVoid."

## Root cause

`Detector::process_frame` in `src/dsp/detector.rs` computes each band's
"baseline" (the reference level that triggers suppression) as a smoothed
version of **that same band's own level over time**:

```rust
let bcoef = if target > self.baseline[i] {
    baseline_attack   // derived from the Attack param, default 10ms
} else {
    baseline_release
};
self.baseline[i] += (target - self.baseline[i]) * bcoef;
```

This is a pure per-band temporal envelope follower. When a band's level
rises — e.g. a sustained note lands on a resonant frequency — the
baseline chases it upward at the **attack** rate (10ms by default). Within
roughly one attack-time constant, `baseline[i]` has converged on the
resonance itself, `excess = level - (baseline + threshold)` collapses
back toward zero, and gain reduction relaxes back to ~unity.

Net effect: the detector can only react to *very short* transients before
its own baseline "learns" the resonance as normal and stops flagging it.
For anything sustained — which is the primary use case for a resonance
suppressor — suppression is audible for at most a few tens of
milliseconds per onset, if at all, and then silently disengages. This
fully explains "no audible change."

### Why this diverges from the documented algorithm

`README.md` advertises:

> Intelligent resonance detection with **median-filtered threshold
> calculation**

That description implies a **spectral** (cross-band) reference: comparing
each frequency bin/band against its *neighboring bands at the same
instant*, so a narrow resonant peak stands out against the surrounding
spectral envelope regardless of how long it persists in time. This is
the Soothe2-style approach and is fundamentally different from a
per-band time-domain follower. The current `detector.rs` does no
cross-band comparison at all — the implementation and the documented
behavior have diverged.

## Diagnostic steps (do first, before changing code)

Confirm the diagnosis experimentally before touching `detector.rs`:

1. Set Depth = 100%, Selectivity = 100%, Attack = 40–50ms (slow), Release
   = fast. Play a sustained resonant tone/note. Expect: a brief dip in
   level right at onset, then it disappears — confirms detection fires
   but doesn't hold.
2. Enable Delta mode (outputs only the removed content). If Delta is
   near-silent even at extreme settings, there may be a *second*,
   independent bug (e.g. gains never leaving ~1.0, a band-mapping issue
   in `bands.rs`, or the biquad `set_peak` gain math) — investigate that
   before or alongside the detector rewrite. If Delta produces short
   clicks/pips at onsets but nothing sustained, that confirms the root
   cause above and no other bug is masking it.
3. Log (`nice_dbg!` or temporary `eprintln!`) `self.baseline[i]` and
   `levels_db[i]` for one band over a few seconds of sustained tone to
   directly observe the baseline converging on the signal.

## Fix plan

### Goal
Replace (or augment) the per-band temporal baseline with a **spectral**
reference computed across bands within each analysis frame, so sustained
resonances stay flagged for as long as they stick out from the
surrounding spectral shape — not just at onset.

### Design

For each analysis frame (64 band levels in `levels_db`):

1. Compute a **spectral envelope / local median** per band using
   neighboring bands rather than time history. Practical options, in
   increasing complexity:
   - **Option A — sliding median (recommended starting point):** for
     band `i`, take the median of `levels_db` over a window of `±N`
     neighboring bands (e.g. N = 4–8, tunable, roughly matching how
     "wide" a resonance vs. the surrounding spectral shape should be
     judged). This directly matches the README's "median-filtered
     threshold" description.
   - **Option B — smoothed spectral envelope:** a simple moving average
     or exponential smoothing *across the band axis* (not across time)
     to get a low-order approximation of the local spectral trend.
   - Median (A) is more robust to a resonance itself skewing the
     reference (a moving average across bands would be dragged upward by
     a wide/loud resonance; median is far less sensitive to a single
     outlier band).
2. Keep a **light time-domain smoothing on top** of the spectral
   reference (not on the raw per-frame value) so the reference doesn't
   jitter wildly frame-to-frame — but this smoothing should track the
   *spectral median*, not the band's own raw level, so it no longer
   converges on a sustained resonance.
3. `excess = levels_db[i] - (spectral_reference[i] + threshold)` as
   before; downstream gain-reduction and attack/release smoothing on the
   *gain* (already implemented in `process_frame`) can stay mostly
   unchanged — that part of the ballistics design is fine, it's only the
   baseline that's wrong.
4. Keep `Selectivity` controlling the threshold offset as it does now.
   `Depth`/`Sharpness` continue to scale reduction amount as now.

### Concrete steps for opencode's AI model

1. **Confirm the bug** using the diagnostic steps above; report findings
   before changing code (especially whether Delta mode reveals a second
   bug).
2. In `src/dsp/detector.rs`:
   - Add a per-frame spectral median (or moving-average) computation
     over `levels_db` — a free function, e.g. `fn spectral_median(levels:
     &[f32; BANDS], window: usize) -> [f32; BANDS]`, using a
     small-window median over neighboring band indices (mind edge
     bands — clamp the window at the array boundaries, don't wrap).
   - Replace `self.baseline[i]`'s per-band-history update with logic
     that smooths *toward the spectral median value at band i*, not
     toward `levels_db[i]` itself. Keep using `baseline_attack` /
     `baseline_release` coefficients for this smoothing (rename if it
     clarifies intent, e.g. `self.reference[i]`).
   - Everything below the reference calculation in `process_frame`
     (`excess`, `reduction_db`, `target_gain`, gain smoothing) can stay
     as-is structurally; just feed it the new reference instead of
     `self.baseline[i] + thr`.
   - Update `Detector::reset()` to clear whatever new state field(s) are
     added.
3. Add a `window` concept — either a new `DetectParams` field (e.g.
   `spectral_width: f32` mapped to a band-count window, possibly tied to
   existing `Selectivity` or a new control) or a fixed constant to start
   (e.g. `const MEDIAN_WINDOW_BANDS: usize = 6;`) — ship the fixed
   constant first, only expose a parameter if testing shows it's worth
   tuning by ear.
4. Rebuild and re-run the diagnostic steps above. Expect: suppression
   should now persist for the duration of a sustained resonant tone, not
   just its onset.
5. Sanity-check CPU cost: computing a per-band median over a small window
   (e.g. 6–13 bands) once per analysis frame (not per sample) is cheap —
   confirm no allocation is introduced (use a fixed-size stack array or
   in-place partial sort over a small slice, not a `Vec`/heap alloc, to
   preserve the allocation-free audio-thread guarantee noted in
   `PROJECT_CONTEXT.md`).
6. Update `README.md` if the final algorithm differs from "median-
   filtered threshold" wording, or leave it as-is if the fix now matches
   that description (it should, if Option A is implemented as described).
7. Cross-reference `PROJECT_CONTEXT.md` issue list — this fix is
   unrelated to the latency-reporting bug (issue #1) or the STFT/overlap-
   add discrepancy (issue #3), but note in that document once this is
   resolved, since issue #3 (non-overlapping windows) affects how often
   this new spectral detection can react (once per full FFT window,
   same cadence as today).

## Acceptance criteria

- Sustained resonant tones show continuous gain reduction for as long as
  they remain spectrally distinct from their neighboring bands, not just
  a brief onset dip.
- Delta mode produces audible, sustained content (not just clicks) when
  a resonance is present in the source material.
- No new heap allocation introduced in `process_frame` or its helpers
  (verify with `--features assert_process_allocs`).
- Broadband/non-resonant material (e.g. pink noise, full mixes without a
  dominant peak) should NOT trigger heavy suppression — spot-check this
  to avoid over-triggering now that detection is more sensitive to
  genuinely narrow spectral peaks.
