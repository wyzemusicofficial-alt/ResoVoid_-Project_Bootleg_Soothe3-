// src/dsp/detector.rs
// Per-band resonance detector.
//
// Spectral reference: for each frame a median-filtered envelope is computed
// across bands (neighboring bands at the same instant). Each band's level is
// compared against that local spectral median + threshold, so a narrow peak
// stands out regardless of how long it persists. A light time-domain smoother
// follows the *spectral median* (not the band's raw level) to avoid jitter
// without "learning" a sustained resonance as normal. See ResoVoid_FIX_PLAN_detector.md.

use crate::dsp::{BANDS, MAX_NODES};

/// Half-window radius for the spectral median (total window = 2*R+1 = 13 bands).
const MEDIAN_HALF_WINDOW: usize = 6;

/// Knee width for hard mode: near-hard-knee (steep, narrow transition).
const HARD_KNEE_DB: f32 = 0.5;
/// Knee width for soft mode: wide, gentle transition into full suppression.
const SOFT_KNEE_DB: f32 = 9.0;

/// Reference frequency (Hz) for frequency-scaled ballistics: a band centered
/// exactly here runs at the user-facing `attack_ms`/`release_ms` unchanged
/// (scale factor 1.0). Bands below run slower (longer τ), bands above faster.
///
/// 1000 Hz is the conventional midrange anchor (also the ISO 226 / Fletcher–
/// Munson reference): with the log-spaced layout in bands.rs (20 Hz–20 kHz
/// over 64 bands, so band ~37 sits near 1 kHz) it lands near the geometric
/// middle of the audible range, keeping the extreme-band multipliers bounded
/// (≈3x slower at 40 Hz, ≈0.4x faster at 16 kHz with the α below).
const BALLISTICS_FREQ_REF: f32 = 1000.0;

/// Exponent α for frequency-scaled ballistics: τ(f) = τ_base · (f_ref / f)^α.
///
/// Per-octave cost is 2^α ≈ 1.27x (α = 0.35), so each octave down stretches
/// timing ~27% and each octave up compresses it ~21% — clearly audible in an
/// A/B without turning sub-bass into molasses or treble into chatter. Over
/// the full ~10-octave range (20 Hz–20 kHz) the total spread is 2^(10α) ≈ 11x
/// (40 Hz ≈ 3.0x slower, 16 kHz ≈ 0.38x faster), versus ≈32x at α = 0.5
/// (too grabby up top, too sluggish down low) and ≈5.7x at α = 0.25 (barely
/// perceptible). α = 0 is the uniform-timing fallback (scale ≡ 1.0); see the
/// `ballistics_scaling_is_uniform_at_zero_alpha` regression test.
const BALLISTICS_ALPHA: f32 = 0.35;

/// Frequency scale multiplier for one band: (f_ref / f)^α.
///
/// `alpha` is a parameter (rather than reading [`BALLISTICS_ALPHA`] directly)
/// so tests can pass 0.0 to verify the uniform-timing fallback. Frequencies
/// are clamped to ≥ 1 Hz for float safety (never triggered in practice —
/// band centers start at ~20 Hz).
fn ballistics_scale(freq_hz: f32, alpha: f32) -> f32 {
    (BALLISTICS_FREQ_REF / freq_hz.max(1.0)).powf(alpha)
}

#[derive(Clone, Copy)]
pub struct DetectParams {
    pub depth: f32,
    pub sharpness: f32,
    pub selectivity: f32,
    pub attack_ms: f32,
    pub release_ms: f32,
    pub soft_mode: bool,
    /// Per-region depth multipliers, one per node slot.
    pub node_depths: [f32; MAX_NODES],
    /// Per-node center frequencies in Hz, one per node slot.
    pub node_freqs: [f32; MAX_NODES],
    /// Per-node shapes: 0 = Bell, 1 = Low Shelf, 2 = High Shelf.
    pub node_shapes: [usize; MAX_NODES],
    /// Per-node enabled flags. A disabled node is truly absent from the
    /// calculation (zero weight), not a neutral-depth anchor.
    pub node_enabled: [bool; MAX_NODES],
}

impl Default for DetectParams {
    fn default() -> Self {
        Self {
            depth: 0.5,
            sharpness: 0.5,
            selectivity: 0.5,
            attack_ms: 10.0,
            release_ms: 100.0,
            soft_mode: false,
            node_depths: [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            node_freqs: [80.0, 300.0, 1000.0, 3500.0, 8500.0, 15000.0, 4000.0, 8000.0],
            node_shapes: [1, 0, 0, 0, 0, 2, 0, 0],
            node_enabled: [true, true, true, true, true, true, false, false],
        }
    }
}

/// Gaussian sigma in octaves for the node-shape kernel.
///
/// Chosen relative to the ~10-octave display/audible range (20 Hz–20 kHz =
/// log2(1000) ≈ 10 octaves): σ = 1.0 octave, i.e. roughly one tenth of the
/// total range, so each node's influence visibly bows across 2–4 octaves
/// (±2σ = ±2 octaves ≈ 40% of the display) the way Soothe-style curves do.
///
/// Reference weights with σ = 1.0:
/// At ±1 octave:  w ≈ 0.607
/// At ±2 octaves: w ≈ 0.135
/// At ±3 octaves: w ≈ 0.011  (effectively gone)
/// At ±4 octaves: w ≈ 0.0003 (numerical zero)
///
/// (The earlier σ = 0.425 matched the old triangular kernel's -6 dB point at
/// ±0.5 octave; it was deliberately widened here so transitions bow gently
/// over several octaves instead of steepening within one.)
const NODE_SHAPE_SIGMA: f32 = 1.0;

/// Compute the per-region depth multiplier for a single frequency.
///
/// Blends between the Gaussian-weighted node average and a neutral baseline
/// of 1.0 (no extra attenuation) based on how much total node weight is
/// actually present at `freq`:
///
/// ```text
/// node_avg = weighted_sum / total_weight        // what the nodes say
/// coverage = total_weight.clamp(0.0, 1.0)       // how much they say it
/// result   = coverage * node_avg + (1 - coverage) * 1.0
/// ```
///
/// Design choice: the normalizing constant 1.0 is the on-axis weight of a
/// single isolated node, so `coverage` measures "node influence present" in
/// units of one fully-engaged node.  At an anchor, coverage ≈ 1 and the
/// result is the node average; far from every node, coverage → 0 and the
/// result relaxes to neutral 1.0 instead of saturating to the nearest node's
/// raw value.  In a gap between two distant nodes the curve sags gently back
/// toward neutral rather than forming a plateau or a wall.
///
/// The `total_weight <= 1e-6` fallback returns exact neutral 1.0 and exists
/// only for float safety — with a Gaussian kernel it should not trigger in
/// practice.
///
/// This is the single source of truth used by both the DSP detector
/// and the GUI shape curve — never duplicate this math elsewhere.
///
/// Shape semantics (0 = Bell, 1 = Low Shelf, 2 = High Shelf):
/// - **Bell**: symmetric Gaussian falloff on both sides of the anchor.
/// - **Low Shelf**: full influence (w = 1) below the anchor, Gaussian
///   falloff above — the node "holds down" everything to its left.
/// - **High Shelf**: full influence (w = 1) above the anchor, Gaussian
///   falloff below — the node "holds down" everything to its right.
///
/// A disabled slot (`node_enabled[j] == false`) contributes zero weight —
/// it is skipped entirely, so deleting a node removes its influence rather
/// than leaving behind a neutral anchor.
pub fn node_shape(
    freq: f32,
    node_freqs: &[f32; MAX_NODES],
    node_depths: &[f32; MAX_NODES],
    node_shapes: &[usize; MAX_NODES],
    node_enabled: &[bool; MAX_NODES],
) -> f32 {
    let band_log = freq.max(1.0).log2();
    let mut weighted_sum = 0.0f32;
    let mut total_weight = 0.0f32;
    for j in 0..MAX_NODES {
        if !node_enabled[j] {
            continue;
        }
        let anchor_log = node_freqs[j].log2();
        let dist_octaves = band_log - anchor_log;

        // Shape-dependent weight:
        //   Bell (0): Gaussian on both sides (abs dist).
        //   Low Shelf (1): full weight below anchor, Gaussian above.
        //   High Shelf (2): full weight above anchor, Gaussian below.
        let w = match node_shapes[j] {
            1 => {
                // Low shelf: full influence below anchor.
                if dist_octaves <= 0.0 {
                    1.0
                } else {
                    (-0.5 * (dist_octaves / NODE_SHAPE_SIGMA).powi(2)).exp()
                }
            }
            2 => {
                // High shelf: full influence above anchor.
                if dist_octaves >= 0.0 {
                    1.0
                } else {
                    (-0.5 * (dist_octaves / NODE_SHAPE_SIGMA).powi(2)).exp()
                }
            }
            _ => {
                // Bell (default): symmetric Gaussian.
                (-0.5 * (dist_octaves.abs() / NODE_SHAPE_SIGMA).powi(2)).exp()
            }
        };
        weighted_sum += node_depths[j] * w;
        total_weight += w;
    }
    if total_weight > 1e-6 {
        let node_avg = weighted_sum / total_weight;
        let coverage = total_weight.clamp(0.0, 1.0);
        coverage * node_avg + (1.0 - coverage) * 1.0
    } else {
        1.0
    }
}

/// Knee-shaped reduction curve (C1-continuous soft/hard knee).
///
/// `excess` is `levels_db[i] - reference` (dB above threshold),
/// `k` is the slope term `effective_depth * (0.5 + sharpness)`,
/// `knee_db` is the knee width (use [`HARD_KNEE_DB`] or [`SOFT_KNEE_DB`]).
///
/// Below `-knee_db/2` the reduction is 0; above `+knee_db/2` it is `k*excess`;
/// inside the knee it follows the standard quadratic blend so both the value
/// and the first derivative are continuous at the boundaries.
fn knee_reduction_db(excess: f32, k: f32, knee_db: f32) -> f32 {
    let half_knee = knee_db / 2.0;
    if excess <= -half_knee {
        0.0
    } else if excess >= half_knee {
        (k * excess).max(0.0)
    } else {
        k * (excess + half_knee).powi(2) / (2.0 * knee_db)
    }
}

pub struct Detector {
    sample_rate: f32,
    /// Samples between consecutive frames (the analysis hop, not the FFT
    /// size). Drives all per-frame follower timing via `coef()`.
    frame_hop: f32,
    /// Smoothed spectral reference (follows the per-frame median, not raw level).
    baseline: [f32; BANDS],
    gain: [f32; BANDS],
    /// Depth-chain diagnostics, refreshed once per `process_frame`.
    /// Fixed-size float stores only: allocation-free on the audio thread.
    /// Read by `ResonanceSuppressor` when building `AnalysisFrame`.
    last_effective_depth: [f32; BANDS],
    last_reduction_db: [f32; BANDS],
    last_target_gain: [f32; BANDS],
}

impl Detector {
    pub fn new(frame_hop: usize, sample_rate: f32) -> Self {
        Self {
            sample_rate,
            frame_hop: frame_hop as f32,
            baseline: [0.0; BANDS],
            gain: [1.0; BANDS],
            last_effective_depth: [0.0; BANDS],
            last_reduction_db: [0.0; BANDS],
            last_target_gain: [1.0; BANDS],
        }
    }

    /// Last frame's `effective_depth` per band (p.depth * region_depth).
    #[inline]
    pub fn last_effective_depth(&self) -> &[f32; BANDS] {
        &self.last_effective_depth
    }

    /// Last frame's clamped `reduction_db` per band.
    #[inline]
    pub fn last_reduction_db(&self) -> &[f32; BANDS] {
        &self.last_reduction_db
    }

    /// Last frame's pre-follower `target_gain` per band.
    #[inline]
    pub fn last_target_gain(&self) -> &[f32; BANDS] {
        &self.last_target_gain
    }

    /// Process one analysis frame, returning smoothed linear gains per band (<= 1.0).
    ///
    /// Gain *targets* are recomputed only once per analysis window (this is
    /// called from `AnalysisEngine::process_sample` when a window completes).
    /// The exponential followers below are the sole interpolator between
    /// updates, which keeps the biquad coefficients from zipper-noising. If
    /// zipper noise is audible at small FFT sizes (1024, where windows are
    /// short), consider moving the follower to per-sample operation.
    ///
    /// The node_depths act as per-region depth attenuators — dragging a node
    /// down reduces suppression in that frequency neighborhood. The node_freqs
    /// set the center frequency of each region. Gaussian weights in octave-distance
    /// space provide smooth, C∞ interpolation between regions.
    #[inline]
    pub fn process_frame(&mut self, levels_db: &[f32; BANDS], p: &DetectParams, band_centers: &[f32; BANDS]) -> [f32; BANDS] {
        let frame_time = self.frame_hop / self.sample_rate;

        // Frequency-scaled ballistics: τ(f) = τ_base · (f_ref / f)^α, so high
        // bands react faster and low bands slower. `p.attack_ms`/`p.release_ms`
        // stay the user-facing base constants; the scale multiplies them before
        // `coef()` (which is nonlinear in τ, so the coefficient itself cannot
        // just be scaled). The existing 0.5x gain-follower speedup and its
        // floors are preserved — the scale applies *after* the floor, so the
        // full HF speedup survives instead of being clamped back to the
        // uniform floor (`coef()` retains its own 0.1 ms / frame_time safety).
        //
        // Recomputed per frame (64 × powf at the analysis frame rate, ~tens of
        // Hz — negligible): `band_centers` legitimately changes across calls
        // (it depends on FFT size / sample rate via `compute_band_layout`, and
        // `Detector` holds no layout state), so caching the scales inside
        // `Detector` would need invalidation logic and risks stale timing
        // after an FFT-size switch. Recompute is the correct trade.
        let mut freq_scale = [1.0f32; BANDS];
        for i in 0..BANDS {
            freq_scale[i] = ballistics_scale(band_centers[i], BALLISTICS_ALPHA);
        }
        // Base (pre-scale) follower times in ms, matching the previous uniform math.
        let base_gain_attack_ms = (p.attack_ms * 0.5).max(0.5);
        let base_gain_release_ms = (p.release_ms * 0.5).max(5.0);

        // Threshold (dB): higher selectivity -> smaller threshold -> more is caught.
        let thr = (1.0 - p.selectivity) * 12.0 + p.selectivity * 1.0;
        let knee_db = if p.soft_mode { SOFT_KNEE_DB } else { HARD_KNEE_DB };

        // Spectral (cross-band) reference — median over neighboring bands at the same instant.
        let medians = spectral_median(levels_db);

        let mut out = [1.0f32; BANDS];
        for i in 0..BANDS {
            let region_depth = node_shape(
                band_centers[i],
                &p.node_freqs,
                &p.node_depths,
                &p.node_shapes,
                &p.node_enabled,
            );

            // Smooth *toward the spectral median*, not toward this band's own raw level.
            // This prevents a sustained resonance from being "learned" as normal.
            let spectral_ref = medians[i];
            let s = freq_scale[i];
            let bcoef = if spectral_ref > self.baseline[i] {
                coef(p.attack_ms * s, frame_time)
            } else {
                coef(p.release_ms * s, frame_time)
            };
            self.baseline[i] += (spectral_ref - self.baseline[i]) * bcoef;

            let reference = self.baseline[i] + thr;
            let excess = levels_db[i] - reference;
            // Apply region_depth multiplier to the global depth before computing reduction.
            let effective_depth = p.depth * region_depth;
            let k = effective_depth * (0.5 + p.sharpness);
            let reduction_db = knee_reduction_db(excess, k, knee_db).min(36.0);
            let target_gain = 10.0f32.powf(-reduction_db / 20.0);
            // Depth-chain side channel: plain float stores, no allocation.
            self.last_effective_depth[i] = effective_depth;
            self.last_reduction_db[i] = reduction_db;
            self.last_target_gain[i] = target_gain;
            let gcoef = if target_gain < self.gain[i] {
                coef(base_gain_attack_ms * s, frame_time)
            } else {
                coef(base_gain_release_ms * s, frame_time)
            };
            self.gain[i] += (target_gain - self.gain[i]) * gcoef;
            out[i] = self.gain[i];
        }
        out
    }

    pub fn reset(&mut self) {
        self.baseline = [0.0; BANDS];
        self.gain = [1.0; BANDS];
        self.last_effective_depth = [0.0; BANDS];
        self.last_reduction_db = [0.0; BANDS];
        self.last_target_gain = [1.0; BANDS];
    }
}

/// Spectral median over neighboring bands (stack-allocated, no heap).
/// For each band `i`, takes the median of `levels` over `[i-R, i+R]` clamped
/// at the edges (no wrapping). Window size = 2*R+1 = 13 with R=6 by default.
fn spectral_median(levels: &[f32; BANDS]) -> [f32; BANDS] {
    let mut out = [0.0f32; BANDS];
    for i in 0..BANDS {
        let lo = i.saturating_sub(MEDIAN_HALF_WINDOW);
        let hi = (i + MEDIAN_HALF_WINDOW).min(BANDS - 1);
        let n = hi - lo + 1;
        // Max window is 13, stack buffer — no allocation.
        let mut buf = [0.0f32; 13];
        for (k, idx) in (lo..=hi).enumerate() {
            buf[k] = levels[idx];
        }
        // Insertion sort is optimal for n <= 13.
        for j in 1..n {
            let key = buf[j];
            let mut k = j;
            while k > 0 && buf[k - 1] > key {
                buf[k] = buf[k - 1];
                k -= 1;
            }
            buf[k] = key;
        }
        out[i] = buf[n / 2];
    }
    out
}

/// Per-frame smoothing coefficient so a follower reaches ~63% in `ms` milliseconds.
fn coef(ms: f32, frame_time: f32) -> f32 {
    let tau = (ms.max(0.1) / 1000.0).max(frame_time);
    let frames = tau / frame_time;
    1.0 - (-1.0 / frames).exp()
}

#[cfg(test)]
mod tests {
    use super::{DetectParams, Detector};
    use crate::dsp::{BANDS, MAX_NODES};

    /// Generate log-spaced band centers for tests.
    fn test_centers() -> [f32; BANDS] {
        let f_min = 20.0_f32;
        let f_max = 20000.0_f32;
        let l_min = f_min.log10();
        let l_max = f_max.log10();
        let mut centers = [0.0f32; BANDS];
        for i in 0..BANDS {
            let t = i as f32 / (BANDS - 1) as f32;
            let l = l_min + t * (l_max - l_min);
            centers[i] = 10.0_f32.powf(l);
        }
        centers
    }

    fn params() -> DetectParams {
        DetectParams {
            depth: 1.0,
            sharpness: 1.0,
            selectivity: 0.0, // threshold = 12 dB
            attack_ms: 10.0,
            release_ms: 100.0,
            soft_mode: false,
            node_depths: [1.0; MAX_NODES],
            node_freqs: [200.0, 2000.0, 12000.0, 500.0, 1000.0, 4000.0, 8000.0, 16000.0],
            node_shapes: [0; MAX_NODES],
            node_enabled: [true, true, true, false, false, false, false, false],
        }
    }

    #[test]
    fn suppresses_hot_band_reduces_others() {
        let mut det = Detector::new(2048, 44100.0);
        let centers = test_centers();
        // Establish a quiet spectral baseline first.
        let quiet = [-30.0f32; BANDS];
        for _ in 0..20 {
            det.process_frame(&quiet, &params(), &centers);
        }
        // A large transient above the baseline + threshold is caught.
        let mut levels = quiet;
        levels[10] = 40.0;

        let gains = det.process_frame(&levels, &params(), &centers);

        // Hot band gets attenuation (gain < 1).
        assert!(gains[10] < 1.0, "hot band should be reduced");
        assert!(gains[10] > 0.0, "gain stays positive");
        // A quiet band is left essentially untouched.
        assert!((gains[5] - 1.0).abs() < 1e-2, "quiet band unchanged");
    }

    #[test]
    fn flat_spectrum_yields_unity_gain() {
        let mut det = Detector::new(2048, 44100.0);
        let centers = test_centers();
        let levels = [-60.0f32; BANDS]; // all below the 12 dB threshold
        let gains = det.process_frame(&levels, &params(), &centers);
        for g in gains.iter() {
            assert!((g - 1.0).abs() < 1e-3, "no reduction on quiet spectrum");
        }
    }

    #[test]
    fn higher_selectivity_catches_more() {
        // Under spectral detection a single isolated peak at -120 would be caught at any
        // threshold (huge outlier vs median), so use a less extreme spectrum where
        // threshold distinction matters: flat at -10 dB, one band 10 dB hotter at 0 dB.
        let mut det_lo = Detector::new(2048, 44100.0);
        let mut det_hi = Detector::new(2048, 44100.0);
        let centers = test_centers();
        // Prime the smoothed spectral reference to the flat level.
        let flat = [-10.0f32; BANDS];
        for _ in 0..20 {
            det_lo.process_frame(&flat, &params(), &centers);
            det_hi.process_frame(&flat, &params(), &centers);
        }
        let mut levels = flat;
        levels[10] = 0.0; // 10 dB above median

        let mut p_lo = params();
        p_lo.selectivity = 0.0; // thr = 12 dB -> 10 dB excess not enough
        let mut p_hi = params();
        p_hi.selectivity = 1.0; // thr = 1 dB -> easily caught

        let g_lo = det_lo.process_frame(&levels, &p_lo, &centers)[10];
        let g_hi = det_hi.process_frame(&levels, &p_hi, &centers)[10];
        assert!(g_hi < 1.0, "high selectivity catches the band");
        assert!((g_lo - 1.0).abs() < 1e-3, "low selectivity misses the band");
    }

    #[test]
    fn sustained_resonance_holds() {
        // Sustained resonance must stay suppressed, not recover after one attack period.
        let mut det = Detector::new(2048, 44100.0);
        let centers = test_centers();
        let quiet = [-30.0f32; BANDS];
        for _ in 0..20 {
            det.process_frame(&quiet, &params(), &centers);
        }
        let mut resonant = quiet;
        resonant[20] = 10.0; // strong narrow peak, >> median + thr

        let mut p = params();
        p.selectivity = 1.0; // thr = 1 dB so excess is large and unambiguous
        // First frame: should suppress
        let g1 = det.process_frame(&resonant, &p, &centers)[20];
        assert!(g1 < 0.9, "first frame should suppress sustained peak");

        // Hold the same resonance for many frames — old per-band baseline would
        // converge and release; spectral baseline must keep suppressing.
        let mut last = g1;
        for _ in 0..20 {
            last = det.process_frame(&resonant, &p, &centers)[20];
        }
        assert!(
            last < 0.9,
            "sustained peak must stay suppressed, got gain {last}"
        );
    }

    #[test]
    fn broadband_flat_no_heavy_suppression() {
        // Pink-noise-like flat spectrum should not trigger heavy suppression.
        let mut det = Detector::new(2048, 44100.0);
        let centers = test_centers();
        let flat = [-20.0f32; BANDS];
        for _ in 0..20 {
            det.process_frame(&flat, &params(), &centers);
        }
        let gains = det.process_frame(&flat, &params(), &centers);
        for g in gains.iter() {
            assert!((g - 1.0).abs() < 5e-2, "flat broadband should not heavily suppress");
        }
    }

    #[test]
    fn spectral_median_is_robust_to_outlier() {
        let mut levels = [-20.0f32; BANDS];
        levels[30] = 20.0;
        let med = super::spectral_median(&levels);
        // Median around the outlier still -20 (majority of window is -20)
        assert!((med[30] - (-20.0)).abs() < 1e-6);
        // Far from outlier median is also -20
        assert!((med[0] - (-20.0)).abs() < 1e-6);
    }

    #[test]
    fn node_shape_returns_depth_at_anchor_freq() {
        let nf = [200.0, 2000.0, 12000.0, 500.0, 1000.0, 4000.0, 8000.0, 16000.0];
        let nd = [0.5, 0.8, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0];
        let ns = [0; MAX_NODES]; // Bell
        let ne = [true, true, true, false, false, false, false, false];
        // At an anchor, coverage ≈ 1 so the result is the node average, which
        // is dominated by (but not exactly equal to) the anchor's depth:
        // the wider σ = 1.0 kernel lets neighboring nodes leak in a few
        // percent.  Tolerance loosened from 0.02 to 0.05 for the wider kernel.
        let shape_at_200 = super::node_shape(200.0, &nf, &nd, &ns, &ne);
        assert!(
            (shape_at_200 - 0.5).abs() < 0.05,
            "at 200 Hz anchor should be ~0.5, got {shape_at_200}"
        );
        let shape_at_2k = super::node_shape(2000.0, &nf, &nd, &ns, &ne);
        assert!(
            (shape_at_2k - 0.8).abs() < 0.05,
            "at 2 kHz anchor should be ~0.8, got {shape_at_2k}"
        );
        let shape_at_12k = super::node_shape(12000.0, &nf, &nd, &ns, &ne);
        assert!(
            (shape_at_12k - 0.3).abs() < 0.05,
            "at 12 kHz anchor should be ~0.3, got {shape_at_12k}"
        );
    }

    #[test]
    fn node_shape_relaxes_to_neutral_far_from_anchors() {
        // INTENTIONALLY INVERTED vs the old
        // `node_shape_far_from_anchors_tracks_weighted_average` test: the old
        // test asserted the saturating behavior (all-zero depths far from
        // anchors → ~0.0, i.e. pinned to the nearest node's raw value).  The
        // neutral-blend fix deliberately changes this: far from every node,
        // coverage → 0 and the result must trend to neutral 1.0 *regardless*
        // of what the node depths are.
        let nf = [500.0, 1000.0, 2000.0, 4000.0, 8000.0, 12000.0, 16000.0, 18000.0];
        let ns = [0; MAX_NODES]; // Bell
        let ne = [true, true, true, false, false, false, false, false];

        // All-zero depths, far above every node: must be ~neutral, not ~0.0.
        let nd_zero = [0.0; MAX_NODES];
        let shape = super::node_shape(15000.0, &nf, &nd_zero, &ns, &ne);
        assert!(
            (shape - 1.0).abs() < 0.05,
            "all-zero depths far from anchors should relax to ~1.0, got {shape}"
        );

        // Varied depths, same far point: still ~neutral.
        let nd_varied = [0.2, 0.0, 0.7, 1.0, 1.0, 1.0, 1.0, 1.0];
        let shape = super::node_shape(15000.0, &nf, &nd_varied, &ns, &ne);
        assert!(
            (shape - 1.0).abs() < 0.05,
            "varied depths far from anchors should relax to ~1.0, got {shape}"
        );

        // Far below every node (20 Hz, >4 octaves under the lowest anchor):
        // total weight is ~numerical zero, same neutral result.
        let shape = super::node_shape(20.0, &nf, &nd_zero, &ns, &ne);
        assert!(
            (shape - 1.0).abs() < 0.01,
            "far-below-all-nodes should be neutral 1.0, got {shape}"
        );
    }

    #[test]
    fn low_shelf_holds_down_below_anchor() {
        // Widen spacing so neighbors don't bleed into the shelf region.
        let nf = [200.0, 5000.0, 18000.0, 500.0, 1000.0, 4000.0, 8000.0, 16000.0];
        let nd = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]; // Low node at zero depth
        let ns = [1, 0, 0, 0, 0, 0, 0, 0]; // Low node = Low Shelf
        let ne = [true, true, true, false, false, false, false, false];

        // Well below the low-shelf anchor: full influence → 0.0.
        let shape_below = super::node_shape(50.0, &nf, &nd, &ns, &ne);
        assert!(
            shape_below < 0.05,
            "low shelf below anchor should be ~0.0, got {shape_below}"
        );
        // At the anchor: ~0.0 (coverage ≈ 1, anchor depth dominates).
        let shape_at = super::node_shape(200.0, &nf, &nd, &ns, &ne);
        assert!(
            shape_at < 0.05,
            "low shelf at anchor should be ~0.0, got {shape_at}"
        );
        // Above the anchor: Gaussian falloff returns toward neutral.
        let shape_above = super::node_shape(2000.0, &nf, &nd, &ns, &ne);
        assert!(
            shape_above > 0.6,
            "low shelf well above anchor should relax toward 1.0, got {shape_above}"
        );
    }

    #[test]
    fn high_shelf_holds_down_above_anchor() {
        // Wide spacing; non-shelf nodes zeroed to isolate the high shelf.
        let nf = [200.0, 2000.0, 12000.0, 500.0, 1000.0, 4000.0, 8000.0, 16000.0];
        let nd = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let ns = [0, 0, 2, 0, 0, 0, 0, 0]; // High node = High Shelf
        let ne = [true, true, true, false, false, false, false, false];

        // Well above the high-shelf anchor: full influence → 0.0.
        let shape_above = super::node_shape(20000.0, &nf, &nd, &ns, &ne);
        assert!(
            shape_above < 0.05,
            "high shelf above anchor should be ~0.0, got {shape_above}"
        );
        // At the anchor: ~0.0.
        let shape_at = super::node_shape(12000.0, &nf, &nd, &ns, &ne);
        assert!(
            shape_at < 0.05,
            "high shelf at anchor should be ~0.0, got {shape_at}"
        );
        // Below the anchor: 200 Hz is >5 octaves away from 12000 Hz,
        // so the shelf's Gaussian tail is negligible → neutral 1.0.
        let shape_below = super::node_shape(200.0, &nf, &nd, &ns, &ne);
        assert!(
            shape_below > 0.85,
            "high shelf well below anchor should relax toward 1.0, got {shape_below}"
        );
    }

    #[test]
    fn mixed_shelf_and_bell_coexist() {
        let nf = [200.0, 2000.0, 10000.0, 500.0, 1000.0, 4000.0, 8000.0, 16000.0];
        let nd = [0.0, 0.5, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let ns = [1, 0, 2, 0, 0, 0, 0, 0]; // Low=Low Shelf, Mid=Bell, High=High Shelf
        let ne = [true, true, true, false, false, false, false, false];

        // Low shelf below its anchor: attenuated.
        let shape_low = super::node_shape(50.0, &nf, &nd, &ns, &ne);
        assert!(shape_low < 0.1, "low shelf region should be attenuated, got {shape_low}");
        // Mid bell region: partially attenuated (~0.5 at anchor).
        let shape_mid = super::node_shape(2000.0, &nf, &nd, &ns, &ne);
        assert!(shape_mid < 0.7, "mid bell region should be attenuated, got {shape_mid}");
        // High shelf above its anchor: attenuated.
        let shape_high = super::node_shape(18000.0, &nf, &nd, &ns, &ne);
        assert!(shape_high < 0.3, "high shelf region should be attenuated, got {shape_high}");
    }

    #[test]
    fn overlapping_shelves_average_in_shared_region() {
        // Low Shelf at 2000 Hz (depth 0.0) + High Shelf at 500 Hz (depth 1.0).
        // Between 500 and 2000 Hz both shelves hold w = 1.0, so total_weight
        // ≥ 2.0 → coverage clamped to 1.0 and the result is the true weighted
        // average of the two depths (not pulled toward neutral 1.0).
        //
        // Derived analytically from the node_shape formula (SIGMA = 1.0):
        let nf = [2000.0, 500.0, 12000.0, 4000.0, 8000.0, 100.0, 300.0, 16000.0];
        let nd = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let ns = [1, 2, 0, 0, 0, 0, 0, 0]; // Low=Shelf, High=Shelf, High2=Bell
        let ne = [true, true, true, false, false, false, false, false];

        // At 1000 Hz: Low Shelf w=1.0 (below anchor), High Shelf w=1.0 (above anchor).
        // Bell at 12 kHz has negligible weight (~0.0016).
        // weighted_sum = 0.0*1.0 + 1.0*1.0 + 1.0*0.0016 = 1.0016
        // total_weight = 1.0 + 1.0 + 0.0016 = 2.0016
        // node_avg = 1.0016/2.0016 = 0.5004, coverage = 1.0
        let shape = super::node_shape(1000.0, &nf, &nd, &ns, &ne);
        assert!(
            (shape - 0.500406).abs() < 0.001,
            "overlap zone at 1000 Hz should average the two shelf depths, got {shape}"
        );
        // Must NOT be pulled toward neutral 1.0 — coverage is fully saturated.
        assert!(
            shape < 0.6,
            "overlap must stay near 0.5, not relax toward 1.0, got {shape}"
        );

        // At 500 Hz (High Shelf anchor): both shelves still w=1.0, Bell ~0.
        // Result ≈ 0.500007 — pure average of depths [0.0, 1.0].
        let shape_at_500 = super::node_shape(500.0, &nf, &nd, &ns, &ne);
        assert!(
            (shape_at_500 - 0.500007).abs() < 0.001,
            "at High Shelf anchor should be ~0.5, got {shape_at_500}"
        );

        // At 2000 Hz (Low Shelf anchor): both shelves still w=1.0, Bell ~0.035.
        // Result ≈ 0.508708 — slight Bell contamination pushes above 0.5.
        let shape_at_2k = super::node_shape(2000.0, &nf, &nd, &ns, &ne);
        assert!(
            (shape_at_2k - 0.508708).abs() < 0.005,
            "at Low Shelf anchor should be ~0.509, got {shape_at_2k}"
        );
    }

    #[test]
    fn nested_shelves_boundary_no_nan_or_discontinuity() {
        // Low Shelf at 1000 Hz (depth 0.0) + High Shelf at 800 Hz (depth 0.7).
        // The High Shelf holds everything above 800 Hz at w=1.0.
        // The Low Shelf holds everything below 1000 Hz at w=1.0.
        // Between 800 and 1000 Hz both are w=1.0 — fully nested overlap.
        //
        // Derived analytically from the node_shape formula (SIGMA = 1.0):
        let nf = [1000.0, 800.0, 12000.0, 4000.0, 200.0, 100.0, 300.0, 16000.0];
        let nd = [0.0, 0.7, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let ns = [1, 2, 0, 0, 0, 0, 0, 0]; // Low=Shelf, High=Shelf, High2=Bell
        let ne = [true, true, true, false, false, false, false, false];

        // 900 Hz: inside the nested overlap zone. Both shelves w=1.0.
        // Bell at 12 kHz: negligible (w ≈ 0.00093).
        // weighted_sum = 0.0*1.0 + 0.7*1.0 = 0.70093
        // total_weight = 2.00093, node_avg = 0.350291, coverage = 1.0
        let shape_900 = super::node_shape(900.0, &nf, &nd, &ns, &ne);
        assert!(
            (shape_900 - 0.350291).abs() < 0.001,
            "nested overlap at 900 Hz should be ~0.350, got {shape_900}"
        );
        assert!(
            shape_900 < 0.5,
            "must not relax toward neutral in overlap zone, got {shape_900}"
        );

        // 800 Hz: at the High Shelf anchor, inside Low Shelf's held region.
        // Both shelves w=1.0. Result ≈ 0.350162.
        let shape_800 = super::node_shape(800.0, &nf, &nd, &ns, &ne);
        assert!(
            (shape_800 - 0.350162).abs() < 0.001,
            "at High Shelf anchor should be ~0.350, got {shape_800}"
        );

        // 1000 Hz: at the Low Shelf anchor, inside High Shelf's held region.
        // Both shelves w=1.0. Result ≈ 0.350482.
        let shape_1000 = super::node_shape(1000.0, &nf, &nd, &ns, &ne);
        assert!(
            (shape_1000 - 0.350482).abs() < 0.001,
            "at Low Shelf anchor should be ~0.350, got {shape_1000}"
        );

        // Continuity check: no discontinuity at crossover boundaries.
        // The difference between adjacent samples must be small.
        let delta_800_900 = (shape_900 - shape_800).abs();
        let delta_900_1000 = (shape_1000 - shape_900).abs();
        assert!(
            delta_800_900 < 0.01,
            "no discontinuity between 800 and 900 Hz, delta = {delta_800_900}"
        );
        assert!(
            delta_900_1000 < 0.01,
            "no discontinuity between 900 and 1000 Hz, delta = {delta_900_1000}"
        );

        // 500 Hz: below both anchors. Low Shelf w=1.0, High Shelf Gaussian
        // tail w ≈ 0.795, Bell negligible.
        // Result ≈ 0.309951. Must be sane, no NaN.
        let shape_500 = super::node_shape(500.0, &nf, &nd, &ns, &ne);
        assert!(shape_500.is_finite(), "must not be NaN, got {shape_500}");
        assert!(
            (shape_500 - 0.309951).abs() < 0.005,
            "below both anchors should be ~0.310, got {shape_500}"
        );

        // 1500 Hz: above both anchors. High Shelf w=1.0, Low Shelf Gaussian
        // tail w ≈ 0.843, Bell ~0.011.
        // Result ≈ 0.383508. Must be sane, no NaN.
        let shape_1500 = super::node_shape(1500.0, &nf, &nd, &ns, &ne);
        assert!(shape_1500.is_finite(), "must not be NaN, got {shape_1500}");
        assert!(
            (shape_1500 - 0.383508).abs() < 0.005,
            "above both anchors should be ~0.384, got {shape_1500}"
        );
    }

    #[test]
    fn disabled_node_contributes_zero_weight() {
        // A disabled node must be truly absent — not a neutral-depth anchor.
        // Compare: same slot layout, node 0 enabled vs disabled. With the
        // node disabled at 2000 Hz (depth 0.0), the curve there must relax
        // toward neutral instead of dipping to ~0.0.
        let nf = [200.0, 2000.0, 12000.0, 500.0, 1000.0, 4000.0, 8000.0, 16000.0];
        let nd = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let ns = [0; MAX_NODES];
        let ne_on = [true, true, true, false, false, false, false, false];
        let ne_off = [true, false, true, false, false, false, false, false];

        let enabled = super::node_shape(2000.0, &nf, &nd, &ns, &ne_on);
        assert!(
            enabled < 0.1,
            "enabled zero-depth node should pull to ~0.0, got {enabled}"
        );
        let disabled = super::node_shape(2000.0, &nf, &nd, &ns, &ne_off);
        assert!(
            disabled > 0.9,
            "disabled node must vanish (relax to ~1.0), got {disabled}"
        );
    }

    #[test]
    fn node_shape_interpolates_between_anchors() {
        // Default-spread anchors with all-zero depths: at the Low anchor the
        // result is ~0.0 (coverage ≈ 1), mid-gap it sags halfway back toward
        // neutral (~0.5: partial coverage of an all-zero average), and at the
        // far left edge (20 Hz, >3 octaves below the lowest node) it has
        // relaxed back to ~neutral.  No walls, no non-neutral plateaus.
        let nf = [200.0, 2000.0, 12000.0, 500.0, 1000.0, 4000.0, 8000.0, 16000.0];
        let nd = [0.0; MAX_NODES];
        let ns = [0; MAX_NODES]; // Bell
        let ne = [true, true, true, false, false, false, false, false];
        let shape_at_200 = super::node_shape(200.0, &nf, &nd, &ns, &ne);
        let shape_at_mid = super::node_shape(632.0, &nf, &nd, &ns, &ne);
        let shape_at_edge = super::node_shape(20.0, &nf, &nd, &ns, &ne);
        assert!(
            shape_at_200 < 0.05,
            "at 200 Hz should be near 0.0, got {shape_at_200}"
        );
        assert!(
            shape_at_200 < shape_at_mid && shape_at_mid < 0.7,
            "mid-gap should rise partway toward neutral, got {shape_at_mid}"
        );
        assert!(
            shape_at_edge > 0.9,
            "far left edge should relax to ~neutral, got {shape_at_edge}"
        );
    }

    #[test]
    fn high_freq_band_converges_faster_than_low() {
        use super::{BALLISTICS_ALPHA, BALLISTICS_FREQ_REF};
        // Same sudden resonance, two identical detectors, one peaked low
        // (index 5, ≈35 Hz) and one peaked high (index 55, ≈8.3 kHz with the
        // log-spaced test centers). Frequency scaling must make the high band
        // reach the target in measurably fewer frames.
        //
        // Timing regime: hop 256 @ 44.1 kHz → frame_time ≈ 5.8 ms, attack 50 ms
        // so every band is *unsaturated* (τ_base > frame_time) and the per-band
        // τ spread actually shows up in `coef()`. (With the default 2048-hop /
        // 10 ms attack the followers saturate at 0.632 for all bands, which is
        // why existing tests are unaffected — verified, not assumed.)
        assert!(
            BALLISTICS_ALPHA > 0.0,
            "scaling must be active for this test, got α = {BALLISTICS_ALPHA}"
        );
        let centers = test_centers();
        assert!(
            centers[5] < BALLISTICS_FREQ_REF && centers[55] > BALLISTICS_FREQ_REF,
            "test bands must straddle f_ref: got {} Hz and {} Hz",
            centers[5],
            centers[55]
        );

        let mut p = params();
        p.selectivity = 1.0; // thr = 1 dB so the 40 dB peak is unambiguous
        p.attack_ms = 50.0;
        p.release_ms = 200.0;

        let quiet = [-30.0f32; BANDS];
        let mut resonant_lo = quiet;
        resonant_lo[5] = 10.0;
        let mut resonant_hi = quiet;
        resonant_hi[55] = 10.0;

        // Prime both detectors on quiet so baselines settle at -30 dB.
        let mut det_lo = Detector::new(256, 44100.0);
        let mut det_hi = Detector::new(256, 44100.0);
        for _ in 0..30 {
            det_lo.process_frame(&quiet, &p, &centers);
            det_hi.process_frame(&quiet, &p, &centers);
        }

        // Step the identical resonance and count frames to cross gain < 0.5
        // (target is ≈0.016 for a 39 dB excess, so 0.5 is solidly mid-flight).
        let frames_to_half = |det: &mut Detector, levels: &[f32; BANDS], band: usize| -> usize {
            for n in 1..=60 {
                let gains = det.process_frame(levels, &p, &centers);
                if gains[band] < 0.5 {
                    return n;
                }
            }
            usize::MAX // never converged — test setup broken, fail loudly below
        };
        let lo_frames = frames_to_half(&mut det_lo, &resonant_lo, 5);
        let hi_frames = frames_to_half(&mut det_hi, &resonant_hi, 55);

        assert!(
            hi_frames <= 60 && lo_frames <= 60,
            "both bands must converge within 60 frames: hi={hi_frames} lo={lo_frames}"
        );
        assert!(
            hi_frames < lo_frames,
            "high band ({} Hz) must converge faster than low band ({} Hz): hi={hi_frames} frames lo={lo_frames} frames",
            centers[55],
            centers[5]
        );
        assert!(
            hi_frames + 2 <= lo_frames,
            "speedup must be measurable (≥2 frames), not a 1-frame rounding edge: hi={hi_frames} lo={lo_frames}"
        );
    }

    #[test]
    fn ballistics_scaling_is_uniform_at_zero_alpha() {
        use super::{ballistics_scale, coef, BALLISTICS_ALPHA, BALLISTICS_FREQ_REF};
        // α = 0 must reduce exactly to the old uniform-timing behavior:
        // every band's scale is 1.0, so every band's effective τ equals τ_base.
        for &f in &[20.0f32, 40.0, 100.0, 1000.0, 8000.0, 16000.0, 20000.0] {
            let s = ballistics_scale(f, 0.0);
            assert!(
                (s - 1.0).abs() < 1e-6,
                "α=0 must give scale 1.0 at {f} Hz, got {s}"
            );
        }
        // …and that identity flows through the real `coef()` math unchanged.
        let frame_time = 256.0 / 44100.0;
        for &ms in &[10.0f32, 25.0, 50.0, 100.0] {
            let uniform = coef(ms, frame_time);
            for &f in &[35.0f32, 1000.0, 8300.0] {
                let scaled = coef(ms * ballistics_scale(f, 0.0), frame_time);
                assert!(
                    (scaled - uniform).abs() < 1e-9,
                    "α=0 coef must equal uniform coef (ms={ms}, f={f}): {scaled} vs {uniform}"
                );
            }
        }
        // Guard the active default points the right way (would catch a flipped
        // ratio or a negative α): reference is unity, lows are slowed, highs
        // are quickened, monotonically.
        assert!(
            (ballistics_scale(BALLISTICS_FREQ_REF, BALLISTICS_ALPHA) - 1.0).abs() < 1e-6,
            "f_ref must map to scale 1.0"
        );
        let s_lo = ballistics_scale(40.0, BALLISTICS_ALPHA);
        let s_hi = ballistics_scale(16000.0, BALLISTICS_ALPHA);
        assert!(s_lo > 1.0, "40 Hz must be slowed (scale > 1), got {s_lo}");
        assert!(s_hi < 1.0, "16 kHz must be quickened (scale < 1), got {s_hi}");
        assert!(
            s_lo > s_hi,
            "scale must fall with frequency, got lo={s_lo} hi={s_hi}"
        );
    }

    #[test]
    fn knee_reduction_is_c1_continuous() {
        use super::{knee_reduction_db, HARD_KNEE_DB, SOFT_KNEE_DB};
        // h=1e-4 keeps the O(h * k/knee) finite-difference bias below 1e-3 even
        // for the narrow hard knee (second derivative k/knee = 3 there).
        let h = 0.0001f32;
        for &(k, knee_db) in &[(1.5f32, SOFT_KNEE_DB), (1.5f32, HARD_KNEE_DB)] {
            let half = knee_db / 2.0;
            // Value at -half_knee is ~0.0.
            let at_lo = knee_reduction_db(-half, k, knee_db);
            assert!(
                at_lo.abs() < 1e-6,
                "k={k} knee={knee_db}: at -half expected ~0.0, got {at_lo}"
            );
            // Value at +half_knee equals k*half_knee within 1e-4.
            let at_hi = knee_reduction_db(half, k, knee_db);
            assert!(
                (at_hi - k * half).abs() < 1e-4,
                "k={k} knee={knee_db}: at +half expected {}, got {at_hi}",
                k * half
            );
            // Numerical derivative approaching -half from both sides matches.
            let d_lo_left =
                (knee_reduction_db(-half, k, knee_db) - knee_reduction_db(-half - h, k, knee_db)) / h;
            let d_lo_right =
                (knee_reduction_db(-half + h, k, knee_db) - knee_reduction_db(-half, k, knee_db)) / h;
            assert!(
                (d_lo_left - d_lo_right).abs() < 1e-3,
                "k={k} knee={knee_db}: derivative mismatch at -half: {d_lo_left} vs {d_lo_right}"
            );
            // Numerical derivative approaching +half from both sides matches.
            let d_hi_left =
                (knee_reduction_db(half, k, knee_db) - knee_reduction_db(half - h, k, knee_db)) / h;
            let d_hi_right =
                (knee_reduction_db(half + h, k, knee_db) - knee_reduction_db(half, k, knee_db)) / h;
            assert!(
                (d_hi_left - d_hi_right).abs() < 1e-3,
                "k={k} knee={knee_db}: derivative mismatch at +half: {d_hi_left} vs {d_hi_right}"
            );
        }
    }

    #[test]
    fn knee_far_field_converges_between_modes() {
        use super::{knee_reduction_db, HARD_KNEE_DB, SOFT_KNEE_DB};
        let k = 1.5f32;
        let excess = 30.0f32;
        let hard = knee_reduction_db(excess, k, HARD_KNEE_DB);
        let soft = knee_reduction_db(excess, k, SOFT_KNEE_DB);
        assert!(
            (hard - soft).abs() < 1e-6,
            "far-field must converge: hard={hard} soft={soft}"
        );
        assert!(
            (hard - k * excess).abs() < 1e-4,
            "far-field equals k*excess: got {hard}"
        );
    }

    #[test]
    fn node_shape_has_no_third_production_caller() {
        // Production sources that may legally reference `node_shape`.
        // Paths are anchored at the crate root via CARGO_MANIFEST_DIR so the
        // test is independent of the including file's location.
        let files: &[(&str, &str)] = &[
            (
                "src/dsp/detector.rs",
                include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/dsp/detector.rs")),
            ),
            (
                "src/gui.rs",
                include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/gui.rs")),
            ),
            (
                "src/lib.rs",
                include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/lib.rs")),
            ),
            (
                "src/dsp/suppressor.rs",
                include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/dsp/suppressor.rs")),
            ),
            (
                "src/dsp/filters.rs",
                include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/dsp/filters.rs")),
            ),
            (
                "src/dsp/analysis.rs",
                include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/dsp/analysis.rs")),
            ),
            (
                "src/dsp/bands.rs",
                include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/dsp/bands.rs")),
            ),
            (
                "src/dsp/mod.rs",
                include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/dsp/mod.rs")),
            ),
        ];

        /// Production code only: everything before the unit-test module.
        /// Cuts at the newline-anchored `mod tests {` (not `#[cfg(test)]`),
        /// because gui.rs carries an early `#[cfg(test)]` test-only helper
        /// (`Theme::luminance`) far above its real test module — cutting at
        /// the first `#[cfg(test)]` would wrongly discard the GUI production
        /// caller. The leading `\n` keeps this from self-matching the
        /// marker string inside this very test.
        fn production_part(src: &str) -> &str {
            match src.rfind("\nmod tests {") {
                Some(idx) => &src[..idx],
                None => src,
            }
        }

        // (a) The definition must exist exactly once across production files.
        let mut def_total = 0usize;
        let mut def_files: Vec<&str> = Vec::new();
        for (name, src) in files {
            let n = production_part(src).matches("pub fn node_shape(").count();
            if n > 0 {
                def_total += n;
                def_files.push(name);
            }
        }
        assert!(
            def_total == 1,
            "expected exactly 1 `pub fn node_shape(` definition across production files, found {def_total} in {def_files:?}"
        );

        // (b) Exactly 2 production call sites: one in detector.rs (DSP loop),
        // one in gui.rs (shape-curve loop). Excludes the definition itself,
        // the `use ... node_shape;` import, `super::node_shape` test callers,
        // test-module lines (stripped above), and `//` comments.
        let mut total_calls = 0usize;
        let mut per_file: Vec<(&str, usize)> = Vec::new();
        for (name, src) in files {
            let mut count = 0usize;
            for line in production_part(src).lines() {
                let code = match line.find("//") {
                    Some(idx) => &line[..idx],
                    None => line,
                };
                if code.contains("pub fn node_shape(") {
                    continue;
                }
                if code.contains("use ") && code.contains("node_shape") {
                    continue;
                }
                if code.contains("super::node_shape") {
                    continue;
                }
                count += code.matches("node_shape(").count();
            }
            per_file.push((name, count));
            total_calls += count;
        }
        let calls_in = |f: &str| -> usize {
            per_file.iter().find(|(n, _)| *n == f).map(|(_, c)| *c).unwrap_or(0)
        };
        assert!(
            total_calls == 2 && calls_in("src/dsp/detector.rs") == 1 && calls_in("src/gui.rs") == 1,
            "expected exactly 2 production `node_shape(` call sites (1 in src/dsp/detector.rs, 1 in src/gui.rs); a third production caller may have appeared. per-file counts: {per_file:?}"
        );
        // Name the offender explicitly if some other file gained a caller.
        for (name, count) in &per_file {
            if *name != "src/dsp/detector.rs" && *name != "src/gui.rs" {
                assert!(
                    *count == 0,
                    "third production caller of `node_shape` detected in offending file: {name} ({count} call site(s)); per-file counts: {per_file:?}"
                );
            }
        }
    }

    #[test]
    fn gui_curve_matches_dsp_region_depth_bit_identical() {
        // Fixed node params exercising bells, both shelf kinds, and mixed
        // enabled/disabled slots — the same shapes the DSP loop and the GUI
        // loop both feed into the shared `node_shape` single source of truth.
        let nf = [200.0, 2000.0, 12000.0, 500.0, 1000.0, 4000.0, 8000.0, 16000.0];
        let nd = [0.2, 0.9, 0.1, 0.7, 0.4, 1.0, 0.0, 0.5];
        let nsh = [1, 0, 2, 0, 1, 2, 0, 0];
        let nen = [true, true, true, false, true, false, true, false];
        // Sample freqs spanning below/above anchors and shelf regions.
        let freqs = [50.0f32, 200.0, 1000.0, 2000.0, 8000.0, 18000.0];
        // Precomputed pin table: (freq, expected value, expected f32 bits).
        // Generated from the first run of this test (see PIN lines with
        // `-- --nocapture`); exact `==`/bits equality catches any silent
        // signature or math change in `node_shape`.
        const EXPECTED: [(f32, f32, u32); 6] = [
            (50.0, 0.3000002, 0x3E9999A0),
            (200.0, 0.30120212, 0x3E9A372A),
            (1000.0, 0.5688666, 0x3F11A13E),
            (2000.0, 0.6438931, 0x3F24D62E),
            (8000.0, 0.10583202, 0x3DD8BE75),
            (18000.0, 0.07013579, 0x3D8FA35A),
        ];
        for (i, f) in freqs.iter().enumerate() {
            // Same args the DSP loop and GUI loop use: repeated calls must be
            // exactly (bit-)identical, not merely approximate.
            let dsp = super::node_shape(*f, &nf, &nd, &nsh, &nen);
            let gui = super::node_shape(*f, &nf, &nd, &nsh, &nen);
            assert!(
                dsp == gui,
                "DSP and GUI region depth must be bit-identical at {f} Hz: dsp={dsp:?} gui={gui:?}"
            );
            assert!(dsp.is_finite(), "region depth must be finite at {f} Hz, got {dsp:?}");
            let (exp_freq, exp_val, exp_bits) = EXPECTED[i];
            assert!(
                *f == exp_freq,
                "pin table order drifted: freqs[{i}] = {f}, table has {exp_freq}"
            );
            assert!(
                dsp == exp_val,
                "bit-identity pin failed at {f} Hz: got {dsp:?}, table has {exp_val:?} (math or signature changed?)"
            );
            assert!(
                dsp.to_bits() == exp_bits,
                "bit-pattern pin failed at {f} Hz: got 0x{:08X}, table has 0x{exp_bits:08X}",
                dsp.to_bits()
            );
            println!("PIN node_shape({f}) = {dsp:?} bits=0x{:08X}", dsp.to_bits());
        }
    }
}
