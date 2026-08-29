// src/dsp/detector.rs
// Per-band resonance detector.
//
// Spectral reference: for each frame a median-filtered envelope is computed
// across bands (neighboring bands at the same instant). Each band's level is
// compared against that local spectral median + threshold, so a narrow peak
// stands out regardless of how long it persists. A light time-domain smoother
// follows the *spectral median* (not the band's raw level) to avoid jitter
// without "learning" a sustained resonance as normal. See ResoVoid_FIX_PLAN_detector.md.

use crate::dsp::BANDS;

/// Half-window radius for the spectral median (total window = 2*R+1 = 13 bands).
/// Fixed constant per fix plan — expose as param only if ear-testing warrants it.
const MEDIAN_HALF_WINDOW: usize = 6;

#[derive(Clone, Copy)]
pub struct DetectParams {
    pub depth: f32,
    pub sharpness: f32,
    pub selectivity: f32,
    pub attack_ms: f32,
    pub release_ms: f32,
    pub soft_mode: bool,
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
        }
    }
}

pub struct Detector {
    sample_rate: f32,
    frame_samples: f32,
    /// Smoothed spectral reference (follows the per-frame median, not raw level).
    baseline: [f32; BANDS],
    gain: [f32; BANDS],
}

impl Detector {
    pub fn new(fft_size: usize, sample_rate: f32) -> Self {
        Self {
            sample_rate,
            frame_samples: fft_size as f32,
            baseline: [0.0; BANDS],
            gain: [1.0; BANDS],
        }
    }

    /// Process one analysis frame, returning smoothed linear gains per band (<= 1.0).
    ///
    /// Gain *targets* are recomputed only once per analysis window (this is
    /// called from `AnalysisEngine::process_sample` when a window completes).
    /// The exponential followers below are the sole interpolator between
    /// updates, which keeps the biquad coefficients from zipper-noising. If
    /// zipper noise is audible at small FFT sizes (1024, where windows are
    /// short), consider moving the follower to per-sample operation.
    pub fn process_frame(&mut self, levels_db: &[f32; BANDS], p: &DetectParams) -> [f32; BANDS] {
        let frame_time = self.frame_samples / self.sample_rate;

        let baseline_attack = coef(p.attack_ms, frame_time);
        let baseline_release = coef(p.release_ms, frame_time);
        let gain_attack = coef((p.attack_ms * 0.5).max(0.5), frame_time);
        let gain_release = coef((p.release_ms * 0.5).max(5.0), frame_time);

        // Threshold (dB): higher selectivity -> smaller threshold -> more is caught.
        let thr = (1.0 - p.selectivity) * 12.0 + p.selectivity * 1.0;
        let scale = if p.soft_mode { 0.7 } else { 1.0 };

        // Spectral (cross-band) reference — median over neighboring bands at the same instant.
        let medians = spectral_median(levels_db);

        let mut out = [1.0f32; BANDS];
        for i in 0..BANDS {
            // Smooth *toward the spectral median*, not toward this band's own raw level.
            // This prevents a sustained resonance from being "learned" as normal.
            let spectral_ref = medians[i];
            let bcoef = if spectral_ref > self.baseline[i] {
                baseline_attack
            } else {
                baseline_release
            };
            self.baseline[i] += (spectral_ref - self.baseline[i]) * bcoef;

            let reference = self.baseline[i] + thr;
            let excess = levels_db[i] - reference;
            let mut reduction_db = 0.0f32;
            if excess > 0.0 {
                reduction_db = (excess * p.depth * scale * (0.5 + p.sharpness)).min(36.0);
            }
            let target_gain = 10.0f32.powf(-reduction_db / 20.0);
            let gcoef = if target_gain < self.gain[i] {
                gain_attack
            } else {
                gain_release
            };
            self.gain[i] += (target_gain - self.gain[i]) * gcoef;
            out[i] = self.gain[i];
        }
        out
    }

    pub fn reset(&mut self) {
        self.baseline = [0.0; BANDS];
        self.gain = [1.0; BANDS];
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
    use crate::dsp::BANDS;

    fn params() -> DetectParams {
        DetectParams {
            depth: 1.0,
            sharpness: 1.0,
            selectivity: 0.0, // threshold = 12 dB
            attack_ms: 10.0,
            release_ms: 100.0,
            soft_mode: false,
        }
    }

    #[test]
    fn suppresses_hot_band_reduces_others() {
        let mut det = Detector::new(2048, 44100.0);
        // Establish a quiet spectral baseline first.
        let quiet = [-30.0f32; BANDS];
        for _ in 0..20 {
            det.process_frame(&quiet, &params());
        }
        // A large transient above the baseline + threshold is caught.
        let mut levels = quiet;
        levels[10] = 40.0;

        let gains = det.process_frame(&levels, &params());

        // Hot band gets attenuation (gain < 1).
        assert!(gains[10] < 1.0, "hot band should be reduced");
        assert!(gains[10] > 0.0, "gain stays positive");
        // A quiet band is left essentially untouched.
        assert!((gains[5] - 1.0).abs() < 1e-2, "quiet band unchanged");
    }

    #[test]
    fn flat_spectrum_yields_unity_gain() {
        let mut det = Detector::new(2048, 44100.0);
        let levels = [-60.0f32; BANDS]; // all below the 12 dB threshold
        let gains = det.process_frame(&levels, &params());
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
        // Prime the smoothed spectral reference to the flat level.
        let flat = [-10.0f32; BANDS];
        for _ in 0..20 {
            det_lo.process_frame(&flat, &params());
            det_hi.process_frame(&flat, &params());
        }
        let mut levels = flat;
        levels[10] = 0.0; // 10 dB above median

        let mut p_lo = params();
        p_lo.selectivity = 0.0; // thr = 12 dB -> 10 dB excess not enough
        let mut p_hi = params();
        p_hi.selectivity = 1.0; // thr = 1 dB -> easily caught

        let g_lo = det_lo.process_frame(&levels, &p_lo)[10];
        let g_hi = det_hi.process_frame(&levels, &p_hi)[10];
        assert!(g_hi < 1.0, "high selectivity catches the band");
        assert!((g_lo - 1.0).abs() < 1e-3, "low selectivity misses the band");
    }

    #[test]
    fn sustained_resonance_holds() {
        // Sustained resonance must stay suppressed, not recover after one attack period.
        let mut det = Detector::new(2048, 44100.0);
        let quiet = [-30.0f32; BANDS];
        for _ in 0..20 {
            det.process_frame(&quiet, &params());
        }
        let mut resonant = quiet;
        resonant[20] = 10.0; // strong narrow peak, >> median + thr

        let mut p = params();
        p.selectivity = 1.0; // thr = 1 dB so excess is large and unambiguous
        // First frame: should suppress
        let g1 = det.process_frame(&resonant, &p)[20];
        assert!(g1 < 0.9, "first frame should suppress sustained peak");

        // Hold the same resonance for many frames — old per-band baseline would
        // converge and release; spectral baseline must keep suppressing.
        let mut last = g1;
        for _ in 0..20 {
            last = det.process_frame(&resonant, &p)[20];
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
        let flat = [-20.0f32; BANDS];
        for _ in 0..20 {
            det.process_frame(&flat, &params());
        }
        let gains = det.process_frame(&flat, &params());
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
}
