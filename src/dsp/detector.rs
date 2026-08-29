// src/dsp/detector.rs
// Per-band resonance detector.
//
// The detector tracks a slow spectral baseline (the "desired" envelope). Whenever a
// band's level exceeds the baseline by more than a selectivity-dependent threshold,
// the excess is turned into downward gain reduction. Attack/release ballistics are
// applied to both the baseline follower and the final gain.

use crate::dsp::BANDS;

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
    pub fn process_frame(&mut self, levels_db: &[f32; BANDS], p: &DetectParams) -> [f32; BANDS] {
        let frame_time = self.frame_samples / self.sample_rate;

        let baseline_attack = coef(p.attack_ms, frame_time);
        let baseline_release = coef(p.release_ms, frame_time);
        let gain_attack = coef((p.attack_ms * 0.5).max(0.5), frame_time);
        let gain_release = coef((p.release_ms * 0.5).max(5.0), frame_time);

        // Threshold (dB): higher selectivity -> smaller threshold -> more is caught.
        let thr = (1.0 - p.selectivity) * 12.0 + p.selectivity * 1.0;
        let scale = if p.soft_mode { 0.7 } else { 1.0 };

        let mut out = [1.0f32; BANDS];
        for i in 0..BANDS {
            let target = levels_db[i];
            let bcoef = if target > self.baseline[i] {
                baseline_attack
            } else {
                baseline_release
            };
            self.baseline[i] += (target - self.baseline[i]) * bcoef;

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

/// Per-frame smoothing coefficient so a follower reaches ~63% in `ms` milliseconds.
fn coef(ms: f32, frame_time: f32) -> f32 {
    let tau = (ms.max(0.1) / 1000.0).max(frame_time);
    let frames = tau / frame_time;
    1.0 - (-1.0 / frames).exp()
}
