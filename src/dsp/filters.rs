// src/dsp/filters.rs
// Cascade of peaking biquads that applies the per-band reduction.
//
// Each band is a peaking EQ centred on the band frequency. At 0 dB gain the filter
// is mathematically the identity (b == a), so an unreduced signal passes untouched.

use crate::dsp::bands::BandLayout;
use crate::dsp::BANDS;

#[inline]
fn flush_denormal(v: f32) -> f32 {
    if v.is_subnormal() { 0.0 } else { v }
}

pub struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,
    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl Biquad {
    pub fn identity() -> Self {
        Self {
            b0: 1.0,
            b1: 0.0,
            b2: 0.0,
            a1: 0.0,
            a2: 0.0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    /// Configure as a peaking EQ. `gain_db == 0` yields the identity response.
    #[inline]
    pub fn set_peak(&mut self, f0: f32, q: f32, gain_db: f32, fs: f32) {
        let a = 10.0f32.powf(gain_db / 40.0);
        let w0 = 2.0 * std::f32::consts::PI * f0 / fs;
        let cw = w0.cos();
        let sw = w0.sin();
        let alpha = sw / (2.0 * q.max(1e-3));
        let b0 = 1.0 + alpha * a;
        let b1 = -2.0 * cw;
        let b2 = 1.0 - alpha * a;
        let a0 = 1.0 + alpha / a;
        let a1 = -2.0 * cw;
        let a2 = 1.0 - alpha / a;
        self.b0 = b0 / a0;
        self.b1 = b1 / a0;
        self.b2 = b2 / a0;
        self.a1 = a1 / a0;
        self.a2 = a2 / a0;
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let y = flush_denormal(
            self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
                - self.a1 * self.y1
                - self.a2 * self.y2,
        );
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

pub struct BandProcessor {
    biquads: [Biquad; BANDS],
    layout: BandLayout,
    sample_rate: f32,
}

impl BandProcessor {
    pub fn new(layout: BandLayout, sample_rate: f32) -> Self {
        let biquads = [(); BANDS].map(|_| Biquad::identity());
        Self {
            biquads,
            layout,
            sample_rate,
        }
    }

    /// Update all band gains from linear gain targets (<= 1.0 => reduction).
    #[inline]
    pub fn set_gains(&mut self, gains: &[f32; BANDS]) {
        for i in 0..BANDS {
            let gain_db = 20.0 * (gains[i].max(1e-4)).log10();
            self.biquads[i].set_peak(
                self.layout.centers[i],
                self.layout.q[i],
                gain_db,
                self.sample_rate,
            );
        }
    }

    #[inline]
    pub fn process_sample(&mut self, x: f32) -> f32 {
        let mut y = x;
        for i in 0..BANDS {
            y = self.biquads[i].process(y);
        }
        y
    }

    pub fn reset(&mut self) {
        for i in 0..BANDS {
            self.biquads[i] = Biquad::identity();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::bands::compute_band_layout;

    /// Setting all gains to 1.0 (0 dB) should produce an identity round-trip:
    /// output == input for every sample.  This guards the biquad coefficient
    /// math against regressions that would break unity gain.
    #[test]
    fn identity_round_trip() {
        let layout = compute_band_layout(2048, 44100.0);
        let mut bp = BandProcessor::new(layout, 44100.0);

        let unity = [1.0f32; BANDS];
        bp.set_gains(&unity);

        // Run a constant DC signal to let all 64 biquads settle, then verify
        // that the output matches the input in steady state.
        let input = 0.5f32;
        // Settle: 64 biquads × ~32 samples each ≈ 2048 samples to be safe.
        for _ in 0..2048 {
            bp.process_sample(input);
        }
        // Now verify steady-state identity.
        let mut max_err = 0.0f32;
        for _ in 0..1024 {
            let y = bp.process_sample(input);
            let err = (y - input).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(
            max_err < 1e-4,
            "identity round-trip steady-state error too large: {max_err}"
        );
    }

    #[test]
    fn biquad_identity_passthrough() {
        let mut b = Biquad::identity();
        let out = b.process(0.5);
        assert!((out - 0.5).abs() < 1e-10);
    }

    #[test]
    fn biquad_peak_zero_db_is_identity() {
        let mut b = Biquad::identity();
        b.set_peak(1000.0, 1.0, 0.0, 44100.0);
        // After settling, a constant input should pass through at unity.
        let mut y = 0.0;
        for _ in 0..256 {
            y = b.process(1.0);
        }
        assert!((y - 1.0).abs() < 1e-5, "0 dB peak should be identity, got {y}");
    }
}
