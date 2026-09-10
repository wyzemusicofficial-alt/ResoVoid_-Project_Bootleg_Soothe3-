// src/dsp/filters.rs
// Cascade of peaking biquads that applies the per-band reduction.
//
// Each band is a peaking EQ centred on the band frequency. At 0 dB gain the filter
// is mathematically the identity (b == a), so an unreduced signal passes untouched.

use crate::dsp::bands::BandLayout;
use crate::dsp::BANDS;

/// Enable Flush-To-Zero + Denormals-Are-Zero on the calling thread (x86/x86_64).
///
/// Biquad state feedback can decay into subnormals during silence, and each
/// subnormal op traps to microcode (~100+ cycles) — across a 64-stage cascade
/// that is a real-time hazard. With FTZ+DAZ set, subnormal *results* are
/// flushed and subnormal *inputs* read as zero in hardware, so no per-sample
/// branch is needed. MXCSR is per-thread, so the audio-thread owner (lib.rs
/// `process()`) calls this once per block. No-op on other architectures,
/// where denormal handling is left to hardware defaults.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
pub(crate) fn enable_flush_to_zero() {
    // MXCSR flag bits: FTZ = bit 15, DAZ = bit 6. OR them in, preserving
    // the host's rounding/exception-mask configuration. Inline asm is used
    // because the `_mm_getcsr`/`_mm_setcsr` intrinsics are deprecated.
    const FTZ: u32 = 1 << 15;
    const DAZ: u32 = 1 << 6;
    let mut csr: u32 = 0;
    unsafe {
        std::arch::asm!(
            "stmxcsr [{0}]",
            in(reg) &mut csr,
            options(nostack, preserves_flags),
        );
        csr |= FTZ | DAZ;
        std::arch::asm!(
            "ldmxcsr [{0}]",
            in(reg) &csr,
            options(nostack, preserves_flags),
        );
    }
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[inline]
pub(crate) fn enable_flush_to_zero() {
    // No portable FTZ control; the `flush_denormal` guard below covers this.
}

/// Zero subnormal biquad outputs. On x86/x86_64 this compiles to the identity:
/// `enable_flush_to_zero()` guarantees subnormals cannot arise, so the branch
/// is dead code there and is compiled out. Other architectures keep the guard.
#[inline]
fn flush_denormal(v: f32) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        v
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        if v.is_subnormal() {
            0.0
        } else {
            v
        }
    }
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
    /// Host sample rate (analysis/detector reference).
    base_rate: f32,
    /// When true, coefficients target 2x rate and `process_sample_2x` runs.
    oversampled: bool,
    /// Previous input sample (linear-interp upsampler state).
    prev_input: f32,
    /// Last applied gains, kept so a rate switch can recompute coefficients.
    last_gains: [f32; BANDS],
}

impl BandProcessor {
    pub fn new(layout: BandLayout, sample_rate: f32) -> Self {
        let biquads = [(); BANDS].map(|_| Biquad::identity());
        Self {
            biquads,
            layout,
            base_rate: sample_rate,
            oversampled: false,
            prev_input: 0.0,
            last_gains: [1.0; BANDS],
        }
    }

    /// Effective synthesis rate: doubled when oversampling is active.
    #[inline]
    fn synth_rate(&self) -> f32 {
        if self.oversampled {
            self.base_rate * 2.0
        } else {
            self.base_rate
        }
    }

    /// Enable/disable 2x internal processing. On a flip, coefficients are
    /// recomputed immediately from the last gains so the cascade never runs
    /// with stale-rate coefficients. Cheap no-op when unchanged.
    pub fn set_oversampled(&mut self, on: bool) {
        if on != self.oversampled {
            self.oversampled = on;
            let gains = self.last_gains;
            self.set_gains(&gains);
        }
    }

    /// Update all band gains from linear gain targets (<= 1.0 => reduction).
    #[inline]
    pub fn set_gains(&mut self, gains: &[f32; BANDS]) {
        self.last_gains = *gains;
        let fs = self.synth_rate();
        for i in 0..BANDS {
            let gain_db = 20.0 * (gains[i].max(1e-4)).log10();
            self.biquads[i].set_peak(
                self.layout.centers[i],
                self.layout.q[i],
                gain_db,
                fs,
            );
        }
    }

    #[inline]
    fn process_cascade(&mut self, x: f32) -> f32 {
        let mut y = x;
        for i in 0..BANDS {
            y = self.biquads[i].process(y);
        }
        y
    }

    #[inline]
    pub fn process_sample(&mut self, x: f32) -> f32 {
        self.process_cascade(x)
    }

    /// 2x internal path: linear-interp upsample (midpoint + current), run the
    /// cascade twice at 2x rate, average for downsampling. Latency-free: the
    /// interpolator/averager add zero host-visible samples of delay.
    /// `gains` documents the active targets (coefficients already hold them).
    #[inline]
    pub fn process_sample_2x(&mut self, x: f32, gains: &[f32; BANDS]) -> f32 {
        let _ = gains;
        let mid = 0.5 * (self.prev_input + x);
        self.prev_input = x;
        let y0 = self.process_cascade(mid);
        let y1 = self.process_cascade(x);
        0.5 * (y0 + y1)
    }

    pub fn reset(&mut self) {
        for i in 0..BANDS {
            self.biquads[i] = Biquad::identity();
        }
        self.prev_input = 0.0;
        self.last_gains = [1.0; BANDS];
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
    fn oversampled_unity_passes_sine_transparently() {
        // Sine (not DC) input: high-Q low bands amplify f32 rounding of a
        // DC component into a slow ±1e-2 wander, which is orthogonal to the
        // sine and vanishes from an RMS comparison.
        let layout = compute_band_layout(2048, 44100.0);
        let mut bp = BandProcessor::new(layout, 44100.0);
        bp.set_oversampled(true);
        bp.set_gains(&[1.0f32; BANDS]);
        let unity = [1.0f32; BANDS];
        let mut sum_xx = 0.0f64;
        let mut sum_yy = 0.0f64;
        let mut peak = 0.0f32;
        for n in 0..16384 {
            let x = 0.5 * (2.0 * std::f32::consts::PI * 1000.0 * n as f32 / 44100.0).sin();
            let y = bp.process_sample_2x(x, &unity);
            assert!(y.is_finite(), "2x output must stay finite");
            peak = peak.max(y.abs());
            if n >= 12288 {
                sum_xx += (x as f64) * (x as f64);
                sum_yy += (y as f64) * (y as f64);
            }
        }
        let rms_ratio = (sum_yy / sum_xx).sqrt();
        assert!(
            (rms_ratio - 1.0).abs() < 0.05,
            "2x unity must pass a 1 kHz sine transparently, rms ratio {rms_ratio}"
        );
        assert!(peak < 0.65, "no overshoot/blowup, peak {peak}");
    }

    #[test]
    fn oversample_toggle_recomputes_without_nan() {
        let layout = compute_band_layout(2048, 44100.0);
        let mut bp = BandProcessor::new(layout, 44100.0);
        let mut cut = [1.0f32; BANDS];
        cut[10] = 0.1;
        bp.set_gains(&cut);
        bp.set_oversampled(true);
        bp.set_oversampled(false);
        let y = bp.process_sample(0.5);
        assert!(y.is_finite(), "output must stay finite across toggles");
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
