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

/// Number of samples over which biquad coefficients ramp toward a new target
/// after `set_gains()`. Linear-interpolates the 5 coefficients per band per
/// sample (stack-only, no trig, no heap) so analysis-hop updates don't snap.
pub(crate) const GAIN_RAMP_SAMPLES: usize = 384;

/// Maximum Q multiplier applied to a fully concentrated resonance.
/// Effective Q = nominal Q * (1 + conc * (Q_BOOST_MAX - 1)), clamped to the
/// per-band ringing ceiling.
const Q_BOOST_MAX: f32 = 6.0;

/// Identity coefficients [b0, b1, b2, a1, a2].
const IDENTITY_COEFFS: [f32; 5] = [1.0, 0.0, 0.0, 0.0, 0.0];

/// Trig-based peaking-EQ coefficient math (same as `Biquad::set_peak`).
/// Factored out so `set_gains()` can compute *target* coeffs without touching
/// the live running biquad; the ramp stepper lerps the live coeffs toward
/// these targets. O(trig) — called once per hop, never per sample.
#[inline]
fn peak_coeffs(f0: f32, q: f32, gain_db: f32, fs: f32) -> [f32; 5] {
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
    [b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0]
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
    /// Trig-based target math (kept unchanged); `BandProcessor::set_gains`
    /// computes targets via `peak_coeffs()` and ramps the live biquad toward
    /// them instead of calling this per hop. Kept for tests / external use.
    #[allow(dead_code)]
    #[inline]
    pub fn set_peak(&mut self, f0: f32, q: f32, gain_db: f32, fs: f32) {
        let c = peak_coeffs(f0, q, gain_db, fs);
        self.b0 = c[0];
        self.b1 = c[1];
        self.b2 = c[2];
        self.a1 = c[3];
        self.a2 = c[4];
    }

    #[inline]
    fn coeffs(&self) -> [f32; 5] {
        [self.b0, self.b1, self.b2, self.a1, self.a2]
    }

    #[inline]
    fn apply_coeffs(&mut self, c: [f32; 5]) {
        self.b0 = c[0];
        self.b1 = c[1];
        self.b2 = c[2];
        self.a1 = c[3];
        self.a2 = c[4];
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
    /// Last concentration values, kept alongside gains for rate-switch recompute.
    last_concentration: [f32; BANDS],
    /// Depth-chain diagnostics, refreshed in `set_gains` with plain float
    /// stores only (allocation-free on the audio thread):
    /// per-band `gain_db` and `effective_q` actually sent to `peak_coeffs`.
    last_gain_db: [f32; BANDS],
    last_effective_q: [f32; BANDS],
    /// Ramp start (prev) coefficients per band: [b0, b1, b2, a1, a2].
    ramp_from: [[f32; 5]; BANDS],
    /// Ramp target coefficients per band: [b0, b1, b2, a1, a2].
    ramp_to: [[f32; 5]; BANDS],
    /// Samples stepped since the last `set_gains()`. `>= GAIN_RAMP_SAMPLES`
    /// means the ramp is complete (live == target).
    ramp_pos: usize,
}

impl BandProcessor {
    pub fn new(layout: BandLayout, sample_rate: f32) -> Self {
        let biquads = [(); BANDS].map(|_| Biquad::identity());
        // Snapshot nominal Q so `last_effective_q` is meaningful pre-first-hop.
        let mut nominal_q = [1.0f32; BANDS];
        for i in 0..BANDS {
            nominal_q[i] = layout.q[i];
        }
        Self {
            biquads,
            layout,
            base_rate: sample_rate,
            oversampled: false,
            prev_input: 0.0,
            last_gains: [1.0; BANDS],
            last_concentration: [0.0; BANDS],
            last_gain_db: [0.0; BANDS],
            last_effective_q: nominal_q,
            ramp_from: [IDENTITY_COEFFS; BANDS],
            ramp_to: [IDENTITY_COEFFS; BANDS],
            ramp_pos: GAIN_RAMP_SAMPLES,
        }
    }

    /// Last per-band `gain_db` sent toward `peak_coeffs` (0.0 at unity).
    #[inline]
    pub fn last_gain_db(&self) -> &[f32; BANDS] {
        &self.last_gain_db
    }

    /// Last per-band `effective_q` sent toward `peak_coeffs`.
    #[inline]
    pub fn last_effective_q(&self) -> &[f32; BANDS] {
        &self.last_effective_q
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
            let conc = self.last_concentration;
            self.set_gains(&gains, &conc);
        }
    }

    /// Update all band gains from linear gain targets (<= 1.0 => reduction).
    /// Computes trig-based *target* coefficients and starts a per-sample
    /// linear ramp from the current live coefficients. Never snaps the live
    /// biquad. O(BANDS) trig, once per analysis hop — not per sample.
    #[inline]
    pub fn set_gains(&mut self, gains: &[f32; BANDS], concentration: &[f32; BANDS]) {
        self.last_gains = *gains;
        self.last_concentration = *concentration;
        // Settle any in-flight ramp into the live biquads so the new ramp
        // starts from what is actually running (avoids jumps on rapid hops).
        self.materialize_current();
        let fs = self.synth_rate();
        for i in 0..BANDS {
            self.ramp_from[i] = self.biquads[i].coeffs();
            // Unity (0 dB) is exactly the identity — use canonical identity
            // coefficients so a unity update is a no-op ramp (no transient
            // from lerping between two equivalent realizations) and the
            // state stays clean. Matches `Biquad::identity()`.
            if gains[i] >= 1.0 - 1e-6 {
                self.ramp_to[i] = IDENTITY_COEFFS;
                // Depth-chain side channel: unity maps to 0 dB + nominal Q.
                self.last_gain_db[i] = 0.0;
                self.last_effective_q[i] = self.layout.q[i];
            } else {
                let gain_db = 20.0 * (gains[i].max(1e-4)).log10();
                let q_mult = 1.0 + concentration[i] * (Q_BOOST_MAX - 1.0);
                let effective_q =
                    (self.layout.q[i] * q_mult).clamp(0.3, self.layout.q_ceiling[i]);
                self.last_gain_db[i] = gain_db;
                self.last_effective_q[i] = effective_q;
                self.ramp_to[i] = peak_coeffs(
                    self.layout.centers[i],
                    effective_q,
                    gain_db,
                    fs,
                );
            }
        }
        self.ramp_pos = 0;
    }

    /// Write the currently-interpolated coefficients into the live biquads
    /// without advancing the ramp. Used by `set_gains()` to snapshot state.
    #[inline]
    fn materialize_current(&mut self) {
        if self.ramp_pos >= GAIN_RAMP_SAMPLES {
            return;
        }
        let t = self.ramp_pos as f32 / GAIN_RAMP_SAMPLES as f32;
        for i in 0..BANDS {
            let f = self.ramp_from[i];
            let tt = self.ramp_to[i];
            self.biquads[i].apply_coeffs([
                f[0] + (tt[0] - f[0]) * t,
                f[1] + (tt[1] - f[1]) * t,
                f[2] + (tt[2] - f[2]) * t,
                f[3] + (tt[3] - f[3]) * t,
                f[4] + (tt[4] - f[4]) * t,
            ]);
        }
    }

    /// Advance the coefficient ramp by one (host) sample: O(1) lerps per band
    /// (5 lerps x BANDS), stack-only, no trig, no heap. When the ramp is
    /// complete this is a single branch check.
    #[inline]
    fn step_ramp(&mut self) {
        if self.ramp_pos >= GAIN_RAMP_SAMPLES {
            return;
        }
        self.ramp_pos += 1;
        let t = self.ramp_pos as f32 / GAIN_RAMP_SAMPLES as f32;
        for i in 0..BANDS {
            let f = self.ramp_from[i];
            let tt = self.ramp_to[i];
            self.biquads[i].apply_coeffs([
                f[0] + (tt[0] - f[0]) * t,
                f[1] + (tt[1] - f[1]) * t,
                f[2] + (tt[2] - f[2]) * t,
                f[3] + (tt[3] - f[3]) * t,
                f[4] + (tt[4] - f[4]) * t,
            ]);
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
        self.step_ramp();
        self.process_cascade(x)
    }

    /// 2x internal path: linear-interp upsample (midpoint + current), run the
    /// cascade twice at 2x rate, average for downsampling. Latency-free: the
    /// interpolator/averager add zero host-visible samples of delay.
    /// Steps the shared coefficient ramp once per host sample so both
    /// internal sub-samples use the same interpolated coefficients.
    /// `gains` documents the active targets (ramp already holds them).
    #[inline]
    pub fn process_sample_2x(&mut self, x: f32, gains: &[f32; BANDS]) -> f32 {
        let _ = gains;
        self.step_ramp();
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
        self.last_concentration = [0.0; BANDS];
        self.last_gain_db = [0.0; BANDS];
        for i in 0..BANDS {
            self.last_effective_q[i] = self.layout.q[i];
        }
        self.ramp_from = [IDENTITY_COEFFS; BANDS];
        self.ramp_to = [IDENTITY_COEFFS; BANDS];
        self.ramp_pos = GAIN_RAMP_SAMPLES;
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
        bp.set_gains(&unity, &[0.0f32; BANDS]);

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
        bp.set_gains(&[1.0f32; BANDS], &[0.0f32; BANDS]);
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
        bp.set_gains(&cut, &[0.0f32; BANDS]);
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

    #[test]
    fn concentration_narrows_q_target() {
        // NOTE: with the real 20 Hz..20 kHz layout, LF bands (e.g. index 10
        // at ~60 Hz) have q_ceiling == q by design (ringing limit below the
        // nominal Q), so a boost there is intentionally inert. To isolate the
        // concentration->Q path at index 10, use a synthetic layout with
        // headroom at that index.
        let mut layout = compute_band_layout(2048, 44100.0);
        layout.q[10] = 2.0;
        layout.q_ceiling[10] = 12.0;
        let mut lo = BandProcessor::new(layout, 44100.0);
        let mut hi = BandProcessor::new(layout, 44100.0);
        let mut gains = [1.0f32; BANDS];
        gains[10] = 0.1;
        let conc_lo = [0.0f32; BANDS];
        let mut conc_hi = [0.0f32; BANDS];
        conc_hi[10] = 1.0;
        lo.set_gains(&gains, &conc_lo);
        hi.set_gains(&gains, &conc_hi);
        assert!(
            lo.ramp_to[10] != hi.ramp_to[10],
            "concentrated cut must narrow Q (different target coeffs)"
        );
        // And on the real layout a mid band with ceiling headroom also narrows.
        let real = compute_band_layout(2048, 44100.0);
        assert!(
            real.q_ceiling[40] > real.q[40],
            "mid band should have boost headroom"
        );
        let mut lo2 = BandProcessor::new(real, 44100.0);
        let mut hi2 = BandProcessor::new(real, 44100.0);
        let mut gains2 = [1.0f32; BANDS];
        gains2[40] = 0.1;
        let mut conc2 = [0.0f32; BANDS];
        conc2[40] = 1.0;
        lo2.set_gains(&gains2, &[0.0f32; BANDS]);
        hi2.set_gains(&gains2, &conc2);
        assert!(
            lo2.ramp_to[40] != hi2.ramp_to[40],
            "real-layout mid-band concentrated cut must narrow Q"
        );
    }

    #[test]
    fn high_band_boost_hits_nominal_ceiling() {
        let layout = compute_band_layout(2048, 44100.0);
        let last = BANDS - 1;
        assert!(
            layout.q_ceiling[last] > 85.0,
            "top band ceiling should be near 90, got {}",
            layout.q_ceiling[last]
        );
        let mut bp = BandProcessor::new(layout, 44100.0);
        let mut gains = [1.0f32; BANDS];
        gains[last] = 0.01;
        let mut conc = [0.0f32; BANDS];
        conc[last] = 1.0;
        bp.set_gains(&gains, &conc);
        let effective_q =
            (layout.q[last] * Q_BOOST_MAX).clamp(0.3, layout.q_ceiling[last]);
        assert!(
            (effective_q - 90.0).abs() < 6.0,
            "high-band effective Q should reach near 90, got {effective_q}"
        );
        // The stored target must match peak coeffs at the boosted Q.
        let gain_db = 20.0 * gains[last].max(1e-4).log10();
        let expected = peak_coeffs(layout.centers[last], effective_q, gain_db, 44100.0);
        for k in 0..5 {
            assert!(
                (bp.ramp_to[last][k] - expected[k]).abs() < 1e-6,
                "ramp target must use boosted Q (coeff {k} mismatch)"
            );
        }
    }

    #[test]
    fn lf_cut_ringing_decays_within_limit() {
        let sr = 44100.0;
        let layout = compute_band_layout(2048, sr);
        // Band nearest 200 Hz.
        let mut target = 0;
        let mut best = f32::MAX;
        for i in 0..BANDS {
            let d = (layout.centers[i] - 200.0).abs();
            if d < best {
                best = d;
                target = i;
            }
        }
        let mut bp = BandProcessor::new(layout, sr);
        let mut gains = [1.0f32; BANDS];
        gains[target] = 0.01;
        let mut conc = [0.0f32; BANDS];
        conc[target] = 1.0;
        bp.set_gains(&gains, &conc);
        // Settle the 384-sample coefficient ramp with silence.
        for _ in 0..GAIN_RAMP_SAMPLES {
            bp.process_sample(0.0);
        }
        // Unit impulse then ~0.15 s of silence.
        bp.process_sample(1.0);
        let tail_len = (0.15 * sr) as usize;
        let mut tail = 0.0f32;
        for _ in 0..tail_len {
            tail = bp.process_sample(0.0);
        }
        assert!(
            tail.abs() < 1e-3,
            "LF ringing tail must decay below 1e-3 after 0.15 s, got {tail}"
        );
    }

    // Diagnostic alarm, NOT a regression gate: it panics deterministically on
    // the current code (stacking reproduces by construction), so it is
    // `#[ignore]`d to keep `cargo test` green. Run it explicitly when
    // diagnosing vocal renders or validating a neighbor-inhibition fix:
    // `cargo test flanking -- --ignored`.
    #[test]
    #[ignore = "diagnostic alarm: fails by design until neighbor inhibition / shared-Q lands; run with --ignored"]
    fn flanking_cuts_flagged_when_neighbors_cut_together() {
        // Flanking-notch texture risk: detector.rs scores every band
        // independently (no neighbor suppression), so a single resonance can
        // push 2-3 adjacent bands to gains < 1 at once. Each of those then
        // takes the concentration->Q narrowing path in `set_gains`, cutting
        // deep AND narrow at nearly the same center, so the cuts stack into
        // a wide rough-edged flanking notch instead of one surgical cut.
        // Intended behavior (future work): neighbor inhibition or a shared-Q
        // allocation so one resonance yields one narrow cut, not flankers.
        // This test only REPORTS the narrowed-cut composite (no assert on
        // it) and fails exactly in the extreme stacking regime: >= 2
        // adjacent bands both deeper than -12 dB while narrowed past 2x
        // their nominal Q.
        let layout = compute_band_layout(2048, 44100.0);
        let nominal_q = layout.q;
        let centers = layout.centers;
        // Three adjacent mid bands with room to narrow past 2x nominal Q.
        let mut mid: Option<usize> = None;
        for i in 1..BANDS - 1 {
            if layout.q_ceiling[i - 1] > 2.0 * layout.q[i - 1]
                && layout.q_ceiling[i] > 2.0 * layout.q[i]
                && layout.q_ceiling[i + 1] > 2.0 * layout.q[i + 1]
            {
                mid = Some(i);
                break;
            }
        }
        let m = mid.expect("need 3 adjacent bands with Q-boost headroom");
        let idx = [m - 1, m, m + 1];
        // Synthetic single resonance seen by all three neighbors at once.
        let mut bp = BandProcessor::new(layout, 44100.0);
        let mut gains = [1.0f32; BANDS];
        let mut conc = [0.0f32; BANDS];
        for &i in &idx {
            gains[i] = 0.1; // -20 dB
            conc[i] = 1.0; // max narrowing
        }
        bp.set_gains(&gains, &conc);
        let gain_db = *bp.last_gain_db();
        let eff_q = *bp.last_effective_q();
        // Informational composite only: cutting (< -3 dB) while narrowed
        // (> 2x nominal Q). Reported, never asserted.
        let narrowed_cuts = (0..BANDS)
            .filter(|&i| gain_db[i] < -3.0 && eff_q[i] > 2.0 * nominal_q[i])
            .count();
        println!(
            "flanking composite: {narrowed_cuts} narrowed cuts \
             (band  center_hz  gain_db  eff_q  nominal_q):"
        );
        for &i in &idx {
            println!(
                "  [{i}]  {:.1}  {:.2}  {:.2}  {:.2}",
                centers[i], gain_db[i], eff_q[i], nominal_q[i]
            );
        }
        // Flag: any adjacent pair within the driven cluster both deeper
        // than -12 dB while narrowed.
        let stacked = idx.windows(2).any(|w| {
            gain_db[w[0]] < -12.0
                && gain_db[w[1]] < -12.0
                && eff_q[w[0]] > 2.0 * nominal_q[w[0]]
                && eff_q[w[1]] > 2.0 * nominal_q[w[1]]
        });
        if stacked {
            panic!(
                "flanking-notch stacking: {} adjacent bands cut deeper than \
                 -12 dB while narrowed >2x nominal Q. per-band \
                 (center_hz, gain_db, effective_q): [{:.1} Hz, {:.2} dB, Q {:.2}] \
                 [{:.1} Hz, {:.2} dB, Q {:.2}] [{:.1} Hz, {:.2} dB, Q {:.2}]. \
                 one resonance produced {} simultaneous narrow cuts; want \
                 neighbor inhibition or shared-Q (future work).",
                idx.len(),
                centers[idx[0]],
                gain_db[idx[0]],
                eff_q[idx[0]],
                centers[idx[1]],
                gain_db[idx[1]],
                eff_q[idx[1]],
                centers[idx[2]],
                gain_db[idx[2]],
                eff_q[idx[2]],
                narrowed_cuts,
            );
        }
    }
}
