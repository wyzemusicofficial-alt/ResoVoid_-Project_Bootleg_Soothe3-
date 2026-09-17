// src/dsp/analysis.rs
// STFT analysis front-end: windowed real FFT -> per-band level (dB).
// Allocation-free at runtime: all buffers are preallocated in `new`.

use std::sync::Arc;

use num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};

use crate::dsp::bands::{compute_band_layout, BandLayout};
use crate::dsp::BANDS;

/// PAR (peak-to-average ratio) in dB that maps to zero concentration
/// (broadband energy spread evenly across the band's bins).
const PAR_LOW_DB: f32 = 2.0;
/// PAR in dB that maps to full concentration (a single dominant bin).
///
/// Calibration (no math-reference change - threshold fix): measured in-band
/// PAR at band 40 (fft 2048 @ 44.1 kHz) is ~11.7 dB for a pure sine,
/// ~7.5 dB for a 3-harmonic formant-like cluster with vibrato, ~5.5 dB for
/// a resonant noise bump, and ~4.4 dB mean for white noise. The old 12 dB
/// ceiling sat at/above the pure-tone PAR, stranding formant material at
/// ~0.55 concentration (Q mult ~3.8x of 6x). A 9 dB ceiling keeps the sine
/// saturated at 1.0 while moving the formant cluster to ~0.79 (Q mult ~5x)
/// and the resonant bump to ~0.50, without pushing the broadband floor
/// (mean ~0.34) into the concentrated regime. Concentration never feeds
/// detection - it only narrows filter Q - so the lifted noise floor is
/// inert unless a cut is actually applied there (unity gains bypass Q).
const PAR_HIGH_DB: f32 = 9.0;

/// Per-frame spectral description: band levels plus a tonal-concentration
/// estimate per band. Concentration is derived from the in-band
/// peak-to-average magnitude ratio and is used only to narrow filter Q;
/// it never feeds detection/threshold logic.
#[derive(Clone, Copy)]
pub struct BandAnalysis {
    pub levels: [f32; BANDS],
    pub concentration: [f32; BANDS],
}

/// A single snapshot pushed from the audio thread to the GUI thread.
#[derive(Clone, Copy)]
pub struct AnalysisFrame {
    /// Per-band input spectrum level in dB.
    pub spectrum: [f32; BANDS],
    /// Per-band linear gain applied by the suppressor (<= 1.0).
    /// This is the detector follower output (`smoothed_gain` in depth CSV).
    pub reduction: [f32; BANDS],
    /// Band center frequencies in Hz (cached from the band layout).
    pub centers: [f32; BANDS],
    // TEMPORARY DEBUG FIELD - remove after concentration diagnosis is done
    pub concentration: [f32; BANDS],
    /// Current sample rate.
    pub sample_rate: f32,
    /// Monotonic analysis-frame counter (wrapping). Stamped by
    /// `ResonanceSuppressor` once per pushed frame; lets a test CSV join
    /// one row per (frame, band) without relying on GUI `debug_hop`.
    pub frame_idx: u32,
    /// Gain-reduction depth chain diagnostics (fixed-size, Copy).
    /// Populated on the audio thread with plain float stores only —
    /// no allocation, no String/Vec/file I/O. The CSV rendering below
    /// is `#[cfg(test)]` (test/GUI thread only), never on audio thread.
    /// `effective_depth` = p.depth * region_depth (detector L311).
    pub effective_depth: [f32; BANDS],
    /// `reduction_db` = knee_reduction_db output clamped to 36 dB (detector L313).
    pub reduction_db: [f32; BANDS],
    /// `target_gain` = 10^(-reduction_db/20), pre-follower (detector L314).
    pub target_gain: [f32; BANDS],
    /// `gain_db` = 20*log10(max(smoothed_gain,1e-4)) (filters L254).
    pub gain_db: [f32; BANDS],
    /// `effective_q` = clamp(layout.q * q_mult) (filters L255-257).
    pub effective_q: [f32; BANDS],
}

/// Depth-chain CSV header: one row per (frame, band).
/// `smoothed_gain` is `AnalysisFrame.reduction` (post-follower linear gain).
#[cfg(test)]
pub const DEPTH_CSV_HEADER: &str =
    "frame_idx,band,effective_depth,reduction_db,target_gain,smoothed_gain,gain_db,effective_q";

/// Render depth-chain frames as CSV text. Test/GUI thread only — allocates
/// `String`, so never call on the audio thread. Compiled under `#[cfg(test)]`
/// to keep the audio-thread build free of String/Vec/file IO.
#[cfg(test)]
pub fn format_depth_csv(frames: &[AnalysisFrame]) -> String {
    use std::fmt::Write as _;
    let mut out = String::from(DEPTH_CSV_HEADER);
    out.push('\n');
    for f in frames {
        for b in 0..BANDS {
            let _ = write!(
                out,
                "{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                f.frame_idx,
                b,
                f.effective_depth[b],
                f.reduction_db[b],
                f.target_gain[b],
                f.reduction[b],
                f.gain_db[b],
                f.effective_q[b],
            );
        }
    }
    out
}

/// Preallocated analysis engine for one FFT size.
///
/// Windows overlap by 50%: a frame is emitted every `hop = fft_size / 2`
/// samples from a ring buffer, so the detector refreshes twice per window.
/// Hann @ 50% overlap is COLA (constant overlap-add), keeping successive
/// frames consistent. Output-path latency is unchanged (still `fft_size`,
/// set by the delay line); only the detector/viz refresh rate doubles.
pub struct AnalysisEngine {
    fft_size: usize,
    hop: usize,
    r2c: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    buf: Vec<f32>,
    input: Vec<f32>,
    spectrum: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
    write_pos: usize,
    since_frame: usize,
    bands: BandLayout,
}

impl AnalysisEngine {
    pub fn new(fft_size: usize, sample_rate: f32) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(fft_size);
        let window = make_hann(fft_size);
        let buf = vec![0.0; fft_size];
        let input = r2c.make_input_vec();
        let spectrum = r2c.make_output_vec();
        let scratch = r2c.make_scratch_vec();
        let bands = compute_band_layout(fft_size, sample_rate);

        Self {
            fft_size,
            hop: (fft_size / 2).max(1),
            r2c,
            window,
            buf,
            input,
            spectrum,
            scratch,
            write_pos: 0,
            since_frame: 0,
            bands,
        }
    }

    pub fn bands(&self) -> &BandLayout {
        &self.bands
    }

    /// Samples between consecutive emitted frames (half the FFT size).
    /// The detector uses this (not the FFT size) for its per-frame timing.
    pub fn hop(&self) -> usize {
        self.hop
    }

    /// Feed one input sample. Returns per-band levels in dB every `hop`
    /// samples, once the ring buffer holds its first full window.
    /// Allocation-free: only preallocated buffers are touched.
    #[inline]
    pub fn process_sample(&mut self, x: f32) -> Option<BandAnalysis> {
        self.buf[self.write_pos] = x;
        self.write_pos += 1;
        if self.write_pos >= self.fft_size {
            self.write_pos = 0;
        }
        self.since_frame += 1;

        // First frame needs a full window in the ring; afterwards every hop.
        let primed = self.since_frame >= self.fft_size;
        if !primed || !(self.since_frame - self.fft_size).is_multiple_of(self.hop) {
            return None;
        }

        // Unwrap the ring (oldest sample first) under the Hann window.
        // Branch-on-wrap instead of modulo: one predictable branch per bin.
        let mut idx = self.write_pos;
        for i in 0..self.fft_size {
            self.input[i] = self.buf[idx] * self.window[i];
            idx += 1;
            if idx >= self.fft_size {
                idx = 0;
            }
        }
        let _ = self
            .r2c
            .process_with_scratch(&mut self.input, &mut self.spectrum, &mut self.scratch);

        let mut levels = [0.0f32; BANDS];
        let mut concentration = [0.0f32; BANDS];
        for b in 0..BANDS {
            let lo = self.bands.bin_lo[b];
            let hi = self.bands.bin_hi[b].min(self.spectrum.len() - 1);
            let mut sum = 0.0f32;
            let mut peak = 0.0f32;
            for k in lo..=hi {
                let c = self.spectrum[k];
                let mag = (c.re * c.re + c.im * c.im).sqrt();
                sum += mag;
                if mag > peak {
                    peak = mag;
                }
            }
            let n = (hi - lo + 1).max(1) as f32;
            let mean_mag = sum / n;
            levels[b] = 20.0 * (mean_mag + 1e-9).log10();
            let par_db = 20.0 * ((peak + 1e-9) / (mean_mag + 1e-9)).log10();
            concentration[b] =
                ((par_db - PAR_LOW_DB) / (PAR_HIGH_DB - PAR_LOW_DB)).clamp(0.0, 1.0);
        }
        Some(BandAnalysis {
            levels,
            concentration,
        })
    }

    pub fn reset(&mut self) {
        self.buf.fill(0.0);
        self.write_pos = 0;
        self.since_frame = 0;
    }
}

fn make_hann(n: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; n];
    if n <= 1 {
        return v;
    }
    for i in 0..n {
        v[i] = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n as f32 - 1.0)).cos());
    }
    v
}

#[cfg(test)]
mod tests {
    use super::AnalysisEngine;
    use crate::dsp::BANDS;

    #[test]
    fn sine_tone_is_highly_concentrated() {
        let sr = 44100.0;
        let fft = 2048;
        let mut eng = AnalysisEngine::new(fft, sr);
        let centers = eng.bands().centers;
        // Pick a mid-high band with enough bins for a clear peak-to-average.
        let target = 40;
        let freq = centers[target];
        let mut out = None;
        for n in 0..(fft * 2) {
            let x = (2.0 * std::f32::consts::PI * freq * n as f32 / sr).sin();
            if let Some(a) = eng.process_sample(x) {
                out = Some(a);
            }
        }
        let analysis = out.expect("sine must emit a frame");
        assert!(
            analysis.concentration[target] > 0.7,
            "sine at {} Hz should be concentrated, got {}",
            freq,
            analysis.concentration[target]
        );
    }

    #[test]
    fn white_noise_is_unconcentrated() {
        let sr = 44100.0;
        let fft = 2048;
        let mut eng = AnalysisEngine::new(fft, sr);
        // Deterministic LCG white noise (no dev-dependency on rand).
        let mut state: u32 = 0x12345678;
        let mut next = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / 8_388_608.0) * 2.0 - 1.0
        };
        let mut out = None;
        for _ in 0..(fft * 4) {
            if let Some(a) = eng.process_sample(next()) {
                out = Some(a);
            }
        }
        let analysis = out.expect("noise must emit a frame");
        let mean: f32 =
            analysis.concentration.iter().sum::<f32>() / BANDS as f32;
        // Bound follows the PAR window: under the 2-9 dB mapping the same
        // physical noise (mean PAR ~4.4 dB) reads ~0.34, so 0.3 would be a
        // false failure. Intent is unchanged - broadband stays well below
        // tonal material (sine reads 1.0, formant cluster ~0.79).
        assert!(
            mean < 0.45,
            "white noise mean concentration should be low, got {mean}"
        );
    }

    #[test]
    fn formant_cluster_engages_q_boost() {
        // Vocal-formant proxy: 3-harmonic cluster with slight vibrato inside
        // one band. Under the old 2-12 dB window this read ~0.55 (Q mult
        // ~3.8x of 6x) - too weak to matter on real vocal renders. Under the
        // recalibrated 2-9 dB window it must read well into the upper range
        // so the Q-boost path actually engages on diffuse/formant material.
        let sr = 44100.0;
        let fft = 2048;
        let mut eng = AnalysisEngine::new(fft, sr);
        let centers = eng.bands().centers;
        let target = 40;
        let f0 = centers[target];
        let mut out = None;
        for n in 0..(fft * 4) {
            let t = n as f32 / sr;
            let vib = 1.0 + 0.003 * (2.0 * std::f32::consts::PI * 5.0 * t).sin();
            let x = (2.0 * std::f32::consts::PI * f0 * vib * t).sin()
                + 0.7 * (2.0 * std::f32::consts::PI * f0 * 1.02 * vib * t).sin()
                + 0.5 * (2.0 * std::f32::consts::PI * f0 * 0.98 * vib * t).sin();
            if let Some(a) = eng.process_sample(x * 0.4) {
                out = Some(a);
            }
        }
        let analysis = out.expect("formant must emit a frame");
        assert!(
            analysis.concentration[target] > 0.65,
            "formant cluster at {} Hz should engage Q-boost, got {}",
            f0,
            analysis.concentration[target]
        );
    }

    #[test]
    fn resonant_bump_is_partially_concentrated() {
        // Second diffuse-material proxy: white noise through a ringing
        // resonator at the band center. Must read clearly above the
        // broadband floor (white-noise mean ~0.34) without saturating like
        // a pure tone - the mid-range the recalibration was designed for.
        let sr = 44100.0;
        let fft = 2048;
        let eng = AnalysisEngine::new(fft, sr);
        let centers = eng.bands().centers;
        let target = 40;
        let f0 = centers[target];
        let mut eng2 = AnalysisEngine::new(fft, sr);
        let mut state: u32 = 0xdeadbeef;
        let w0 = 2.0 * std::f32::consts::PI * f0 / sr;
        let r: f32 = 0.985;
        let c = w0.cos();
        let mut y1 = 0.0f32;
        let mut y2 = 0.0f32;
        let mut out2 = None;
        for _ in 0..(fft * 4) {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            let w = ((state >> 8) as f32 / 8_388_608.0) * 2.0 - 1.0;
            let y = w + 2.0 * r * c * y1 - r * r * y2;
            y2 = y1;
            y1 = y;
            if let Some(a) = eng2.process_sample(y * 0.05) {
                out2 = Some(a);
            }
        }
        let a2 = out2.expect("resonant noise must emit a frame");
        assert!(
            a2.concentration[target] > 0.4,
            "resonant bump at {} Hz should read above broadband floor, got {}",
            f0,
            a2.concentration[target]
        );
        assert!(
            a2.concentration[target] < 1.0,
            "resonant bump must not saturate like a pure tone, got {}",
            a2.concentration[target]
        );
    }
}
