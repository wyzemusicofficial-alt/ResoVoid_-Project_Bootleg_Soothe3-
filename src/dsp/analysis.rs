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
const PAR_HIGH_DB: f32 = 12.0;

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
    pub reduction: [f32; BANDS],
    /// Band center frequencies in Hz (cached from the band layout).
    pub centers: [f32; BANDS],
    // TEMPORARY DEBUG FIELD - remove after concentration diagnosis is done
    pub concentration: [f32; BANDS],
    /// Current sample rate.
    pub sample_rate: f32,
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
        assert!(
            mean < 0.3,
            "white noise mean concentration should be low, got {mean}"
        );
    }
}

