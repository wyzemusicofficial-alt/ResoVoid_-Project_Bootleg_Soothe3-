// src/dsp/analysis.rs
// STFT analysis front-end: windowed real FFT -> per-band level (dB).
// Allocation-free at runtime: all buffers are preallocated in `new`.

use std::sync::Arc;

use num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};

use crate::dsp::bands::{compute_band_layout, BandLayout};
use crate::dsp::BANDS;

/// A single snapshot pushed from the audio thread to the GUI thread.
#[derive(Clone, Copy)]
pub struct AnalysisFrame {
    /// Per-band input spectrum level in dB.
    pub spectrum: [f32; BANDS],
    /// Per-band linear gain applied by the suppressor (<= 1.0).
    pub reduction: [f32; BANDS],
    /// Band center frequencies in Hz (cached from the band layout).
    pub centers: [f32; BANDS],
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
    pub fn process_sample(&mut self, x: f32) -> Option<[f32; BANDS]> {
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
        for b in 0..BANDS {
            let lo = self.bands.bin_lo[b];
            let hi = self.bands.bin_hi[b].min(self.spectrum.len() - 1);
            let mut sum = 0.0f32;
            for k in lo..=hi {
                let c = self.spectrum[k];
                sum += (c.re * c.re + c.im * c.im).sqrt();
            }
            let n = (hi - lo + 1).max(1) as f32;
            let mag = sum / n;
            levels[b] = 20.0 * (mag + 1e-9).log10();
        }
        Some(levels)
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

