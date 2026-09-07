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
pub struct AnalysisEngine {
    fft_size: usize,
    r2c: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    buf: Vec<f32>,
    input: Vec<f32>,
    spectrum: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
    write_pos: usize,
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
            r2c,
            window,
            buf,
            input,
            spectrum,
            scratch,
            write_pos: 0,
            bands,
        }
    }

    pub fn bands(&self) -> &BandLayout {
        &self.bands
    }

    /// Feed one input sample. Returns per-band levels in dB once a full
    /// (non-overlapping) analysis window has been collected.
    ///
    /// NOTE: windows are intentionally non-overlapping (`write_pos` resets to 0
    /// after each full block). This is a deliberate tradeoff: cheaper than
    /// hop-based overlap-add, and the per-band biquad cascade smooths gain
    /// changes between updates. The detector consequently refreshes once per
    /// window rather than every hop. See the README "Design Notes".
    #[inline]
    pub fn process_sample(&mut self, x: f32) -> Option<[f32; BANDS]> {
        self.buf[self.write_pos] = x;
        self.write_pos += 1;

        if self.write_pos >= self.fft_size {
            self.write_pos = 0;

            for i in 0..self.fft_size {
                self.input[i] = self.buf[i] * self.window[i];
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
        } else {
            None
        }
    }

    pub fn reset(&mut self) {
        self.buf.fill(0.0);
        self.write_pos = 0;
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

