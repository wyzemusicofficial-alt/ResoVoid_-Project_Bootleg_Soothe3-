// src/dsp/bands.rs
// Log-spaced frequency band layout shared by the analysis and filter stages.

use crate::dsp::BANDS;

/// Per-band geometry derived for a given FFT size and sample rate.
#[derive(Clone, Copy)]
pub struct BandLayout {
    pub centers: [f32; BANDS],
    pub bin_lo: [usize; BANDS],
    pub bin_hi: [usize; BANDS],
    pub q: [f32; BANDS],
}

/// Build a log-spaced set of `BANDS` bands between ~20 Hz and ~96% of Nyquist.
pub fn compute_band_layout(fft_size: usize, sample_rate: f32) -> BandLayout {
    let nyquist = sample_rate * 0.5;
    let f_min = 20.0_f32;
    let f_max = (nyquist * 0.96).min(20000.0);

    let log_min = f_min.ln();
    let log_max = f_max.ln();

    let mut centers = [0.0f32; BANDS];
    for i in 0..BANDS {
        let t = i as f32 / (BANDS - 1) as f32;
        centers[i] = (log_min + t * (log_max - log_min)).exp();
    }

    let bin_res = sample_rate / fft_size as f32;
    let bin_of = |f: f32| -> usize {
        let b = (f / bin_res).round() as usize;
        b.min(fft_size / 2).max(0)
    };

    let mut bin_lo = [0usize; BANDS];
    let mut bin_hi = [0usize; BANDS];
    let mut q = [1.0f32; BANDS];

    for i in 0..BANDS {
        let lo_freq = if i == 0 {
            f_min
        } else {
            (centers[i - 1] * centers[i]).sqrt()
        };
        let hi_freq = if i == BANDS - 1 {
            f_max
        } else {
            (centers[i] * centers[i + 1]).sqrt()
        };

        let lo = bin_of(lo_freq);
        let hi = bin_of(hi_freq).max(lo + 1);
        bin_lo[i] = lo;
        bin_hi[i] = hi;

        // Q from the bandwidth in octaves between the band edges.
        let oct = (hi_freq / lo_freq).ln() / std::f32::consts::LN_2;
        let two_b = 2.0_f32.powf(oct);
        q[i] = (two_b.sqrt() / (two_b - 1.0)).clamp(0.3, 18.0);
    }

    BandLayout {
        centers,
        bin_lo,
        bin_hi,
        q,
    }
}
