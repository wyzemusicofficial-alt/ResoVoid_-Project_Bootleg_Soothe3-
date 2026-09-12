// src/dsp/bands.rs
// Log-spaced frequency band layout shared by the analysis and filter stages.

use crate::dsp::BANDS;

/// Maximum allowed -60 dB ringing time for a surgical cut, in milliseconds.
/// Caps the effective Q at low frequencies where the nominal Q would ring
/// far longer than this.
const MAX_RING_MS: f32 = 100.0;
/// Time in decay-time-constants to reach -60 dB (ln(1000)).
const RING_TIME_LN_FACTOR: f32 = 6.907755;
/// Absolute ceiling for the ringing-limited Q, even at high frequencies.
const NOMINAL_Q_CEILING: f32 = 90.0;

/// Per-band geometry derived for a given FFT size and sample rate.
#[derive(Clone, Copy)]
pub struct BandLayout {
    pub centers: [f32; BANDS],
    pub bin_lo: [usize; BANDS],
    pub bin_hi: [usize; BANDS],
    pub q: [f32; BANDS],
    pub q_ceiling: [f32; BANDS],
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
        b.min(fft_size / 2)
    };

    let mut bin_lo = [0usize; BANDS];
    let mut bin_hi = [0usize; BANDS];
    let mut q = [1.0f32; BANDS];
    let mut q_ceiling = [1.0f32; BANDS];

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
        let ring_ceiling =
            (MAX_RING_MS / 1000.0) * std::f32::consts::PI * centers[i] / RING_TIME_LN_FACTOR;
        q_ceiling[i] = ring_ceiling.min(NOMINAL_Q_CEILING).max(q[i]);
    }

    BandLayout {
        centers,
        bin_lo,
        bin_hi,
        q,
        q_ceiling,
    }
}

#[cfg(test)]
mod tests {
    use super::compute_band_layout;
    use crate::dsp::BANDS;

    #[test]
    fn layout_is_log_spaced_and_in_range() {
        let sr = 44100.0;
        let fft = 2048;
        let layout = compute_band_layout(fft, sr);

        // Lowest band starts at ~20 Hz; highest band stays below ~96% Nyquist
        // (capped at 20 kHz for 44.1 kHz).
        assert!((layout.centers[0] - 20.0).abs() < 1.0, "first center ~20 Hz");
        let nyquist = sr * 0.5;
        let expected_max = (nyquist * 0.96).min(20000.0);
        assert!(
            layout.centers[BANDS - 1] <= expected_max + 1.0,
            "last center within range"
        );

        // Centers must be strictly increasing (log spacing).
        for i in 1..BANDS {
            assert!(layout.centers[i] > layout.centers[i - 1]);
        }

        // Each band's bin range is well-formed and every Q is clamped.
        let half = fft / 2;
        for i in 0..BANDS {
            assert!(layout.bin_lo[i] < layout.bin_hi[i]);
            assert!(layout.bin_hi[i] <= half);
            assert!((0.3..=18.0).contains(&layout.q[i]), "q in bounds");
        }
    }

    #[test]
    fn layout_scales_with_sample_rate() {
        let low = compute_band_layout(2048, 22050.0);
        let high = compute_band_layout(2048, 96000.0);
        // At higher sample rates the top band reaches higher frequencies.
        assert!(high.centers[BANDS - 1] > low.centers[BANDS - 1]);
    }

    #[test]
    fn q_ceiling_covers_nominal_q_and_rises_with_frequency() {
        let layout = compute_band_layout(2048, 44100.0);
        for i in 0..BANDS {
            assert!(
                layout.q_ceiling[i] >= layout.q[i],
                "ceiling must cover nominal q at band {i}"
            );
        }
        assert!(
            layout.q_ceiling[0] < layout.q_ceiling[BANDS - 1],
            "lowest band ceiling ({}) must be meaningfully smaller than highest ({})",
            layout.q_ceiling[0],
            layout.q_ceiling[BANDS - 1]
        );
    }
}
