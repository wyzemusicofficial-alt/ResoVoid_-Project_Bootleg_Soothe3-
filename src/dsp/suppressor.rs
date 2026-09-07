use rtrb::Producer;

use crate::dsp::analysis::{AnalysisEngine, AnalysisFrame};
use crate::dsp::bands::compute_band_layout;
use crate::dsp::detector::{DetectParams, Detector};
use crate::dsp::filters::BandProcessor;
use crate::dsp::BANDS;

#[derive(Clone, Copy)]
pub struct DspParams {
    pub depth: f32,
    pub sharpness: f32,
    pub selectivity: f32,
    pub attack_ms: f32,
    pub release_ms: f32,
    pub mix: f32,
    pub soft_mode: bool,
    pub delta_mode: bool,
    /// Per-region depth multipliers: [Low, Mid, High] at anchors 200 Hz, 2000 Hz, 12000 Hz.
    /// Each value 0.0..1.0 attenuates the global depth in that frequency neighborhood.
    pub node_depths: [f32; 3],
}

impl Default for DspParams {
    fn default() -> Self {
        Self {
            depth: 0.5,
            sharpness: 0.5,
            selectivity: 0.5,
            attack_ms: 10.0,
            release_ms: 100.0,
            mix: 1.0,
            soft_mode: false,
            delta_mode: false,
            node_depths: [1.0, 1.0, 1.0],
        }
    }
}

pub(crate) const FFT_SIZES: [usize; 4] = [1024, 2048, 4096, 8192];

/// Number of samples to linearly ramp the output after an FFT-size switch.
/// At 48 kHz, 512 samples ≈ 10.7 ms — long enough to mask the discontinuity
/// from swapping analysis/delay state, short enough to feel snappy.
const CROSSFADE_LEN: usize = 512;

struct ChannelState {
    analysis: AnalysisEngine,
    detector: Detector,
    bands: BandProcessor,
    delay: DelayLine,
}

impl ChannelState {
    fn new(fft_size: usize, sample_rate: f32) -> Self {
        let layout = compute_band_layout(fft_size, sample_rate);
        Self {
            analysis: AnalysisEngine::new(fft_size, sample_rate),
            detector: Detector::new(fft_size, sample_rate),
            bands: BandProcessor::new(layout, sample_rate),
            delay: DelayLine::new(fft_size),
        }
    }

    fn reset(&mut self) {
        self.analysis.reset();
        self.detector.reset();
        self.bands.reset();
        self.delay.reset();
    }
}

/// Exact-N-sample delay, where N is the buffer length.
///
/// The read index is always `pos`: each slot is read (the value written N
/// samples ago, or 0.0 until the buffer fills) before being overwritten.
/// No modular arithmetic on the read path — a predictable branch on the
/// write pointer is all that's needed.
struct DelayLine {
    buf: Vec<f32>,
    pos: usize,
}

impl DelayLine {
    fn new(len: usize) -> Self {
        Self {
            buf: vec![0.0; len.max(1)],
            pos: 0,
        }
    }

    #[inline]
    fn push_pop(&mut self, x: f32) -> f32 {
        let out = self.buf[self.pos];
        self.buf[self.pos] = x;
        self.pos += 1;
        if self.pos >= self.buf.len() {
            self.pos = 0;
        }
        out
    }

    fn reset(&mut self) {
        self.buf.fill(0.0);
        self.pos = 0;
    }
}

struct ChannelBundle {
    states: [ChannelState; 4],
}

impl ChannelBundle {
    fn new(sample_rate: f32) -> Self {
        Self {
            states: [
                ChannelState::new(FFT_SIZES[0], sample_rate),
                ChannelState::new(FFT_SIZES[1], sample_rate),
                ChannelState::new(FFT_SIZES[2], sample_rate),
                ChannelState::new(FFT_SIZES[3], sample_rate),
            ],
        }
    }

    fn reset(&mut self) {
        for s in &mut self.states {
            s.reset();
        }
    }
}

pub struct ResonanceSuppressor {
    channels: [ChannelBundle; 2],
    fft_index: usize,
    detect_params: DetectParams,
    mix: f32,
    delta_mode: bool,
    sample_rate: f32,
    viz_producer: Producer<AnalysisFrame>,
    /// Linear crossfade ramp: counts down from CROSSFADE_LEN to 0 after an
    /// FFT-size switch. While nonzero the wet output is attenuated to avoid
    /// clicks caused by the discontinuity in analysis/delay state.
    crossfade_remaining: usize,
}

impl ResonanceSuppressor {
    pub fn new(sample_rate: f32, viz_producer: Producer<AnalysisFrame>) -> Self {
        Self {
            channels: [ChannelBundle::new(sample_rate), ChannelBundle::new(sample_rate)],
            fft_index: 1,
            detect_params: DetectParams::default(),
            mix: 1.0,
            delta_mode: false,
            sample_rate,
            viz_producer,
            crossfade_remaining: 0,
        }
    }

    pub fn latency(&self) -> usize {
        FFT_SIZES[self.fft_index]
    }

    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        if (self.sample_rate - sample_rate).abs() < 1.0 {
            return;
        }
        self.sample_rate = sample_rate;
        self.channels = [ChannelBundle::new(sample_rate), ChannelBundle::new(sample_rate)];
        // Fresh delay/analysis state means the same discontinuity as an
        // FFT-size switch, so ramp the output identically.
        self.crossfade_remaining = CROSSFADE_LEN;
    }

    pub fn set_fft_size_index(&mut self, idx: usize) {
        let new = idx.min(FFT_SIZES.len() - 1);
        if new != self.fft_index {
            self.fft_index = new;
            self.crossfade_remaining = CROSSFADE_LEN;
        }
    }

    #[inline]
    pub fn set_params(&mut self, p: DspParams) {
        self.detect_params.depth = p.depth.clamp(0.0, 1.0);
        self.detect_params.sharpness = p.sharpness.clamp(0.0, 1.0);
        self.detect_params.selectivity = p.selectivity.clamp(0.0, 1.0);
        self.detect_params.attack_ms = p.attack_ms.max(0.1);
        self.detect_params.release_ms = p.release_ms.max(1.0);
        self.detect_params.soft_mode = p.soft_mode;
        self.detect_params.node_depths = p.node_depths;
        self.mix = p.mix.clamp(0.0, 1.0);
        self.delta_mode = p.delta_mode;
    }

    pub fn reset(&mut self) {
        for ch in &mut self.channels {
            ch.reset();
        }
        self.crossfade_remaining = 0;
    }

    #[inline]
    pub fn process_sample(&mut self, left: f32, right: f32) -> (f32, f32) {
        let idx = self.fft_index;
        let params = self.detect_params;

        let (wet_l, dry_l, frame_l) =
            Self::process_channel(&mut self.channels[0], idx, left, params);
        let (wet_r, dry_r, _frame_r) =
            Self::process_channel(&mut self.channels[1], idx, right, params);

        if let Some((levels, gains)) = frame_l {
            let centers = self.channels[0].states[idx].analysis.bands().centers;
            let _ = self.viz_producer.push(AnalysisFrame {
                spectrum: levels,
                reduction: gains,
                centers,
                sample_rate: self.sample_rate,
            });
        }

        let out_l = apply_io(dry_l, wet_l, self.mix, self.delta_mode);
        let out_r = apply_io(dry_r, wet_r, self.mix, self.delta_mode);

        // Linear mute-and-ramp after an FFT-size switch or sample-rate
        // rebuild. The freshly selected state has a zeroed delay line, so
        // even the *dry* path jumps; ramping only the wet mix would leave
        // that click (and do nothing at all in delta mode, where the mix is
        // bypassed). Attenuating the whole mixed output covers both.
        if self.crossfade_remaining > 0 {
            self.crossfade_remaining -= 1;
            let g = 1.0 - (self.crossfade_remaining as f32 / CROSSFADE_LEN as f32);
            (out_l * g, out_r * g)
        } else {
            (out_l, out_r)
        }
    }

    #[inline]
    fn process_channel(
        ch: &mut ChannelBundle,
        idx: usize,
        x: f32,
        params: DetectParams,
    ) -> (f32, f32, Option<([f32; BANDS], [f32; BANDS])>) {
        let st = &mut ch.states[idx];
        let mut frame = None;
        if let Some(levels) = st.analysis.process_sample(x) {
            let band_centers = st.analysis.bands().centers;
            let gains = st.detector.process_frame(&levels, &params, &band_centers);
            st.bands.set_gains(&gains);
            frame = Some((levels, gains));
        }
        let dry = st.delay.push_pop(x);
        let wet = st.bands.process_sample(dry);
        (wet, dry, frame)
    }
}

#[inline]
fn apply_io(dry: f32, wet: f32, mix: f32, delta: bool) -> f32 {
    if delta {
        dry - wet
    } else {
        dry * (1.0 - mix) + wet * mix
    }
}

#[inline]
pub fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dsp::analysis::AnalysisEngine;
    use crate::dsp::BANDS;

    // ------------------------------------------------------------------ DelayLine

    #[test]
    fn delay_line_exact_n_sample_delay() {
        // An impulse fed in must emerge exactly `len` samples later.
        let mut dl = DelayLine::new(8);
        assert!((dl.push_pop(1.0) - 0.0).abs() < 1e-10);
        for _ in 0..7 {
            assert!((dl.push_pop(0.0) - 0.0).abs() < 1e-10);
        }
        // 8 samples after the impulse, it reappears.
        let out = dl.push_pop(0.0);
        assert!((out - 1.0).abs() < 1e-10, "impulse delayed by 8 samples");
        // And the line is empty again afterwards.
        assert!((dl.push_pop(0.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn delay_line_preserves_order_over_wrap() {
        let mut dl = DelayLine::new(4);
        // Push 1..=6 through a length-4 line; outputs lag inputs by 4.
        let mut outs = Vec::new();
        for v in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] {
            outs.push(dl.push_pop(v));
        }
        assert_eq!(outs, vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn delay_line_reset_clears_buffer() {
        let mut dl = DelayLine::new(4);
        dl.push_pop(1.0);
        dl.push_pop(2.0);
        dl.push_pop(3.0);
        dl.reset();
        // After reset, reading should return 0.0.
        let out = dl.push_pop(0.0);
        assert!((out - 0.0).abs() < 1e-10);
    }

    #[test]
    fn delay_line_wraps_around_correctly() {
        let mut dl = DelayLine::new(4);
        // Fill the buffer: [1, 2, 3, 4], pos wraps to 0.
        for v in [1.0, 2.0, 3.0, 4.0] {
            dl.push_pop(v);
        }
        // Now pos=0, buf=[1,2,3,4]. Next read = pos = 0 → outputs 1.0.
        let out = dl.push_pop(5.0);
        assert!((out - 1.0).abs() < 1e-10);
        // pos=1, read=1 → outputs 2.0.
        let out = dl.push_pop(6.0);
        assert!((out - 2.0).abs() < 1e-10);
    }

    // -------------------------------------------------------- AnalysisEngine

    #[test]
    fn analysis_returns_none_until_window_fills() {
        let mut eng = AnalysisEngine::new(64, 44100.0);
        // Feed fewer than fft_size samples → should get None.
        for i in 0..63 {
            assert!(eng.process_sample(i as f32).is_none(), "None at sample {i}");
        }
    }

    #[test]
    fn analysis_returns_some_on_window_completion() {
        let fft = 64;
        let mut eng = AnalysisEngine::new(fft, 44100.0);
        // Feed exactly fft_size samples.
        for i in 0..fft - 1 {
            assert!(eng.process_sample(i as f32).is_none());
        }
        // Last sample triggers the FFT.
        let result = eng.process_sample(0.0);
        assert!(result.is_some(), "should return Some after full window");
        let levels = result.unwrap();
        assert_eq!(levels.len(), BANDS);
        // Levels should be finite (not NaN or inf).
        for &v in &levels {
            assert!(v.is_finite(), "level must be finite, got {v}");
        }
    }

    #[test]
    fn analysis_non_overlapping_windows() {
        let fft = 32;
        let mut eng = AnalysisEngine::new(fft, 44100.0);
        // First window → Some.
        for _ in 0..fft - 1 {
            eng.process_sample(0.0);
        }
        assert!(eng.process_sample(0.0).is_some());
        // Immediately after, the next fft-1 samples should be None (new window).
        for i in 0..fft - 1 {
            assert!(
                eng.process_sample(0.0).is_none(),
                "None during second window at sample {i}"
            );
        }
    }

    #[test]
    fn analysis_reset_clears_buffer() {
        let fft = 32;
        let mut eng = AnalysisEngine::new(fft, 44100.0);
        // Fill partway.
        for _ in 0..16 {
            eng.process_sample(1.0);
        }
        eng.reset();
        // After reset, we should need a full fft samples again.
        for i in 0..fft - 1 {
            assert!(eng.process_sample(0.0).is_none(), "None after reset at {i}");
        }
        assert!(eng.process_sample(0.0).is_some());
    }

    #[test]
    fn analysis_dc_signal_levels_are_high() {
        let fft = 256;
        let mut eng = AnalysisEngine::new(fft, 44100.0);
        // A DC signal should put energy in the lowest band.
        for _ in 0..fft - 1 {
            eng.process_sample(1.0);
        }
        let levels = eng.process_sample(1.0).unwrap();
        // The lowest-frequency band should have significant energy.
        assert!(levels[0] > -20.0, "DC should register in band 0, got {}", levels[0]);
    }

    // --------------------------------------------------- Crossfade behaviour

    #[test]
    fn crossfade_triggers_on_fft_size_change() {
        let (tx, _rx) = rtrb::RingBuffer::<AnalysisFrame>::new(16);
        let mut sup = ResonanceSuppressor::new(44100.0, tx);
        assert_eq!(sup.crossfade_remaining, 0);

        sup.set_fft_size_index(0); // 1024 (different from default 1 = 2048)
        assert_eq!(sup.crossfade_remaining, CROSSFADE_LEN);
    }

    #[test]
    fn crossfade_does_not_trigger_on_same_index() {
        let (tx, _rx) = rtrb::RingBuffer::<AnalysisFrame>::new(16);
        let mut sup = ResonanceSuppressor::new(44100.0, tx);
        sup.set_fft_size_index(1); // same as default
        assert_eq!(sup.crossfade_remaining, 0);
    }

    #[test]
    fn crossfade_ramp_completes() {
        let (tx, _rx) = rtrb::RingBuffer::<AnalysisFrame>::new(16);
        let mut sup = ResonanceSuppressor::new(44100.0, tx);
        sup.set_fft_size_index(0);
        assert_eq!(sup.crossfade_remaining, CROSSFADE_LEN);

        // Process enough samples to exhaust the crossfade.
        for _ in 0..CROSSFADE_LEN {
            sup.process_sample(0.0, 0.0);
        }
        assert_eq!(sup.crossfade_remaining, 0);
    }

    #[test]
    fn crossfade_mutes_dry_discontinuity_on_switch() {
        let (tx, _rx) = rtrb::RingBuffer::<AnalysisFrame>::new(16);
        let mut sup = ResonanceSuppressor::new(44100.0, tx);
        // Dry-only so the output would equal the input without the ramp.
        sup.set_params(DspParams {
            mix: 0.0,
            ..DspParams::default()
        });
        // Warm up so the delay line is full of signal.
        for _ in 0..4096 {
            sup.process_sample(1.0, 1.0);
        }
        sup.set_fft_size_index(0);
        // First sample after the switch: newly selected delay line is
        // zeroed, and the ramp gain is ~1/CROSSFADE_LEN, so the output must
        // be near silence rather than a full-scale dry click.
        let (l, r) = sup.process_sample(1.0, 1.0);
        assert!(l.abs() < 0.01, "switched output muted, got {l}");
        assert!(r.abs() < 0.01, "switched output muted, got {r}");
    }

    #[test]
    fn crossfade_ramps_monotonically_to_full_level() {
        let (tx, _rx) = rtrb::RingBuffer::<AnalysisFrame>::new(16);
        let mut sup = ResonanceSuppressor::new(44100.0, tx);
        sup.set_params(DspParams {
            mix: 0.0,
            ..DspParams::default()
        });
        for _ in 0..4096 {
            sup.process_sample(1.0, 1.0);
        }
        sup.set_fft_size_index(0);
        // Feed the new state's own delay line with DC so the underlying
        // dry signal is constant; observed output must rise monotonically
        // as the ramp gain goes 0 → 1.
        let mut prev = -1.0f32;
        for _ in 0..CROSSFADE_LEN {
            let (l, _) = sup.process_sample(1.0, 1.0);
            assert!(l + 1e-6 >= prev, "ramp must not dip: {l} after {prev}");
            prev = l;
        }
        assert_eq!(sup.crossfade_remaining, 0);
    }

    #[test]
    fn crossfade_applies_in_delta_mode() {
        let (tx, _rx) = rtrb::RingBuffer::<AnalysisFrame>::new(16);
        let mut sup = ResonanceSuppressor::new(44100.0, tx);
        sup.set_params(DspParams {
            delta_mode: true,
            ..DspParams::default()
        });
        for _ in 0..4096 {
            sup.process_sample(1.0, 1.0);
        }
        sup.set_fft_size_index(0);
        let (l, r) = sup.process_sample(1.0, 1.0);
        assert!(l.abs() < 0.01, "delta output muted on switch, got {l}");
        assert!(r.abs() < 0.01, "delta output muted on switch, got {r}");
    }

    #[test]
    fn reset_clears_crossfade() {
        let (tx, _rx) = rtrb::RingBuffer::<AnalysisFrame>::new(16);
        let mut sup = ResonanceSuppressor::new(44100.0, tx);
        sup.set_fft_size_index(0);
        assert_eq!(sup.crossfade_remaining, CROSSFADE_LEN);
        sup.reset();
        assert_eq!(sup.crossfade_remaining, 0);
    }

    #[test]
    fn set_sample_rate_triggers_crossfade() {
        let (tx, _rx) = rtrb::RingBuffer::<AnalysisFrame>::new(16);
        let mut sup = ResonanceSuppressor::new(44100.0, tx);
        assert_eq!(sup.crossfade_remaining, 0);
        sup.set_sample_rate(48000.0);
        assert_eq!(sup.crossfade_remaining, CROSSFADE_LEN);
    }

    #[test]
    fn latency_matches_fft_size() {
        let (tx, _rx) = rtrb::RingBuffer::<AnalysisFrame>::new(16);
        let mut sup = ResonanceSuppressor::new(44100.0, tx);
        assert_eq!(sup.latency(), 2048); // default index 1
        sup.set_fft_size_index(0);
        assert_eq!(sup.latency(), 1024);
        sup.set_fft_size_index(3);
        assert_eq!(sup.latency(), 8192);
    }
}
