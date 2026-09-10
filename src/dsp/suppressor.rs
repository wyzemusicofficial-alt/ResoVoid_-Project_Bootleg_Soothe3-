// src/dsp/suppressor.rs
use rtrb::Producer;

use crate::dsp::analysis::{AnalysisEngine, AnalysisFrame};
use crate::dsp::bands::compute_band_layout;
use crate::dsp::detector::{DetectParams, Detector};
use crate::dsp::filters::BandProcessor;
use crate::dsp::{BANDS, MAX_NODES};

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
    /// Per-region depth multipliers, one per node slot.
    /// Each value 0.0..1.0 attenuates the global depth in that frequency neighborhood.
    pub node_depths: [f32; MAX_NODES],
    /// Per-node center frequencies in Hz, one per node slot.
    pub node_freqs: [f32; MAX_NODES],
    /// Per-node shapes: 0 = Bell, 1 = Low Shelf, 2 = High Shelf.
    pub node_shapes: [usize; MAX_NODES],
    /// Per-node enabled flags. Disabled slots contribute zero weight.
    pub node_enabled: [bool; MAX_NODES],
    /// Stereo link: both channels use `min(gains_l, gains_r)` per band.
    pub stereo_linked: bool,
    /// 2x internal oversampling for the biquad synthesis path only.
    pub oversampling: bool,
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
            node_depths: [1.0; MAX_NODES],
            node_freqs: [
                200.0, 2000.0, 12000.0, 500.0, 1000.0, 4000.0, 8000.0, 16000.0,
            ],
            node_shapes: [0; MAX_NODES],
            node_enabled: [true, true, true, false, false, false, false, false],
            stereo_linked: true,
            oversampling: false,
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
        let analysis = AnalysisEngine::new(fft_size, sample_rate);
        let hop = analysis.hop();
        Self {
            analysis,
            detector: Detector::new(hop, sample_rate),
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
    stereo_linked: bool,
    oversampling: bool,
    /// Most recently applied per-channel gains (drives the 2x biquad path).
    applied_gains: [[f32; BANDS]; 2],
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
            channels: [
                ChannelBundle::new(sample_rate),
                ChannelBundle::new(sample_rate),
            ],
            fft_index: 1,
            detect_params: DetectParams::default(),
            mix: 1.0,
            delta_mode: false,
            stereo_linked: true,
            oversampling: false,
            applied_gains: [[1.0; BANDS]; 2],
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
        self.channels = [
            ChannelBundle::new(sample_rate),
            ChannelBundle::new(sample_rate),
        ];
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
        self.detect_params.node_freqs = p.node_freqs;
        self.detect_params.node_shapes = p.node_shapes;
        self.detect_params.node_enabled = p.node_enabled;
        self.mix = p.mix.clamp(0.0, 1.0);
        self.delta_mode = p.delta_mode;
        self.stereo_linked = p.stereo_linked;
        self.oversampling = p.oversampling;
    }

    pub fn reset(&mut self) {
        for ch in &mut self.channels {
            ch.reset();
        }
        self.applied_gains = [[1.0; BANDS]; 2];
        self.crossfade_remaining = 0;
    }

    #[inline]
    pub fn process_sample(&mut self, left: f32, right: f32) -> (f32, f32) {
        let idx = self.fft_index;
        let params = self.detect_params;
        let linked = self.stereo_linked;
        let oversampled = self.oversampling;

        // Both detectors run first (analysis only — no gain application yet)
        // so the stereo link can see both channels' gains before either
        // channel's biquads consume them.
        let (dry_l, frame_l) = Self::process_channel(&mut self.channels[0], idx, left, params);
        let (dry_r, frame_r) = Self::process_channel(&mut self.channels[1], idx, right, params);

        // Apply fresh detector gains. Both channels share the same hop, so in
        // practice frames arrive in lockstep; the one-sided arms are defensive
        // (a channel with no new frame keeps its previous coefficients).
        match (frame_l, frame_r) {
            (Some((levels, gains_l)), Some((_, gains_r))) => {
                let (app_l, app_r) = if linked {
                    let m = linked_gains(&gains_l, &gains_r);
                    (m, m)
                } else {
                    (gains_l, gains_r)
                };
                self.apply_gains(0, idx, &app_l, oversampled);
                self.apply_gains(1, idx, &app_r, oversampled);
                self.applied_gains = [app_l, app_r];
                let centers = self.channels[0].states[idx].analysis.bands().centers;
                let _ = self.viz_producer.push(AnalysisFrame {
                    spectrum: levels,
                    reduction: app_l,
                    centers,
                    sample_rate: self.sample_rate,
                });
            }
            (Some((levels, gains_l)), None) => {
                self.apply_gains(0, idx, &gains_l, oversampled);
                self.applied_gains[0] = gains_l;
                let centers = self.channels[0].states[idx].analysis.bands().centers;
                let _ = self.viz_producer.push(AnalysisFrame {
                    spectrum: levels,
                    reduction: gains_l,
                    centers,
                    sample_rate: self.sample_rate,
                });
            }
            (None, Some((levels, gains_r))) => {
                self.apply_gains(1, idx, &gains_r, oversampled);
                self.applied_gains[1] = gains_r;
                let centers = self.channels[0].states[idx].analysis.bands().centers;
                let _ = self.viz_producer.push(AnalysisFrame {
                    spectrum: levels,
                    reduction: self.applied_gains[0],
                    centers,
                    sample_rate: self.sample_rate,
                });
            }
            (None, None) => {
                // No new detector data: still propagate a toggled oversampling
                // flag so coefficients get recomputed at the new rate.
                self.channels[0].states[idx]
                    .bands
                    .set_oversampled(oversampled);
                self.channels[1].states[idx]
                    .bands
                    .set_oversampled(oversampled);
            }
        }

        let wet_l = {
            let st = &mut self.channels[0].states[idx];
            let g = self.applied_gains[0];
            if oversampled {
                st.bands.process_sample_2x(dry_l, &g)
            } else {
                st.bands.process_sample(dry_l)
            }
        };
        let wet_r = {
            let st = &mut self.channels[1].states[idx];
            let g = self.applied_gains[1];
            if oversampled {
                st.bands.process_sample_2x(dry_r, &g)
            } else {
                st.bands.process_sample(dry_r)
            }
        };

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

    /// Run analysis + detection for one channel. Returns the delay-compensated
    /// dry sample plus fresh detector output when a new analysis frame fired.
    /// Gain application is deliberately left to the caller (`process_sample`)
    /// so stereo linking can combine both channels first.
    #[inline]
    fn process_channel(
        ch: &mut ChannelBundle,
        idx: usize,
        x: f32,
        params: DetectParams,
    ) -> (f32, Option<([f32; BANDS], [f32; BANDS])>) {
        let st = &mut ch.states[idx];
        let mut frame = None;
        if let Some(levels) = st.analysis.process_sample(x) {
            let band_centers = st.analysis.bands().centers;
            let gains = st.detector.process_frame(&levels, &params, &band_centers);
            frame = Some((levels, gains));
        }
        let dry = st.delay.push_pop(x);
        (dry, frame)
    }

    /// Push fresh gains into one channel's biquad cascade, propagating the
    /// oversampling flag first so coefficients compute at the right rate.
    #[inline]
    fn apply_gains(&mut self, ch: usize, idx: usize, gains: &[f32; BANDS], oversampled: bool) {
        let bands = &mut self.channels[ch].states[idx].bands;
        bands.set_oversampled(oversampled);
        bands.set_gains(gains);
    }
}

/// Stereo link: per-band minimum (the more aggressive reduction wins on both
/// channels). Stack-allocated, no heap.
#[inline]
fn linked_gains(a: &[f32; BANDS], b: &[f32; BANDS]) -> [f32; BANDS] {
    let mut out = [1.0f32; BANDS];
    for i in 0..BANDS {
        out[i] = a[i].min(b[i]);
    }
    out
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
    fn analysis_overlaps_by_half_hop() {
        // 50% overlap: after the first full window, a frame is emitted every
        // hop = fft/2 samples, not every fft samples. This test replaces the
        // old non-overlapping-windows expectation.
        let fft = 32;
        let hop = fft / 2;
        let mut eng = AnalysisEngine::new(fft, 44100.0);
        assert_eq!(eng.hop(), hop);
        // First window → Some.
        for _ in 0..fft - 1 {
            eng.process_sample(0.0);
        }
        assert!(eng.process_sample(0.0).is_some());
        // Next hop-1 samples → None, then the overlapped frame → Some.
        for i in 0..hop - 1 {
            assert!(
                eng.process_sample(0.0).is_none(),
                "None during hop at sample {i}"
            );
        }
        assert!(
            eng.process_sample(0.0).is_some(),
            "overlapped frame expected after one hop of {hop}"
        );
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
        assert!(
            levels[0] > -20.0,
            "DC should register in band 0, got {}",
            levels[0]
        );
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
    fn linked_gains_takes_per_band_minimum() {
        let mut a = [1.0f32; BANDS];
        let mut b = [1.0f32; BANDS];
        a[3] = 0.5;
        b[3] = 0.8;
        a[7] = 0.9;
        b[7] = 0.2;
        let m = super::linked_gains(&a, &b);
        assert!((m[3] - 0.5).abs() < 1e-9, "left more aggressive wins");
        assert!((m[7] - 0.2).abs() < 1e-9, "right more aggressive wins");
        assert!((m[0] - 1.0).abs() < 1e-9, "unity elsewhere");
    }

    #[test]
    fn stereo_link_couples_channels_end_to_end() {
        // One-sided resonance: loud sine left, silence right.
        let run = |linked: bool| -> [[f32; BANDS]; 2] {
            let (tx, _rx) = rtrb::RingBuffer::<AnalysisFrame>::new(64);
            let mut sup = ResonanceSuppressor::new(44100.0, tx);
            sup.set_params(DspParams {
                depth: 1.0,
                selectivity: 1.0,
                stereo_linked: linked,
                ..DspParams::default()
            });
            for n in 0..16384 {
                let l = 0.5 * (2.0 * std::f32::consts::PI * 440.0 * n as f32 / 44100.0).sin();
                sup.process_sample(l, 0.0);
            }
            sup.applied_gains
        };
        let linked = run(true);
        // Left channel must be reducing somewhere.
        assert!(
            linked[0].iter().any(|&g| g < 0.95),
            "left channel should reduce a 440 Hz resonance"
        );
        // Linked: both channels carry identical gains.
        for i in 0..BANDS {
            assert!(
                (linked[0][i] - linked[1][i]).abs() < 1e-9,
                "linked gains must match on both channels at band {i}"
            );
        }
        let indep = run(false);
        assert!(
            indep[0].iter().any(|&g| g < 0.95),
            "left channel should still reduce when unlinked"
        );
        assert!(
            indep[1].iter().all(|&g| (g - 1.0).abs() < 1e-3),
            "silent right channel must stay at unity when unlinked"
        );
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
