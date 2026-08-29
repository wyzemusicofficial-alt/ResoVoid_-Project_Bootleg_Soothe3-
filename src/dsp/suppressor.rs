// src/dsp/suppressor.rs
// Hybrid resonance suppressor: STFT analysis + per-band biquad cascade.
//
// For each FFT size the engine keeps one independent state per channel so the
// `fft_size` parameter can be changed at runtime (allocation-free) by simply
// swapping an index. All per-sample work reuses preallocated buffers.

use rtrb::Producer;

use crate::dsp::analysis::{AnalysisEngine, AnalysisFrame};
use crate::dsp::bands::compute_band_layout;
use crate::dsp::detector::{DetectParams, Detector};
use crate::dsp::filters::BandProcessor;
use crate::dsp::BANDS;

const FFT_SIZES: [usize; 4] = [1024, 2048, 4096, 8192];

struct ChannelState {
    analysis: AnalysisEngine,
    detector: Detector,
    bands: BandProcessor,
}

impl ChannelState {
    fn new(fft_size: usize, sample_rate: f32) -> Self {
        let layout = compute_band_layout(fft_size, sample_rate);
        Self {
            analysis: AnalysisEngine::new(fft_size, sample_rate),
            detector: Detector::new(fft_size, sample_rate),
            bands: BandProcessor::new(layout, sample_rate),
        }
    }

    fn reset(&mut self) {
        self.detector.reset();
        self.bands.reset();
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
    mix: f32,           // 0.0 = dry, 1.0 = wet
    delta_mode: bool,   // output only the removed content
    sample_rate: f32,
    viz_producer: Option<Producer<AnalysisFrame>>,
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
            viz_producer: Some(viz_producer),
        }
    }

    /// Current processing latency in samples (one analysis window).
    pub fn latency(&self) -> usize {
        FFT_SIZES[self.fft_index]
    }

    pub fn set_sample_rate(&mut self, sample_rate: f32) {
        if (self.sample_rate - sample_rate).abs() < 1.0 {
            return;
        }
        self.sample_rate = sample_rate;
        self.channels = [ChannelBundle::new(sample_rate), ChannelBundle::new(sample_rate)];
    }

    pub fn set_fft_size_index(&mut self, idx: i64) {
        self.fft_index = idx.clamp(0, 3) as usize;
    }

    pub fn set_depth(&mut self, v: f32) {
        self.detect_params.depth = v.clamp(0.0, 1.0);
    }
    pub fn set_sharpness(&mut self, v: f32) {
        self.detect_params.sharpness = v.clamp(0.0, 1.0);
    }
    pub fn set_selectivity(&mut self, v: f32) {
        self.detect_params.selectivity = v.clamp(0.0, 1.0);
    }
    pub fn set_attack(&mut self, v: f32) {
        self.detect_params.attack_ms = v.max(0.1);
    }
    pub fn set_release(&mut self, v: f32) {
        self.detect_params.release_ms = v.max(1.0);
    }
    pub fn set_soft_mode(&mut self, v: bool) {
        self.detect_params.soft_mode = v;
    }
    pub fn set_mix(&mut self, v: f32) {
        self.mix = v.clamp(0.0, 1.0);
    }
    pub fn set_delta_mode(&mut self, v: bool) {
        self.delta_mode = v;
    }

    pub fn reset(&mut self) {
        for ch in &mut self.channels {
            ch.reset();
        }
    }

    /// Process a single stereo sample pair, returning (left, right).
    pub fn process_sample(&mut self, left: f32, right: f32) -> (f32, f32) {
        let idx = self.fft_index;
        let params = self.detect_params;

        let (wet_l, frame_l) =
            Self::process_channel(&mut self.channels[0], idx, left, params, self.sample_rate);
        let (wet_r, _frame_r) =
            Self::process_channel(&mut self.channels[1], idx, right, params, self.sample_rate);

        if let (Some((levels, gains)), Some(producer)) = (frame_l, self.viz_producer.as_mut()) {
            let centers = self.channels[0].states[idx].analysis.bands().centers;
            let _ = producer.push(AnalysisFrame {
                spectrum: levels,
                reduction: gains,
                centers,
                sample_rate: self.sample_rate,
            });
        }

        let out_l = apply_io(left, wet_l, self.mix, self.delta_mode);
        let out_r = apply_io(right, wet_r, self.mix, self.delta_mode);
        (out_l, out_r)
    }

    #[inline]
    fn process_channel(
        ch: &mut ChannelBundle,
        idx: usize,
        x: f32,
        params: DetectParams,
        _sample_rate: f32,
    ) -> (f32, Option<([f32; BANDS], [f32; BANDS])>) {
        let st = &mut ch.states[idx];
        let mut frame = None;
        if let Some(levels) = st.analysis.process_sample(x) {
            let gains = st.detector.process_frame(&levels, &params);
            st.bands.set_gains(&gains);
            frame = Some((levels, gains));
        }
        let wet = st.bands.process_sample(x);
        (wet, frame)
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


