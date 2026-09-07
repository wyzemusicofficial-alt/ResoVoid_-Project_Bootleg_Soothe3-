use std::sync::Arc;
use std::sync::atomic::Ordering;

use atomic_float::AtomicF32;

use nice_plug::editor::dpi::LogicalSize;
use nice_plug::editor::ResizeHint;
use nice_plug::prelude::*;
use nice_plug_egui::{
    create_egui_editor, EguiEditor, EguiEditorState, EguiNiceSettings, RepaintNotifier,
};
use rtrb::RingBuffer;

mod dsp;
mod gui;

use dsp::{AnalysisFrame, BANDS, ResonanceSuppressor};
use dsp::suppressor::DspParams;
use gui::ResoVoidEditor;

const WINDOW_SIZE: LogicalSize<f32> = LogicalSize::new(900.0, 560.0);

#[derive(Params)]
pub struct ResoVoidParams {
    #[id = "depth"]
    pub depth: FloatParam,

    #[id = "sharp"]
    pub sharpness: FloatParam,

    #[id = "sel"]
    pub selectivity: FloatParam,

    #[id = "atk"]
    pub attack: FloatParam,

    #[id = "rel"]
    pub release: FloatParam,

    #[id = "mix"]
    pub mix: FloatParam,

    #[id = "out"]
    pub output_gain: FloatParam,

    #[id = "fft"]
    pub fft_size: IntParam,

    #[id = "mode"]
    pub soft_mode: BoolParam,

    #[id = "delt"]
    pub delta: BoolParam,
}

impl Default for ResoVoidParams {
    fn default() -> Self {
        Self {
            depth: FloatParam::new("Depth", 0.5, FloatRange::Linear { min: 0.0, max: 1.0 })
                .with_smoother(SmoothingStyle::Exponential(5.0)),

            sharpness: FloatParam::new("Sharpness", 0.5, FloatRange::Linear { min: 0.0, max: 1.0 })
                .with_smoother(SmoothingStyle::Exponential(5.0)),

            selectivity: FloatParam::new("Selectivity", 0.5, FloatRange::Linear { min: 0.0, max: 1.0 })
                .with_smoother(SmoothingStyle::Exponential(5.0)),

            attack: FloatParam::new(
                "Attack",
                10.0,
                FloatRange::Skewed { min: 0.1, max: 50.0, factor: 0.33 },
            )
            .with_unit(" ms")
            .with_smoother(SmoothingStyle::Exponential(5.0)),

            release: FloatParam::new(
                "Release",
                100.0,
                FloatRange::Skewed { min: 10.0, max: 500.0, factor: 0.33 },
            )
            .with_unit(" ms")
            .with_smoother(SmoothingStyle::Exponential(5.0)),

            mix: FloatParam::new("Mix", 1.0, FloatRange::Linear { min: 0.0, max: 1.0 })
                .with_unit("%")
                .with_smoother(SmoothingStyle::Linear(20.0)),

            output_gain: FloatParam::new("Output", 0.0, FloatRange::Linear { min: -12.0, max: 12.0 })
                .with_unit(" dB")
                .with_smoother(SmoothingStyle::Exponential(5.0)),

            fft_size: IntParam::new("FFT Size", 1, IntRange::Linear { min: 0, max: 3 })
                .with_value_to_string(Arc::new(|v| match v {
                    0 => "1024".to_string(),
                    1 => "2048".to_string(),
                    2 => "4096".to_string(),
                    3 => "8192".to_string(),
                    _ => format!("{v}"),
                }
                .to_string())),

            soft_mode: BoolParam::new("Mode", false),
            delta: BoolParam::new("Delta", false),
        }
    }
}

pub struct ResoVoid {
    params: Arc<ResoVoidParams>,
    suppressor: ResonanceSuppressor,

    editor_state: Arc<EguiEditorState>,
    repaint_notifier: RepaintNotifier,

    sample_rate_shared: Arc<AtomicF32>,
    current_fft_index: usize,

    initial_editor: Option<ResoVoidEditor>,
}

impl Default for ResoVoid {
    fn default() -> Self {
        let params = Arc::new(ResoVoidParams::default());

        let (viz_producer, viz_consumer) = RingBuffer::<AnalysisFrame>::new(16);
        let sample_rate_shared = Arc::new(AtomicF32::new(44100.0));

        let initial_editor = ResoVoidEditor {
            params: params.clone(),
            viz_consumer,
            latest_spectrum: [(-120.0); BANDS],
            latest_reduction: [1.0; BANDS],
            centers: [0.0; BANDS],
            sample_rate: sample_rate_shared.clone(),
            gui_ctx: None,
        };

        Self {
            params: params.clone(),
            suppressor: ResonanceSuppressor::new(44100.0, viz_producer),
            editor_state: EguiEditorState::from_size(WINDOW_SIZE, 1.0),
            repaint_notifier: RepaintNotifier::new(),
            sample_rate_shared,
            current_fft_index: 1,
            initial_editor: Some(initial_editor),
        }
    }
}

impl Plugin for ResoVoid {
    const NAME: &'static str = "ResoVoid";
    const VENDOR: &'static str = "Wyz3Music";
    const URL: &'static str = "https://www.youtube.com/@wyzemusic_official";
    const EMAIL: &'static str = "wyzemusicofficial@gmail.com";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(2),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(1),
            main_output_channels: NonZeroU32::new(1),
            ..AudioIOLayout::const_default()
        },
    ];

    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    type Editor = EguiEditor<ResoVoidEditor>;
    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Self::Editor> {
        create_egui_editor(
            self.editor_state.clone(),
            self.repaint_notifier.clone(),
            EguiNiceSettings::new()
                .with_tile("ResoVoid")
                .with_resize_hint(ResizeHint::resizable().with_min_logical_size(WINDOW_SIZE)),
            self.initial_editor.take().unwrap(),
        )
    }

    fn activate(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        context: &mut impl ActivateContext<Self>,
    ) -> bool {
        let sr = buffer_config.sample_rate;
        self.sample_rate_shared.store(sr, Ordering::Relaxed);
        self.suppressor.set_sample_rate(sr);

        let idx = self.params.fft_size.value().clamp(0, 3) as usize;
        self.current_fft_index = idx;
        self.suppressor.set_fft_size_index(idx);
        context.set_latency_samples(self.suppressor.latency() as u32);
        true
    }

    fn reset(&mut self) {
        self.suppressor.reset();
    }

    #[inline]
    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let dsp_params = DspParams {
            depth: self.params.depth.smoothed.next(),
            sharpness: self.params.sharpness.smoothed.next(),
            selectivity: self.params.selectivity.smoothed.next(),
            attack_ms: self.params.attack.smoothed.next(),
            release_ms: self.params.release.smoothed.next(),
            mix: self.params.mix.smoothed.next(),
            soft_mode: self.params.soft_mode.value(),
            delta_mode: self.params.delta.value(),
        };
        self.suppressor.set_params(dsp_params);

        let idx = self.params.fft_size.value().clamp(0, 3) as usize;
        if idx != self.current_fft_index {
            self.current_fft_index = idx;
            self.suppressor.set_fft_size_index(idx);
            context.set_latency_samples(self.suppressor.latency() as u32);
        }

        let output_gain_db = self.params.output_gain.smoothed.next();
        let gain = dsp::suppressor::db_to_linear(output_gain_db);

        let channels = buffer.as_slice();
        let num_samples = channels.first().map(|c| c.len()).unwrap_or(0);
        let num_channels = channels.len();

        for i in 0..num_samples {
            let left = if num_channels >= 1 { channels[0][i] } else { 0.0 };
            let right = if num_channels >= 2 { channels[1][i] } else { left };

            let (out_left, out_right) = self.suppressor.process_sample(left, right);

            if num_channels >= 2 {
                channels[0][i] = out_left * gain;
                channels[1][i] = out_right * gain;
            } else if num_channels == 1 {
                channels[0][i] = out_left * gain;
            }
        }

        ProcessStatus::Normal
    }
}

impl ClapPlugin for ResoVoid {
    const CLAP_ID: &'static str = "com.wyzemusic.resovoid";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("Dynamic Resonance Suppressor");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Stereo,
        ClapFeature::Mono,
        ClapFeature::Utility,
    ];
}

impl Vst3Plugin for ResoVoid {
    const VST3_CLASS_ID: [u8; 16] = *b"ResoVoidWowNice!";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] = &[
        Vst3SubCategory::Fx,
        Vst3SubCategory::Dynamics,
    ];
}

nice_export_clap!(ResoVoid);
nice_export_vst3!(ResoVoid);
