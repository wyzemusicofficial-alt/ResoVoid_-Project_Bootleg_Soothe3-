// src/lib.rs
use std::sync::atomic::Ordering;
use std::sync::Arc;

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
mod presets;

use dsp::suppressor::DspParams;
use dsp::{AnalysisFrame, ResonanceSuppressor, BANDS};
use gui::ResoVoidEditor;

const WINDOW_SIZE: LogicalSize<f32> = LogicalSize::new(1080.0, 672.0);

/// Per-node filter shape. Backed by an IntParam (0/1/2) so host automation
/// and patch recall keep working; the enum gives names instead of magic numbers.
#[derive(Enum, PartialEq, Clone, Copy, Debug)]
pub enum NodeShape {
    #[id = "bell"]
    #[name = "Bell"]
    Bell,
    #[id = "lshelf"]
    #[name = "Low Shelf"]
    LowShelf,
    #[id = "hshelf"]
    #[name = "High Shelf"]
    HighShelf,
}

impl NodeShape {
    /// Index into the DSP's per-node shape arrays (matches node_shape() semantics).
    pub fn to_dsp_index(self) -> usize {
        match self {
            NodeShape::Bell => 0,
            NodeShape::LowShelf => 1,
            NodeShape::HighShelf => 2,
        }
    }
}

/// Default center frequencies for the 8 preallocated node slots. Soothe2
/// factory default: slots 0-5 active (LowShelf@80Hz/depth0, 4 Bells at
/// 300/1000/3500/8500Hz depth1.0, HighShelf@15kHz/depth0); slots 6-7 are
/// spare disabled nodes at 4000/8000Hz.
pub const NODE_DEFAULT_FREQS: [f32; 8] = [
    80.0, 300.0, 1000.0, 3500.0, 8500.0, 15000.0, 4000.0, 8000.0,
];

/// Stereo gain-reduction coupling.
#[derive(Enum, PartialEq, Clone, Copy, Debug)]
pub enum StereoLink {
    #[id = "linked"]
    #[name = "Stereo Link"]
    Linked,
    #[id = "indep"]
    #[name = "Independent"]
    Independent,
}

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

    /// Node 1 depth multiplier.
    #[id = "node0"]
    pub node_depth_0: FloatParam,

    /// Node 2 depth multiplier.
    #[id = "node1"]
    pub node_depth_1: FloatParam,

    /// Node 3 depth multiplier.
    #[id = "node2"]
    pub node_depth_2: FloatParam,

    /// Node 4 depth multiplier.
    #[id = "node3"]
    pub node_depth_3: FloatParam,

    /// Node 5 depth multiplier.
    #[id = "node4"]
    pub node_depth_4: FloatParam,

    /// Node 6 depth multiplier.
    #[id = "node5"]
    pub node_depth_5: FloatParam,

    /// Node 7 depth multiplier.
    #[id = "node6"]
    pub node_depth_6: FloatParam,

    /// Node 8 depth multiplier.
    #[id = "node7"]
    pub node_depth_7: FloatParam,

    /// Node 1 center frequency in Hz.
    #[id = "nf0"]
    pub node_freq_0: FloatParam,

    /// Node 2 center frequency in Hz.
    #[id = "nf1"]
    pub node_freq_1: FloatParam,

    /// Node 3 center frequency in Hz.
    #[id = "nf2"]
    pub node_freq_2: FloatParam,

    /// Node 4 center frequency in Hz.
    #[id = "nf3"]
    pub node_freq_3: FloatParam,

    /// Node 5 center frequency in Hz.
    #[id = "nf4"]
    pub node_freq_4: FloatParam,

    /// Node 6 center frequency in Hz.
    #[id = "nf5"]
    pub node_freq_5: FloatParam,

    /// Node 7 center frequency in Hz.
    #[id = "nf6"]
    pub node_freq_6: FloatParam,

    /// Node 8 center frequency in Hz.
    #[id = "nf7"]
    pub node_freq_7: FloatParam,

    /// Node 1 enabled (deleting a node sets this false; param identity is kept).
    #[id = "nen0"]
    pub node_enabled_0: BoolParam,

    /// Node 2 enabled.
    #[id = "nen1"]
    pub node_enabled_1: BoolParam,

    /// Node 3 enabled.
    #[id = "nen2"]
    pub node_enabled_2: BoolParam,

    /// Node 4 enabled.
    #[id = "nen3"]
    pub node_enabled_3: BoolParam,

    /// Node 5 enabled.
    #[id = "nen4"]
    pub node_enabled_4: BoolParam,

    /// Node 6 enabled.
    #[id = "nen5"]
    pub node_enabled_5: BoolParam,

    /// Node 7 enabled.
    #[id = "nen6"]
    pub node_enabled_6: BoolParam,

    /// Node 8 enabled.
    #[id = "nen7"]
    pub node_enabled_7: BoolParam,

    /// Node 1 shape.
    #[id = "nsh0"]
    pub node_shape_0: EnumParam<NodeShape>,

    /// Node 2 shape.
    #[id = "nsh1"]
    pub node_shape_1: EnumParam<NodeShape>,

    /// Node 3 shape.
    #[id = "nsh2"]
    pub node_shape_2: EnumParam<NodeShape>,

    /// Node 4 shape.
    #[id = "nsh3"]
    pub node_shape_3: EnumParam<NodeShape>,

    /// Node 5 shape.
    #[id = "nsh4"]
    pub node_shape_4: EnumParam<NodeShape>,

    /// Node 6 shape.
    #[id = "nsh5"]
    pub node_shape_5: EnumParam<NodeShape>,

    /// Node 7 shape.
    #[id = "nsh6"]
    pub node_shape_6: EnumParam<NodeShape>,

    /// Node 8 shape.
    #[id = "nsh7"]
    pub node_shape_7: EnumParam<NodeShape>,

    /// Stereo link for gain reduction (linked = min of L/R per band).
    #[id = "slnk"]
    pub stereo_link: EnumParam<StereoLink>,

    /// 2x internal oversampling for the biquad synthesis path.
    #[id = "os2x"]
    pub oversampling: BoolParam,
}

impl Default for ResoVoidParams {
    fn default() -> Self {
        Self {
            depth: FloatParam::new("Depth", 0.5, FloatRange::Linear { min: 0.0, max: 1.0 })
                .with_smoother(SmoothingStyle::Exponential(5.0)),

            sharpness: FloatParam::new("Sharpness", 0.5, FloatRange::Linear { min: 0.0, max: 1.0 })
                .with_smoother(SmoothingStyle::Exponential(5.0)),

            selectivity: FloatParam::new(
                "Selectivity",
                0.5,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_smoother(SmoothingStyle::Exponential(5.0)),

            attack: FloatParam::new(
                "Attack",
                10.0,
                FloatRange::Skewed {
                    min: 0.1,
                    max: 50.0,
                    factor: 0.33,
                },
            )
            .with_unit(" ms")
            .with_smoother(SmoothingStyle::Exponential(5.0)),

            release: FloatParam::new(
                "Release",
                100.0,
                FloatRange::Skewed {
                    min: 10.0,
                    max: 500.0,
                    factor: 0.33,
                },
            )
            .with_unit(" ms")
            .with_smoother(SmoothingStyle::Exponential(5.0)),

            mix: FloatParam::new("Mix", 1.0, FloatRange::Linear { min: 0.0, max: 1.0 })
                .with_unit("%")
                .with_smoother(SmoothingStyle::Linear(20.0)),

            output_gain: FloatParam::new(
                "Output",
                0.0,
                FloatRange::Linear {
                    min: -12.0,
                    max: 12.0,
                },
            )
            .with_unit(" dB")
            .with_smoother(SmoothingStyle::Exponential(5.0)),

            fft_size: IntParam::new("FFT Size", 1, IntRange::Linear { min: 0, max: 3 })
                .with_value_to_string(Arc::new(|v| {
                    match v {
                        0 => "1024".to_string(),
                        1 => "2048".to_string(),
                        2 => "4096".to_string(),
                        3 => "8192".to_string(),
                        _ => format!("{v}"),
                    }
                    .to_string()
                })),

            soft_mode: BoolParam::new("Mode", false),
            delta: BoolParam::new("Delta", false),

            node_depth_0: FloatParam::new(
                "Node Depth 1",
                0.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_smoother(SmoothingStyle::Linear(5.0)),

            node_depth_1: FloatParam::new(
                "Node Depth 2",
                1.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_smoother(SmoothingStyle::Linear(5.0)),

            node_depth_2: FloatParam::new(
                "Node Depth 3",
                1.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_smoother(SmoothingStyle::Linear(5.0)),

            node_depth_3: FloatParam::new(
                "Node Depth 4",
                1.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_smoother(SmoothingStyle::Linear(5.0)),

            node_depth_4: FloatParam::new(
                "Node Depth 5",
                1.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_smoother(SmoothingStyle::Linear(5.0)),

            node_depth_5: FloatParam::new(
                "Node Depth 6",
                0.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_smoother(SmoothingStyle::Linear(5.0)),

            node_depth_6: FloatParam::new(
                "Node Depth 7",
                1.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_smoother(SmoothingStyle::Linear(5.0)),

            node_depth_7: FloatParam::new(
                "Node Depth 8",
                1.0,
                FloatRange::Linear { min: 0.0, max: 1.0 },
            )
            .with_smoother(SmoothingStyle::Linear(5.0)),

            node_freq_0: FloatParam::new(
                "Node Freq 1",
                NODE_DEFAULT_FREQS[0],
                FloatRange::Skewed {
                    min: 20.0,
                    max: 20000.0,
                    factor: 0.3,
                },
            )
            .with_unit(" Hz")
            .with_smoother(SmoothingStyle::Linear(5.0)),

            node_freq_1: FloatParam::new(
                "Node Freq 2",
                NODE_DEFAULT_FREQS[1],
                FloatRange::Skewed {
                    min: 20.0,
                    max: 20000.0,
                    factor: 0.3,
                },
            )
            .with_unit(" Hz")
            .with_smoother(SmoothingStyle::Linear(5.0)),

            node_freq_2: FloatParam::new(
                "Node Freq 3",
                NODE_DEFAULT_FREQS[2],
                FloatRange::Skewed {
                    min: 20.0,
                    max: 20000.0,
                    factor: 0.3,
                },
            )
            .with_unit(" Hz")
            .with_smoother(SmoothingStyle::Linear(5.0)),

            node_freq_3: FloatParam::new(
                "Node Freq 4",
                NODE_DEFAULT_FREQS[3],
                FloatRange::Skewed {
                    min: 20.0,
                    max: 20000.0,
                    factor: 0.3,
                },
            )
            .with_unit(" Hz")
            .with_smoother(SmoothingStyle::Linear(5.0)),

            node_freq_4: FloatParam::new(
                "Node Freq 5",
                NODE_DEFAULT_FREQS[4],
                FloatRange::Skewed {
                    min: 20.0,
                    max: 20000.0,
                    factor: 0.3,
                },
            )
            .with_unit(" Hz")
            .with_smoother(SmoothingStyle::Linear(5.0)),

            node_freq_5: FloatParam::new(
                "Node Freq 6",
                NODE_DEFAULT_FREQS[5],
                FloatRange::Skewed {
                    min: 20.0,
                    max: 20000.0,
                    factor: 0.3,
                },
            )
            .with_unit(" Hz")
            .with_smoother(SmoothingStyle::Linear(5.0)),

            node_freq_6: FloatParam::new(
                "Node Freq 7",
                NODE_DEFAULT_FREQS[6],
                FloatRange::Skewed {
                    min: 20.0,
                    max: 20000.0,
                    factor: 0.3,
                },
            )
            .with_unit(" Hz")
            .with_smoother(SmoothingStyle::Linear(5.0)),

            node_freq_7: FloatParam::new(
                "Node Freq 8",
                NODE_DEFAULT_FREQS[7],
                FloatRange::Skewed {
                    min: 20.0,
                    max: 20000.0,
                    factor: 0.3,
                },
            )
            .with_unit(" Hz")
            .with_smoother(SmoothingStyle::Linear(5.0)),

            node_enabled_0: BoolParam::new("Node Enable 1", true),
            node_enabled_1: BoolParam::new("Node Enable 2", true),
            node_enabled_2: BoolParam::new("Node Enable 3", true),
            node_enabled_3: BoolParam::new("Node Enable 4", true),
            node_enabled_4: BoolParam::new("Node Enable 5", true),
            node_enabled_5: BoolParam::new("Node Enable 6", true),
            node_enabled_6: BoolParam::new("Node Enable 7", false),
            node_enabled_7: BoolParam::new("Node Enable 8", false),

            node_shape_0: EnumParam::new("Node Shape 1", NodeShape::LowShelf),
            node_shape_1: EnumParam::new("Node Shape 2", NodeShape::Bell),
            node_shape_2: EnumParam::new("Node Shape 3", NodeShape::Bell),
            node_shape_3: EnumParam::new("Node Shape 4", NodeShape::Bell),
            node_shape_4: EnumParam::new("Node Shape 5", NodeShape::Bell),
            node_shape_5: EnumParam::new("Node Shape 6", NodeShape::HighShelf),
            node_shape_6: EnumParam::new("Node Shape 7", NodeShape::Bell),
            node_shape_7: EnumParam::new("Node Shape 8", NodeShape::Bell),

            stereo_link: EnumParam::new("Stereo Link", StereoLink::Linked),
            oversampling: BoolParam::new("Oversampling", false),
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

        // Plot-depth default: 0 dB mapped into the visualizer's dB range
        // (-36..+36 dB): (0 - (-36)) / (36 - (-36)) = 36/72 = 0.5.
        // NOTE: node_depth_N FloatParams keep their 1.0 defaults (depth
        // multipliers, intentionally unchanged), and NEW_DEPTH=0.5 in gui.rs
        // stays mid-plot for newly added nodes — do not conflate either
        // with this plot-position seed.
        const NODE_DEFAULT_DEPTH: f32 = 36.0 / 72.0; // = 0.5, 0 dB line for -36..+36 range: (0-(-36))/(36-(-36))
        let initial_editor = ResoVoidEditor {
            params: params.clone(),
            viz_consumer,
            latest_spectrum: [(-120.0); BANDS],
            latest_reduction: [1.0; BANDS],
            centers: [0.0; BANDS],
            sample_rate: sample_rate_shared.clone(),
            gui_ctx: None,
            node_positions: [
                (NODE_DEFAULT_DEPTH, params.node_freq_0.value()),
                (NODE_DEFAULT_DEPTH, params.node_freq_1.value()),
                (NODE_DEFAULT_DEPTH, params.node_freq_2.value()),
                (NODE_DEFAULT_DEPTH, params.node_freq_3.value()),
                (NODE_DEFAULT_DEPTH, params.node_freq_4.value()),
                (NODE_DEFAULT_DEPTH, params.node_freq_5.value()),
                (NODE_DEFAULT_DEPTH, params.node_freq_6.value()),
                (NODE_DEFAULT_DEPTH, params.node_freq_7.value()),
            ],
            preset_names: Vec::new(),
            selected_preset: String::new(),
            show_save_dialog: false,
            save_name: String::new(),
            show_overwrite_confirm: false,
            presets_refresh_at: f64::NEG_INFINITY,
            dark_mode: false,
            // TEMPORARY DEBUG FIELDS - remove after concentration diagnosis is done
            debug_log: Vec::new(),
            debug_hop: 0,
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
        // MXCSR FTZ+DAZ is per-thread and the audio thread belongs to the
        // host, so (re)assert it here rather than in initialize()/activate().
        // One stmxcsr/ldmxcsr pair per block is negligible next to the DSP.
        dsp::filters::enable_flush_to_zero();

        let idx = self.params.fft_size.value().clamp(0, 3) as usize;
        if idx != self.current_fft_index {
            self.current_fft_index = idx;
            self.suppressor.set_fft_size_index(idx);
            context.set_latency_samples(self.suppressor.latency() as u32);
        }

        let soft_mode = self.params.soft_mode.value();
        let delta_mode = self.params.delta.value();

        let channels = buffer.as_slice();
        let num_samples = channels.first().map(|c| c.len()).unwrap_or(0);
        let num_channels = channels.len();

        // Smoothers MUST advance once per sample: stepping them once per
        // block ties the smoothing time to the host buffer size (a 512-sample
        // block would finish a "5 ms" smoother ~512x too fast, producing
        // zipper jumps instead of a glide). Per-sample stepping keeps the
        // time constant host-independent at the cost of a few flops.
        for i in 0..num_samples {
            self.suppressor.set_params(DspParams {
                depth: self.params.depth.smoothed.next(),
                sharpness: self.params.sharpness.smoothed.next(),
                selectivity: self.params.selectivity.smoothed.next(),
                attack_ms: self.params.attack.smoothed.next(),
                release_ms: self.params.release.smoothed.next(),
                mix: self.params.mix.smoothed.next(),
                soft_mode,
                delta_mode,
                node_depths: [
                    self.params.node_depth_0.smoothed.next(),
                    self.params.node_depth_1.smoothed.next(),
                    self.params.node_depth_2.smoothed.next(),
                    self.params.node_depth_3.smoothed.next(),
                    self.params.node_depth_4.smoothed.next(),
                    self.params.node_depth_5.smoothed.next(),
                    self.params.node_depth_6.smoothed.next(),
                    self.params.node_depth_7.smoothed.next(),
                ],
                node_freqs: [
                    self.params.node_freq_0.smoothed.next(),
                    self.params.node_freq_1.smoothed.next(),
                    self.params.node_freq_2.smoothed.next(),
                    self.params.node_freq_3.smoothed.next(),
                    self.params.node_freq_4.smoothed.next(),
                    self.params.node_freq_5.smoothed.next(),
                    self.params.node_freq_6.smoothed.next(),
                    self.params.node_freq_7.smoothed.next(),
                ],
                node_shapes: [
                    self.params.node_shape_0.value().to_dsp_index(),
                    self.params.node_shape_1.value().to_dsp_index(),
                    self.params.node_shape_2.value().to_dsp_index(),
                    self.params.node_shape_3.value().to_dsp_index(),
                    self.params.node_shape_4.value().to_dsp_index(),
                    self.params.node_shape_5.value().to_dsp_index(),
                    self.params.node_shape_6.value().to_dsp_index(),
                    self.params.node_shape_7.value().to_dsp_index(),
                ],
                node_enabled: [
                    self.params.node_enabled_0.value(),
                    self.params.node_enabled_1.value(),
                    self.params.node_enabled_2.value(),
                    self.params.node_enabled_3.value(),
                    self.params.node_enabled_4.value(),
                    self.params.node_enabled_5.value(),
                    self.params.node_enabled_6.value(),
                    self.params.node_enabled_7.value(),
                ],
                stereo_linked: self.params.stereo_link.value() == StereoLink::Linked,
                oversampling: self.params.oversampling.value(),
            });

            let gain = dsp::suppressor::db_to_linear(self.params.output_gain.smoothed.next());

            let left = if num_channels >= 1 {
                channels[0][i]
            } else {
                0.0
            };
            let right = if num_channels >= 2 {
                channels[1][i]
            } else {
                left
            };

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
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Dynamics];
}

nice_export_clap!(ResoVoid);
nice_export_vst3!(ResoVoid);
