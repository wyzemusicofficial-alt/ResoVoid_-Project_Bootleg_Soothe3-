// src/gui.rs
// native egui editor for ResoVoid, built on nice-plug-egui.
//
// The audio thread pushes `AnalysisFrame` snapshots into a lock-free rtrb ring buffer.
// This editor drains that buffer on the GUI thread and draws the input spectrum plus
// the suppression (reduction) curve. No FFT or analysis ever runs on the GUI thread.
//
// Visual design: light and airy. Soft blue-gray page, white cards with subtle borders,
// cyan/teal primary accent and a violet/magenta secondary (replaces the old red).

use std::sync::Arc;
use std::sync::atomic::Ordering;

use atomic_float::AtomicF32;

use nice_plug::context::gui::{GuiContext, ParamSetter};
use nice_plug::prelude::{BoolParam, FloatParam};
use nice_plug_egui::baseview::HandlerError;
use nice_plug_egui::widgets;
use nice_plug_egui::{Frame, NiceEguiApp};

use egui::epaint::{PathShape, RectShape};
use egui::CornerRadius;
use egui::Color32;

use rtrb::Consumer;

use crate::dsp::analysis::AnalysisFrame;
use crate::dsp::BANDS;
use crate::ResoVoidParams;

// ----- Light and airy palette ----------------------------------------------
const BG_PAGE: Color32  = Color32::from_rgb(238, 242, 248); // soft blue-gray page
const CARD: Color32     = Color32::from_rgb(255, 255, 255); // white card
const BORDER: Color32   = Color32::from_rgb(226, 232, 240); // card border
const TEXT: Color32     = Color32::from_rgb(30, 37, 50);    // primary text
const TEXT_DIM: Color32 = Color32::from_rgb(107, 118, 137); // muted text
const CYAN: Color32     = Color32::from_rgb(10, 196, 182);  // primary accent
const CYAN_TRACK: Color32 = Color32::from_rgb(225, 231, 240); // knob track
const VIOLET: Color32   = Color32::from_rgb(139, 92, 255);  // secondary accent
const GRID: Color32     = Color32::from_rgb(234, 239, 245);  // visualizer grid

/// Soft translucent cyan used to fill the area under the spectrum curve.
fn cyan_fill() -> Color32 {
    Color32::from_rgba_unmultiplied(10, 196, 182, 38)
}

/// State for the editor window. Created once in `ResoVoid::default()` and moved into
/// the editor the first time the host opens it.
pub struct ResoVoidEditor {
    pub params: Arc<ResoVoidParams>,
    pub viz_consumer: Consumer<AnalysisFrame>,
    pub latest_spectrum: [f32; BANDS],
    pub latest_reduction: [f32; BANDS],
    pub centers: [f32; BANDS],
    /// Shared with the plugin so the GUI axis knows the current sample rate.
    pub sample_rate: Arc<AtomicF32>,
    /// Captured in `build()`, used to issue parameter changes from widgets.
    pub gui_ctx: Option<GuiContext>,
}

impl NiceEguiApp for ResoVoidEditor {
    fn build(
        &mut self,
        egui_ctx: egui::Context,
        gui_ctx: GuiContext,
        _frame: &mut Frame,
    ) -> Result<(), HandlerError> {
        apply_theme(&egui_ctx);
        self.gui_ctx = Some(gui_ctx);
        Ok(())
    }

    fn editor_closed(&mut self) {
        self.gui_ctx = None;
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut Frame) {
        // Drain the latest analysis frame(s) from the audio thread.
        while let Ok(frame) = self.viz_consumer.pop() {
            self.latest_spectrum = frame.spectrum;
            self.latest_reduction = frame.reduction;
            self.centers = frame.centers;
            self.sample_rate.store(frame.sample_rate, Ordering::Relaxed);
        }
        let sample_rate = self.sample_rate.load(Ordering::Relaxed);

        let mut page = egui::Frame::default();
        page.fill = BG_PAGE;
        page.inner_margin = egui::Margin::same(14);
        page.show(ui, |ui| {
            ui.add_space(8.0);

            // Header
            ui.horizontal(|ui| {
                let (r, _) = ui.allocate_exact_size(egui::vec2(16.0, 16.0), egui::Sense::hover());
                ui.painter_at(r).circle_filled(r.center(), 7.0, CYAN);
                ui.add_space(8.0);
                ui.heading(
                    egui::RichText::new("ResoVoid")
                        .size(20.0)
                        .strong()
                        .color(TEXT),
                );
                ui.add_space(8.0);
                ui.label(
                    egui::RichText::new("Dynamic Resonance Suppressor")
                        .size(12.0)
                        .color(TEXT_DIM),
                );
            });
            ui.add_space(8.0);

            // Visualizer (draws its own white card).
            render_visualizer(
                ui,
                &self.latest_spectrum,
                &self.latest_reduction,
                &self.centers,
                sample_rate,
            );

            ui.add_space(12.0);

            if let Some(setter) = self.gui_ctx.as_ref().map(|ctx| ctx.param_setter()) {
                // Parameters card
                let mut card = egui::Frame::default();
                card.fill = CARD;
                card.stroke = egui::Stroke::new(1.0, BORDER);
                card.corner_radius = CornerRadius::same(12);
                card.inner_margin = egui::Margin::same(14);
                card.show(ui, |ui| {
                    section_header(ui, "Parameters");
                    ui.add_space(8.0);

                    ui.horizontal(|ui| {
                        param_knob(ui, "Depth", &self.params.depth, &setter, 0.0, 1.0);
                        param_knob(ui, "Sharpness", &self.params.sharpness, &setter, 0.0, 1.0);
                        param_knob(ui, "Selectivity", &self.params.selectivity, &setter, 0.0, 1.0);
                        param_knob(ui, "Attack (ms)", &self.params.attack, &setter, 0.1, 50.0);
                        param_knob(ui, "Release (ms)", &self.params.release, &setter, 10.0, 500.0);
                        param_knob(ui, "Mix", &self.params.mix, &setter, 0.0, 1.0);
                        param_knob(ui, "Output (dB)", &self.params.output_gain, &setter, -12.0, 12.0);
                    });

                    ui.add_space(10.0);
                    ui.separator();
                    ui.add_space(6.0);

                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("FFT Size").color(TEXT));
                        ui.add(widgets::ParamSlider::for_param(&self.params.fft_size, &setter));
                    });

                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        bool_checkbox(ui, "Soft Mode", &self.params.soft_mode, &setter);
                        ui.add_space(18.0);
                        bool_checkbox(ui, "Delta", &self.params.delta, &setter);
                    });
                });
            }
        });

        // Keep the visualizer animating while the editor is open.
        ui.ctx().request_repaint();
    }
}

/// Configure egui's global visuals for the light and airy look.
fn apply_theme(ctx: &egui::Context) {
    let mut v = egui::Visuals::light();
    v.override_text_color = Some(TEXT);
    v.panel_fill = BG_PAGE;
    v.extreme_bg_color = BG_PAGE;
    v.window_corner_radius = CornerRadius::same(12);

    v.widgets.inactive.bg_fill = CARD;
    v.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, BORDER);
    v.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, TEXT_DIM);
    v.widgets.inactive.corner_radius = CornerRadius::same(8);

    v.widgets.hovered.bg_fill = CARD;
    v.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, CYAN);
    v.widgets.hovered.fg_stroke = egui::Stroke::new(1.0, CYAN);
    v.widgets.hovered.corner_radius = CornerRadius::same(8);

    v.widgets.active.bg_fill = CARD;
    v.widgets.active.bg_stroke = egui::Stroke::new(1.5, CYAN);
    v.widgets.active.fg_stroke = egui::Stroke::new(1.5, CYAN);
    v.widgets.active.corner_radius = CornerRadius::same(8);

    v.widgets.noninteractive.bg_fill = BG_PAGE;
    v.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, TEXT);

    v.selection.bg_fill = Color32::from_rgba_unmultiplied(10, 196, 182, 60);
    v.selection.stroke = egui::Stroke::new(1.0, CYAN);

    v.slider_trailing_fill = true;

    ctx.set_visuals(v);
}

/// A small section title with a cyan accent bar.
fn section_header(ui: &mut egui::Ui, title: &str) {
    ui.horizontal(|ui| {
        let (r, _) = ui.allocate_exact_size(egui::vec2(4.0, 16.0), egui::Sense::hover());
        ui.painter_at(r).rect_filled(r, 2.0, CYAN);
        ui.add_space(6.0);
        ui.label(egui::RichText::new(title).size(14.0).strong().color(TEXT));
    });
}

/// Draw a native egui rotary knob.
fn draw_knob(ui: &mut egui::Ui, label: &str, value: f32, min: f32, max: f32) -> Option<f32> {
    let size = egui::vec2(70.0, 70.0);
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::drag());
    let painter = ui.painter_at(rect);
    let center = rect.center();
    let radius = (rect.width().min(rect.height()) / 2.0) - 8.0;

    let t = ((value - min) / (max - min)).clamp(0.0, 1.0);
    let a_min = std::f32::consts::PI * 0.75;
    let a_max = std::f32::consts::PI * 2.25;
    let angle = a_min + t * (a_max - a_min);

    // Knob body.
    painter.circle_filled(center, radius, CARD);
    painter.circle_stroke(center, radius, egui::Stroke::new(2.0, BORDER));
    // Background track arc.
    draw_arc(
        &painter,
        center,
        radius,
        a_min,
        a_max,
        egui::Stroke::new(3.0, CYAN_TRACK),
    );
    // Active arc.
    draw_arc(
        &painter,
        center,
        radius,
        a_min,
        angle,
        egui::Stroke::new(3.0, CYAN),
    );
    let tip = center + radius * egui::Vec2::new(angle.cos(), angle.sin());
    painter.line_segment([center, tip], egui::Stroke::new(2.5, CYAN));

    let mut new_value = value;
    if response.dragged() {
        let delta = (-response.drag_delta().y / 200.0) * (max - min);
        new_value = (value + delta).clamp(min, max);
    }

    painter.text(
        center + egui::vec2(0.0, radius + 10.0),
        egui::Align2::CENTER_CENTER,
        label,
        egui::FontId::proportional(11.0),
        TEXT,
    );
    painter.text(
        center + egui::vec2(0.0, radius + 24.0),
        egui::Align2::CENTER_CENTER,
        format!("{value:.2}"),
        egui::FontId::proportional(10.0),
        TEXT_DIM,
    );

    if response.dragged() {
        Some(new_value)
    } else {
        None
    }
}

/// Draw an arc segment from `start` to `end` (radians) around `center`.
fn draw_arc(
    painter: &egui::Painter,
    center: egui::Pos2,
    radius: f32,
    start: f32,
    end: f32,
    stroke: egui::Stroke,
) {
    let span = (end - start).abs();
    let steps = ((span * 32.0 / std::f32::consts::PI) as usize + 1).clamp(1, 64);
    let mut prev = center + radius * egui::Vec2::new(start.cos(), start.sin());
    for i in 1..=steps {
        let a = start + (end - start) * (i as f32 / steps as f32);
        let p = center + radius * egui::Vec2::new(a.cos(), a.sin());
        painter.line_segment([prev, p], stroke);
        prev = p;
    }
}

/// Map a parameter to a knob and apply changes through the `ParamSetter`.
fn param_knob(
    ui: &mut egui::Ui,
    label: &str,
    param: &FloatParam,
    setter: &ParamSetter,
    min: f32,
    max: f32,
) {
    let value = param.value();
    if let Some(new_value) = draw_knob(ui, label, value, min, max) {
        setter.begin_set_parameter(param);
        setter.set_parameter(param, new_value);
        setter.end_set_parameter(param);
    }
}

/// Checkbox wired to a boolean parameter via the `ParamSetter`.
fn bool_checkbox(ui: &mut egui::Ui, label: &str, param: &BoolParam, setter: &ParamSetter) {
    let mut v = param.value();
    if ui
        .checkbox(&mut v, egui::RichText::new(label).color(TEXT))
        .clicked()
    {
        setter.begin_set_parameter(param);
        setter.set_parameter(param, v);
        setter.end_set_parameter(param);
    }
}

/// Render the input spectrum (cyan) and reduction curve (violet) on a logarithmic
/// frequency axis. Called from the GUI thread only. The top of the axis is
/// Nyquist-aware so it never plots bands above the actual sample rate's limit.
fn render_visualizer(
    ui: &mut egui::Ui,
    spectrum: &[f32; BANDS],
    reduction: &[f32; BANDS],
    centers: &[f32; BANDS],
    sample_rate: f32,
) {
    let (rect, _response) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), 240.0),
        egui::Sense::hover(),
    );
    let painter = ui.painter_at(rect);

    // Card background + border.
    painter.add(RectShape::new(
        rect,
        CornerRadius::same(12),
        CARD,
        egui::Stroke::new(1.0, BORDER),
        egui::StrokeKind::Outside,
    ));

    let plot = rect.shrink(12.0);

    let f_min = 20.0_f32;
    let f_max = (sample_rate * 0.5).min(20000.0).max(f_min * 2.0);
    let l_min = f_min.log10();
    let l_max = f_max.log10();
    let db_min = -80.0_f32;
    let db_max = 12.0_f32;

    let x_of = |freq: f32| -> f32 {
        let l = freq.max(f_min).log10();
        plot.left() + ((l - l_min) / (l_max - l_min)) * plot.width()
    };
    let y_of = |db: f32| -> f32 {
        plot.bottom() - ((db - db_min) / (db_max - db_min)) * plot.height()
    };

    // Frequency grid lines.
    for &freq in &[100.0_f32, 1000.0, 10000.0] {
        let x = x_of(freq);
        painter.line_segment(
            [egui::pos2(x, plot.top()), egui::pos2(x, plot.bottom())],
            egui::Stroke::new(1.0, GRID),
        );
    }
    // dB grid lines.
    for &db in &[-60.0_f32, -40.0, -20.0, 0.0] {
        let y = y_of(db);
        painter.line_segment(
            [egui::pos2(plot.left(), y), egui::pos2(plot.right(), y)],
            egui::Stroke::new(1.0, GRID),
        );
    }

    // Input spectrum polyline + soft filled area.
    let mut spec_pts: Vec<egui::Pos2> = Vec::with_capacity(BANDS);
    for i in 0..BANDS {
        if centers[i] <= 0.0 {
            continue;
        }
        if centers[i] < f_min || centers[i] > f_max {
            continue;
        }
        spec_pts.push(egui::pos2(x_of(centers[i]), y_of(spectrum[i].clamp(db_min, db_max))));
    }
    if spec_pts.len() >= 2 {
        let mut area = spec_pts.clone();
        area.push(egui::pos2(plot.right(), plot.bottom()));
        area.push(egui::pos2(plot.left(), plot.bottom()));
        painter.add(egui::Shape::Path(PathShape {
            points: area,
            closed: true,
            fill: cyan_fill(),
            stroke: egui::Stroke::NONE.into(),
        }));
        painter.add(egui::Shape::line(spec_pts, egui::Stroke::new(1.8, CYAN)));
    }

    // Reduction curve: linear gain -> dB (<= 0). Violet line.
    let mut red_pts: Vec<egui::Pos2> = Vec::with_capacity(BANDS);
    for i in 0..BANDS {
        if centers[i] <= 0.0 {
            continue;
        }
        if centers[i] < f_min || centers[i] > f_max {
            continue;
        }
        let rdb = 20.0 * (reduction[i].max(1e-4)).log10();
        red_pts.push(egui::pos2(x_of(centers[i]), y_of(rdb.clamp(-40.0, 0.0))));
    }
    if red_pts.len() >= 2 {
        painter.add(egui::Shape::line(red_pts, egui::Stroke::new(2.0, VIOLET)));
    }

    // Axis tick labels.
    for &freq in &[100.0_f32, 1000.0, 10000.0] {
        let label = match freq {
            100.0 => "100",
            1000.0 => "1k",
            10000.0 => "10k",
            _ => "",
        };
        painter.text(
            egui::pos2(x_of(freq), plot.bottom() - 12.0),
            egui::Align2::CENTER_CENTER,
            label,
            egui::FontId::proportional(10.0),
            TEXT_DIM,
        );
    }
    for &db in &[-60.0_f32, -40.0, -20.0, 0.0] {
        painter.text(
            egui::pos2(plot.left() + 4.0, y_of(db)),
            egui::Align2::LEFT_CENTER,
            format!("{db}"),
            egui::FontId::proportional(9.0),
            TEXT_DIM,
        );
    }

    painter.text(
        egui::pos2(plot.left() + 6.0, plot.top() + 4.0),
        egui::Align2::LEFT_TOP,
        format!(
            "Spectrum (cyan) / Reduction (violet), 20 Hz - {} kHz, log",
            (f_max / 1000.0).round() as u32
        ),
        egui::FontId::proportional(11.0),
        TEXT_DIM,
    );
}
