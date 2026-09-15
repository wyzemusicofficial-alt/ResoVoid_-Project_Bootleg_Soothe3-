// src/gui.rs
// native egui editor for ResoVoid, built on nice-plug-egui.
//
// The audio thread pushes `AnalysisFrame` snapshots into a lock-free rtrb ring buffer.
// This editor drains that buffer on the GUI thread and draws the input spectrum plus
// the suppression (reduction) curve. No FFT or analysis ever runs on the GUI thread.
//
// Visual design: light and airy. Soft blue-gray page, white cards with subtle borders,
// cyan/teal primary accent and a violet/magenta secondary (replaces the old red).

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use atomic_float::AtomicF32;

use nice_plug::context::gui::{GuiContext, ParamSetter};
use nice_plug::prelude::{BoolParam, EnumParam, FloatParam, Param};
use nice_plug_egui::baseview::HandlerError;
use nice_plug_egui::widgets;
use nice_plug_egui::{Frame, NiceEguiApp};

use egui::epaint::{PathShape, RectShape};
use egui::CornerRadius;
use egui::Color32;

use rtrb::Consumer;

use crate::dsp::analysis::AnalysisFrame;
use crate::dsp::detector::node_shape;
use crate::dsp::suppressor::FFT_SIZES;
use crate::dsp::{BANDS, MAX_NODES};
use crate::{NodeShape, ResoVoidParams, StereoLink};

// ----- Theme ----------------------------------------------------------------
// Every painted color flows through `Theme` so the whole GUI can flip between
// the light look and an eye-strain-friendly dark look. Signal hues (CYAN,
// VIOLET, node dots, GR tint) are identical in both themes — they read well
// on either background; only surfaces, text, and grids invert.
#[derive(Clone, Copy)]
pub struct Theme {
    pub bg_page: Color32,
    pub card: Color32,
    pub border: Color32,
    pub text: Color32,
    pub text_dim: Color32,
    pub cyan: Color32,
    pub cyan_track: Color32,
    pub violet: Color32,
    pub grid: Color32,
    pub shape: Color32,
    pub pill_bg: Color32,
}

impl Theme {
    pub fn light() -> Self {
        Self {
            bg_page: Color32::from_rgb(238, 242, 248), // soft blue-gray page
            card: Color32::from_rgb(255, 255, 255),    // white card
            border: Color32::from_rgb(226, 232, 240),  // card border
            text: Color32::from_rgb(30, 37, 50),       // primary text
            text_dim: Color32::from_rgb(107, 118, 137), // muted text
            cyan: Color32::from_rgb(10, 196, 182),     // primary accent
            cyan_track: Color32::from_rgb(225, 231, 240), // knob track
            violet: Color32::from_rgb(139, 92, 255),   // secondary accent
            grid: Color32::from_rgb(234, 239, 245),    // visualizer grid
            shape: Color32::from_rgb(180, 160, 120),   // warm amber/taupe curve
            pill_bg: Color32::from_rgb(225, 231, 240), // toggle pill backdrop
        }
    }

    pub fn dark() -> Self {
        Self {
            bg_page: Color32::from_rgb(15, 18, 26),   // deep slate page
            card: Color32::from_rgb(26, 30, 43),      // raised surface
            border: Color32::from_rgb(47, 55, 76),    // card border
            text: Color32::from_rgb(232, 236, 244),   // primary text
            text_dim: Color32::from_rgb(148, 158, 178), // muted text
            cyan: Color32::from_rgb(10, 196, 182),    // primary accent (kept)
            cyan_track: Color32::from_rgb(47, 55, 76), // knob track
            violet: Color32::from_rgb(139, 92, 255),  // secondary accent (kept)
            grid: Color32::from_rgb(38, 44, 60),      // visualizer grid
            shape: Color32::from_rgb(180, 160, 120),  // warm amber/taupe curve
            pill_bg: Color32::from_rgb(47, 55, 76),   // toggle pill backdrop
        }
    }

    /// Relative luminance of an sRGB color (0 = black, 1 = white).
    /// Test-only helper for the theme contrast test.
    #[cfg(test)]
    pub fn luminance(c: Color32) -> f32 {
        let lin = |u: u8| {
            let s = u as f32 / 255.0;
            if s <= 0.04045 {
                s / 12.92
            } else {
                ((s + 0.055) / 1.055).powf(2.4)
            }
        };
        0.2126 * lin(c.r()) + 0.7152 * lin(c.g()) + 0.0722 * lin(c.b())
    }
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
    /// Per-node positions: (depth 0.0..1.0, freq_hz), one slot per node.
    /// Whether a slot is drawn/active comes from the node's BoolParam.
    pub node_positions: [(f32, f32); MAX_NODES],
    /// Dark-mode toggle. GUI-local state (not a host parameter): it only
    /// repaints, never touches DSP or automation.
    pub dark_mode: bool,
    /// Preset browser state (GUI thread only; refreshed at most every 2 s).
    pub preset_names: Vec<String>,
    pub selected_preset: String,
    pub show_save_dialog: bool,
    pub save_name: String,
    pub show_overwrite_confirm: bool,
    pub presets_refresh_at: f64,
    // TEMPORARY DEBUG FIELDS - remove after concentration diagnosis is done
    /// Captured viz frames: (hop_index, centers, levels, concentration, gain_linear).
    /// GUI thread only; capped at 5000 entries (stop appending when full).
    pub debug_log: Vec<(u32, [f32; BANDS], [f32; BANDS], [f32; BANDS], [f32; BANDS])>,
    /// Monotonic hop counter stamped onto each `debug_log` entry.
    pub debug_hop: u32,
}

impl NiceEguiApp for ResoVoidEditor {
    fn build(
        &mut self,
        egui_ctx: egui::Context,
        gui_ctx: GuiContext,
        _frame: &mut Frame,
    ) -> Result<(), HandlerError> {
        apply_theme(&egui_ctx, &Theme::light(), false);
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
            // TEMPORARY DEBUG CAPTURE - remove after concentration diagnosis is done.
            // Data-capture side channel only: no DSP behavior changes.
            // Capped at 5000 entries (stop appending when full). GUI thread only.
            if self.debug_log.len() < 5000 {
                self.debug_log.push((
                    self.debug_hop,
                    frame.centers,
                    frame.spectrum,
                    frame.concentration,
                    frame.reduction,
                ));
            }
            self.debug_hop = self.debug_hop.wrapping_add(1);
        }
        // Unconditional repaint for smooth spectrogram rendering.
        // The audio thread pushes frames asynchronously — gating repaint
        // on frame arrival causes micro-stutters vs the monitor refresh rate.
        ui.ctx().request_repaint();
        let sample_rate = self.sample_rate.load(Ordering::Relaxed);

        // Active theme: re-applied every frame so the dark-mode toggle (and
        // any egui-owned popup/menu) repaints instantly.
        let theme = if self.dark_mode {
            Theme::dark()
        } else {
            Theme::light()
        };
        apply_theme(ui.ctx(), &theme, self.dark_mode);

        let mut page = egui::Frame::default();
        page.fill = theme.bg_page;
        page.inner_margin = egui::Margin::same(14);
        page.show(ui, |ui| {
            ui.add_space(8.0);

            // Header (plain horizontal: *_centered variants claim all
            // remaining vertical space and push the cards off-screen).
            ui.horizontal(|ui| {
                let (r, _) = ui.allocate_exact_size(egui::vec2(16.0, 16.0), egui::Sense::hover());
                ui.painter_at(r).circle_filled(r.center(), 7.0, theme.cyan);
                ui.add_space(8.0);
                ui.heading(
                    egui::RichText::new("ResoVoid")
                        .size(20.0)
                        .strong()
                        .color(theme.text),
                );
                ui.add_space(12.0);
                // Nudge the small byline down so it optically centers against
                // the 20 px title (nested vertical with top padding).
                ui.vertical(|ui| {
                    ui.add_space(4.0);
                    ui.label(
                        egui::RichText::new("By Wyz3DSP")
                            .size(12.0)
                            .color(theme.text_dim),
                    );
                });
                ui.add_space(8.0);
                ui.label(
                    egui::RichText::new("Dynamic Resonance Suppressor")
                        .size(12.0)
                        .color(theme.text_dim),
                );
            });
            ui.add_space(8.0);

            // Visualizer (draws its own themed card). Height is flexible so
            // a taller host window stretches the plot instead of leaving
            // blank page space below the Parameters card. Reserve ~372px
            // for the params card + gaps; default 672px window keeps 240.
            if let Some(setter) = self.gui_ctx.as_ref().map(|ctx| ctx.param_setter()) {
                let viz_h = (ui.available_height() - 372.0).max(240.0);
                render_visualizer(
                    ui,
                    &self.latest_spectrum,
                    &self.latest_reduction,
                    &self.centers,
                    sample_rate,
                    &mut self.node_positions,
                    &self.params,
                    &setter,
                    &theme,
                    viz_h,
                );
            }

            ui.add_space(12.0);

            if let Some(setter) = self.gui_ctx.as_ref().map(|ctx| ctx.param_setter()) {
                // Parameters card
                let mut card = egui::Frame::default();
                card.fill = theme.card;
                card.stroke = egui::Stroke::new(1.0, theme.border);
                card.corner_radius = CornerRadius::same(12);
                card.inner_margin = egui::Margin::same(14);
                card.show(ui, |ui| {
                    section_header(ui, "Parameters", &theme);
                    ui.add_space(8.0);

                    ui.horizontal(|ui| {
                        param_knob(ui, "Depth", &self.params.depth, &setter, 0.0, 1.0, &theme);
                        param_knob(ui, "Sharpness", &self.params.sharpness, &setter, 0.0, 1.0, &theme);
                        param_knob(ui, "Selectivity", &self.params.selectivity, &setter, 0.0, 1.0, &theme);
                        param_knob(ui, "Attack (ms)", &self.params.attack, &setter, 0.1, 50.0, &theme);
                        param_knob(ui, "Release (ms)", &self.params.release, &setter, 10.0, 500.0, &theme);
                        param_knob(ui, "Mix", &self.params.mix, &setter, 0.0, 1.0, &theme);
                        param_knob(ui, "Output (dB)", &self.params.output_gain, &setter, -12.0, 12.0, &theme);
                    });

                    ui.add_space(10.0);
                    ui.separator();
                    ui.add_space(6.0);

                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("FFT Size").color(theme.text));
                        ui.add(widgets::ParamSlider::for_param(&self.params.fft_size, &setter));
                        let fft_idx = self.params.fft_size.value().clamp(0, 3) as usize;
                        let fft_size = FFT_SIZES[fft_idx];
                        let latency_ms = fft_size as f32 / sample_rate * 1000.0;
                        ui.label(
                            egui::RichText::new(format!("({latency_ms:.1} ms latency)"))
                                .size(11.0)
                                .color(theme.text_dim),
                        );
                    });

                    // Re-scan the presets dir at most every 2 s (fs IO stays
                    // off the per-frame hot path).
                    let now = ui.input(|i| i.time);
                    if now >= self.presets_refresh_at {
                        self.preset_names = crate::presets::list_presets();
                        if self.selected_preset.is_empty() {
                            if let Some(first) = self.preset_names.first().cloned() {
                                self.selected_preset = first;
                            }
                        }
                        self.presets_refresh_at = now + 2.0;
                    }

                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        pill_toggle(ui, "Soft Mode", &self.params.soft_mode, &setter, &theme);
                        ui.add_space(18.0);
                        bool_checkbox(ui, "Delta", &self.params.delta, &setter, &theme);
                        ui.add_space(18.0);
                        pill_toggle(ui, "Oversampling", &self.params.oversampling, &setter, &theme);
                        ui.add_space(18.0);
                        ui.separator();
                        ui.add_space(8.0);
                        egui::ComboBox::from_label("Preset")
                            .selected_text(self.selected_preset.clone())
                            .show_ui(ui, |ui| {
                                for idx in 0..self.preset_names.len() {
                                    let name = self.preset_names[idx].clone();
                                    ui.selectable_value(
                                        &mut self.selected_preset,
                                        name.clone(),
                                        &name,
                                    );
                                }
                            });
                        if ui.button("Load").clicked() && !self.selected_preset.is_empty() {
                            if let Ok(preset) =
                                crate::presets::load_preset(&self.selected_preset)
                            {
                                apply_preset(&self.params, &setter, &preset.values);
                            }
                        }
                        if ui.button("Save").clicked() {
                            self.save_name = self.selected_preset.clone();
                            self.show_save_dialog = true;
                        }
                        if ui.button("Delete").clicked() && !self.selected_preset.is_empty() {
                            let target = self.selected_preset.clone();
                            if crate::presets::delete_preset(&target).is_ok() {
                                self.selected_preset.clear();
                                // Force a rescan so the deleted file disappears.
                                self.presets_refresh_at = f64::NEG_INFINITY;
                            }
                        }
                        // Dark-mode switch pinned to the right end of the same
                        // row (right_to_left only flips direction — it claims
                        // no vertical space, so the cards can't be pushed away).
                        ui.with_layout(
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| {
                                ui.checkbox(&mut self.dark_mode, "Dark mode");
                            },
                        );
                    });

                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        ui.label(egui::RichText::new("Stereo").color(theme.text));
                        let mut link = self.params.stereo_link.value();
                        ui.radio_value(&mut link, StereoLink::Linked, "Stereo Link");
                        ui.radio_value(&mut link, StereoLink::Independent, "Independent");
                        if link != self.params.stereo_link.value() {
                            setter.begin_set_parameter(&self.params.stereo_link);
                            setter.set_parameter(&self.params.stereo_link, link);
                            setter.end_set_parameter(&self.params.stereo_link);
                        }
                    });

                    // TEMPORARY DEBUG UI - remove after concentration diagnosis is done.
                    // CSV write happens here on the GUI thread only (same threading
                    // model as presets.rs); never any file I/O on the audio thread.
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new(format!(
                                "Conc. debug (TEMP): {} / 5000 hops",
                                self.debug_log.len()
                            ))
                            .color(theme.text_dim),
                        );
                        if ui.button("Dump debug CSV").clicked() {
                            let path = debug_csv_path();
                            let _ = write_debug_csv(&path, &self.debug_log);
                        }
                        if ui.button("Clear debug log").clicked() {
                            self.debug_log.clear();
                            self.debug_hop = 0;
                        }
                    });
                });

                if self.show_save_dialog {
                    egui::Window::new("Save Preset").show(ui.ctx(), |ui| {
                        ui.label("Preset name:");
                        ui.text_edit_singleline(&mut self.save_name);
                        ui.horizontal(|ui| {
                            if ui.button("OK").clicked() {
                                let trimmed = self.save_name.trim().to_string();
                                // Duplicate-name detection on the trimmed name,
                                // using the same list the combo box shows.
                                // Empty names fall through to the existing
                                // direct-save path (unchanged behavior).
                                if !trimmed.is_empty()
                                    && self.preset_names.contains(&trimmed)
                                {
                                    // Do NOT write; open the overwrite confirm.
                                    self.show_overwrite_confirm = true;
                                } else {
                                    let values = capture_preset(&self.params);
                                    if crate::presets::save_preset(&self.save_name, values).is_ok()
                                    {
                                        self.selected_preset = self.save_name.clone();
                                        // Force a rescan so the new file appears.
                                        self.presets_refresh_at = f64::NEG_INFINITY;
                                    }
                                    self.show_save_dialog = false;
                                }
                            }
                            if ui.button("Cancel").clicked() {
                                self.show_save_dialog = false;
                            }
                        });
                    });
                }

                if self.show_overwrite_confirm {
                    egui::Window::new("Overwrite existing preset?").show(ui.ctx(), |ui| {
                        ui.label(format!(
                            "Preset \"{}\" already exists. Overwrite it?",
                            self.save_name.trim()
                        ));
                        ui.horizontal(|ui| {
                            if ui.button("Overwrite").clicked() {
                                let values = capture_preset(&self.params);
                                if crate::presets::save_preset(&self.save_name, values).is_ok()
                                {
                                    self.selected_preset = self.save_name.clone();
                                    // Force a rescan so the new file appears.
                                    self.presets_refresh_at = f64::NEG_INFINITY;
                                }
                                self.show_overwrite_confirm = false;
                                self.show_save_dialog = false;
                            }
                            if ui.button("Cancel").clicked() {
                                self.show_overwrite_confirm = false;
                            }
                        });
                    });
                }
            }
        });

    }
}

/// Configure egui's global visuals from the active theme. Re-applied every
/// frame (plain struct assignment) so the dark-mode toggle takes effect
/// instantly, including popups and menus owned by egui itself.
fn apply_theme(ctx: &egui::Context, theme: &Theme, dark: bool) {
    let mut v = if dark {
        egui::Visuals::dark()
    } else {
        egui::Visuals::light()
    };
    v.override_text_color = Some(theme.text);
    v.panel_fill = theme.bg_page;
    v.extreme_bg_color = theme.bg_page;
    v.window_fill = theme.card;
    v.window_corner_radius = CornerRadius::same(12);

    v.widgets.inactive.bg_fill = theme.card;
    v.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, theme.border);
    v.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, theme.text_dim);
    v.widgets.inactive.corner_radius = CornerRadius::same(8);

    v.widgets.hovered.bg_fill = theme.card;
    v.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, theme.cyan);
    v.widgets.hovered.fg_stroke = egui::Stroke::new(1.0, theme.cyan);
    v.widgets.hovered.corner_radius = CornerRadius::same(8);

    v.widgets.active.bg_fill = theme.card;
    v.widgets.active.bg_stroke = egui::Stroke::new(1.5, theme.cyan);
    v.widgets.active.fg_stroke = egui::Stroke::new(1.5, theme.cyan);
    v.widgets.active.corner_radius = CornerRadius::same(8);

    v.widgets.noninteractive.bg_fill = theme.bg_page;
    v.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, theme.text);

    v.selection.bg_fill = Color32::from_rgba_unmultiplied(10, 196, 182, 60);
    v.selection.stroke = egui::Stroke::new(1.0, theme.cyan);

    v.slider_trailing_fill = true;

    ctx.set_visuals(v);
}

/// A small section title with a cyan accent bar.
fn section_header(ui: &mut egui::Ui, title: &str, theme: &Theme) {
    ui.horizontal(|ui| {
        let (r, _) = ui.allocate_exact_size(egui::vec2(4.0, 16.0), egui::Sense::hover());
        ui.painter_at(r).rect_filled(r, 2.0, theme.cyan);
        ui.add_space(6.0);
        ui.label(
            egui::RichText::new(title)
                .size(14.0)
                .strong()
                .color(theme.text),
        );
    });
}

/// Parse a knob type-entry string into a clamped parameter value.
/// Returns `None` for garbage (non-numeric) input so the caller can reject it
/// without touching the parameter. Pure helper — unit-tested below.
fn parse_knob_entry(text: &str, min: f32, max: f32) -> Option<f32> {
    text.trim()
        .parse::<f32>()
        .ok()
        .filter(|v| v.is_finite())
        .map(|v| v.clamp(min, max))
}

/// Draw a native egui rotary knob.
///
/// Interactions (GUI thread only):
/// - drag vertically → continuous change (returned, as before),
/// - right-click → reset to `default` (returned the same way),
/// - double-click → inline text entry (Enter commits a parsed f32 clamped to
///   `[min, max]`, Escape or click-elsewhere cancels).
fn draw_knob(
    ui: &mut egui::Ui,
    label: &str,
    value: f32,
    min: f32,
    max: f32,
    default: f32,
    theme: &Theme,
) -> Option<f32> {
    let size = egui::vec2(70.0, 110.0);
    let (rect, response) =
        ui.allocate_exact_size(size, egui::Sense::click_and_drag());
    let painter = ui.painter_at(rect);
    let center = rect.center();
    let radius = (rect.width().min(rect.height()) / 2.0) - 8.0;

    // Per-knob edit state (7 knobs share this widget): a bool flag plus the
    // text buffer, keyed by label-derived persistent ids in egui temp memory.
    let edit_flag_id = ui.make_persistent_id(format!("{label}::knob_edit"));
    let edit_text_id = ui.make_persistent_id(format!("{label}::knob_text"));

    // Right-click resets to the param default and exits any edit session.
    if response.secondary_clicked() {
        ui.ctx().memory_mut(|m| {
            m.data.insert_temp(edit_flag_id, false);
        });
        return Some(default.clamp(min, max));
    }

    // Double-click opens the inline editor, pre-filled with the current value.
    if response.double_clicked() {
        ui.ctx().memory_mut(|m| {
            m.data.insert_temp(edit_flag_id, true);
            m.data
                .insert_temp(edit_text_id, format!("{value:.2}"));
        });
    }

    let t = ((value - min) / (max - min)).clamp(0.0, 1.0);
    let a_min = std::f32::consts::PI * 0.75;
    let a_max = std::f32::consts::PI * 2.25;
    let angle = a_min + t * (a_max - a_min);

    // Knob body.
    painter.circle_filled(center, radius, theme.card);
    painter.circle_stroke(center, radius, egui::Stroke::new(2.0, theme.border));
    // Background track arc.
    draw_arc(
        &painter,
        center,
        radius,
        a_min,
        a_max,
        egui::Stroke::new(3.0, theme.cyan_track),
    );
    // Active arc.
    draw_arc(
        &painter,
        center,
        radius,
        a_min,
        angle,
        egui::Stroke::new(3.0, theme.cyan),
    );
    let tip = center + radius * egui::Vec2::new(angle.cos(), angle.sin());
    painter.line_segment([center, tip], egui::Stroke::new(2.5, theme.cyan));

    // Inline text entry while this knob is in edit mode. Dragging is ignored
    // for the edited knob so the two interactions can't fight.
    let editing: bool = ui
        .ctx()
        .memory(|m| m.data.get_temp(edit_flag_id).unwrap_or(false));
    if editing {
        let mut text: String = ui.ctx().memory(|m| {
            m.data
                .get_temp(edit_text_id)
                .unwrap_or_else(|| format!("{value:.2}"))
        });
        let edit_rect = egui::Rect::from_center_size(center, egui::vec2(64.0, 20.0));
        let edit_resp = ui.put(
            edit_rect,
            egui::TextEdit::singleline(&mut text).desired_width(60.0),
        );
        if response.double_clicked() {
            edit_resp.request_focus();
        }
        ui.ctx()
            .memory_mut(|m| m.data.insert_temp(edit_text_id, text.clone()));

        let enter = ui.input(|i| i.key_pressed(egui::Key::Enter));
        let esc = ui.input(|i| i.key_pressed(egui::Key::Escape));
        if esc {
            ui.ctx().memory_mut(|m| {
                m.data.insert_temp(edit_flag_id, false);
            });
        } else if enter {
            // Garbage input parses to `None` and is rejected: stay in edit
            // mode, leave the parameter untouched, never panic.
            if let Some(v) = parse_knob_entry(&text, min, max) {
                ui.ctx().memory_mut(|m| {
                    m.data.insert_temp(edit_flag_id, false);
                });
                return Some(v);
            }
        } else if edit_resp.lost_focus() {
            // Clicked elsewhere: cancel without changing the parameter.
            ui.ctx().memory_mut(|m| {
                m.data.insert_temp(edit_flag_id, false);
            });
        }
        // Fall through to paint the label/value; no drag handling this frame.
    } else {
        let mut new_value = value;
        if response.dragged() {
            let delta = (-response.drag_delta().y / 200.0) * (max - min);
            new_value = (value + delta).clamp(min, max);
        }
        if response.dragged() {
            return Some(new_value);
        }
    }

    painter.text(
        center + egui::vec2(0.0, radius + 10.0),
        egui::Align2::CENTER_CENTER,
        label,
        egui::FontId::proportional(11.0),
        theme.text,
    );
    painter.text(
        center + egui::vec2(0.0, radius + 24.0),
        egui::Align2::CENTER_CENTER,
        format!("{value:.2}"),
        egui::FontId::proportional(10.0),
        theme.text_dim,
    );

    None
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

/// Interpolate smooth curve points from sparse log-spaced band centers/values.
/// Uses linear interpolation in log-frequency space to generate `target_points` render points.
fn interpolate_curve(
    centers: &[f32; BANDS],
    values: &[f32; BANDS],
    f_min: f32,
    f_max: f32,
    x_of: &dyn Fn(f32) -> f32,
    y_of: &dyn Fn(f32) -> f32,
    target_points: usize,
) -> Vec<egui::Pos2> {
    let mut result = Vec::with_capacity(target_points);
    let l_min = f_min.log10();
    let l_max = f_max.log10();

    for i in 0..target_points {
        let t = i as f32 / (target_points - 1) as f32;
        let l = l_min + t * (l_max - l_min);
        let freq = 10.0_f32.powf(l);

        // `centers` is strictly ascending, so binary search O(log N) instead
        // of the old linear `.position()` scan O(N) — 512 eval points × 64
        // bands per frame adds up on the UI thread.
        let idx = centers.partition_point(|&c| c <= freq);

        if idx == 0 || idx >= BANDS {
            let x = x_of(freq);
            let y = y_of(values[idx.clamp(0, BANDS - 1)]);
            result.push(egui::pos2(x, y));
            continue;
        }

        let c0 = centers[idx - 1];
        let c1 = centers[idx];
        let v0 = values[idx - 1];
        let v1 = values[idx];

        let l0 = c0.max(f_min).log10();
        let l1 = c1.max(f_min).log10();
        let t_local = if l1 > l0 {
            (l - l0) / (l1 - l0)
        } else {
            0.0
        };

        let v_interp = v0 + t_local * (v1 - v0);
        let x = x_of(freq);
        let y = y_of(v_interp);
        result.push(egui::pos2(x, y));
    }
    result
}

/// Nearest band index to `freq` by log-distance (pure arithmetic, no alloc).
/// Returns 0 when `centers` is unusable (e.g. all zeros before the first
/// analysis frame arrives).
fn nearest_band(centers: &[f32; BANDS], freq: f32) -> usize {
    let target = freq.max(1.0).log2();
    let mut best = 0;
    let mut best_dist = f32::INFINITY;
    for i in 0..BANDS {
        if centers[i] <= 0.0 {
            continue;
        }
        let d = (centers[i].log2() - target).abs();
        if d < best_dist {
            best_dist = d;
            best = i;
        }
    }
    best
}

/// Gain-reduction tint for a node handle: `base` RGB at 0 dB, CYAN at -3 dB,
/// VIOLET at -6 dB or more. Linear RGB lerp, pure arithmetic.
fn gr_tint(base: [u8; 3], gr_db: f32) -> [u8; 3] {
    const CYAN_RGB: [f32; 3] = [10.0, 196.0, 182.0];
    const VIOLET_RGB: [f32; 3] = [139.0, 92.0, 255.0];
    let t = (gr_db / 6.0).clamp(0.0, 1.0);
    let b = [base[0] as f32, base[1] as f32, base[2] as f32];
    let (from, to, k) = if t <= 0.5 {
        (b, CYAN_RGB, t * 2.0)
    } else {
        (CYAN_RGB, VIOLET_RGB, (t - 0.5) * 2.0)
    };
    [
        (from[0] + (to[0] - from[0]) * k) as u8,
        (from[1] + (to[1] - from[1]) * k) as u8,
        (from[2] + (to[2] - from[2]) * k) as u8,
    ]
}

/// Snapshot every preset-relevant parameter into an ID -> value map.
/// Enum/bool/int params are stored as plain f32 (enum variant index, 0.0/1.0,
/// int value); `apply_preset` decodes them back.
fn capture_preset(params: &ResoVoidParams) -> HashMap<String, f32> {
    let mut m = HashMap::new();
    let floats: [(&FloatParam, &str); 23] = [
        (&params.depth, "depth"),
        (&params.sharpness, "sharp"),
        (&params.selectivity, "sel"),
        (&params.attack, "atk"),
        (&params.release, "rel"),
        (&params.mix, "mix"),
        (&params.output_gain, "out"),
        (&params.node_depth_0, "node0"),
        (&params.node_depth_1, "node1"),
        (&params.node_depth_2, "node2"),
        (&params.node_depth_3, "node3"),
        (&params.node_depth_4, "node4"),
        (&params.node_depth_5, "node5"),
        (&params.node_depth_6, "node6"),
        (&params.node_depth_7, "node7"),
        (&params.node_freq_0, "nf0"),
        (&params.node_freq_1, "nf1"),
        (&params.node_freq_2, "nf2"),
        (&params.node_freq_3, "nf3"),
        (&params.node_freq_4, "nf4"),
        (&params.node_freq_5, "nf5"),
        (&params.node_freq_6, "nf6"),
        (&params.node_freq_7, "nf7"),
    ];
    for (p, id) in floats {
        m.insert(id.to_string(), p.value());
    }
    m.insert("fft".to_string(), params.fft_size.value() as f32);
    let bools: [(&BoolParam, &str); 11] = [
        (&params.soft_mode, "mode"),
        (&params.delta, "delt"),
        (&params.oversampling, "os2x"),
        (&params.node_enabled_0, "nen0"),
        (&params.node_enabled_1, "nen1"),
        (&params.node_enabled_2, "nen2"),
        (&params.node_enabled_3, "nen3"),
        (&params.node_enabled_4, "nen4"),
        (&params.node_enabled_5, "nen5"),
        (&params.node_enabled_6, "nen6"),
        (&params.node_enabled_7, "nen7"),
    ];
    for (p, id) in bools {
        m.insert(id.to_string(), if p.value() { 1.0 } else { 0.0 });
    }
    let shapes: [(&EnumParam<NodeShape>, &str); 8] = [
        (&params.node_shape_0, "nsh0"),
        (&params.node_shape_1, "nsh1"),
        (&params.node_shape_2, "nsh2"),
        (&params.node_shape_3, "nsh3"),
        (&params.node_shape_4, "nsh4"),
        (&params.node_shape_5, "nsh5"),
        (&params.node_shape_6, "nsh6"),
        (&params.node_shape_7, "nsh7"),
    ];
    for (p, id) in shapes {
        m.insert(id.to_string(), p.value().to_dsp_index() as f32);
    }
    m.insert(
        "slnk".to_string(),
        match params.stereo_link.value() {
            StereoLink::Linked => 0.0,
            StereoLink::Independent => 1.0,
        },
    );
    m
}

/// Push every entry of a preset map back into the live parameters.
/// Unknown IDs are ignored so old presets stay loadable after new params
/// are added.
fn apply_preset(
    params: &ResoVoidParams,
    setter: &ParamSetter,
    values: &HashMap<String, f32>,
) {
    let floats: [(&FloatParam, &str); 23] = [
        (&params.depth, "depth"),
        (&params.sharpness, "sharp"),
        (&params.selectivity, "sel"),
        (&params.attack, "atk"),
        (&params.release, "rel"),
        (&params.mix, "mix"),
        (&params.output_gain, "out"),
        (&params.node_depth_0, "node0"),
        (&params.node_depth_1, "node1"),
        (&params.node_depth_2, "node2"),
        (&params.node_depth_3, "node3"),
        (&params.node_depth_4, "node4"),
        (&params.node_depth_5, "node5"),
        (&params.node_depth_6, "node6"),
        (&params.node_depth_7, "node7"),
        (&params.node_freq_0, "nf0"),
        (&params.node_freq_1, "nf1"),
        (&params.node_freq_2, "nf2"),
        (&params.node_freq_3, "nf3"),
        (&params.node_freq_4, "nf4"),
        (&params.node_freq_5, "nf5"),
        (&params.node_freq_6, "nf6"),
        (&params.node_freq_7, "nf7"),
    ];
    for (p, id) in floats {
        if let Some(&v) = values.get(id) {
            setter.begin_set_parameter(p);
            setter.set_parameter(p, v);
            setter.end_set_parameter(p);
        }
    }
    if let Some(&v) = values.get("fft") {
        let iv = (v.round() as i32).clamp(0, 3);
        setter.begin_set_parameter(&params.fft_size);
        setter.set_parameter(&params.fft_size, iv);
        setter.end_set_parameter(&params.fft_size);
    }
    let bools: [(&BoolParam, &str); 11] = [
        (&params.soft_mode, "mode"),
        (&params.delta, "delt"),
        (&params.oversampling, "os2x"),
        (&params.node_enabled_0, "nen0"),
        (&params.node_enabled_1, "nen1"),
        (&params.node_enabled_2, "nen2"),
        (&params.node_enabled_3, "nen3"),
        (&params.node_enabled_4, "nen4"),
        (&params.node_enabled_5, "nen5"),
        (&params.node_enabled_6, "nen6"),
        (&params.node_enabled_7, "nen7"),
    ];
    for (p, id) in bools {
        if let Some(&v) = values.get(id) {
            setter.begin_set_parameter(p);
            setter.set_parameter(p, v > 0.5);
            setter.end_set_parameter(p);
        }
    }
    let shapes: [(&EnumParam<NodeShape>, &str); 8] = [
        (&params.node_shape_0, "nsh0"),
        (&params.node_shape_1, "nsh1"),
        (&params.node_shape_2, "nsh2"),
        (&params.node_shape_3, "nsh3"),
        (&params.node_shape_4, "nsh4"),
        (&params.node_shape_5, "nsh5"),
        (&params.node_shape_6, "nsh6"),
        (&params.node_shape_7, "nsh7"),
    ];
    for (p, id) in shapes {
        if let Some(&v) = values.get(id) {
            let s = match v.round() as i32 {
                1 => NodeShape::LowShelf,
                2 => NodeShape::HighShelf,
                _ => NodeShape::Bell,
            };
            setter.begin_set_parameter(p);
            setter.set_parameter(p, s);
            setter.end_set_parameter(p);
        }
    }
    if let Some(&v) = values.get("slnk") {
        let s = if v.round() as i32 == 1 {
            StereoLink::Independent
        } else {
            StereoLink::Linked
        };
        setter.begin_set_parameter(&params.stereo_link);
        setter.set_parameter(&params.stereo_link, s);
        setter.end_set_parameter(&params.stereo_link);
    }
}

// TEMPORARY DEBUG CSV DUMP - remove after concentration diagnosis is done.
// Piggybacks on the existing AnalysisFrame rtrb pipeline: frames are captured
// into `debug_log` on the GUI thread during the normal viz drain, and the CSV
// write below also runs only on the GUI thread (same threading model as
// presets.rs — `std::fs` here, never on the audio thread).
// One row per (hop, band): hop,band,center_hz,level_db,concentration,gain_linear.
type DebugLogEntry = (u32, [f32; BANDS], [f32; BANDS], [f32; BANDS], [f32; BANDS]);

/// Destination for the temporary concentration CSV (GUI thread only).
/// Lives in ResoVoid's own presets folder, per spec.
fn debug_csv_path() -> std::path::PathBuf {
    crate::presets::presets_dir().join("resovoid_concentration.csv")
}

/// Render the debug log as CSV text. Pure helper (no I/O) — unit-tested below.
fn format_debug_csv(log: &[DebugLogEntry]) -> String {
    let mut out = String::from("hop,band,center_hz,level_db,concentration,gain_linear\n");
    for (hop, centers, levels, concentration, gains) in log {
        for b in 0..BANDS {
            out.push_str(&format!(
                "{hop},{b},{:.3},{:.3},{:.6},{:.6}\n",
                centers[b], levels[b], concentration[b], gains[b]
            ));
        }
    }
    out
}

/// Write the debug log to `path`. GUI thread only — mirrors how presets.rs
/// does directory creation + `std::fs` write on the GUI thread.
fn write_debug_csv(
    path: &std::path::Path,
    log: &[DebugLogEntry],
) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, format_debug_csv(log))
}

/// Map a parameter to a knob and apply changes through the `ParamSetter`.
/// The reset default is pulled from the `FloatParam` itself via
/// `Param::default_plain_value()` (nice-plug API), never hardcoded, so all
/// seven Parameters-card knobs reset to their documented lib.rs defaults.
fn param_knob(
    ui: &mut egui::Ui,
    label: &str,
    param: &FloatParam,
    setter: &ParamSetter,
    min: f32,
    max: f32,
    theme: &Theme,
) {
    let value = param.value();
    let default = param.default_plain_value();
    if let Some(new_value) = draw_knob(ui, label, value, min, max, default, theme) {
        setter.begin_set_parameter(param);
        setter.set_parameter(param, new_value);
        setter.end_set_parameter(param);
    }
}

/// Checkbox wired to a boolean parameter via the `ParamSetter`.
fn bool_checkbox(
    ui: &mut egui::Ui,
    label: &str,
    param: &BoolParam,
    setter: &ParamSetter,
    theme: &Theme,
) {
    let mut v = param.value();
    if ui
        .checkbox(&mut v, egui::RichText::new(label).color(theme.text))
        .clicked()
    {
        setter.begin_set_parameter(param);
        setter.set_parameter(param, v);
        setter.end_set_parameter(param);
    }
}

/// Pill-shaped toggle for a BoolParam. Draws "SOFT" | "HARD" with the active
/// segment highlighted in CYAN. Fixed size 120×28.
fn pill_toggle(
    ui: &mut egui::Ui,
    _label: &str,
    param: &BoolParam,
    setter: &ParamSetter,
    theme: &Theme,
) {
    let size = egui::vec2(120.0, 28.0);
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click());
    let painter = ui.painter_at(rect);

    // Draw pill background.
    painter.add(RectShape::new(
        rect,
        CornerRadius::same(14),
        theme.pill_bg,
        egui::Stroke::new(1.0, theme.border),
        egui::StrokeKind::Outside,
    ));

    let is_soft = !param.value(); // false = SOFT mode (default), true = HARD mode
    let half_w = rect.width() / 2.0;
    let left_rect = egui::Rect::from_min_size(rect.min, egui::vec2(half_w, rect.height()));
    let right_rect = egui::Rect::from_min_size(
        egui::pos2(rect.left() + half_w, rect.top()),
        egui::vec2(half_w, rect.height()),
    );

    // Highlight active side.
    if is_soft {
        painter.add(RectShape::new(
            left_rect,
            CornerRadius {
                nw: 14,
                ne: 0,
                sw: 14,
                se: 0,
            },
            theme.cyan,
            egui::Stroke::NONE,
            egui::StrokeKind::Outside,
        ));
    } else {
        painter.add(RectShape::new(
            right_rect,
            CornerRadius {
                nw: 0,
                ne: 14,
                sw: 0,
                se: 14,
            },
            theme.cyan,
            egui::Stroke::NONE,
            egui::StrokeKind::Outside,
        ));
    }

    // Active-side label is white-on-cyan in both themes; the idle side
    // follows the theme's muted text.
    let label_color_soft = if is_soft {
        Color32::WHITE
    } else {
        theme.text_dim
    };
    let label_color_hard = if is_soft {
        theme.text_dim
    } else {
        Color32::WHITE
    };
    painter.text(
        left_rect.center(),
        egui::Align2::CENTER_CENTER,
        "SOFT",
        egui::FontId::proportional(11.0),
        label_color_soft,
    );
    painter.text(
        right_rect.center(),
        egui::Align2::CENTER_CENTER,
        "HARD",
        egui::FontId::proportional(11.0),
        label_color_hard,
    );

    // Handle click.
    if response.clicked() {
        let new_val = !param.value();
        setter.begin_set_parameter(param);
        setter.set_parameter(param, new_val);
        setter.end_set_parameter(param);
    }
}

/// Render the input spectrum (cyan) and reduction curve (violet) on a logarithmic
/// frequency axis. Called from the GUI thread only. The top of the axis is
/// Nyquist-aware so it never plots bands above the actual sample rate's limit.
///
/// Includes up to MAX_NODES draggable node markers that act as per-region
/// depth multipliers with user-repositionable center frequencies. Nodes are
/// hard-clamped against their frequency-sorted neighbors while dragging so
/// enabled nodes never cross. Double-clicking empty plot space creates a
/// node in the first disabled slot; left-clicking a node's shape badge opens
/// a shape popup, while the right-click menu only deletes the node (clears
/// its enabled flag).
fn render_visualizer(
    ui: &mut egui::Ui,
    spectrum: &[f32; BANDS],
    reduction: &[f32; BANDS],
    centers: &[f32; BANDS],
    sample_rate: f32,
    node_positions: &mut [(f32, f32); MAX_NODES],
    params: &ResoVoidParams,
    setter: &ParamSetter,
    theme: &Theme,
    height: f32,
) {
    // Sense::click so double-clicks on empty plot area can create nodes.
    // (Single clicks do nothing; node dots have their own interact rects.)
    let (rect, plot_response) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), height),
        egui::Sense::click(),
    );
    let painter = ui.painter_at(rect);

    // Card background + border with embedded panel look (corner_radius 16, bottom shadow).
    painter.add(RectShape::new(
        rect,
        CornerRadius::same(16),
        theme.card,
        egui::Stroke::new(1.0, theme.border),
        egui::StrokeKind::Outside,
    ));
    // Subtle bottom shadow / darker border for embedded panel look.
    let shadow_rect = egui::Rect::from_min_size(
        egui::pos2(rect.left() + 1.0, rect.bottom() - 2.0),
        egui::vec2(rect.width() - 2.0, 2.0),
    );
    painter.add(RectShape::new(
        shadow_rect,
        CornerRadius::same(0),
        Color32::from_rgba_unmultiplied(0, 0, 0, 20),
        egui::Stroke::NONE,
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
    // Inverse log mapping: pixel x -> frequency Hz.
    let freq_of_x = |x: f32| -> f32 {
        let t = ((x - plot.left()) / plot.width()).clamp(0.0, 1.0);
        10.0_f32.powf(l_min + t * (l_max - l_min))
    };
    let y_of = |db: f32| -> f32 {
        plot.bottom() - ((db - db_min) / (db_max - db_min)) * plot.height()
    };

    // Frequency grid lines and labels: 100, 250, 500, 1k, 2k, 4k, 8k, 16k.
    let freq_ticks: &[(f32, &str)] = &[
        (100.0, "100"),
        (250.0, "250"),
        (500.0, "500"),
        (1000.0, "1k"),
        (2000.0, "2k"),
        (4000.0, "4k"),
        (8000.0, "8k"),
        (16000.0, "16k"),
    ];
    for &(freq, label) in freq_ticks {
        if freq >= f_min && freq <= f_max {
            let x = x_of(freq);
            // Thin vertical gridline.
            painter.line_segment(
                [egui::pos2(x, plot.top()), egui::pos2(x, plot.bottom())],
                egui::Stroke::new(1.0, theme.grid),
            );
            // Frequency label centered under tick mark.
            painter.text(
                egui::pos2(x, plot.bottom() + 10.0),
                egui::Align2::CENTER_TOP,
                label,
                egui::FontId::proportional(9.0),
                theme.text_dim,
            );
        }
    }
    // dB grid lines.
    for &db in &[-60.0_f32, -40.0, -20.0, 0.0] {
        let y = y_of(db);
        painter.line_segment(
            [egui::pos2(plot.left(), y), egui::pos2(plot.right(), y)],
            egui::Stroke::new(1.0, theme.grid),
        );
    }

    const INTERP_POINTS: usize = 512;

    // Input spectrum: interpolate smooth curve from 64 band centers.
    let spec_pts = interpolate_curve(
        centers,
        spectrum,
        f_min,
        f_max,
        &x_of,
        &|db| y_of(db.clamp(db_min, db_max)),
        INTERP_POINTS,
    );

    // --- Gradient fill under spectrum (per-x-column vertical strips) ---
    // Each sampled curve point owns the vertical column from its y down to
    // the plot baseline. Columns can never cross-connect, so no diagonal
    // streaks are possible even when the live curve is jagged/noisy.
    if spec_pts.len() >= 2 {
        const MAX_ALPHA: f32 = 38.0;
        const SEGMENTS: usize = 7;
        let step = if spec_pts.len() > 300 { 2 } else { 1 };
        let sampled: Vec<usize> = (0..spec_pts.len()).step_by(step).collect();
        for (k, &idx) in sampled.iter().enumerate() {
            let pt = spec_pts[idx];
            let width = if k + 1 < sampled.len() {
                (spec_pts[sampled[k + 1]].x - pt.x).abs().max(1.0)
            } else if k > 0 {
                (pt.x - spec_pts[sampled[k - 1]].x).abs().max(1.0)
            } else {
                1.0
            };
            let height = plot.bottom() - pt.y;
            if height <= 0.0 {
                continue;
            }
            for seg in 0..SEGMENTS {
                let d0 = seg as f32 / SEGMENTS as f32;
                let d1 = (seg + 1) as f32 / SEGMENTS as f32;
                let d_mid = (d0 + d1) * 0.5;
                let alpha = (MAX_ALPHA * (1.0 - d_mid).powf(1.5)) as u8;
                let y0 = pt.y + d0 * height;
                let y1 = pt.y + d1 * height;
                let seg_rect = egui::Rect::from_min_max(
                    egui::pos2(pt.x, y0),
                    egui::pos2(pt.x + width, y1 + 0.5),
                );
                painter.rect_filled(
                    seg_rect,
                    0.0,
                    Color32::from_rgba_unmultiplied(10, 196, 182, alpha),
                );
            }
        }

        // --- Filled envelope band (spectrum minus ~6dB, semi-transparent cyan) ---
        let envelope_db_offset = 6.0; // dB below spectrum
        let envelope_pts = interpolate_curve(
            centers,
            spectrum,
            f_min,
            f_max,
            &x_of,
            &|db| y_of((db - envelope_db_offset).clamp(db_min, db_max)),
            INTERP_POINTS,
        );
        if envelope_pts.len() >= 2 {
            // Build area between spectrum curve (top) and envelope curve (bottom).
            let mut area: Vec<egui::Pos2> = Vec::with_capacity(spec_pts.len() + envelope_pts.len());
            area.extend_from_slice(&spec_pts);
            // Reverse the envelope to close the shape.
            for pt in envelope_pts.iter().rev() {
                area.push(*pt);
            }
            area.push(*spec_pts.first().unwrap()); // close back to start

            painter.add(egui::Shape::Path(PathShape {
                points: area,
                closed: true,
                fill: Color32::from_rgba_unmultiplied(10, 196, 182, 24),
                stroke: egui::Stroke::NONE.into(),
            }));
        }

        // Draw the main spectrum line on top.
        painter.add(egui::Shape::line(
            spec_pts.clone(),
            egui::Stroke::new(1.8, theme.cyan),
        ));
    }

    // Reduction curve: linear gain -> dB (<= 0). Interpolated smooth violet line.
    let reduction_db: [f32; BANDS] = {
        let mut arr = [0.0_f32; BANDS];
        for i in 0..BANDS {
            arr[i] = 20.0 * (reduction[i].max(1e-4)).log10();
        }
        arr
    };
    let red_pts = interpolate_curve(
        centers,
        &reduction_db,
        f_min,
        f_max,
        &x_of,
        &|db| y_of(db.clamp(-40.0, 0.0)),
        INTERP_POINTS,
    );
    if red_pts.len() >= 2 {
        painter.add(egui::Shape::line(
            red_pts,
            egui::Stroke::new(2.0, theme.violet),
        ));
    }

    // --- Node shape curve (Soothe3-style white tilt-curve equivalent) ---
    // Visual mapping: region_depth is a 0..1 multiplier on the global depth.
    // We map it to plot-y so that 1.0 (full depth, no attenuation) sits at
    // `shape_baseline_y` (near the top, aligned with depth=1.0 node dots),
    // and 0.0 (zero depth) drops to `shape_baseline_y + shape_range`.
    // Departures from 1.0 bow downward, connecting the 3 node dots into one
    // continuous shape — the Soothe3-style tilt curve.
    //
    // Design choice: the curve uses the exact same `node_shape()` function
    // that the DSP detector calls — single source of truth, no visual drift.
    const SHAPE_SAMPLES: usize = 1000;
    const SHAPE_BASELINE_FRAC: f32 = 0.05; // 5% from top = neutral line (depth=1.0)
    const SHAPE_RANGE_FRAC: f32 = 0.90;    // full drop covers 90% of plot height

    let shape_baseline_y = plot.top() + plot.height() * SHAPE_BASELINE_FRAC;
    let shape_range = plot.height() * SHAPE_RANGE_FRAC;

    // Collect the node params once for sampling.
    // Depths/freqs read from node_positions (not FloatParams) so the shape
    // curve tracks the live GUI state including hard-clamped frequencies.
    // Shapes/enabled read from params (no per-sample smoothing on those).
    let nd = [
        node_positions[0].0,
        node_positions[1].0,
        node_positions[2].0,
        node_positions[3].0,
        node_positions[4].0,
        node_positions[5].0,
        node_positions[6].0,
        node_positions[7].0,
    ];
    let nf = [
        node_positions[0].1,
        node_positions[1].1,
        node_positions[2].1,
        node_positions[3].1,
        node_positions[4].1,
        node_positions[5].1,
        node_positions[6].1,
        node_positions[7].1,
    ];
    let nsh = [
        params.node_shape_0.value().to_dsp_index(),
        params.node_shape_1.value().to_dsp_index(),
        params.node_shape_2.value().to_dsp_index(),
        params.node_shape_3.value().to_dsp_index(),
        params.node_shape_4.value().to_dsp_index(),
        params.node_shape_5.value().to_dsp_index(),
        params.node_shape_6.value().to_dsp_index(),
        params.node_shape_7.value().to_dsp_index(),
    ];
    let nen = [
        params.node_enabled_0.value(),
        params.node_enabled_1.value(),
        params.node_enabled_2.value(),
        params.node_enabled_3.value(),
        params.node_enabled_4.value(),
        params.node_enabled_5.value(),
        params.node_enabled_6.value(),
        params.node_enabled_7.value(),
    ];

    let mut shape_pts: Vec<egui::Pos2> = Vec::with_capacity(SHAPE_SAMPLES);
    let l_min_s = f_min.log10();
    let l_max_s = f_max.log10();
    for s in 0..SHAPE_SAMPLES {
        let t = s as f32 / (SHAPE_SAMPLES - 1) as f32;
        let freq = 10.0_f32.powf(l_min_s + t * (l_max_s - l_min_s));
        let depth = node_shape(freq, &nf, &nd, &nsh, &nen);
        let y = shape_baseline_y + (1.0 - depth) * shape_range;
        shape_pts.push(egui::pos2(x_of(freq), y));
    }

    // Draw a subtle fill from the neutral baseline down to the shape curve.
    if shape_pts.len() >= 2 {
        let mut fill_area: Vec<egui::Pos2> = Vec::with_capacity(shape_pts.len() + 2);
        fill_area.push(egui::pos2(shape_pts.first().unwrap().x, shape_baseline_y));
        fill_area.extend_from_slice(&shape_pts);
        fill_area.push(egui::pos2(shape_pts.last().unwrap().x, shape_baseline_y));
        painter.add(egui::Shape::Path(PathShape {
            points: fill_area,
            closed: true,
                fill: Color32::from_rgba_unmultiplied(
                    theme.shape.r(),
                    theme.shape.g(),
                    theme.shape.b(),
                    18,
                ),
            stroke: egui::Stroke::NONE.into(),
        }));
    }
    // Draw the shape line itself.
    if shape_pts.len() >= 2 {
        painter.add(egui::Shape::line(
            shape_pts,
            egui::Stroke::new(1.5, theme.shape),
        ));
    }

    // --- Interactive node markers (up to MAX_NODES draggable handles) ---
    // Slots 0–2 keep the original colors; the rest rotate through distinct hues.
    const NODE_COLORS: [Color32; MAX_NODES] = [
        Color32::from_rgb(10, 196, 182),
        Color32::from_rgb(236, 72, 153),
        Color32::from_rgb(250, 204, 21),
        Color32::from_rgb(34, 197, 94),
        Color32::from_rgb(59, 130, 246),
        Color32::from_rgb(139, 92, 255),
        Color32::from_rgb(249, 115, 22),
        Color32::from_rgb(239, 68, 68),
    ];
    const NODE_LABELS: [&str; MAX_NODES] = ["1", "2", "3", "4", "5", "6", "7", "8"];
    let node_radius = 8.0;

    // Param handles per slot so the loop below needs no per-index matches.
    let depth_params: [&FloatParam; MAX_NODES] = [
        &params.node_depth_0,
        &params.node_depth_1,
        &params.node_depth_2,
        &params.node_depth_3,
        &params.node_depth_4,
        &params.node_depth_5,
        &params.node_depth_6,
        &params.node_depth_7,
    ];
    let freq_params: [&FloatParam; MAX_NODES] = [
        &params.node_freq_0,
        &params.node_freq_1,
        &params.node_freq_2,
        &params.node_freq_3,
        &params.node_freq_4,
        &params.node_freq_5,
        &params.node_freq_6,
        &params.node_freq_7,
    ];
    let enabled_params: [&BoolParam; MAX_NODES] = [
        &params.node_enabled_0,
        &params.node_enabled_1,
        &params.node_enabled_2,
        &params.node_enabled_3,
        &params.node_enabled_4,
        &params.node_enabled_5,
        &params.node_enabled_6,
        &params.node_enabled_7,
    ];
    let shape_params: [&EnumParam<NodeShape>; MAX_NODES] = [
        &params.node_shape_0,
        &params.node_shape_1,
        &params.node_shape_2,
        &params.node_shape_3,
        &params.node_shape_4,
        &params.node_shape_5,
        &params.node_shape_6,
        &params.node_shape_7,
    ];

    // Enabled slots sorted by frequency — the ordering reference for the
    // drag hard-clamp. Neighbors are whoever is adjacent in this sorted
    // subset, so ordering holds no matter which slots are active.
    let mut freq_order: Vec<usize> = (0..MAX_NODES).filter(|&j| nen[j]).collect();
    freq_order.sort_by(|&a, &b| {
        node_positions[a]
            .1
            .partial_cmp(&node_positions[b].1)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for i in 0..MAX_NODES {
        if !nen[i] {
            continue;
        }
        let (depth, freq) = node_positions[i];
        if freq < f_min || freq > f_max {
            continue;
        }

        let node_y = plot.bottom() - depth * plot.height();
        let node_center = egui::pos2(x_of(freq), node_y);

        // Gain-reduction meter: tint + swell by the reduction at the nearest
        // band (neutral at 0 dB, cyan at -3 dB, violet at -6 dB or more).
        let band = nearest_band(centers, freq);
        let gr_db = -20.0 * reduction[band].max(1e-4).log10();
        let gr_t = (gr_db / 6.0).clamp(0.0, 1.0);
        let base = NODE_COLORS[i];
        let tint = gr_tint([base.r(), base.g(), base.b()], gr_db);
        let handle_color = Color32::from_rgb(tint[0], tint[1], tint[2]);
        let handle_radius = node_radius + 2.0 * gr_t;

        painter.circle_filled(node_center, handle_radius, handle_color);
        painter.circle_stroke(
            node_center,
            handle_radius,
            egui::Stroke::new(2.0, theme.card),
        );

        painter.text(
            node_center,
            egui::Align2::CENTER_CENTER,
            NODE_LABELS[i],
            egui::FontId::proportional(10.0),
            Color32::WHITE,
        );

        // Persistent shape badge: 16×12 visible rect, 24×20 oversized hit
        // target centered on the same badge center (easier to hit, visuals
        // unchanged). Gesture split: shape = common/left-click on the badge
        // popup below; delete = destructive/right-click on the node dot
        // (context menu further below). Glyph keeps reflecting `nsh[i]` live.
        let badge_center = node_center + egui::vec2(14.0, -14.0);
        let badge_rect =
            egui::Rect::from_center_size(badge_center, egui::vec2(16.0, 12.0));
        let badge_hit =
            egui::Rect::from_center_size(badge_center, egui::vec2(24.0, 20.0));
        {
            painter.add(RectShape::new(
                badge_rect,
                CornerRadius::same(3),
                NODE_COLORS[i],
                egui::Stroke::new(1.0, theme.card),
                egui::StrokeKind::Outside,
            ));
            let l = badge_rect.left() + 3.0;
            let r = badge_rect.right() - 3.0;
            let t = badge_rect.top() + 3.0;
            let b = badge_rect.bottom() - 3.0;
            let mid_x = badge_rect.center().x;
            let glyph = egui::Stroke::new(1.2, Color32::WHITE);
            match nsh[i] {
                // Bell: small bump ∩ as a 5-point polyline.
                0 => {
                    let w = r - l;
                    let h = b - t;
                    let pts = [
                        egui::pos2(l, b),
                        egui::pos2(l + w * 0.25, t + h * 0.25),
                        egui::pos2(mid_x, t),
                        egui::pos2(l + w * 0.75, t + h * 0.25),
                        egui::pos2(r, b),
                    ];
                    for w2 in pts.windows(2) {
                        painter.line_segment([w2[0], w2[1]], glyph);
                    }
                }
                // Low Shelf: flat-left then slopes down right.
                1 => {
                    painter.line_segment([egui::pos2(l, t), egui::pos2(mid_x, t)], glyph);
                    painter.line_segment([egui::pos2(mid_x, t), egui::pos2(r, b)], glyph);
                }
                // High Shelf: slopes up from bottom-left to flat-right.
                _ => {
                    painter.line_segment([egui::pos2(l, b), egui::pos2(mid_x, t)], glyph);
                    painter.line_segment([egui::pos2(mid_x, t), egui::pos2(r, t)], glyph);
                }
            }
        }

        let node_response = ui.interact(
            egui::Rect::from_center_size(node_center, egui::vec2(24.0, 24.0)),
            egui::Id::new(format!("node_{i}")),
            egui::Sense::click_and_drag(),
        );

        // Shape badge hit target (created after the node dot so it wins the
        // overlap region). Left-click toggles the shape popup below.
        let badge_response = ui.interact(
            badge_hit,
            egui::Id::new(format!("shape_badge_{i}")),
            egui::Sense::click(),
        );
        if badge_response.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }
        // Left-click shape popup. Anchored below the badge; the explicit
        // alternative aligns flip it above/beside the badge when the node
        // sits near the left/right/top/bottom plot edges so the menu is
        // never clipped outside the visible window.
        egui::Popup::menu(&badge_response)
            .gap(4.0)
            .align(egui::RectAlign::BOTTOM_START)
            .align_alternatives(&[
                egui::RectAlign::BOTTOM_START,
                egui::RectAlign::BOTTOM_END,
                egui::RectAlign::TOP_START,
                egui::RectAlign::TOP_END,
                egui::RectAlign::RIGHT_START,
                egui::RectAlign::LEFT_START,
            ])
            .show(|ui| {
                let shape_param = shape_params[i];
                let mut current_shape = shape_param.value();
                let mut changed = false;
                if ui.radio_value(&mut current_shape, NodeShape::Bell, "Bell").clicked() {
                    changed = true;
                }
                if ui.radio_value(&mut current_shape, NodeShape::LowShelf, "Low Shelf").clicked() {
                    changed = true;
                }
                if ui.radio_value(&mut current_shape, NodeShape::HighShelf, "High Shelf").clicked() {
                    changed = true;
                }
                if changed {
                    setter.begin_set_parameter(shape_param);
                    setter.set_parameter(shape_param, current_shape);
                    setter.end_set_parameter(shape_param);
                    ui.close();
                }
            });

        // Right-click context menu: delete only. Shape lives on the badge
        // popup (left-click) since it is the common, non-destructive action;
        // the dot's right-click menu keeps just the destructive action.
        node_response.context_menu(|ui| {
            // "Deleting" a node clears its enabled flag; param identity
            // (and the host's automation/recall of it) is preserved.
            if ui.button("Delete node").clicked() {
                let en_param = enabled_params[i];
                setter.begin_set_parameter(en_param);
                setter.set_parameter(en_param, false);
                setter.end_set_parameter(en_param);
                ui.close();
            }
        });

        // A badge click must never move the node: the 24×20 badge hit rect
        // overlaps the 24×24 node-dot rect, so skip the drag update when the
        // pointer press originated inside the badge hit rect.
        let press_in_badge = ui
            .input(|i| i.pointer.press_origin())
            .is_some_and(|p| badge_hit.contains(p));
        if node_response.dragged() && !press_in_badge {
            let delta = node_response.drag_delta();

            // Vertical drag → depth (0.0..1.0).
            let dy_normalized = -delta.y / plot.height();
            let new_depth = (depth + dy_normalized).clamp(0.0, 1.0);

            // Horizontal drag → frequency via inverse log mapping, hard-clamped
            // against the adjacent nodes in frequency-sorted order (standard
            // EQ paradigm: enabled nodes never cross, whichever slots they are).
            let rank = freq_order.iter().position(|&j| j == i).unwrap_or(0);
            let lo = if rank > 0 {
                node_positions[freq_order[rank - 1]].1
            } else {
                f_min
            };
            let hi = if rank + 1 < freq_order.len() {
                node_positions[freq_order[rank + 1]].1
            } else {
                f_max
            };
            let new_freq = freq_of_x(x_of(freq) + delta.x).clamp(lo, hi).clamp(f_min, f_max);

            node_positions[i] = (new_depth, new_freq);

            // Write depth + frequency to this slot's FloatParams.
            let depth_param = depth_params[i];
            setter.begin_set_parameter(depth_param);
            setter.set_parameter(depth_param, new_depth);
            setter.end_set_parameter(depth_param);

            let freq_param = freq_params[i];
            setter.begin_set_parameter(freq_param);
            setter.set_parameter(freq_param, new_freq);
            setter.end_set_parameter(freq_param);
        }
    }

    // Double-click empty plot area → create a node in the first disabled slot
    // (Bell shape, mid depth so the effect is visible). Clicks landing on an
    // existing dot are ignored — those belong to the node's own interact rect.
    if plot_response.double_clicked() {
        if let Some(click_pos) = plot_response.interact_pointer_pos() {
            if plot.contains(click_pos) {
                let near_node = (0..MAX_NODES).filter(|&j| nen[j]).any(|j| {
                    let (d, f) = node_positions[j];
                    let c = egui::pos2(x_of(f), plot.bottom() - d * plot.height());
                    c.distance(click_pos) < 24.0
                });
                if !near_node {
                    if let Some(slot) = (0..MAX_NODES).find(|&j| !nen[j]) {
                        const NEW_DEPTH: f32 = 0.5;
                        let freq = freq_of_x(click_pos.x).clamp(f_min, f_max);
                        node_positions[slot] = (NEW_DEPTH, freq);
                        let depth_param = depth_params[slot];
                        setter.begin_set_parameter(depth_param);
                        setter.set_parameter(depth_param, NEW_DEPTH);
                        setter.end_set_parameter(depth_param);
                        let freq_param = freq_params[slot];
                        setter.begin_set_parameter(freq_param);
                        setter.set_parameter(freq_param, freq);
                        setter.end_set_parameter(freq_param);
                        let en_param = enabled_params[slot];
                        setter.begin_set_parameter(en_param);
                        setter.set_parameter(en_param, true);
                        setter.end_set_parameter(en_param);
                    }
                }
            }
        }
    }

    // No post-drag fixup needed: the hard-clamp above preserves sorted order
    // at drag time, and params are written only on user gestures
    // (drag / double-click / menu), never re-spammed every frame.

    // Axis tick labels for dB.
    for &db in &[-60.0_f32, -40.0, -20.0, 0.0] {
        painter.text(
            egui::pos2(plot.left() + 4.0, y_of(db)),
            egui::Align2::LEFT_CENTER,
            format!("{db}"),
            egui::FontId::proportional(9.0),
            theme.text_dim,
        );
    }

    painter.text(
        egui::pos2(plot.left() + 6.0, plot.top() + 4.0),
        egui::Align2::LEFT_TOP,
        format!(
            "Spectrum (cyan) / Reduction (violet), 20 Hz - {} kHz, log — drag nodes to move, double-click empty space to add, click badge for shape, right-click for delete",
            (f_max / 1000.0).round() as u32
        ),
        egui::FontId::proportional(11.0),
        theme.text_dim,
    );
}

#[cfg(test)]
mod tests {
    use super::{format_debug_csv, gr_tint, nearest_band, parse_knob_entry, Theme};
    use crate::dsp::BANDS;
    use crate::ResoVoidParams;
    use nice_plug::prelude::Param;

    fn log_centers() -> [f32; BANDS] {
        let mut c = [0.0f32; BANDS];
        for i in 0..BANDS {
            let t = i as f32 / (BANDS - 1) as f32;
            c[i] = 10.0_f32.powf(1.30103 + t * (4.30103 - 1.30103));
        }
        c
    }

    #[test]
    fn gr_tint_zero_db_is_base() {
        let base = [200u8, 150, 100];
        assert_eq!(gr_tint(base, 0.0), base);
        assert_eq!(gr_tint(base, -1.0), base, "negative GR clamps to base");
    }

    #[test]
    fn gr_tint_three_db_is_cyan() {
        let tint = gr_tint([255, 255, 255], 3.0);
        assert_eq!(tint, [10, 196, 182]);
    }

    #[test]
    fn gr_tint_six_db_is_violet() {
        let tint = gr_tint([255, 255, 255], 6.0);
        assert_eq!(tint, [139, 92, 255]);
    }

    #[test]
    fn gr_tint_clamps_beyond_six_db() {
        let tint = gr_tint([255, 255, 255], 20.0);
        assert_eq!(tint, [139, 92, 255]);
    }

    #[test]
    fn nearest_band_exact_and_edges() {
        let c = log_centers();
        assert_eq!(nearest_band(&c, c[10]), 10);
        assert_eq!(nearest_band(&c, 1.0), 0, "below range clamps low");
        assert_eq!(nearest_band(&c, 1e9), BANDS - 1, "above range clamps high");
    }

    #[test]
    fn nearest_band_degenerate_centers() {
        assert_eq!(nearest_band(&[0.0; BANDS], 1000.0), 0);
    }

    #[test]
    fn themes_have_readable_contrast() {
        // Light: dark text on a bright page. Dark: bright text on a deep page.
        // Accent hues are shared so the spectrum/GR language never changes.
        let light = Theme::light();
        let dark = Theme::dark();
        assert!(Theme::luminance(light.text) < Theme::luminance(light.bg_page));
        assert!(Theme::luminance(dark.text) > Theme::luminance(dark.bg_page));
        assert!(Theme::luminance(dark.text_dim) > Theme::luminance(dark.bg_page));
        assert_eq!(light.cyan, dark.cyan);
        assert_eq!(light.violet, dark.violet);
        assert_ne!(light.card, dark.card);
        assert_ne!(light.bg_page, dark.bg_page);
    }

    #[test]
    fn knob_reset_defaults_match_documented_values() {
        // Right-click reset pulls these via `Param::default_plain_value()`;
        // guard against copy-pasted wrong constants per knob.
        let p = ResoVoidParams::default();
        assert_eq!(p.depth.default_plain_value(), 0.5);
        assert_eq!(p.sharpness.default_plain_value(), 0.5);
        assert_eq!(p.selectivity.default_plain_value(), 0.5);
        assert_eq!(p.attack.default_plain_value(), 10.0);
        assert_eq!(p.release.default_plain_value(), 100.0);
        assert_eq!(p.mix.default_plain_value(), 1.0);
        assert_eq!(p.output_gain.default_plain_value(), 0.0);
    }

    #[test]
    fn knob_text_entry_parses_and_clamps() {
        assert_eq!(parse_knob_entry("0.75", 0.0, 1.0), Some(0.75));
        assert_eq!(parse_knob_entry(" 10.0 ", 0.1, 50.0), Some(10.0));
        // Out-of-range input clamps instead of escaping [min, max].
        assert_eq!(parse_knob_entry("99.0", 0.0, 1.0), Some(1.0));
        assert_eq!(parse_knob_entry("-5.0", 0.0, 1.0), Some(0.0));
    }

    #[test]
    fn knob_text_entry_rejects_garbage_without_panic() {
        assert_eq!(parse_knob_entry("", 0.0, 1.0), None);
        assert_eq!(parse_knob_entry("abc", 0.0, 1.0), None);
        assert_eq!(parse_knob_entry("1.0.0", 0.0, 1.0), None);
        assert_eq!(parse_knob_entry("NaN-ish!", -12.0, 12.0), None);
        // Infinities / NaN parse as f32 but must not corrupt the param:
        // they never compare usefully, so treat them as rejected.
        assert_eq!(parse_knob_entry("inf", 0.0, 1.0), None);
        assert_eq!(parse_knob_entry("NaN", 0.0, 1.0), None);
    }

    #[test]
    fn debug_csv_formats_one_row_per_hop_band() {
        // TEMPORARY DEBUG TEST - remove with the concentration debug log.
        let log = vec![(7u32, [100.0; BANDS], [-20.0; BANDS], [0.5; BANDS], [0.8; BANDS])];
        let csv = format_debug_csv(&log);
        let mut lines = csv.lines();
        assert_eq!(lines.next().unwrap(), "hop,band,center_hz,level_db,concentration,gain_linear");
        let first = lines.next().unwrap();
        assert!(first.starts_with("7,0,100.000,-20.000,0.500000,0.800000"), "got {first}");
        assert_eq!(csv.lines().count(), 1 + BANDS);
        assert_eq!(format_debug_csv(&[]).lines().count(), 1);
    }
}
