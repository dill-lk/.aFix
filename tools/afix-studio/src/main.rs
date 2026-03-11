//! # aFix Studio
//!
//! A native Windows desktop application for working with `.aFix` image files.
//!
//! ## Features
//!
//! - **Convert** — Convert JPEG / PNG images to `.aFix` via a drag-and-drop interface.
//! - **View**    — Inspect `.aFix` files: header info, chunk table, and embedded preview.

// On Windows, suppress the console window so the app starts as a proper GUI app.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::PathBuf;

use eframe::egui;
use eframe::egui::{ColorImage, TextureHandle};

use afix_encoder::{encode_file, EncodeOptions};
use libafix::{AfixFile, Profile};

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("aFix Studio")
            .with_min_inner_size([700.0, 520.0])
            .with_inner_size([900.0, 640.0]),
        ..Default::default()
    };
    eframe::run_native(
        "aFix Studio",
        options,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
}

// ── App state ─────────────────────────────────────────────────────────────────

/// Top-level tab selection.
#[derive(Default, PartialEq)]
enum Tab {
    #[default]
    Convert,
    View,
}

/// The root application state.
struct App {
    tab: Tab,
    convert: ConvertState,
    view: ViewState,
}

impl App {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        App {
            tab: Tab::default(),
            convert: ConvertState::new(),
            view: ViewState::default(),
        }
    }
}

// ── Convert tab ───────────────────────────────────────────────────────────────

/// State for the Convert tab.
#[derive(Default)]
struct ConvertState {
    /// Source files selected by the user.
    input_files: Vec<PathBuf>,
    /// Output directory (defaults to same directory as source if None).
    output_dir: Option<PathBuf>,
    /// Encoding options.
    quality: u8,
    profile: ProfileChoice,
    no_preview: bool,
    /// Results from the last conversion run.
    results: Vec<ConvertResult>,
    /// True while a conversion is in progress.
    busy: bool,
}

impl ConvertState {
    fn new() -> Self {
        ConvertState {
            quality: 85,
            ..Default::default()
        }
    }
}

#[derive(Default, PartialEq, Clone, Copy)]
enum ProfileChoice {
    #[default]
    WebLossy,
    WebLossless,
    Spatial,
    Trusted,
    Full,
}

impl ProfileChoice {
    fn label(self) -> &'static str {
        match self {
            ProfileChoice::WebLossy    => "web-lossy",
            ProfileChoice::WebLossless => "web-lossless",
            ProfileChoice::Spatial     => "spatial",
            ProfileChoice::Trusted     => "trusted",
            ProfileChoice::Full        => "full",
        }
    }

    fn to_profile(self) -> Profile {
        match self {
            ProfileChoice::WebLossy    => Profile::WebLossy,
            ProfileChoice::WebLossless => Profile::WebLossless,
            ProfileChoice::Spatial     => Profile::Spatial,
            ProfileChoice::Trusted     => Profile::Trusted,
            ProfileChoice::Full        => Profile::Full,
        }
    }
}

struct ConvertResult {
    source: PathBuf,
    output: Option<PathBuf>,
    error:  Option<String>,
    bytes:  u64,
}

// ── View tab ──────────────────────────────────────────────────────────────────

/// State for the View / Inspect tab.
#[derive(Default)]
struct ViewState {
    /// Currently loaded `.aFix` file path.
    path: Option<PathBuf>,
    /// Parsed file (if successful).
    file: Option<AfixFile>,
    /// Error message if the file could not be opened.
    error: Option<String>,
    /// Decoded JPEG preview texture, if a PREV chunk is present.
    preview_texture: Option<TextureHandle>,
}

// ── eframe::App impl ──────────────────────────────────────────────────────────

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("header").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.heading("🔷 aFix Studio");
                ui.add_space(24.0);
                ui.selectable_value(&mut self.tab, Tab::Convert, "⚙ Convert");
                ui.selectable_value(&mut self.tab, Tab::View,    "🔍 View");
            });
            ui.add_space(4.0);
        });

        match self.tab {
            Tab::Convert => draw_convert(ctx, &mut self.convert),
            Tab::View    => draw_view(ctx, &mut self.view),
        }
    }
}

// ── Convert tab drawing ───────────────────────────────────────────────────────

fn draw_convert(ctx: &egui::Context, state: &mut ConvertState) {
    egui::CentralPanel::default().show(ctx, |ui| {
        ui.add_space(8.0);
        ui.heading("Convert Images to .aFix");
        ui.add_space(8.0);
        ui.separator();
        ui.add_space(8.0);

        // ── Source file selection ─────────────────────────────────────────────
        ui.horizontal(|ui| {
            if ui.button("➕ Add Files…").clicked() {
                if let Some(paths) = rfd::FileDialog::new()
                    .set_title("Select images to convert")
                    .add_filter("Images", &["jpg", "jpeg", "png"])
                    .pick_files()
                {
                    for p in paths {
                        if !state.input_files.contains(&p) {
                            state.input_files.push(p);
                        }
                    }
                }
            }
            if ui.button("🗑 Clear List").clicked() {
                state.input_files.clear();
            }
        });

        ui.add_space(6.0);

        // ── File list ─────────────────────────────────────────────────────────
        let list_height = 120.0;
        egui::ScrollArea::vertical()
            .id_salt("convert_list")
            .max_height(list_height)
            .show(ui, |ui| {
                if state.input_files.is_empty() {
                    ui.label(
                        egui::RichText::new("No files selected.  Click \"Add Files…\" above.")
                            .color(egui::Color32::GRAY)
                    );
                } else {
                    for path in &state.input_files {
                        ui.label(path.display().to_string());
                    }
                }
            });

        ui.add_space(8.0);
        ui.separator();
        ui.add_space(8.0);

        // ── Options ───────────────────────────────────────────────────────────
        ui.heading("Options");
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            ui.label("Profile:");
            for choice in [
                ProfileChoice::WebLossy,
                ProfileChoice::WebLossless,
                ProfileChoice::Spatial,
                ProfileChoice::Trusted,
                ProfileChoice::Full,
            ] {
                ui.selectable_value(&mut state.profile, choice, choice.label());
            }
        });

        ui.horizontal(|ui| {
            ui.label("Quality (0–100):");
            ui.add(egui::Slider::new(&mut state.quality, 0..=100));
        });

        ui.checkbox(&mut state.no_preview, "Omit JPEG preview (PREV chunk)");

        ui.add_space(6.0);

        // ── Output directory ──────────────────────────────────────────────────
        ui.horizontal(|ui| {
            ui.label("Output directory:");
            let out_label = state
                .output_dir
                .as_deref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "(same as source)".into());
            ui.label(egui::RichText::new(out_label).monospace());
            if ui.button("Browse…").clicked() {
                if let Some(dir) = rfd::FileDialog::new()
                    .set_title("Select output directory")
                    .pick_folder()
                {
                    state.output_dir = Some(dir);
                }
            }
            if state.output_dir.is_some() && ui.button("✖ Reset").clicked() {
                state.output_dir = None;
            }
        });

        ui.add_space(8.0);
        ui.separator();
        ui.add_space(8.0);

        // ── Convert button ────────────────────────────────────────────────────
        let can_convert = !state.input_files.is_empty() && !state.busy;
        if ui
            .add_enabled(can_convert, egui::Button::new("▶ Convert"))
            .clicked()
        {
            run_conversion(state);
        }

        // ── Results ───────────────────────────────────────────────────────────
        if !state.results.is_empty() {
            ui.add_space(8.0);
            ui.separator();
            ui.add_space(8.0);
            ui.heading("Results");
            ui.add_space(4.0);

            egui::ScrollArea::vertical()
                .id_salt("convert_results")
                .show(ui, |ui| {
                    for r in &state.results {
                        ui.horizontal(|ui| {
                            if let Some(ref err) = r.error {
                                ui.label(
                                    egui::RichText::new("✗")
                                        .color(egui::Color32::RED)
                                );
                                ui.label(r.source.display().to_string());
                                ui.label(
                                    egui::RichText::new(format!("— {err}"))
                                        .color(egui::Color32::RED)
                                );
                            } else {
                                ui.label(
                                    egui::RichText::new("✓")
                                        .color(egui::Color32::GREEN)
                                );
                                ui.label(r.source.display().to_string());
                                ui.label("→");
                                if let Some(ref out) = r.output {
                                    ui.label(
                                        egui::RichText::new(out.display().to_string())
                                            .monospace()
                                    );
                                }
                                ui.label(format!("({} bytes)", r.bytes));
                            }
                        });
                    }
                });
        }
    });
}

/// Run the conversion synchronously (the file set is typically small for
/// a desktop tool; async is not required here).
fn run_conversion(state: &mut ConvertState) {
    state.results.clear();
    state.busy = true;

    let opts = EncodeOptions {
        profile:         state.profile.to_profile(),
        quality:         state.quality,
        preview:         !state.no_preview,
        preview_quality: 60,
        semantic:        true,
    };

    for src in &state.input_files {
        let out_path = {
            let stem = src.file_stem().unwrap_or_default();
            let dir = state
                .output_dir
                .as_deref()
                .unwrap_or_else(|| src.parent().unwrap_or(std::path::Path::new(".")));
            dir.join(stem).with_extension("afix")
        };

        match encode_file(src, &out_path, &opts) {
            Ok(()) => {
                let bytes = std::fs::metadata(&out_path)
                    .map(|m| m.len())
                    .unwrap_or(0);
                state.results.push(ConvertResult {
                    source: src.clone(),
                    output: Some(out_path),
                    error:  None,
                    bytes,
                });
            }
            Err(e) => {
                state.results.push(ConvertResult {
                    source: src.clone(),
                    output: None,
                    error:  Some(e.to_string()),
                    bytes:  0,
                });
            }
        }
    }

    state.busy = false;
}

// ── View tab drawing ──────────────────────────────────────────────────────────

fn draw_view(ctx: &egui::Context, state: &mut ViewState) {
    egui::CentralPanel::default().show(ctx, |ui| {
        ui.add_space(8.0);
        ui.heading("Inspect .aFix File");
        ui.add_space(8.0);

        // ── Open button ───────────────────────────────────────────────────────
        if ui.button("📂 Open .aFix File…").clicked() {
            if let Some(path) = rfd::FileDialog::new()
                .set_title("Open .aFix file")
                .add_filter(".aFix files", &["afix"])
                .pick_file()
            {
                load_afix(state, &path, ctx);
            }
        }

        if let Some(ref err) = state.error {
            ui.add_space(8.0);
            ui.label(egui::RichText::new(format!("⚠ {err}")).color(egui::Color32::RED));
            return;
        }

        let Some(ref afix) = state.file else {
            ui.add_space(16.0);
            ui.label(
                egui::RichText::new("No file open.  Click \"Open .aFix File…\" above.")
                    .color(egui::Color32::GRAY)
            );
            return;
        };

        ui.add_space(8.0);
        ui.separator();
        ui.add_space(8.0);

        // ── Two-column layout: info left, preview right ───────────────────────
        ui.columns(2, |cols| {
            // ── Left column: header + chunk table ────────────────────────────
            let left = &mut cols[0];

            left.heading("Header");
            left.add_space(4.0);

            egui::Grid::new("header_grid")
                .num_columns(2)
                .spacing([16.0, 4.0])
                .show(left, |ui| {
                    ui.label("File");
                    if let Some(ref p) = state.path {
                        ui.label(egui::RichText::new(p.display().to_string()).monospace());
                    }
                    ui.end_row();

                    ui.label("Version");
                    ui.label(
                        egui::RichText::new(afix.header.version.to_string())
                            .monospace()
                    );
                    ui.end_row();

                    ui.label("Dimensions");
                    ui.label(egui::RichText::new(format!(
                        "{} × {} (logical)",
                        afix.header.dimensions.width,
                        afix.header.dimensions.height
                    )).monospace());
                    ui.end_row();

                    ui.label("Chunks");
                    ui.label(afix.chunks.len().to_string());
                    ui.end_row();
                });

            left.add_space(12.0);
            left.heading("Chunks");
            left.add_space(4.0);

            egui::ScrollArea::vertical()
                .id_salt("chunk_list")
                .show(left, |ui| {
                    egui::Grid::new("chunk_grid")
                        .num_columns(3)
                        .spacing([16.0, 4.0])
                        .striped(true)
                        .show(ui, |ui| {
                            // Header row
                            ui.label(egui::RichText::new("ID").strong());
                            ui.label(egui::RichText::new("Size").strong());
                            ui.label(egui::RichText::new("Flags").strong());
                            ui.end_row();

                            for chunk in &afix.chunks {
                                let encrypted =
                                    if chunk.is_encrypted() { " 🔒" } else { "" };
                                ui.label(
                                    egui::RichText::new(
                                        format!("{}{}", chunk.id.name(), encrypted)
                                    )
                                    .monospace()
                                );
                                ui.label(format_bytes(chunk.data.len() as u64));
                                ui.label(format!("{:#06x}", chunk.flags));
                                ui.end_row();
                            }
                        });
                });

            // ── Right column: JPEG preview ────────────────────────────────────
            let right = &mut cols[1];
            right.heading("Preview");
            right.add_space(4.0);

            if let Some(ref tex) = state.preview_texture {
                let available = right.available_size();
                let max_side = available.x.min(available.y - 32.0).max(64.0);
                right.add(
                    egui::Image::new(tex)
                        .max_size(egui::vec2(max_side, max_side))
                );
            } else {
                right.label(
                    egui::RichText::new("(no PREV chunk — preview not available)")
                        .color(egui::Color32::GRAY)
                );
            }
        });
    });
}

/// Load and parse an `.aFix` file, populating `state`.
fn load_afix(state: &mut ViewState, path: &std::path::Path, ctx: &egui::Context) {
    state.error           = None;
    state.file            = None;
    state.preview_texture = None;
    state.path            = Some(path.to_path_buf());

    let file = match std::fs::File::open(path) {
        Ok(f)  => f,
        Err(e) => { state.error = Some(e.to_string()); return; }
    };

    let afix = match AfixFile::read(std::io::BufReader::new(file)) {
        Ok(a)  => a,
        Err(e) => { state.error = Some(e.to_string()); return; }
    };

    // Try to decode the PREV chunk into an egui texture.
    if let Some(prev) = afix.get_chunk(libafix::ChunkId::Preview) {
        if let Ok(img) = image::load_from_memory(&prev.data) {
            let rgba = img.to_rgba8();
            let size = [rgba.width() as usize, rgba.height() as usize];
            let color_img = ColorImage::from_rgba_unmultiplied(size, rgba.as_raw());
            state.preview_texture = Some(
                ctx.load_texture("afix_preview", color_img, Default::default())
            );
        }
    }

    state.file = Some(afix);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Format a byte count as a human-readable string.
fn format_bytes(n: u64) -> String {
    if n < 1024 {
        format!("{n} B")
    } else if n < 1024 * 1024 {
        format!("{:.1} KB", n as f64 / 1024.0)
    } else {
        format!("{:.1} MB", n as f64 / (1024.0 * 1024.0))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_bytes_under_1k() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1023), "1023 B");
    }

    #[test]
    fn format_bytes_kilobytes() {
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(2048), "2.0 KB");
    }

    #[test]
    fn format_bytes_megabytes() {
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
    }

    #[test]
    fn profile_choice_roundtrip() {
        for &choice in &[
            ProfileChoice::WebLossy,
            ProfileChoice::WebLossless,
            ProfileChoice::Spatial,
            ProfileChoice::Trusted,
            ProfileChoice::Full,
        ] {
            // Ensure label is non-empty and profile conversion does not panic.
            assert!(!choice.label().is_empty());
            let _ = choice.to_profile();
        }
    }

    #[test]
    fn convert_state_default_quality_is_85() {
        let state = ConvertState::new();
        assert_eq!(state.quality, 85);
    }
}
