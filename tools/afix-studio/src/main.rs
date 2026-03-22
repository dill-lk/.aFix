//! # aFix Studio
//!
//! A native desktop application for working with `.aFix` image files.
//!
//! ## Features
//!
//! - **Convert** — Convert JPEG / PNG images to `.aFix` via a simple file picker.
//! - **View**    — Inspect `.aFix` files: rendered image, header info, and chunk table.

// On Windows, suppress the console window so the app starts as a proper GUI app.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::PathBuf;

use eframe::egui;
use eframe::egui::{ColorImage, RichText, TextureHandle};

use afix_encoder::{dct::decode_dct, encode_file, EncodeOptions};
use libafix::{AfixFile, Profile};

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("aFix Studio")
            .with_min_inner_size([800.0, 600.0])
            .with_inner_size([1100.0, 720.0]),
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
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Apply a dark theme with custom accent colours.
        let mut visuals = egui::Visuals::dark();
        visuals.override_text_color = Some(egui::Color32::from_rgb(220, 220, 230));
        visuals.panel_fill = egui::Color32::from_rgb(22, 22, 30);
        visuals.window_fill = egui::Color32::from_rgb(28, 28, 38);
        visuals.extreme_bg_color = egui::Color32::from_rgb(14, 14, 20);
        visuals.widgets.noninteractive.bg_fill = egui::Color32::from_rgb(32, 32, 46);
        visuals.widgets.inactive.bg_fill = egui::Color32::from_rgb(40, 40, 58);
        visuals.widgets.hovered.bg_fill = egui::Color32::from_rgb(55, 55, 80);
        visuals.widgets.active.bg_fill = egui::Color32::from_rgb(70, 100, 180);
        visuals.selection.bg_fill = egui::Color32::from_rgb(60, 90, 170);
        cc.egui_ctx.set_visuals(visuals);

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
    /// Decoded JPEG preview texture from the PREV chunk, if present.
    preview_texture: Option<TextureHandle>,
    /// Decoded image rendered from the LAT_ (DCT) chunk, if present.
    lat_texture: Option<TextureHandle>,
}

// ── eframe::App impl ──────────────────────────────────────────────────────────

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("header")
            .frame(
                egui::Frame::default()
                    .fill(egui::Color32::from_rgb(18, 18, 28))
                    .inner_margin(egui::Margin::symmetric(12, 8)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("⬡ aFix Studio")
                            .size(20.0)
                            .color(egui::Color32::from_rgb(120, 160, 255))
                            .strong(),
                    );
                    ui.add_space(32.0);
                    ui.selectable_value(&mut self.tab, Tab::Convert, "⚙  Convert");
                    ui.add_space(4.0);
                    ui.selectable_value(&mut self.tab, Tab::View, "🔍  View / Inspect");
                });
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
        ui.add_space(12.0);

        ui.label(
            RichText::new("Convert Images to .aFix")
                .size(18.0)
                .strong()
                .color(egui::Color32::from_rgb(140, 180, 255)),
        );
        ui.add_space(4.0);
        ui.label(
            RichText::new("Select JPEG or PNG files to encode into the .aFix format.")
                .color(egui::Color32::from_rgb(160, 160, 180)),
        );
        ui.add_space(12.0);
        ui.separator();
        ui.add_space(10.0);

        // ── Source file selection ─────────────────────────────────────────────
        ui.horizontal(|ui| {
            if ui
                .add(
                    egui::Button::new("➕  Add Files…")
                        .fill(egui::Color32::from_rgb(45, 65, 110)),
                )
                .clicked()
            {
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
            if !state.input_files.is_empty()
                && ui
                    .add(
                        egui::Button::new("🗑  Clear List")
                            .fill(egui::Color32::from_rgb(80, 30, 30)),
                    )
                    .clicked()
            {
                state.input_files.clear();
                state.results.clear();
            }
            ui.label(
                RichText::new(format!("{} file(s) selected", state.input_files.len()))
                    .color(egui::Color32::from_rgb(140, 140, 160)),
            );
        });

        ui.add_space(8.0);

        // ── File list ─────────────────────────────────────────────────────────
        let list_height = 110.0;
        egui::Frame::default()
            .fill(egui::Color32::from_rgb(18, 18, 26))
            .corner_radius(6.0)
            .inner_margin(egui::Margin::same(8))
            .show(ui, |ui| {
                egui::ScrollArea::vertical()
                    .id_salt("convert_list")
                    .max_height(list_height)
                    .show(ui, |ui| {
                        if state.input_files.is_empty() {
                            ui.add_space(16.0);
                            ui.centered_and_justified(|ui| {
                                ui.label(
                                    RichText::new("No files selected. Click \"Add Files…\" above.")
                                        .color(egui::Color32::from_rgb(100, 100, 120)),
                                );
                            });
                            ui.add_space(16.0);
                        } else {
                            for path in &state.input_files {
                                ui.horizontal(|ui| {
                                    ui.label(
                                        RichText::new("📄")
                                            .color(egui::Color32::from_rgb(100, 140, 220)),
                                    );
                                    ui.label(
                                        RichText::new(path.display().to_string())
                                            .monospace()
                                            .color(egui::Color32::from_rgb(200, 200, 210)),
                                    );
                                });
                            }
                        }
                    });
            });

        ui.add_space(10.0);
        ui.separator();
        ui.add_space(10.0);

        // ── Options ───────────────────────────────────────────────────────────
        ui.label(
            RichText::new("Encoding Options")
                .size(14.0)
                .strong()
                .color(egui::Color32::from_rgb(140, 180, 255)),
        );
        ui.add_space(6.0);

        ui.horizontal(|ui| {
            ui.label(
                RichText::new("Profile:")
                    .color(egui::Color32::from_rgb(180, 180, 200)),
            );
            ui.add_space(4.0);
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

        ui.add_space(4.0);

        ui.horizontal(|ui| {
            ui.label(
                RichText::new("Quality:")
                    .color(egui::Color32::from_rgb(180, 180, 200)),
            );
            ui.add_space(4.0);
            ui.add(egui::Slider::new(&mut state.quality, 0..=100).suffix("%"));
            ui.label(
                RichText::new(quality_label(state.quality))
                    .color(quality_color(state.quality)),
            );
        });

        ui.add_space(4.0);
        ui.checkbox(&mut state.no_preview, "Omit JPEG backward-compatible preview (PREV chunk)");

        ui.add_space(8.0);

        // ── Output directory ──────────────────────────────────────────────────
        ui.horizontal(|ui| {
            ui.label(
                RichText::new("Output:")
                    .color(egui::Color32::from_rgb(180, 180, 200)),
            );
            let out_label = state
                .output_dir
                .as_deref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "Same as source".into());
            ui.label(RichText::new(out_label).monospace().color(egui::Color32::from_rgb(160, 200, 160)));
            if ui.small_button("📁  Browse…").clicked() {
                if let Some(dir) = rfd::FileDialog::new()
                    .set_title("Select output directory")
                    .pick_folder()
                {
                    state.output_dir = Some(dir);
                }
            }
            if state.output_dir.is_some() && ui.small_button("✖").clicked() {
                state.output_dir = None;
            }
        });

        ui.add_space(12.0);
        ui.separator();
        ui.add_space(10.0);

        // ── Convert button ────────────────────────────────────────────────────
        let can_convert = !state.input_files.is_empty() && !state.busy;
        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    can_convert,
                    egui::Button::new(
                        RichText::new("▶  Convert Now")
                            .size(15.0)
                            .strong()
                            .color(egui::Color32::WHITE),
                    )
                    .fill(egui::Color32::from_rgb(50, 100, 200))
                    .min_size(egui::vec2(160.0, 36.0)),
                )
                .clicked()
            {
                run_conversion(state);
            }
            if state.busy {
                ui.spinner();
                ui.label(RichText::new("Converting…").color(egui::Color32::from_rgb(180, 200, 255)));
            }
        });

        // ── Results ───────────────────────────────────────────────────────────
        if !state.results.is_empty() {
            ui.add_space(12.0);
            ui.separator();
            ui.add_space(8.0);

            let ok_count  = state.results.iter().filter(|r| r.error.is_none()).count();
            let err_count = state.results.iter().filter(|r| r.error.is_some()).count();

            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("Results")
                        .size(14.0)
                        .strong()
                        .color(egui::Color32::from_rgb(140, 180, 255)),
                );
                if ok_count > 0 {
                    ui.label(
                        RichText::new(format!("  ✓ {ok_count} succeeded"))
                            .color(egui::Color32::from_rgb(80, 200, 100)),
                    );
                }
                if err_count > 0 {
                    ui.label(
                        RichText::new(format!("  ✗ {err_count} failed"))
                            .color(egui::Color32::from_rgb(220, 80, 80)),
                    );
                }
            });
            ui.add_space(4.0);

            egui::Frame::default()
                .fill(egui::Color32::from_rgb(18, 18, 26))
                .corner_radius(6.0)
                .inner_margin(egui::Margin::same(8))
                .show(ui, |ui| {
                    egui::ScrollArea::vertical()
                        .id_salt("convert_results")
                        .show(ui, |ui| {
                            for r in &state.results {
                                ui.horizontal(|ui| {
                                    if let Some(ref err) = r.error {
                                        ui.label(
                                            RichText::new("✗")
                                                .color(egui::Color32::from_rgb(220, 80, 80)),
                                        );
                                        ui.label(
                                            RichText::new(r.source.file_name()
                                                .map(|n| n.to_string_lossy().into_owned())
                                                .unwrap_or_default())
                                                .monospace()
                                                .color(egui::Color32::from_rgb(200, 200, 210)),
                                        );
                                        ui.label(
                                            RichText::new(format!("— {err}"))
                                                .color(egui::Color32::from_rgb(220, 80, 80)),
                                        );
                                    } else {
                                        ui.label(
                                            RichText::new("✓")
                                                .color(egui::Color32::from_rgb(80, 200, 100)),
                                        );
                                        ui.label(
                                            RichText::new(r.source.file_name()
                                                .map(|n| n.to_string_lossy().into_owned())
                                                .unwrap_or_default())
                                                .monospace()
                                                .color(egui::Color32::from_rgb(200, 200, 210)),
                                        );
                                        ui.label(
                                            RichText::new("→")
                                                .color(egui::Color32::from_rgb(140, 140, 160)),
                                        );
                                        if let Some(ref out) = r.output {
                                            ui.label(
                                                RichText::new(out.file_name()
                                                    .map(|n| n.to_string_lossy().into_owned())
                                                    .unwrap_or_else(|| out.display().to_string()))
                                                    .monospace()
                                                    .color(egui::Color32::from_rgb(140, 200, 140)),
                                            );
                                        }
                                        ui.label(
                                            RichText::new(format!("({})", format_bytes(r.bytes)))
                                                .color(egui::Color32::from_rgb(130, 130, 150)),
                                        );
                                    }
                                });
                            }
                        });
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
        ui.add_space(12.0);

        ui.horizontal(|ui| {
            ui.label(
                RichText::new("View / Inspect .aFix File")
                    .size(18.0)
                    .strong()
                    .color(egui::Color32::from_rgb(140, 180, 255)),
            );
            ui.add_space(16.0);
            if ui
                .add(
                    egui::Button::new("📂  Open .aFix File…")
                        .fill(egui::Color32::from_rgb(45, 65, 110)),
                )
                .clicked()
            {
                if let Some(path) = rfd::FileDialog::new()
                    .set_title("Open .aFix file")
                    .add_filter(".aFix files", &["afix"])
                    .pick_file()
                {
                    load_afix(state, &path, ctx);
                }
            }
        });

        if let Some(ref err) = state.error {
            ui.add_space(12.0);
            egui::Frame::default()
                .fill(egui::Color32::from_rgb(60, 20, 20))
                .corner_radius(6.0)
                .inner_margin(egui::Margin::same(10))
                .show(ui, |ui| {
                    ui.label(
                        RichText::new(format!("⚠  {err}"))
                            .color(egui::Color32::from_rgb(255, 120, 120)),
                    );
                });
            return;
        }

        let Some(ref afix) = state.file else {
            ui.add_space(40.0);
            ui.centered_and_justified(|ui| {
                ui.label(
                    RichText::new("No file open.  Click \"Open .aFix File…\" above.")
                        .size(15.0)
                        .color(egui::Color32::from_rgb(100, 100, 120)),
                );
            });
            return;
        };

        ui.add_space(10.0);
        ui.separator();
        ui.add_space(8.0);

        // ── Main layout: rendered image (left-centre) + metadata (right) ──────
        //    Use a fixed right-panel for metadata and give the rest to the image.
        egui::SidePanel::right("metadata_panel")
            .min_width(260.0)
            .max_width(380.0)
            .frame(
                egui::Frame::default()
                    .fill(egui::Color32::from_rgb(20, 20, 30))
                    .inner_margin(egui::Margin::same(12)),
            )
            .show_inside(ui, |ui| {
                draw_metadata_panel(ui, state, afix);
            });

        // ── Centre: rendered image ────────────────────────────────────────────
        egui::CentralPanel::default()
            .frame(
                egui::Frame::default()
                    .fill(egui::Color32::from_rgb(12, 12, 18))
                    .inner_margin(egui::Margin::same(12)),
            )
            .show_inside(ui, |ui| {
                draw_image_panel(ui, state);
            });
    });
}

/// Draw the metadata / chunk info panel (right side of View tab).
fn draw_metadata_panel(ui: &mut egui::Ui, state: &ViewState, afix: &AfixFile) {
    ui.label(
        RichText::new("File Info")
            .size(13.0)
            .strong()
            .color(egui::Color32::from_rgb(140, 180, 255)),
    );
    ui.add_space(6.0);

    egui::Grid::new("header_grid")
        .num_columns(2)
        .spacing([12.0, 4.0])
        .show(ui, |ui| {
            ui.label(RichText::new("File").color(egui::Color32::from_rgb(160, 160, 180)));
            if let Some(ref p) = state.path {
                let name = p.file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| p.display().to_string());
                ui.label(RichText::new(name).monospace().color(egui::Color32::from_rgb(200, 210, 230)));
            }
            ui.end_row();

            ui.label(RichText::new("Version").color(egui::Color32::from_rgb(160, 160, 180)));
            ui.label(
                RichText::new(afix.header.version.to_string())
                    .monospace()
                    .color(egui::Color32::from_rgb(200, 210, 230)),
            );
            ui.end_row();

            ui.label(RichText::new("Dimensions").color(egui::Color32::from_rgb(160, 160, 180)));
            ui.label(
                RichText::new(format!(
                    "{} × {}",
                    afix.header.dimensions.width,
                    afix.header.dimensions.height
                ))
                .monospace()
                .color(egui::Color32::from_rgb(200, 210, 230)),
            );
            ui.end_row();

            ui.label(RichText::new("Chunks").color(egui::Color32::from_rgb(160, 160, 180)));
            ui.label(
                RichText::new(afix.chunks.len().to_string())
                    .monospace()
                    .color(egui::Color32::from_rgb(200, 210, 230)),
            );
            ui.end_row();
        });

    ui.add_space(14.0);
    ui.separator();
    ui.add_space(10.0);

    ui.label(
        RichText::new("Chunks")
            .size(13.0)
            .strong()
            .color(egui::Color32::from_rgb(140, 180, 255)),
    );
    ui.add_space(6.0);

    egui::ScrollArea::vertical()
        .id_salt("chunk_list")
        .show(ui, |ui| {
            egui::Grid::new("chunk_grid")
                .num_columns(3)
                .spacing([12.0, 4.0])
                .striped(true)
                .show(ui, |ui| {
                    ui.label(RichText::new("ID").strong().color(egui::Color32::from_rgb(180, 180, 200)));
                    ui.label(RichText::new("Size").strong().color(egui::Color32::from_rgb(180, 180, 200)));
                    ui.label(RichText::new("Flags").strong().color(egui::Color32::from_rgb(180, 180, 200)));
                    ui.end_row();

                    for chunk in &afix.chunks {
                        let encrypted = if chunk.is_encrypted() { " 🔒" } else { "" };
                        let id_color = chunk_id_color(chunk.id);
                        ui.label(
                            RichText::new(format!("{}{}", chunk.id.name(), encrypted))
                                .monospace()
                                .color(id_color),
                        );
                        ui.label(
                            RichText::new(format_bytes(chunk.data.len() as u64))
                                .color(egui::Color32::from_rgb(180, 180, 200)),
                        );
                        ui.label(
                            RichText::new(format!("{:#06x}", chunk.flags))
                                .monospace()
                                .color(egui::Color32::from_rgb(140, 140, 160)),
                        );
                        ui.end_row();
                    }
                });
        });
}

/// Draw the rendered image panel (centre of View tab).
fn draw_image_panel(ui: &mut egui::Ui, state: &ViewState) {
    // Prefer the full decoded LAT_ image; fall back to the PREV thumbnail.
    let (texture, label) = if let Some(ref tex) = state.lat_texture {
        (Some(tex), "Decoded Image (LAT_ chunk)")
    } else if let Some(ref tex) = state.preview_texture {
        (Some(tex), "JPEG Preview (PREV chunk)")
    } else {
        (None, "")
    };

    if let Some(tex) = texture {
        ui.vertical_centered(|ui| {
            ui.label(
                RichText::new(label)
                    .size(11.0)
                    .color(egui::Color32::from_rgb(120, 140, 180)),
            );
            ui.add_space(6.0);

            let available = ui.available_size();
            let max_w = available.x.max(64.0);
            let max_h = (available.y - 32.0).max(64.0);

            ui.add(
                egui::Image::new(tex)
                    .max_size(egui::vec2(max_w, max_h))
                    .maintain_aspect_ratio(true),
            );
        });
    } else {
        ui.centered_and_justified(|ui| {
            ui.label(
                RichText::new("No renderable image data found in this file.\n\
                               Ensure the file contains a LAT_ or PREV chunk.")
                    .size(13.0)
                    .color(egui::Color32::from_rgb(100, 100, 120)),
            );
        });
    }
}

/// Load and parse an `.aFix` file, populating `state`.
fn load_afix(state: &mut ViewState, path: &std::path::Path, ctx: &egui::Context) {
    state.error           = None;
    state.file            = None;
    state.preview_texture = None;
    state.lat_texture     = None;
    state.path            = Some(path.to_path_buf());

    let file = match std::fs::File::open(path) {
        Ok(f)  => f,
        Err(e) => { state.error = Some(e.to_string()); return; }
    };

    let afix = match AfixFile::read(std::io::BufReader::new(file)) {
        Ok(a)  => a,
        Err(e) => { state.error = Some(e.to_string()); return; }
    };

    // Try to decode the PREV chunk into an egui texture (quick JPEG preview).
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

    // Decode the LAT_ chunk (DCT-based) to produce the full rendered image.
    // This is the primary visual representation of the encoded image data.
    if let Some(lat) = afix.get_chunk(libafix::ChunkId::Lat) {
        if let Some((rgb, w, h)) = decode_dct(&lat.data) {
            let size = [w as usize, h as usize];
            let rgba: Vec<u8> = rgb
                .chunks_exact(3)
                .flat_map(|p| [p[0], p[1], p[2], 255u8])
                .collect();
            let color_img = ColorImage::from_rgba_unmultiplied(size, &rgba);
            state.lat_texture = Some(
                ctx.load_texture("afix_lat", color_img, Default::default())
            );
        }
    }

    state.file = Some(afix);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Return a colour for a chunk ID badge in the View tab.
fn chunk_id_color(id: libafix::ChunkId) -> egui::Color32 {
    use libafix::ChunkId;
    match id {
        ChunkId::Meta        => egui::Color32::from_rgb(200, 180, 100),
        ChunkId::Vec         => egui::Color32::from_rgb(100, 200, 140),
        ChunkId::Lat         => egui::Color32::from_rgb(100, 160, 255),
        ChunkId::Res         => egui::Color32::from_rgb(160, 120, 220),
        ChunkId::Preview     => egui::Color32::from_rgb(220, 160, 80),
        ChunkId::ObjManifest => egui::Color32::from_rgb(200, 100, 140),
        _                    => egui::Color32::from_rgb(160, 160, 180),
    }
}

/// Return a short description of the quality level.
fn quality_label(q: u8) -> &'static str {
    match q {
        0..=30  => "Low",
        31..=60 => "Medium",
        61..=85 => "High",
        _       => "Maximum",
    }
}

/// Return a colour for the quality label.
fn quality_color(q: u8) -> egui::Color32 {
    match q {
        0..=30  => egui::Color32::from_rgb(220, 100, 80),
        31..=60 => egui::Color32::from_rgb(220, 180, 60),
        61..=85 => egui::Color32::from_rgb(100, 200, 120),
        _       => egui::Color32::from_rgb(80, 160, 255),
    }
}

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

    #[test]
    fn quality_label_ranges() {
        assert_eq!(quality_label(0),   "Low");
        assert_eq!(quality_label(30),  "Low");
        assert_eq!(quality_label(31),  "Medium");
        assert_eq!(quality_label(60),  "Medium");
        assert_eq!(quality_label(61),  "High");
        assert_eq!(quality_label(85),  "High");
        assert_eq!(quality_label(86),  "Maximum");
        assert_eq!(quality_label(100), "Maximum");
    }

    #[test]
    fn chunk_id_color_returns_distinct_colors() {
        use libafix::ChunkId;
        // Ensure different well-known chunks get distinct colours.
        let meta_c = chunk_id_color(ChunkId::Meta);
        let lat_c  = chunk_id_color(ChunkId::Lat);
        let vec_c  = chunk_id_color(ChunkId::Vec);
        assert_ne!(meta_c, lat_c);
        assert_ne!(meta_c, vec_c);
        assert_ne!(lat_c,  vec_c);
    }
}
