//! `afix-convert` — Convert JPEG/PNG/WebP images to `.aFix` format.
//!
//! # Usage
//!
//! ```text
//! afix-convert [OPTIONS] <INPUT> <OUTPUT>
//! afix-convert --batch <INPUT_DIR> --output <OUTPUT_DIR> [OPTIONS]
//! ```

use std::path::{Path, PathBuf};
use std::process;

use afix_encoder::{encode_file, EncodeOptions};
use libafix::Profile;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if let Err(e) = run(&args[1..]) {
        eprintln!("error: {e}");
        process::exit(1);
    }
}

fn run(args: &[String]) -> Result<(), String> {
    // ── Parse flags ───────────────────────────────────────────────────────────
    let mut profile = Profile::WebLossy;
    let mut quality: u8 = 85;
    let mut semantic = true;
    let mut preview = true;
    let mut preview_quality: u8 = 60;
    let mut batch = false;
    let mut output: Option<PathBuf> = None;
    let mut positional: Vec<String> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--profile" | "-p" => {
                i += 1;
                let s = args.get(i).ok_or("--profile requires a value")?;
                profile = s.parse::<Profile>().map_err(|e| e)?;
            }
            "--quality" | "-q" => {
                i += 1;
                let s = args.get(i).ok_or("--quality requires a value")?;
                quality = s.parse::<u8>().map_err(|e| format!("invalid quality: {e}"))?;
            }
            "--no-semantic" => semantic = false,
            "--no-preview"  => preview = false,
            "--preview-quality" => {
                i += 1;
                let s = args.get(i).ok_or("--preview-quality requires a value")?;
                preview_quality = s.parse::<u8>().map_err(|e| format!("invalid preview-quality: {e}"))?;
            }
            "--batch" => batch = true,
            "--output" | "-o" => {
                i += 1;
                let s = args.get(i).ok_or("--output requires a value")?;
                output = Some(PathBuf::from(s));
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            flag if flag.starts_with('-') => {
                return Err(format!("unknown flag '{flag}'"));
            }
            _ => positional.push(args[i].clone()),
        }
        i += 1;
    }

    let options = EncodeOptions { profile, quality, semantic, preview, preview_quality };

    if batch {
        // ── Batch mode ────────────────────────────────────────────────────────
        let input_dir = positional.first().ok_or("batch mode requires an input directory")?;
        let out_dir = output.ok_or("batch mode requires --output <DIR>")?;
        std::fs::create_dir_all(&out_dir)
            .map_err(|e| format!("cannot create output directory: {e}"))?;
        convert_batch(Path::new(input_dir), &out_dir, &options)
    } else {
        // ── Single-file mode ──────────────────────────────────────────────────
        let input = positional.first().ok_or("missing input file")?;
        let out_path: PathBuf = match output {
            Some(p) => p,
            None => {
                // Use second positional arg as output if provided.
                if positional.len() >= 2 {
                    PathBuf::from(&positional[1])
                } else {
                    let stem = Path::new(input)
                        .file_stem()
                        .ok_or("cannot determine output filename")?;
                    PathBuf::from(stem).with_extension("afix")
                }
            }
        };
        convert_one(Path::new(input), &out_path, &options)
    }
}

fn convert_one(input: &Path, output: &Path, opts: &EncodeOptions) -> Result<(), String> {
    println!("Converting {} → {} [profile={} quality={}]",
        input.display(), output.display(), opts.profile, opts.quality);
    encode_file(input, output, opts).map_err(|e| e.to_string())?;
    let out_size = std::fs::metadata(output)
        .map(|m| m.len())
        .unwrap_or(0);
    println!("  ✓  Written {} bytes", out_size);
    Ok(())
}

fn convert_batch(input_dir: &Path, output_dir: &Path, opts: &EncodeOptions) -> Result<(), String> {
    let entries = std::fs::read_dir(input_dir)
        .map_err(|e| format!("cannot read directory '{}': {e}", input_dir.display()))?;

    let mut converted = 0usize;
    let mut errors = 0usize;

    for entry in entries.flatten() {
        let path = entry.path();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if matches!(ext.to_lowercase().as_str(), "jpg" | "jpeg" | "png" | "webp") {
                let stem = path.file_stem().unwrap_or_default();
                let out = output_dir.join(stem).with_extension("afix");
                if let Err(e) = convert_one(&path, &out, opts) {
                    eprintln!("  ✗  {}: {e}", path.display());
                    errors += 1;
                } else {
                    converted += 1;
                }
            }
        }
    }

    println!("\nBatch complete: {converted} converted, {errors} errors");
    if errors > 0 {
        Err(format!("{errors} file(s) failed to convert"))
    } else {
        Ok(())
    }
}

fn print_help() {
    println!(concat!(
        "afix-convert — Convert JPEG/PNG/WebP images to .aFix format\n",
        "\n",
        "USAGE:\n",
        "    afix-convert [OPTIONS] <INPUT> [OUTPUT]\n",
        "    afix-convert --batch <INPUT_DIR> --output <OUTPUT_DIR> [OPTIONS]\n",
        "\n",
        "OPTIONS:\n",
        "    --profile, -p       Encoding profile (web-lossy*, web-lossless, spatial,\n",
        "                        trusted, full)\n",
        "    --quality, -q       Neural/DCT quality 0-100 (default: 85)\n",
        "    --no-semantic       Disable semantic object detection\n",
        "    --no-preview        Omit the JPEG backward-compat preview (PREV chunk)\n",
        "    --preview-quality   JPEG quality for the preview, 1-100 (default: 60)\n",
        "    --batch             Convert all images in INPUT directory\n",
        "    --output, -o        Output file or directory\n",
        "    --help, -h          Show this help message\n",
        "\n",
        "EXAMPLES:\n",
        "    afix-convert photo.jpg\n",
        "    afix-convert --profile web-lossless photo.png photo.afix\n",
        "    afix-convert --batch ./images/ --output ./afix-images/\n",
        "    afix-convert --no-preview photo.jpg photo.afix\n",
    ));
}
