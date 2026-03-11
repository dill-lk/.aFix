//! `afix-info` — Display metadata and chunk information for `.aFix` files.
//!
//! # Usage
//!
//! ```text
//! afix-info <FILE> [FILE...]
//! ```

use std::fs::File;
use std::io::BufReader;
use std::process;

use libafix::{AfixFile, ChunkId, ObjectManifest};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        println!("Usage: afix-info <FILE> [FILE...]");
        return;
    }

    let mut any_error = false;
    for path in &args {
        if let Err(e) = print_info(path) {
            eprintln!("error reading '{path}': {e}");
            any_error = true;
        }
    }
    if any_error {
        process::exit(1);
    }
}

fn print_info(path: &str) -> Result<(), String> {
    let f = File::open(path).map_err(|e| e.to_string())?;
    let reader = BufReader::new(f);
    let afix = AfixFile::read(reader).map_err(|e| e.to_string())?;

    println!("╔══════════════════════════════════════════════════╗");
    println!("║  .aFix File Info                                 ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!("  File      : {path}");
    println!("  Version   : {}", afix.header.version);
    println!("  Dimensions: {} × {} (logical)",
        afix.header.dimensions.width, afix.header.dimensions.height);
    println!("  Chunks    : {}", afix.chunks.len());
    println!();

    for chunk in &afix.chunks {
        let encrypted = if chunk.is_encrypted() { " [ENCRYPTED]" } else { "" };
        println!("  ├─ {:<6}  {:>8} bytes{}", chunk.id.name(), chunk.data.len(), encrypted);

        // Print extra info for known chunk types.
        match chunk.id {
            ChunkId::Meta => {
                if let Ok(json) = serde_json::from_slice::<serde_json::Value>(&chunk.data) {
                    if let Some(creator) = json["creator"].as_str() {
                        println!("  │   creator  : {creator}");
                    }
                    if let Some(profile) = json["profile"].as_str() {
                        println!("  │   profile  : {profile}");
                    }
                }
            }
            ChunkId::Vec => {
                if chunk.data.len() >= 4 {
                    let count = u32::from_le_bytes(chunk.data[..4].try_into().unwrap_or([0; 4]));
                    println!("  │   edges    : {count} detected");
                }
            }
            ChunkId::Lat => {
                if chunk.data.len() >= 12 {
                    let w = u32::from_le_bytes(chunk.data[0..4].try_into().unwrap_or([0; 4]));
                    let h = u32::from_le_bytes(chunk.data[4..8].try_into().unwrap_or([0; 4]));
                    let c = u32::from_le_bytes(chunk.data[8..12].try_into().unwrap_or([0; 4]));
                    println!("  │   tensor   : {w}×{h}×{c}");
                }
            }
            ChunkId::ObjManifest => {
                if let Ok(manifest) = ObjectManifest::from_chunk_data(&chunk.data) {
                    println!("  │   objects  : {}", manifest.objects.len());
                    for obj in &manifest.objects {
                        println!("  │     - {} ({})", obj.id, obj.category);
                    }
                }
            }
            _ => {}
        }
    }
    println!();
    Ok(())
}
