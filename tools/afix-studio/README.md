# aFix Studio

A native Windows desktop application for working with `.aFix` image files.

## Features

| Tab | Description |
|-----|-------------|
| **Convert** | Open JPEG / PNG files and convert them to `.aFix` format. Adjust quality, profile, and output directory before running. |
| **View** | Open any `.aFix` file and inspect its header, protocol version, logical dimensions, and chunk table. If a `PREV` chunk (embedded JPEG preview) is present, it is rendered directly in the app. |

## Building

```bash
# Debug build (console window visible — useful during development)
cargo build -p afix-studio

# Release build (no console window on Windows)
cargo build --release -p afix-studio
```

The binary is written to `target/release/afix-studio.exe` (Windows) or
`target/release/afix-studio` (Linux/macOS).

## Usage

Launch `afix-studio.exe` — no command-line arguments are needed.

### Convert tab

1. Click **Add Files…** to pick one or more JPEG / PNG images.
2. Adjust **Profile** and **Quality** as required.
3. Optionally pick an **Output directory**; if left blank, `.afix` files are
   written next to the original images.
4. Click **▶ Convert** — results appear in the scrollable list at the bottom.

### View tab

1. Click **📂 Open .aFix File…** and select an `.afix` file.
2. The header information and chunk table are shown on the left.
3. If the file contains a `PREV` (JPEG preview) chunk, the image is rendered on
   the right.
