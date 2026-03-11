# libafix

Core C++ and Rust library for reading and writing `.aFix` files. MIT-licensed.

## API Overview

### Rust

```rust
use libafix::{AfixDecoder, AfixEncoder, Profile};

// Decode
let decoder = AfixDecoder::from_path("photo.afix")?;
let image = decoder.decode_full()?;
let layers = decoder.semantic_layers();

// Encode
let encoder = AfixEncoder::new(Profile::WebLossy);
encoder.encode_from_path("source.jpg", "output.afix")?;
```

### C++ (via FFI)

```cpp
#include "libafix.h"

afix_decoder_t* dec = afix_decoder_open("photo.afix");
afix_image_t*   img = afix_decode_full(dec);
afix_decoder_close(dec);
```

## Profiles

| Profile | Description |
|---------|-------------|
| `WebLossy` | S1 + S2 only. Best for consumer web. |
| `WebLossless` | S1 + S2 + S3. For design/print. |
| `Spatial` | S1 + S2 + DPTH. For AR/VR. |
| `Trusted` | S1 + S2 + SIGB. For journalism/legal. |
| `Full` | All chunks. Professional archival. |

## Status

🚧 **Phase 1 — In Development**
