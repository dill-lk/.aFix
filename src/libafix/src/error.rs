//! Error types for `libafix`.

use std::io;

/// All errors that can occur when reading or writing an `.aFix` file.
#[derive(Debug)]
pub enum AfixError {
    /// An I/O error occurred.
    Io(io::Error),
    /// The file does not start with the expected magic bytes (`AFIXK`).
    InvalidMagic([u8; 5]),
    /// A chunk's CRC-32 does not match the stored value.
    CrcMismatch {
        id: [u8; 4],
        stored: u32,
        computed: u32,
    },
    /// A chunk claims to be larger than the allowed maximum (512 MB).
    ChunkTooLarge(usize),
    /// A required chunk was not found in the file.
    MissingChunk(String),
    /// The JSON/BSON payload inside a chunk could not be parsed.
    InvalidChunkData(String),
    /// The file version is not supported by this library.
    UnsupportedVersion { major: u8, minor: u8 },
}

impl std::fmt::Display for AfixError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AfixError::Io(e) => write!(f, "I/O error: {e}"),
            AfixError::InvalidMagic(m) => write!(
                f,
                "invalid magic bytes: {:02X} {:02X} {:02X} {:02X} {:02X}",
                m[0], m[1], m[2], m[3], m[4]
            ),
            AfixError::CrcMismatch { id, stored, computed } => write!(
                f,
                "CRC mismatch in chunk '{}': stored={stored:#010x} computed={computed:#010x}",
                String::from_utf8_lossy(id)
            ),
            AfixError::ChunkTooLarge(n) => write!(f, "chunk too large: {n} bytes"),
            AfixError::MissingChunk(id) => write!(f, "required chunk '{id}' not found"),
            AfixError::InvalidChunkData(msg) => write!(f, "invalid chunk data: {msg}"),
            AfixError::UnsupportedVersion { major, minor } => {
                write!(f, "unsupported format version {major}.{minor}")
            }
        }
    }
}

impl std::error::Error for AfixError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if let AfixError::Io(e) = self {
            Some(e)
        } else {
            None
        }
    }
}

impl From<io::Error> for AfixError {
    fn from(e: io::Error) -> Self {
        AfixError::Io(e)
    }
}

/// Convenience `Result` type for `libafix`.
pub type Result<T> = std::result::Result<T, AfixError>;
