//! # libafix
//!
//! Core library for reading and writing `.aFix` (Adaptive Flexible Image X) files.
//!
//! ## File Format Overview
//!
//! An `.aFix` file has the following structure:
//!
//! ```text
//! ┌────────────────────────────────┐
//! │  HEADER  (33 bytes)            │
//! │    Magic "AFIXK" (5 B)         │
//! │    Version VSN_ (4 B)          │
//! │    Dimensions DESC (24 B)      │
//! ├────────────────────────────────┤
//! │  ATOM_MAP (144 bytes)          │
//! │    6 × 24-byte chunk pointers  │
//! ├────────────────────────────────┤
//! │  PAYLOAD (variable)            │
//! │    Atom chunks (see ChunkId)   │
//! └────────────────────────────────┘
//! ```

use std::io::{self, Read, Seek, SeekFrom, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use crc32fast::Hasher as Crc32Hasher;

// ── Public re-exports ─────────────────────────────────────────────────────────

pub use chunk::{Chunk, ChunkId};
pub use error::{AfixError, Result};
pub use header::{AfixHeader, Dimensions, Version};
pub use manifest::ObjectManifest;
pub use profile::Profile;

// ── Modules ───────────────────────────────────────────────────────────────────

pub mod chunk;
pub mod error;
pub mod header;
pub mod manifest;
pub mod profile;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Magic bytes at the start of every `.aFix` file (`AFIXK` in ASCII).
pub const MAGIC: &[u8; 5] = b"AFIXK";

/// Byte offset where the header ends and ATOM_MAP begins (`0x21`).
pub const ATOM_MAP_OFFSET: u64 = 0x21;

/// Size of the ATOM_MAP in bytes (144 B = 6 × 24 B).
pub const ATOM_MAP_SIZE: usize = 144;

/// Byte offset where the PAYLOAD begins (`0xB1`).
pub const PAYLOAD_OFFSET: u64 = 0xB1;

/// Each atom-map entry is 24 bytes: stream_id (4B) + byte_offset (8B) + byte_length (8B) + checksum (4B).
pub const ATOM_ENTRY_SIZE: usize = 24;

/// Maximum number of atom-map entries.
pub const MAX_ATOM_ENTRIES: usize = 6;

// ── AfixFile ──────────────────────────────────────────────────────────────────

/// A parsed `.aFix` file containing a header and a list of atom chunks.
#[derive(Debug, Clone)]
pub struct AfixFile {
    pub header: AfixHeader,
    pub chunks: Vec<Chunk>,
}

impl AfixFile {
    /// Create a new, empty `.aFix` file with the given dimensions.
    pub fn new(width: f64, height: f64, _profile: Profile) -> Self {
        AfixFile {
            header: AfixHeader {
                version: Version::current(),
                dimensions: Dimensions { width, height },
            },
            chunks: Vec::new(),
        }
    }

    /// Add a chunk to the file.
    pub fn add_chunk(&mut self, chunk: Chunk) {
        self.chunks.push(chunk);
    }

    /// Find the first chunk with the given ID, if present.
    pub fn get_chunk(&self, id: ChunkId) -> Option<&Chunk> {
        self.chunks.iter().find(|c| c.id == id)
    }

    /// Parse an `.aFix` file from a reader.
    pub fn read<R: Read + Seek>(mut reader: R) -> Result<Self> {
        // ── 1. Magic ──────────────────────────────────────────────────────────
        let mut magic = [0u8; 5];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(AfixError::InvalidMagic(magic));
        }

        // ── 2. Version ────────────────────────────────────────────────────────
        let major = reader.read_u8()?;
        let minor = reader.read_u8()?;
        let patch = reader.read_u8()?;
        let flag = reader.read_u8()?;
        let version = Version { major, minor, patch, flag };

        // ── 3. DESC — Dimensions (24 B) ───────────────────────────────────────
        let width = reader.read_f64::<LittleEndian>()?;
        let height = reader.read_f64::<LittleEndian>()?;
        let mut _reserved = [0u8; 8];
        reader.read_exact(&mut _reserved)?;

        let header = AfixHeader {
            version,
            dimensions: Dimensions { width, height },
        };

        // ── 4. ATOM_MAP (144 B) ───────────────────────────────────────────────
        // We read the atom map to locate chunks in the payload, but we also
        // parse them sequentially below, so here we just skip it.
        let mut atom_map_raw = [0u8; ATOM_MAP_SIZE];
        reader.read_exact(&mut atom_map_raw)?;

        // ── 5. PAYLOAD — read chunks sequentially ────────────────────────────
        reader.seek(SeekFrom::Start(PAYLOAD_OFFSET))?;
        let mut chunks = Vec::new();

        loop {
            // Try to read a chunk ID (4 bytes). EOF here is normal.
            let mut id_bytes = [0u8; 4];
            match reader.read_exact(&mut id_bytes) {
                Ok(_) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(AfixError::Io(e)),
            }

            let chunk_len = reader.read_u32::<LittleEndian>()? as usize;
            let flags = reader.read_u16::<LittleEndian>()?;
            let _reserved2 = reader.read_u16::<LittleEndian>()?;

            // Safety: reject absurdly large chunks (512 MB cap).
            const MAX_CHUNK_SIZE: usize = 512 * 1024 * 1024;
            if chunk_len > MAX_CHUNK_SIZE {
                return Err(AfixError::ChunkTooLarge(chunk_len));
            }

            let mut data = vec![0u8; chunk_len];
            reader.read_exact(&mut data)?;

            let stored_crc = reader.read_u32::<LittleEndian>()?;
            let computed_crc = crc32_of(&data);
            if stored_crc != computed_crc {
                return Err(AfixError::CrcMismatch {
                    id: id_bytes,
                    stored: stored_crc,
                    computed: computed_crc,
                });
            }

            let id = ChunkId::from_bytes(id_bytes);
            chunks.push(Chunk { id, flags, data });
        }

        Ok(AfixFile { header, chunks })
    }

    /// Write the `.aFix` file to a writer.
    pub fn write<W: Write + Seek>(&self, mut writer: W) -> Result<()> {
        // ── 1. Magic ──────────────────────────────────────────────────────────
        writer.write_all(MAGIC)?;

        // ── 2. Version ────────────────────────────────────────────────────────
        let v = &self.header.version;
        writer.write_u8(v.major)?;
        writer.write_u8(v.minor)?;
        writer.write_u8(v.patch)?;
        writer.write_u8(v.flag)?;

        // ── 3. DESC (24 B) ────────────────────────────────────────────────────
        writer.write_f64::<LittleEndian>(self.header.dimensions.width)?;
        writer.write_f64::<LittleEndian>(self.header.dimensions.height)?;
        writer.write_all(&[0u8; 8])?; // 8 bytes reserved

        // ── 4. ATOM_MAP (144 B) ───────────────────────────────────────────────
        // Build atom map: for each chunk record its stream_id, offset, length,
        // and CRC so readers can seek directly.
        let mut atom_map = vec![0u8; ATOM_MAP_SIZE];
        let mut current_offset: u64 = PAYLOAD_OFFSET;

        for (i, chunk) in self.chunks.iter().enumerate().take(MAX_ATOM_ENTRIES) {
            // Each payload chunk occupies: 4 (id) + 4 (len) + 2 (flags) + 2 (res) + data + 4 (crc)
            let entry_start = i * ATOM_ENTRY_SIZE;
            let id_bytes = chunk.id.to_bytes();
            atom_map[entry_start..entry_start + 4].copy_from_slice(&id_bytes);
            let offset_bytes = current_offset.to_le_bytes();
            atom_map[entry_start + 4..entry_start + 12].copy_from_slice(&offset_bytes);
            let length = chunk.data.len() as u64;
            let length_bytes = length.to_le_bytes();
            atom_map[entry_start + 12..entry_start + 20].copy_from_slice(&length_bytes);
            let crc = crc32_of(&chunk.data);
            let crc_bytes = crc.to_le_bytes();
            atom_map[entry_start + 20..entry_start + 24].copy_from_slice(&crc_bytes);

            current_offset += 4 + 4 + 2 + 2 + chunk.data.len() as u64 + 4;
        }

        writer.write_all(&atom_map)?;

        // ── 5. PAYLOAD — write chunks ─────────────────────────────────────────
        for chunk in &self.chunks {
            writer.write_all(&chunk.id.to_bytes())?;
            writer.write_u32::<LittleEndian>(chunk.data.len() as u32)?;
            writer.write_u16::<LittleEndian>(chunk.flags)?;
            writer.write_u16::<LittleEndian>(0u16)?; // reserved
            writer.write_all(&chunk.data)?;
            let crc = crc32_of(&chunk.data);
            writer.write_u32::<LittleEndian>(crc)?;
        }

        Ok(())
    }
}

// ── Utility functions ─────────────────────────────────────────────────────────

/// Compute CRC-32 of a byte slice.
pub fn crc32_of(data: &[u8]) -> u32 {
    let mut h = Crc32Hasher::new();
    h.update(data);
    h.finalize()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn make_simple_file() -> AfixFile {
        let mut f = AfixFile::new(1920.0, 1080.0, Profile::WebLossy);
        f.add_chunk(Chunk {
            id: ChunkId::Meta,
            flags: 0,
            data: br#"{"version":"1.0","creator":"test"}"#.to_vec(),
        });
        f.add_chunk(Chunk {
            id: ChunkId::Vec,
            flags: 0,
            data: vec![0xDE, 0xAD, 0xBE, 0xEF],
        });
        f
    }

    #[test]
    fn roundtrip_write_read() {
        let original = make_simple_file();
        let mut buf = Cursor::new(Vec::new());
        original.write(&mut buf).expect("write failed");

        buf.seek(SeekFrom::Start(0)).unwrap();
        let parsed = AfixFile::read(&mut buf).expect("read failed");

        assert_eq!(parsed.header.dimensions.width, 1920.0);
        assert_eq!(parsed.header.dimensions.height, 1080.0);
        assert_eq!(parsed.chunks.len(), 2);
        assert_eq!(parsed.chunks[0].id, ChunkId::Meta);
        assert_eq!(parsed.chunks[1].id, ChunkId::Vec);
    }

    #[test]
    fn bad_magic_is_rejected() {
        let mut buf = Cursor::new(b"NOPE\x00".to_vec());
        assert!(matches!(AfixFile::read(&mut buf), Err(AfixError::InvalidMagic(_))));
    }

    #[test]
    fn crc_mismatch_is_rejected() {
        let f = make_simple_file();
        let mut buf = Cursor::new(Vec::new());
        f.write(&mut buf).unwrap();

        // Flip a byte in the first chunk's data region (after header/atom_map/chunk_header).
        // PAYLOAD starts at 0xB1. First chunk: 4(id)+4(len)+2(flags)+2(res) = 12 bytes before data.
        let flip_pos = (PAYLOAD_OFFSET + 12) as usize;
        let raw = buf.get_mut();
        raw[flip_pos] ^= 0xFF;

        buf.seek(SeekFrom::Start(0)).unwrap();
        assert!(matches!(AfixFile::read(&mut buf), Err(AfixError::CrcMismatch { .. })));
    }

    #[test]
    fn magic_bytes_are_correct() {
        assert_eq!(MAGIC, b"AFIXK");
    }

    #[test]
    fn payload_offset_is_correct() {
        assert_eq!(PAYLOAD_OFFSET, 0xB1);
    }
}
