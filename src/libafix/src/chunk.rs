//! Atom chunk types for `.aFix`.

/// A recognised chunk identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChunkId {
    /// `META` — JSON/BSON creator & licence metadata.
    Meta,
    /// `VEC_` — S1 Geometric Skeleton (B-Spline vectors, LOD-0).
    Vec,
    /// `LAT_` — S2 Latent Texture Field (neural latents, LOD-1).
    Lat,
    /// `RES_` — S3 Parity Residual (lossless correction, LOD-2, optional).
    Res,
    /// `DPTH` — 16-bit unsigned depth map.
    Depth,
    /// `SIGB` — C2PA Ed25519 cryptographic signature block.
    Sig,
    /// `OBJM` — Semantic Object Manifest (BSON).
    ObjManifest,
    /// `PREV` — JPEG preview for backward compatibility with legacy viewers.
    ///
    /// Old systems that do not understand `.aFix` can extract this chunk and
    /// display the embedded JPEG directly.  New systems use it for instant
    /// first-frame display before the neural layers are decoded.
    Preview,
    /// Any other four-byte chunk ID.
    Unknown([u8; 4]),
}

impl ChunkId {
    /// Parse a `ChunkId` from four ASCII bytes.
    pub fn from_bytes(b: [u8; 4]) -> Self {
        match &b {
            b"META" => ChunkId::Meta,
            b"VEC_" => ChunkId::Vec,
            b"LAT_" => ChunkId::Lat,
            b"RES_" => ChunkId::Res,
            b"DPTH" => ChunkId::Depth,
            b"SIGB" => ChunkId::Sig,
            b"OBJM" => ChunkId::ObjManifest,
            b"PREV" => ChunkId::Preview,
            _ => ChunkId::Unknown(b),
        }
    }

    /// Serialise a `ChunkId` to four ASCII bytes.
    pub fn to_bytes(self) -> [u8; 4] {
        match self {
            ChunkId::Meta => *b"META",
            ChunkId::Vec => *b"VEC_",
            ChunkId::Lat => *b"LAT_",
            ChunkId::Res => *b"RES_",
            ChunkId::Depth => *b"DPTH",
            ChunkId::Sig => *b"SIGB",
            ChunkId::ObjManifest => *b"OBJM",
            ChunkId::Preview => *b"PREV",
            ChunkId::Unknown(b) => b,
        }
    }

    /// Human-readable name for display.
    pub fn name(self) -> &'static str {
        match self {
            ChunkId::Meta => "META",
            ChunkId::Vec => "VEC_",
            ChunkId::Lat => "LAT_",
            ChunkId::Res => "RES_",
            ChunkId::Depth => "DPTH",
            ChunkId::Sig => "SIGB",
            ChunkId::ObjManifest => "OBJM",
            ChunkId::Preview => "PREV",
            ChunkId::Unknown(_) => "????",
        }
    }
}

impl std::fmt::Display for ChunkId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// A single atom chunk inside the `.aFix` PAYLOAD.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Four-byte chunk identifier.
    pub id: ChunkId,
    /// Chunk flags (bit 0 = AES-256-GCM encrypted).
    pub flags: u16,
    /// Raw chunk data (already CRC-validated on read).
    pub data: Vec<u8>,
}

impl Chunk {
    /// Returns `true` if this chunk is marked as AES-256-GCM encrypted (flag bit 0).
    pub fn is_encrypted(&self) -> bool {
        self.flags & 0x0001 != 0
    }
}
