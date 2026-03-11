//! File header types for `.aFix`.

/// The `.aFix` file header (33 bytes before the ATOM_MAP).
#[derive(Debug, Clone, PartialEq)]
pub struct AfixHeader {
    /// Protocol version.
    pub version: Version,
    /// Logical image dimensions (resolution-independent).
    pub dimensions: Dimensions,
}

/// Protocol version packed into 4 bytes (`MAJOR.MINOR.PATCH.FLAG`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Version {
    pub major: u8,
    pub minor: u8,
    pub patch: u8,
    /// Reserved extension flags (compression envelope, encryption).
    pub flag: u8,
}

impl Version {
    /// The current protocol version (1.0.4, no flags).
    ///
    /// This matches the `Version` field in the `.aFix` specification (v1.0.4-B).
    /// The library crate version is tracked separately in `Cargo.toml`.
    pub fn current() -> Self {
        Version { major: 1, minor: 0, patch: 4, flag: 0 }
    }
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Logical image dimensions stored as IEEE 754 doubles (resolution-independent).
#[derive(Debug, Clone, PartialEq)]
pub struct Dimensions {
    /// Logical width (not bound to pixels).
    pub width: f64,
    /// Logical height (not bound to pixels).
    pub height: f64,
}
