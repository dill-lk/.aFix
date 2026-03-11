//! Encoding profiles for `.aFix`.

/// An `.aFix` encoding profile defines which chunks are written.
///
/// | Profile      | S1 | S2 | S3 | DPTH | SIGB |
/// |--------------|----|----|----|----- |------|
/// | WebLossy     | ✓  | ✓  | ✗  | opt  | opt  |
/// | WebLossless  | ✓  | ✓  | ✓  | opt  | opt  |
/// | Spatial      | ✓  | ✓  | opt| ✓    | opt  |
/// | Trusted      | ✓  | ✓  | opt| opt  | ✓    |
/// | Full         | ✓  | ✓  | ✓  | ✓    | ✓    |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Profile {
    /// S1 + S2 only. Best file-size for consumer web.
    #[default]
    WebLossy,
    /// S1 + S2 + S3. Pixel-perfect for design & print.
    WebLossless,
    /// S1 + S2 + DPTH. Native depth for AR/VR.
    Spatial,
    /// S1 + S2 + SIGB. C2PA provenance for journalism/legal.
    Trusted,
    /// All chunks. Professional archival.
    Full,
}

impl Profile {
    /// Whether this profile requires the S3 Parity Residual chunk.
    pub fn requires_residual(self) -> bool {
        matches!(self, Profile::WebLossless | Profile::Full)
    }

    /// Whether this profile requires the DPTH depth-map chunk.
    pub fn requires_depth(self) -> bool {
        matches!(self, Profile::Spatial | Profile::Full)
    }

    /// Whether this profile requires the SIGB signature chunk.
    pub fn requires_signature(self) -> bool {
        matches!(self, Profile::Trusted | Profile::Full)
    }
}

impl std::fmt::Display for Profile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Profile::WebLossy => "web-lossy",
            Profile::WebLossless => "web-lossless",
            Profile::Spatial => "spatial",
            Profile::Trusted => "trusted",
            Profile::Full => "full",
        };
        write!(f, "{s}")
    }
}

impl std::str::FromStr for Profile {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "web-lossy" => Ok(Profile::WebLossy),
            "web-lossless" => Ok(Profile::WebLossless),
            "spatial" => Ok(Profile::Spatial),
            "trusted" => Ok(Profile::Trusted),
            "full" => Ok(Profile::Full),
            other => Err(format!("unknown profile '{other}'")),
        }
    }
}
