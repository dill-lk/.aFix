//! Semantic Object Manifest (`OBJM` chunk) types.

use serde::{Deserialize, Serialize};

/// The Semantic Object Manifest stored in the `OBJM` chunk.
///
/// At encode time a segmentation model tags every significant region of the
/// image. The result is serialised as JSON (or BSON) inside the `OBJM` chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectManifest {
    /// Manifest schema version (currently `"1.0"`).
    pub version: String,
    /// List of detected semantic objects.
    pub objects: Vec<SemanticObject>,
}

impl ObjectManifest {
    /// Parse an `ObjectManifest` from the raw bytes of an `OBJM` chunk.
    pub fn from_chunk_data(data: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(data)
    }

    /// Serialise to JSON bytes suitable for storing in an `OBJM` chunk.
    pub fn to_chunk_data(&self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(self)
    }
}

/// A single semantic object within the image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticObject {
    /// Unique identifier within this file (e.g. `"sky"`, `"face_0"`).
    pub id: String,
    /// Human-readable label (e.g. `"sky"`, `"human_face"`).
    pub label: String,
    /// Object category: `"background"`, `"subject"`, or `"overlay"`.
    pub category: String,
    /// Run-length encoded bitmask covering this object's pixels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mask_rle: Option<String>,
    /// Axis-aligned bounding box `[x, y, width, height]` in logical pixels.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bbox: Option<[f64; 4]>,
    /// Segmentation model confidence in `[0, 1]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    /// Facial landmark positions (present for `human_face` objects only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub landmarks: Option<FaceLandmarks>,
}

/// Facial landmark pixel coordinates for a detected face.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceLandmarks {
    pub left_eye: [f64; 2],
    pub right_eye: [f64; 2],
    pub nose: [f64; 2],
    pub mouth: [f64; 2],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_manifest_json() {
        let manifest = ObjectManifest {
            version: "1.0".into(),
            objects: vec![
                SemanticObject {
                    id: "sky".into(),
                    label: "sky".into(),
                    category: "background".into(),
                    mask_rle: None,
                    bbox: Some([0.0, 0.0, 1920.0, 400.0]),
                    confidence: Some(0.97),
                    landmarks: None,
                },
                SemanticObject {
                    id: "face_0".into(),
                    label: "human_face".into(),
                    category: "subject".into(),
                    mask_rle: None,
                    bbox: Some([760.0, 200.0, 400.0, 500.0]),
                    confidence: Some(0.99),
                    landmarks: Some(FaceLandmarks {
                        left_eye: [860.0, 310.0],
                        right_eye: [960.0, 310.0],
                        nose: [910.0, 380.0],
                        mouth: [910.0, 450.0],
                    }),
                },
            ],
        };

        let data = manifest.to_chunk_data().unwrap();
        let parsed = ObjectManifest::from_chunk_data(&data).unwrap();
        assert_eq!(parsed.version, "1.0");
        assert_eq!(parsed.objects.len(), 2);
        assert_eq!(parsed.objects[0].id, "sky");
        assert_eq!(parsed.objects[1].id, "face_0");
        assert!(parsed.objects[1].landmarks.is_some());
    }
}
