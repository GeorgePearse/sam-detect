//! Shared type definitions for sam-detect Rust CLI.

use serde::{Deserialize, Serialize};

/// Bounding box in format (x1, y1, x2, y2)
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

/// Binary segmentation mask
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mask {
    pub width: u32,
    pub height: u32,
    pub data: Vec<bool>, // Flattened row-major format
}

impl Mask {
    /// Create a new mask
    pub fn new(width: u32, height: u32, data: Vec<bool>) -> Self {
        assert_eq!(
            data.len(),
            (width * height) as usize,
            "Mask data size must match width * height"
        );
        Self { width, height, data }
    }

    /// Get mask value at (x, y)
    pub fn get(&self, x: u32, y: u32) -> bool {
        if x >= self.width || y >= self.height {
            return false;
        }
        self.data[(y * self.width + x) as usize]
    }

    /// Get bounding box from mask
    pub fn bbox(&self) -> Option<BBox> {
        let mut min_x = self.width;
        let mut max_x = 0;
        let mut min_y = self.height;
        let mut max_y = 0;
        let mut found = false;

        for y in 0..self.height {
            for x in 0..self.width {
                if self.get(x, y) {
                    found = true;
                    min_x = min_x.min(x);
                    max_x = max_x.max(x);
                    min_y = min_y.min(y);
                    max_y = max_y.max(y);
                }
            }
        }

        if found {
            Some(BBox {
                x1: min_x as f32,
                y1: min_y as f32,
                x2: (max_x + 1) as f32,
                y2: (max_y + 1) as f32,
            })
        } else {
            None
        }
    }

    /// Get number of true pixels
    pub fn pixel_count(&self) -> usize {
        self.data.iter().filter(|&&b| b).count()
    }
}

/// Vector search result from Qdrant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: u64,
    pub label: String,
    pub score: f32,
}

/// Complete detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    pub mask: Mask,
    pub bbox: Option<BBox>,
    pub label: Option<String>,
    pub confidence: f32,
    pub matches: Vec<SearchResult>,
}

impl Detection {
    /// Create a new detection
    pub fn new(mask: Mask) -> Self {
        let bbox = mask.bbox();
        Self {
            mask,
            bbox,
            label: None,
            confidence: 0.0,
            matches: Vec::new(),
        }
    }

    /// Set the label and confidence from search results
    pub fn set_classification(&mut self, matches: Vec<SearchResult>) {
        if let Some(top_match) = matches.first() {
            self.label = Some(top_match.label.clone());
            self.confidence = top_match.score;
        }
        self.matches = matches;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_creation() {
        let data = vec![true, false, true, false];
        let mask = Mask::new(2, 2, data);
        assert_eq!(mask.get(0, 0), true);
        assert_eq!(mask.get(1, 0), false);
    }

    #[test]
    fn test_mask_out_of_bounds() {
        let data = vec![true, false, true, false];
        let mask = Mask::new(2, 2, data);
        // Out of bounds should return false
        assert_eq!(mask.get(10, 10), false);
        assert_eq!(mask.get(2, 2), false);
    }

    #[test]
    fn test_mask_bbox() {
        let mut data = vec![false; 9];
        // Mark a 3x3 region with a 2x2 box in the center
        data[4] = true;
        data[5] = true;
        data[7] = true;
        data[8] = true;

        let mask = Mask::new(3, 3, data);
        if let Some(bbox) = mask.bbox() {
            assert_eq!(bbox.x1, 1.0);
            assert_eq!(bbox.y1, 1.0);
            assert_eq!(bbox.x2, 3.0);
            assert_eq!(bbox.y2, 3.0);
        } else {
            panic!("Expected bbox");
        }
    }

    #[test]
    fn test_mask_bbox_empty() {
        let data = vec![false; 9];
        let mask = Mask::new(3, 3, data);
        // Empty mask should return None
        assert_eq!(mask.bbox(), None);
    }

    #[test]
    fn test_mask_bbox_full() {
        let data = vec![true; 9];
        let mask = Mask::new(3, 3, data);
        if let Some(bbox) = mask.bbox() {
            assert_eq!(bbox.x1, 0.0);
            assert_eq!(bbox.y1, 0.0);
            assert_eq!(bbox.x2, 3.0);
            assert_eq!(bbox.y2, 3.0);
        } else {
            panic!("Expected bbox");
        }
    }

    #[test]
    fn test_mask_pixel_count() {
        let data = vec![true, false, true, false];
        let mask = Mask::new(2, 2, data);
        assert_eq!(mask.pixel_count(), 2);
    }

    #[test]
    fn test_mask_pixel_count_empty() {
        let data = vec![false; 4];
        let mask = Mask::new(2, 2, data);
        assert_eq!(mask.pixel_count(), 0);
    }

    #[test]
    fn test_detection_creation() {
        let data = vec![true, false, true, false];
        let mask = Mask::new(2, 2, data);
        let detection = Detection::new(mask);
        assert_eq!(detection.label, None);
        assert_eq!(detection.confidence, 0.0);
        assert_eq!(detection.matches.len(), 0);
    }

    #[test]
    fn test_detection_set_classification() {
        let data = vec![true; 4];
        let mask = Mask::new(2, 2, data);
        let mut detection = Detection::new(mask);

        let matches = vec![SearchResult {
            id: 1,
            label: "person".to_string(),
            score: 0.95,
        }];

        detection.set_classification(matches);
        assert_eq!(detection.label, Some("person".to_string()));
        assert_eq!(detection.confidence, 0.95);
        assert_eq!(detection.matches.len(), 1);
    }

    #[test]
    fn test_bbox_creation() {
        let bbox = BBox {
            x1: 10.0,
            y1: 20.0,
            x2: 30.0,
            y2: 40.0,
        };
        assert_eq!(bbox.x1, 10.0);
        assert_eq!(bbox.x2, 30.0);
    }

    #[test]
    fn test_search_result_creation() {
        let result = SearchResult {
            id: 42,
            label: "cat".to_string(),
            score: 0.87,
        };
        assert_eq!(result.id, 42);
        assert_eq!(result.label, "cat");
        assert_eq!(result.score, 0.87);
    }
}
