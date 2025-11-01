//! CLIP model inference with TensorRT acceleration via ONNX Runtime.

use anyhow::{Context, Result};
use ndarray::Array4;
use ort::{GraphOptimizationLevel, Session, SessionBuilder};
use tracing::debug;

/// CLIP ViT-Base-32 model for image embeddings
pub struct CLIPModel {
    session: Session,
    embedding_dim: usize,
}

impl CLIPModel {
    /// Create a new CLIP model from ONNX file
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the ONNX model file
    ///
    /// # Example
    ///
    /// ```ignore
    /// let model = CLIPModel::new("models/clip_vit_base.onnx")?;
    /// ```
    pub fn new(model_path: &str) -> Result<Self> {
        Self::with_embedding_dim(model_path, 512)
    }

    /// Create a new CLIP model with specified embedding dimension
    pub fn with_embedding_dim(model_path: &str, embedding_dim: usize) -> Result<Self> {
        debug!("Loading CLIP model from: {}", model_path);

        let session = SessionBuilder::new()
            .context("Failed to create ONNX Runtime session builder")?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .context("Failed to set optimization level")?
            .with_intra_threads(4)
            .context("Failed to set intra threads")?
            .commit_from_file(model_path)
            .context("Failed to load ONNX model")?;

        debug!("CLIP model loaded successfully");
        Ok(Self {
            session,
            embedding_dim,
        })
    }

    /// Generate embedding for an image
    ///
    /// # Arguments
    ///
    /// * `image` - Input image as [1, 3, 224, 224] normalized tensor
    ///
    /// # Returns
    ///
    /// Vector of floating point embedding values
    pub fn embed(&self, image: Array4<f32>) -> Result<Vec<f32>> {
        debug!(
            "Running CLIP embedding on image of shape {:?}",
            image.shape()
        );

        // Run inference
        let outputs = self.session.run(ort::inputs!["pixel_values" => image.view()]?)?;

        // Extract embeddings
        let embeddings = outputs["embeddings"].try_extract_array::<f32>()?;

        // Convert to Vec<f32>
        // CLIP outputs [1, embedding_dim], we want just the embedding vector
        let embedding_vec: Vec<f32> = embeddings
            .iter()
            .copied()
            .collect::<Vec<f32>>();

        debug!(
            "Generated embedding of size {} (expected {})",
            embedding_vec.len(),
            self.embedding_dim
        );

        Ok(embedding_vec)
    }

    /// Get the embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Normalize an embedding to unit length
    pub fn normalize_embedding(embedding: &[f32]) -> Vec<f32> {
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm == 0.0 {
            embedding.to_vec()
        } else {
            embedding.iter().map(|x| x / norm).collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation_fails_with_missing_file() {
        let result = CLIPModel::new("nonexistent_model.onnx");
        assert!(result.is_err());
    }

    #[test]
    fn test_normalize_embedding() {
        let embedding = vec![3.0, 4.0];
        let normalized = CLIPModel::normalize_embedding(&embedding);
        assert!((normalized[0] - 0.6).abs() < 0.001);
        assert!((normalized[1] - 0.8).abs() < 0.001);
    }
}
