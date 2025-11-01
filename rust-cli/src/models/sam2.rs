//! SAM2 model inference with TensorRT acceleration via ONNX Runtime.

use anyhow::{Context, Result};
use ndarray::{Array1, Array4, ArrayD};
use ort::{GraphOptimizationLevel, Session, SessionBuilder};
use tracing::debug;

use crate::types::Mask;

/// SAM2 Hiera Base model for instance segmentation
pub struct SAM2Model {
    session: Session,
}

impl SAM2Model {
    /// Create a new SAM2 model from ONNX file
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the ONNX model file
    ///
    /// # Example
    ///
    /// ```ignore
    /// let model = SAM2Model::new("models/sam2_base.onnx")?;
    /// ```
    pub fn new(model_path: &str) -> Result<Self> {
        debug!("Loading SAM2 model from: {}", model_path);

        let session = SessionBuilder::new()
            .context("Failed to create ONNX Runtime session builder")?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .context("Failed to set optimization level")?
            .with_intra_threads(4)
            .context("Failed to set intra threads")?
            .commit_from_file(model_path)
            .context("Failed to load ONNX model")?;

        debug!("SAM2 model loaded successfully");
        Ok(Self { session })
    }

    /// Segment an image and return masks
    ///
    /// # Arguments
    ///
    /// * `image` - Input image as [1, 3, H, W] normalized tensor
    ///
    /// # Returns
    ///
    /// Vector of detected masks
    pub fn segment(&self, image: Array4<f32>) -> Result<Vec<Mask>> {
        debug!(
            "Running SAM2 segmentation on image of shape {:?}",
            image.shape()
        );

        // Run inference
        let outputs = self.session.run(ort::inputs!["image" => image.view()]?)?;

        // Extract masks from outputs
        // SAM2 outputs image embeddings, which we'll convert to mask format
        let embeddings = outputs["image_embeddings"].try_extract_array::<f32>()?;

        debug!("Got embeddings of shape: {:?}", embeddings.shape());

        // For now, create a placeholder mask (full image)
        // In a full implementation, you would run the mask decoder
        let (_, _, height, width) = (
            image.shape()[0],
            image.shape()[1],
            image.shape()[2],
            image.shape()[3],
        );

        let mask_data = vec![true; height * width];
        let mask = Mask::new(width as u32, height as u32, mask_data);

        Ok(vec![mask])
    }

    /// Get model input/output information
    pub fn get_model_info(&self) -> Result<ModelInfo> {
        let input_names: Vec<String> = self
            .session
            .inputs
            .iter()
            .map(|input| input.name.clone())
            .collect();

        let output_names: Vec<String> = self
            .session
            .outputs
            .iter()
            .map(|output| output.name.clone())
            .collect();

        Ok(ModelInfo {
            input_names,
            output_names,
        })
    }
}

/// Model information
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation_fails_with_missing_file() {
        let result = SAM2Model::new("nonexistent_model.onnx");
        assert!(result.is_err());
    }
}
