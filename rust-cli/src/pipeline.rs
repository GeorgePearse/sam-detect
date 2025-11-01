//! Detection pipeline orchestrating SAM2 + CLIP + Qdrant.

use anyhow::{Context, Result};
use std::path::Path;
use tracing::{debug, info};

use crate::models::{SAM2Model, CLIPModel, preprocess_for_clip, preprocess_for_sam2};
use crate::types::Detection;
use crate::vector_store::QdrantStore;

/// Complete detection pipeline
pub struct DetectionPipeline {
    sam2: SAM2Model,
    clip: CLIPModel,
    vector_store: Option<QdrantStore>,
}

impl DetectionPipeline {
    /// Create a new detection pipeline
    ///
    /// # Arguments
    ///
    /// * `sam2_model_path` - Path to SAM2 ONNX model
    /// * `clip_model_path` - Path to CLIP ONNX model
    /// * `qdrant_url` - Qdrant server URL (None to skip)
    /// * `collection_name` - Qdrant collection name
    ///
    /// # Example
    ///
    /// ```ignore
    /// let pipeline = DetectionPipeline::new(
    ///     "models/sam2_base.onnx",
    ///     "models/clip_vit_base.onnx",
    ///     Some("http://localhost:6333"),
    ///     "sam_detect"
    /// ).await?;
    /// ```
    pub async fn new(
        sam2_model_path: &str,
        clip_model_path: &str,
        qdrant_url: Option<&str>,
        collection_name: &str,
    ) -> Result<Self> {
        info!("Initializing detection pipeline");

        // Load models
        let sam2 = SAM2Model::new(sam2_model_path).context("Failed to load SAM2 model")?;
        let clip = CLIPModel::new(clip_model_path).context("Failed to load CLIP model")?;

        // Initialize vector store if URL provided
        let vector_store = if let Some(url) = qdrant_url {
            Some(
                QdrantStore::new(url, collection_name.to_string(), clip.embedding_dim())
                    .await
                    .context("Failed to connect to Qdrant")?,
            )
        } else {
            debug!("Qdrant not configured, operating in memory-only mode");
            None
        };

        info!("Detection pipeline initialized successfully");
        Ok(Self {
            sam2,
            clip,
            vector_store,
        })
    }

    /// Run detection on an image
    ///
    /// # Arguments
    ///
    /// * `image_path` - Path to input image
    /// * `top_k` - Number of nearest neighbors to retrieve
    ///
    /// # Returns
    ///
    /// Vector of detections
    pub async fn detect(&self, image_path: &str, top_k: usize) -> Result<Vec<Detection>> {
        info!("Running detection on: {}", image_path);

        // Load image
        let image =
            image::open(image_path).context(format!("Failed to load image: {}", image_path))?;

        debug!("Image loaded: {}x{}", image.width(), image.height());

        // Run SAM2 segmentation
        let preprocessed = preprocess_for_sam2(&image)?;
        let masks = self.sam2.segment(preprocessed)?;

        debug!("Detected {} object(s)", masks.len());

        // For each mask, embed and search
        let mut detections = Vec::new();

        for mask in masks {
            let mut detection = Detection::new(mask);

            // Skip embedding if we don't have a vector store
            if self.vector_store.is_none() {
                detections.push(detection);
                continue;
            }

            // Extract crop from image using bbox
            if let Some(bbox) = detection.bbox {
                let crop = crate::models::preprocessing::crop_image(
                    &image,
                    bbox.x1 as u32,
                    bbox.y1 as u32,
                    bbox.x2 as u32,
                    bbox.y2 as u32,
                )?;

                // Preprocess for CLIP
                let preprocessed = preprocess_for_clip(&crop)?;

                // Generate embedding
                let embedding = self.clip.embed(preprocessed)?;

                // Search in Qdrant
                if let Some(ref store) = self.vector_store {
                    let matches = store.search(embedding, top_k).await?;
                    detection.set_classification(matches);
                }
            }

            detections.push(detection);
        }

        info!("Detection complete: {} objects", detections.len());
        Ok(detections)
    }

    /// Add an example to the vector database
    ///
    /// # Arguments
    ///
    /// * `image_path` - Path to example image
    /// * `label` - Classification label
    pub async fn add_example(&self, image_path: &str, label: &str) -> Result<()> {
        let store = self
            .vector_store
            .as_ref()
            .context("Vector store not initialized")?;

        let image = image::open(image_path)
            .context(format!("Failed to load image: {}", image_path))?;

        let preprocessed = preprocess_for_clip(&image)?;
        let embedding = self.clip.embed(preprocessed)?;

        // Use image filename hash as ID
        let id = fxhash::hash64(&image_path) as u64;

        store.insert(id, embedding, label.to_string()).await?;

        info!(
            "Added example: {} with label: {}",
            image_path, label
        );

        Ok(())
    }

    /// Get vector store statistics
    pub async fn get_stats(&self) -> Result<String> {
        if let Some(store) = &self.vector_store {
            let stats = store.get_stats().await?;
            Ok(format!(
                "Collection: {}, Points: {}, Vector Size: {}",
                stats.name, stats.points_count, stats.vector_size
            ))
        } else {
            Ok("Memory-only mode (no Qdrant)".to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pipeline_creation_fails_with_missing_models() {
        let result = DetectionPipeline::new(
            "nonexistent.onnx",
            "nonexistent.onnx",
            None,
            "test",
        )
        .await;
        assert!(result.is_err());
    }
}
