//! Qdrant vector database client wrapper.

use anyhow::{Context, Result};
use qdrant_client::prelude::*;
use qdrant_client::qdrant::PointStruct;
use serde_json::json;
use tracing::{debug, info};

use crate::types::SearchResult;

/// Qdrant vector store for similarity search
pub struct QdrantStore {
    client: QdrantClient,
    collection_name: String,
    vector_size: usize,
}

impl QdrantStore {
    /// Create a new Qdrant client
    ///
    /// # Arguments
    ///
    /// * `url` - Qdrant server URL (e.g., "http://localhost:6333")
    /// * `collection_name` - Name of the collection
    /// * `vector_size` - Dimensionality of vectors
    ///
    /// # Example
    ///
    /// ```ignore
    /// let store = QdrantStore::new("http://localhost:6333", "sam_detect", 512).await?;
    /// ```
    pub async fn new(url: &str, collection_name: String, vector_size: usize) -> Result<Self> {
        debug!("Connecting to Qdrant at: {}", url);

        let client = QdrantClient::from_url(url).build().context(
            "Failed to create Qdrant client - ensure Qdrant server is running",
        )?;

        let store = Self {
            client,
            collection_name: collection_name.clone(),
            vector_size,
        };

        // Ensure collection exists
        store.ensure_collection().await?;

        info!("Connected to Qdrant collection: {}", collection_name);
        Ok(store)
    }

    /// Ensure the collection exists, create if needed
    async fn ensure_collection(&self) -> Result<()> {
        let collections = self.client.get_collections().await?;

        let exists = collections
            .collections
            .iter()
            .any(|c| c.name == self.collection_name);

        if !exists {
            debug!(
                "Creating collection: {} with vector size: {}",
                self.collection_name, self.vector_size
            );

            self.client
                .create_collection(&self.collection_name, VectorsConfig {
                    config: Some(Config {
                        params: Some(Params {
                            size: self.vector_size as u64,
                            distance: Distance::Cosine.into(),
                            ..Default::default()
                        }),
                        ..Default::default()
                    }),
                })
                .await
                .context("Failed to create collection")?;

            info!("Created collection: {}", self.collection_name);
        }

        Ok(())
    }

    /// Search for similar vectors
    ///
    /// # Arguments
    ///
    /// * `embedding` - Query embedding vector
    /// * `top_k` - Number of nearest neighbors to return
    ///
    /// # Returns
    ///
    /// Vector of search results
    pub async fn search(&self, embedding: Vec<f32>, top_k: usize) -> Result<Vec<SearchResult>> {
        debug!("Searching for {} nearest neighbors", top_k);

        let results = self
            .client
            .search_points(&SearchPoints {
                collection_name: self.collection_name.clone(),
                vector: embedding,
                limit: top_k as u64,
                with_payload: Some(true.into()),
                ..Default::default()
            })
            .await
            .context("Failed to search in Qdrant")?;

        let search_results = results
            .result
            .iter()
            .map(|scored_point| {
                let label = scored_point
                    .payload
                    .get("label")
                    .and_then(|v| match v {
                        qdrant_client::qdrant::Value {
                            kind: Some(qdrant_client::qdrant::value::Kind::StringValue(s)),
                        } => Some(s.clone()),
                        _ => None,
                    })
                    .unwrap_or_else(|| "unknown".to_string());

                SearchResult {
                    id: scored_point.id.num.unwrap_or(0),
                    label,
                    score: scored_point.score,
                }
            })
            .collect();

        debug!("Found {} similar vectors", search_results.len());
        Ok(search_results)
    }

    /// Insert or update a vector with a label
    ///
    /// # Arguments
    ///
    /// * `id` - Unique ID for this vector
    /// * `embedding` - Vector embedding
    /// * `label` - Classification label
    pub async fn insert(&self, id: u64, embedding: Vec<f32>, label: String) -> Result<()> {
        debug!("Inserting vector {} with label: {}", id, label);

        let point = PointStruct {
            id: id.into(),
            vectors: embedding.into(),
            payload: {
                let mut map = std::collections::HashMap::new();
                map.insert(
                    "label".to_string(),
                    Value {
                        kind: Some(qdrant_client::qdrant::value::Kind::StringValue(label)),
                    },
                );
                map
            },
        };

        self.client
            .upsert_points(&self.collection_name, vec![point], None)
            .await
            .context("Failed to upsert point in Qdrant")?;

        Ok(())
    }

    /// Get collection statistics
    pub async fn get_stats(&self) -> Result<CollectionStats> {
        let collection = self
            .client
            .get_collection(&self.collection_name)
            .await
            .context("Failed to get collection info")?;

        Ok(CollectionStats {
            name: self.collection_name.clone(),
            points_count: collection.points_count as usize,
            vector_size: self.vector_size,
        })
    }

    /// Clear all vectors in the collection
    pub async fn clear(&self) -> Result<()> {
        self.client
            .delete_collection(&self.collection_name)
            .await
            .context("Failed to delete collection")?;

        self.ensure_collection().await?;

        Ok(())
    }
}

/// Collection statistics
#[derive(Debug, Clone)]
pub struct CollectionStats {
    pub name: String,
    pub points_count: usize,
    pub vector_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_connection_fails_without_server() {
        let result = QdrantStore::new("http://localhost:9999", "test".to_string(), 512).await;
        assert!(result.is_err());
    }
}
