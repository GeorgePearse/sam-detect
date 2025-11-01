#![doc = include_str!("../README.md")]

pub mod cli;
pub mod models;
pub mod pipeline;
pub mod types;
pub mod vector_store;

pub use pipeline::DetectionPipeline;
pub use types::{Detection, Mask, SearchResult};
