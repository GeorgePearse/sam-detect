//! Model implementations for SAM2 and CLIP inference.

// TODO: Fix ONNX Runtime API compatibility issues
// pub mod sam2;
// pub mod clip;
pub mod preprocessing;

// pub use sam2::SAM2Model;
// pub use clip::CLIPModel;
pub use preprocessing::{preprocess_for_sam2, preprocess_for_clip};
