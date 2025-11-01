//! Image preprocessing utilities for SAM2 and CLIP models.

use anyhow::{Context, Result};
use image::DynamicImage;
use ndarray::{Array2, Array4, s};
use tracing::debug;

/// Preprocess image for SAM2 model (1024x1024)
///
/// # Arguments
///
/// * `image` - Input DynamicImage
///
/// # Returns
///
/// Array of shape [1, 3, 1024, 1024] with normalized values
pub fn preprocess_for_sam2(image: &DynamicImage) -> Result<Array4<f32>> {
    debug!(
        "Preprocessing image for SAM2: {}x{}",
        image.width(),
        image.height()
    );

    // Resize to 1024x1024
    let resized = image.resize_exact(1024, 1024, image::imageops::FilterType::Lanczos3);

    // Convert to RGB
    let rgb = resized.to_rgb8();

    // Convert to ndarray [1, 3, 1024, 1024]
    let mut array = Array4::<f32>::zeros((1, 3, 1024, 1024));

    for (x, y, pixel) in rgb.enumerate_pixels() {
        let x = x as usize;
        let y = y as usize;
        array[[0, 0, y, x]] = pixel[0] as f32 / 255.0;
        array[[0, 1, y, x]] = pixel[1] as f32 / 255.0;
        array[[0, 2, y, x]] = pixel[2] as f32 / 255.0;
    }

    debug!("SAM2 preprocessing complete");
    Ok(array)
}

/// Preprocess image for CLIP model (224x224)
///
/// # Arguments
///
/// * `image` - Input DynamicImage
///
/// # Returns
///
/// Array of shape [1, 3, 224, 224] with CLIP-normalized values
pub fn preprocess_for_clip(image: &DynamicImage) -> Result<Array4<f32>> {
    debug!(
        "Preprocessing image for CLIP: {}x{}",
        image.width(),
        image.height()
    );

    // Resize to 224x224
    let resized = image.resize_exact(224, 224, image::imageops::FilterType::Lanczos3);

    // Convert to RGB
    let rgb = resized.to_rgb8();

    // CLIP normalization parameters (from ImageNet)
    let mean = [0.48145466, 0.4578275, 0.40821073];
    let std = [0.26862954, 0.26130258, 0.27577711];

    // Convert to ndarray [1, 3, 224, 224]
    let mut array = Array4::<f32>::zeros((1, 3, 224, 224));

    for (x, y, pixel) in rgb.enumerate_pixels() {
        let x = x as usize;
        let y = y as usize;
        for c in 0..3 {
            let normalized = (pixel[c] as f32 / 255.0 - mean[c]) / std[c];
            array[[0, c, y, x]] = normalized;
        }
    }

    debug!("CLIP preprocessing complete");
    Ok(array)
}

/// Crop an image region based on bounding box coordinates
///
/// # Arguments
///
/// * `image` - Input image
/// * `x1` - Left coordinate
/// * `y1` - Top coordinate
/// * `x2` - Right coordinate
/// * `y2` - Bottom coordinate
///
/// # Returns
///
/// Cropped DynamicImage
pub fn crop_image(
    image: &DynamicImage,
    x1: u32,
    y1: u32,
    x2: u32,
    y2: u32,
) -> Result<DynamicImage> {
    let width = x2 - x1;
    let height = y2 - y1;

    let cropped = image.crop_imm(x1, y1, width, height);

    Ok(cropped)
}

/// Preprocess cropped image region for CLIP embedding
///
/// # Arguments
///
/// * `image` - Cropped image region
///
/// # Returns
///
/// Preprocessed array for CLIP
pub fn preprocess_crop_for_clip(image: &DynamicImage) -> Result<Array4<f32>> {
    preprocess_for_clip(image)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_image() -> DynamicImage {
        // Create a simple 100x100 RGB image with red color
        let img = image::ImageBuffer::from_fn(100, 100, |_x, _y| {
            image::Rgb([255u8, 0u8, 0u8])
        });
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn test_preprocess_for_sam2() {
        let img = create_test_image();
        let result = preprocess_for_sam2(&img);
        assert!(result.is_ok());
        let array = result.unwrap();
        assert_eq!(array.shape(), &[1, 3, 1024, 1024]);
    }

    #[test]
    fn test_preprocess_for_clip() {
        let img = create_test_image();
        let result = preprocess_for_clip(&img);
        assert!(result.is_ok());
        let array = result.unwrap();
        assert_eq!(array.shape(), &[1, 3, 224, 224]);
    }

    #[test]
    fn test_crop_image() {
        let img = create_test_image();
        let cropped = crop_image(&img, 10, 10, 50, 50);
        assert!(cropped.is_ok());
        let cropped = cropped.unwrap();
        assert_eq!(cropped.width(), 40);
        assert_eq!(cropped.height(), 40);
    }

    #[test]
    fn test_crop_image_full() {
        let img = create_test_image();
        // Crop entire image
        let cropped = crop_image(&img, 0, 0, 100, 100);
        assert!(cropped.is_ok());
        let cropped = cropped.unwrap();
        assert_eq!(cropped.width(), 100);
        assert_eq!(cropped.height(), 100);
    }

    #[test]
    fn test_crop_image_corner() {
        let img = create_test_image();
        // Crop just one pixel corner
        let cropped = crop_image(&img, 0, 0, 1, 1);
        assert!(cropped.is_ok());
        let cropped = cropped.unwrap();
        assert_eq!(cropped.width(), 1);
        assert_eq!(cropped.height(), 1);
    }

    #[test]
    fn test_preprocess_crop_for_clip() {
        let img = create_test_image();
        let cropped = crop_image(&img, 10, 10, 50, 50).unwrap();
        let result = preprocess_crop_for_clip(&cropped);
        assert!(result.is_ok());
        let array = result.unwrap();
        assert_eq!(array.shape(), &[1, 3, 224, 224]);
    }

    #[test]
    fn test_sam2_normalization_range() {
        let img = create_test_image();
        let array = preprocess_for_sam2(&img).unwrap();
        // Values should be in [0, 1] range
        let min = array.iter().copied().fold(f32::INFINITY, f32::min);
        let max = array.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(min >= 0.0);
        assert!(max <= 1.0);
    }

    #[test]
    fn test_clip_output_shape() {
        let img = create_test_image();
        let array = preprocess_for_clip(&img).unwrap();
        // Should be [1, 3, 224, 224] with batch size 1
        assert_eq!(array[[0, 0, 0, 0]].is_finite(), true);
    }
}
