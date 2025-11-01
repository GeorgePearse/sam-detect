//! sam-detect Rust CLI - Fast instance segmentation with SAM2 + CLIP + TensorRT

use anyhow::Result;
use clap::Parser;
use serde_json::json;

mod cli;
mod models;
mod pipeline;
mod types;
mod vector_store;

use cli::{get_log_level, Cli, OutputFormat};
use pipeline::DetectionPipeline;

#[tokio::main]
async fn main() -> Result<()> {
    // Parse CLI arguments
    let cli = Cli::parse();

    // Initialize logging
    let log_level = get_log_level(cli.verbose);
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(log_level.parse().unwrap_or_else(|_| "info".parse().unwrap())),
        )
        .init();

    tracing::info!("sam-detect Rust CLI started");

    // Initialize pipeline
    let qdrant_url = if cli.skip_qdrant {
        None
    } else {
        Some(cli.qdrant_url.as_str())
    };

    let pipeline = DetectionPipeline::new(
        cli.sam2_model.to_str().unwrap_or("models/sam2_base.onnx"),
        cli.clip_model.to_str().unwrap_or("models/clip_vit_base.onnx"),
        qdrant_url,
        &cli.collection_name,
    )
    .await?;

    // Add example if requested
    if let Some(label) = &cli.label {
        if let Some(image_path) = cli.images.first() {
            pipeline
                .add_example(image_path.to_str().unwrap_or(""), label)
                .await?;
        }
    }

    // Process each image
    for image_path in &cli.images {
        let image_str = image_path.to_str().unwrap_or("unknown");
        tracing::info!("Processing: {}", image_str);

        let detections = pipeline.detect(image_str, cli.top_k).await?;

        // Output results
        match cli.format {
            OutputFormat::Json => {
                let json = json!({
                    "image": image_str,
                    "detections": detections.iter().map(|d| {
                        json!({
                            "label": d.label,
                            "confidence": d.confidence,
                            "bbox": d.bbox.map(|b| {
                                json!({
                                    "x1": b.x1,
                                    "y1": b.y1,
                                    "x2": b.x2,
                                    "y2": b.y2
                                })
                            }),
                            "pixel_count": d.mask.pixel_count(),
                            "matches": d.matches.iter().map(|m| {
                                json!({
                                    "label": m.label,
                                    "score": m.score
                                })
                            }).collect::<Vec<_>>()
                        })
                    }).collect::<Vec<_>>()
                });
                println!("{}", serde_json::to_string_pretty(&json)?);
            }
            OutputFormat::Text => {
                println!("Image: {}", image_str);
                println!("Detections: {}", detections.len());
                for (i, detection) in detections.iter().enumerate() {
                    println!("  Detection {}:", i + 1);
                    println!("    Label: {:?}", detection.label);
                    println!("    Confidence: {:.2}", detection.confidence);
                    if let Some(bbox) = detection.bbox {
                        println!(
                            "    BBox: ({:.0}, {:.0}, {:.0}, {:.0})",
                            bbox.x1, bbox.y1, bbox.x2, bbox.y2
                        );
                    }
                    println!("    Pixels: {}", detection.mask.pixel_count());
                    if !detection.matches.is_empty() {
                        println!("    Similar examples:");
                        for m in detection.matches.iter().take(3) {
                            println!("      - {} ({:.2})", m.label, m.score);
                        }
                    }
                }
            }
        }
    }

    // Print stats
    let stats = pipeline.get_stats().await?;
    tracing::info!("Pipeline stats: {}", stats);

    tracing::info!("Detection complete");
    Ok(())
}
