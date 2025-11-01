//! Command-line interface for sam-detect Rust CLI.

use clap::Parser;
use std::path::PathBuf;

/// sam-detect Rust CLI - Fast instance segmentation with TensorRT
#[derive(Parser, Debug)]
#[command(
    name = "sam-detect-rs",
    about = "SAM2 + CLIP detection with TensorRT acceleration",
    version,
    author
)]
pub struct Cli {
    /// Image file(s) to process
    #[arg(required = true, value_name = "IMAGE_PATH")]
    pub images: Vec<PathBuf>,

    /// SAM2 ONNX model path
    #[arg(
        long,
        default_value = "models/sam2_base.onnx",
        value_name = "PATH"
    )]
    pub sam2_model: PathBuf,

    /// CLIP ONNX model path
    #[arg(
        long,
        default_value = "models/clip_vit_base.onnx",
        value_name = "PATH"
    )]
    pub clip_model: PathBuf,

    /// Qdrant server URL
    #[arg(
        long,
        default_value = "http://localhost:6333",
        value_name = "URL"
    )]
    pub qdrant_url: String,

    /// Qdrant collection name
    #[arg(long, default_value = "sam_detect", value_name = "NAME")]
    pub collection_name: String,

    /// Number of nearest neighbors to return
    #[arg(long, default_value = "5", value_name = "K")]
    pub top_k: usize,

    /// Output format
    #[arg(long, default_value = "text", value_name = "FORMAT")]
    pub format: OutputFormat,

    /// Add this label to the database before detection
    #[arg(long, value_name = "LABEL")]
    pub label: Option<String>,

    /// Verbose logging (can be repeated: -v, -vv, -vvv)
    #[arg(short, action = clap::ArgAction::Count)]
    pub verbose: u8,

    /// Skip Qdrant connection (memory-only mode)
    #[arg(long)]
    pub skip_qdrant: bool,
}

/// Output format selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    Json,
    Text,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "json" => Ok(OutputFormat::Json),
            "text" => Ok(OutputFormat::Text),
            _ => Err(format!("Unknown format: {}. Use 'json' or 'text'", s)),
        }
    }
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputFormat::Json => write!(f, "json"),
            OutputFormat::Text => write!(f, "text"),
        }
    }
}

/// Get tracing level from verbosity
pub fn get_log_level(verbose: u8) -> &'static str {
    match verbose {
        0 => "info",
        1 => "debug",
        _ => "trace",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_format_parsing() {
        assert_eq!("json".parse::<OutputFormat>().unwrap(), OutputFormat::Json);
        assert_eq!("text".parse::<OutputFormat>().unwrap(), OutputFormat::Text);
        assert!("invalid".parse::<OutputFormat>().is_err());
    }

    #[test]
    fn test_output_format_parsing_case_insensitive() {
        assert_eq!("JSON".parse::<OutputFormat>().unwrap(), OutputFormat::Json);
        assert_eq!("TEXT".parse::<OutputFormat>().unwrap(), OutputFormat::Text);
        assert_eq!("Json".parse::<OutputFormat>().unwrap(), OutputFormat::Json);
    }

    #[test]
    fn test_output_format_display() {
        assert_eq!(OutputFormat::Json.to_string(), "json");
        assert_eq!(OutputFormat::Text.to_string(), "text");
    }

    #[test]
    fn test_log_level() {
        assert_eq!(get_log_level(0), "info");
        assert_eq!(get_log_level(1), "debug");
        assert_eq!(get_log_level(2), "trace");
        assert_eq!(get_log_level(3), "trace");
        assert_eq!(get_log_level(255), "trace");
    }

    #[test]
    fn test_output_format_equality() {
        let json1 = OutputFormat::Json;
        let json2 = OutputFormat::Json;
        let text = OutputFormat::Text;

        assert_eq!(json1, json2);
        assert_ne!(json1, text);
    }

    #[test]
    fn test_output_format_clone_copy() {
        let fmt = OutputFormat::Json;
        let fmt2 = fmt; // Copy trait
        assert_eq!(fmt, fmt2);
    }

    #[test]
    fn test_output_format_parsing_empty_string() {
        assert!("".parse::<OutputFormat>().is_err());
    }

    #[test]
    fn test_output_format_parsing_partial_match() {
        assert!("jso".parse::<OutputFormat>().is_err());
        assert!("js".parse::<OutputFormat>().is_err());
    }
}
