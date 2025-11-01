#!/usr/bin/env python3
"""Generate performance visualization plots for sam-detect.

Generates dummy data showing speed vs accuracy trade-offs across different
configurations, with both static (SVG) and interactive (HTML) outputs.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def generate_dummy_data() -> list[dict]:
    """Generate dummy performance data for different configurations.

    Returns:
        List of configuration data with speed and accuracy metrics
    """
    return [
        {
            "name": "Naive Segmenter\n+ Avg Color",
            "speed_ms": 15,
            "accuracy": 28,
            "group": "Naive Baseline",
            "color": "#1f77b4",
        },
        {
            "name": "Naive Segmenter\n+ CLIP",
            "speed_ms": 155,
            "accuracy": 42,
            "group": "Naive + Better Embeddings",
            "color": "#1f77b4",
        },
        {
            "name": "SAM2 Small\n+ Avg Color",
            "speed_ms": 95,
            "accuracy": 58,
            "group": "SAM2 Small",
            "color": "#2ca02c",
        },
        {
            "name": "SAM2 Small\n+ CLIP",
            "speed_ms": 245,
            "accuracy": 72,
            "group": "SAM2 Small + CLIP",
            "color": "#2ca02c",
        },
        {
            "name": "SAM2 Base\n+ CLIP",
            "speed_ms": 410,
            "accuracy": 81,
            "group": "SAM2 Base",
            "color": "#ff7f0e",
        },
        {
            "name": "SAM2 Large\n+ CLIP",
            "speed_ms": 620,
            "accuracy": 86,
            "group": "SAM2 Large",
            "color": "#d62728",
        },
    ]


def create_matplotlib_plot(data: list[dict], output_path: Path) -> None:
    """Create static scatter plot using matplotlib and save as SVG.

    Args:
        data: List of configuration data
        output_path: Path to save SVG file
    """
    plt.figure(figsize=(12, 8))

    # Set style
    sns.set_style("whitegrid")
    sns.set_palette("husl")

    # Extract data
    speeds = [d["speed_ms"] for d in data]
    accuracies = [d["accuracy"] for d in data]
    colors = [d["color"] for d in data]
    names = [d["name"] for d in data]

    # Create scatter plot
    plt.scatter(
        speeds,
        accuracies,
        s=300,
        c=colors,
        alpha=0.6,
        edgecolors="black",
        linewidth=1.5,
    )

    # Add labels for each point
    for i, (speed, accuracy, name) in enumerate(zip(speeds, accuracies, names)):
        offset_x = 20 if i % 2 == 0 else -20
        plt.annotate(
            name,
            (speed, accuracy),
            textcoords="offset points",
            xytext=(offset_x, 10),
            ha="center" if i % 2 == 0 else "right",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.1),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", lw=0.5),
        )

    # Labels and title
    plt.xlabel("Inference Time per Image (ms)", fontsize=12, fontweight="bold")
    plt.ylabel("Detection Accuracy / mAP (%)", fontsize=12, fontweight="bold")
    plt.title(
        "sam-detect: Speed vs Accuracy Trade-off Across Configurations",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Grid
    plt.grid(True, alpha=0.3)

    # Set axis limits with some padding
    plt.xlim(-50, max(speeds) + 100)
    plt.ylim(15, 95)

    # Add annotation explaining the plot
    plt.text(
        0.02,
        0.02,
        (
            "Trade-off visualization: Faster methods (left) sacrifice accuracy for speed;\n"
            "More accurate methods (right) require longer inference times.\n"
            "Dummy data for illustration purposes."
        ),
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Save as SVG
    plt.tight_layout()
    plt.savefig(output_path, format="svg", dpi=100, bbox_inches="tight")
    print(f"✓ Saved SVG plot to {output_path}")
    plt.close()


def create_plotly_plot(data: list[dict], output_path: Path) -> None:
    """Create interactive scatter plot using plotly.

    Args:
        data: List of configuration data
        output_path: Path to save HTML file
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("⚠️  Plotly not installed. Skipping interactive HTML plot.")
        print("   Install with: pip install 'sam-detect[viz]'")
        return

    # Prepare data for plotly
    groups = [d["group"] for d in data]

    # Create figure
    fig = go.Figure()

    # Add scatter traces grouped by color for legend
    unique_groups = list(dict.fromkeys(groups))  # Preserve order

    for group in unique_groups:
        group_data = [d for d in data if d["group"] == group]
        group_speeds = [d["speed_ms"] for d in group_data]
        group_accuracies = [d["accuracy"] for d in group_data]
        group_names = [d["name"] for d in group_data]
        group_colors = [d["color"] for d in group_data]

        fig.add_trace(
            go.Scatter(
                x=group_speeds,
                y=group_accuracies,
                mode="markers",
                name=group,
                marker=dict(
                    size=12,
                    color=group_colors[0],
                    opacity=0.7,
                    line=dict(color="black", width=1),
                ),
                text=group_names,
                hovertemplate="<b>%{text}</b><br>Speed: %{x:.0f}ms<br>Accuracy: %{y:.0f}%<extra></extra>",
            )
        )

    # Update layout
    fig.update_layout(
        title={
            "text": "sam-detect: Speed vs Accuracy Trade-off<br><sub>Dummy data for illustration</sub>",
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title="Inference Time per Image (ms)",
        yaxis_title="Detection Accuracy / mAP (%)",
        hovermode="closest",
        template="plotly_white",
        width=1000,
        height=700,
        font=dict(size=12),
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
    )

    # Set axis ranges
    max_speed = max(d["speed_ms"] for d in data)
    fig.update_xaxes(range=[-50, max_speed + 100])
    fig.update_yaxes(range=[15, 95])

    # Save as HTML
    fig.write_html(str(output_path))
    print(f"✓ Saved interactive HTML plot to {output_path}")


def main() -> None:
    """Generate all performance plots."""
    # Get output paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    docs_dir = project_root / "docs" / "images"
    docs_dir.mkdir(parents=True, exist_ok=True)

    svg_output = docs_dir / "latency_vs_accuracy.svg"
    html_output = docs_dir / "latency_vs_accuracy.html"

    # Generate data
    print("Generating performance data...")
    data = generate_dummy_data()
    print(f"  {len(data)} configurations")

    # Create visualizations
    print("\nGenerating visualizations...")
    create_matplotlib_plot(data, svg_output)
    create_plotly_plot(data, html_output)

    print("\n✅ Performance plots generated successfully!")
    print("\nFiles created:")
    print(f"  - {svg_output.relative_to(project_root)}")
    print(f"  - {html_output.relative_to(project_root)}")


if __name__ == "__main__":
    main()
