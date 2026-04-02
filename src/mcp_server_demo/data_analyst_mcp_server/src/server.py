#!/usr/bin/env python3
"""
Data Analyst MCP Server
Provides t-SNE analysis and visualization for LeRobot datasets
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pyarrow.parquet as pq
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import anyio
import torch
import torch.nn as nn

from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("data_analyst_mcp_server")

# Global cache for analysis results
analysis_cache = {}


class SequenceEncoder(nn.Module):
    """Simple LSTM-based encoder for sequence-to-vector conversion"""

    def __init__(self, input_size: int, hidden_size: int = 64, latent_dim: int = 16):
        super(SequenceEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_size)
        _, (h_n, _) = self.lstm(x)
        # h_n shape: (1, batch_size, hidden_size)
        return self.fc(h_n.squeeze(0))  # (batch_size, latent_dim)


class LeRobotDatasetAnalyzer:
    """Analyzer for LeRobot datasets with t-SNE visualization"""

    def __init__(self, dataset_base_path: str = "/home/agiuser/datasets_lerobot"):
        self.dataset_base_path = Path(dataset_base_path)
        self.raw_dataset_base_path = Path("/home/agiuser/datasets")
        self.datasets = []
        self.current_data = None
        self.tsne_results = {}  # Single dataset results
        self.multi_tsne_results = {}  # Multiple datasets: {dataset_name: {feature_type: tsne_data}}

    def _map_to_raw_dataset_name(self, dataset_name: str) -> str:
        """Map lerobot dataset name to raw dataset folder name."""
        if "_lerobot" in dataset_name:
            return dataset_name.split("_lerobot")[0]
        return dataset_name

    def discover_datasets(self) -> List[str]:
        """Discover all available LeRobot datasets"""
        if not self.dataset_base_path.exists():
            logger.warning(f"Dataset path {self.dataset_base_path} does not exist")
            return []

        datasets = []
        for path in self.dataset_base_path.iterdir():
            if path.is_dir():
                parquet_path = path / "train" / "data" / "chunk-000" / "file-000.parquet"
                if parquet_path.exists():
                    datasets.append(path.name)

        self.datasets = datasets
        logger.info(f"Discovered {len(datasets)} datasets: {datasets}")
        return datasets

    def load_dataset(self, dataset_name: str, max_samples: int = None) -> Dict[str, Any]:
        """Load a LeRobot dataset from parquet file

        Args:
            dataset_name: Name of the dataset to load
            max_samples: Maximum number of samples to load. If None, load all samples.
                        If specified, randomly sample that many samples.
        """
        parquet_path = self.dataset_base_path / dataset_name / "train" / "data" / "chunk-000" / "file-000.parquet"

        if not parquet_path.exists():
            raise FileNotFoundError(f"Dataset not found: {parquet_path}")

        logger.info(f"Loading dataset from {parquet_path}")
        table = pq.read_table(str(parquet_path))
        df = table.to_pandas()

        # Limit samples if max_samples is specified
        if max_samples is not None and len(df) > max_samples:
            logger.info(f"Sampling {max_samples} from {len(df)} samples")
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        else:
            logger.info(f"Loading all {len(df)} samples")

        self.current_data = df

        return {
            "dataset_name": dataset_name,
            "total_samples": len(df),
            "columns": df.columns.tolist(),
            "shape": df.shape,
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        }

    def extract_episode_features(self, feature_type: str) -> np.ndarray:
        """Extract features from the dataset per episode using sequence encoding"""
        if self.current_data is None:
            raise ValueError("No dataset loaded. Call load_dataset first.")

        df = self.current_data

        # Get the feature sequence based on feature_type
        if feature_type == "state":
            if 'observation.state' in df.columns:
                logger.info("Extracting state features per episode")
                feature_column = 'observation.state'
            else:
                state_cols = [col for col in df.columns if 'state' in col.lower()]
                if not state_cols:
                    logger.warning("No state columns found")
                    return np.array([])
                logger.info(f"Using state columns: {state_cols}")
                feature_column = None  # Will use columns instead

        elif feature_type == "action":
            if 'action' in df.columns:
                logger.info("Extracting action features per episode")
                feature_column = 'action'
            else:
                action_cols = [col for col in df.columns if 'action' in col.lower()]
                if not action_cols:
                    logger.warning("No action columns found")
                    return np.array([])
                logger.info(f"Using action columns: {action_cols}")
                feature_column = None

        elif feature_type == "image":
            logger.warning("Image feature extraction using state as proxy")
            if 'observation.state' in df.columns:
                feature_column = 'observation.state'
            else:
                return np.array([])
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

        # Group by episode_index
        if 'episode_index' not in df.columns:
            raise ValueError("No episode_index column found in dataset")

        episode_features = []

        # Extract sequences per episode
        for episode_id, group in df.groupby('episode_index', sort=True):
            if feature_column:
                # Extract from single column (contains array-like values)
                sequences = np.vstack(group[feature_column].values)
            else:
                # Extract from multiple columns
                state_cols = [col for col in df.columns if feature_type.lower() in col.lower()]
                sequences = group[state_cols].values

            # Handle NaN values
            sequences = np.nan_to_num(sequences, nan=0.0)
            episode_features.append((episode_id, sequences))

        logger.info(f"Extracted {len(episode_features)} episodes for {feature_type}")

        # Encode sequences to fixed-size vectors using neural network
        encoded_features = self._encode_sequences(episode_features, feature_type)

        return encoded_features

    def _encode_sequences(self, episode_features: List[tuple], feature_type: str) -> np.ndarray:
        """Encode variable-length sequences to fixed-size vectors using LSTM"""
        if not episode_features:
            return np.array([])

        # Determine input size from first sequence
        first_seq = episode_features[0][1]
        input_size = first_seq.shape[1]

        # Create encoder
        encoder = SequenceEncoder(input_size=input_size, hidden_size=64, latent_dim=16)
        encoder.eval()

        # Pad sequences to max length
        max_len = max(seq.shape[0] for _, seq in episode_features)
        padded_sequences = []

        for _, seq in episode_features:
            if seq.shape[0] < max_len:
                padding = np.zeros((max_len - seq.shape[0], seq.shape[1]))
                padded_seq = np.vstack([seq, padding])
            else:
                padded_seq = seq
            padded_sequences.append(padded_seq)

        # Convert to tensor
        sequences_tensor = torch.FloatTensor(np.array(padded_sequences))

        # Encode
        with torch.no_grad():
            encoded = encoder(sequences_tensor).numpy()

        logger.info(f"Encoded {len(episode_features)} episodes to shape: {encoded.shape}")
        return encoded

    def compute_tsne(self, feature_type: str, perplexity: int = 30, n_iter: int = 1000) -> Dict[str, Any]:
        """Compute t-SNE for the specified feature type using episode-level features

        Args:
            feature_type: Type of features to analyze ('state', 'action', 'image')
            perplexity: t-SNE perplexity parameter (automatically adjusted for small datasets)
            n_iter: Number of t-SNE iterations
        """
        features = self.extract_episode_features(feature_type)

        if len(features) == 0:
            raise ValueError(f"No features extracted for type: {feature_type}")

        logger.info(f"Computing t-SNE for {feature_type} features, shape: {features.shape}")

        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Compute t-SNE with adaptive perplexity
        # Perplexity should be less than (n_samples - 1) / 3
        n_samples = len(features)
        max_perplexity = max(5, min(perplexity, (n_samples - 1) // 3))

        logger.info(f"Using perplexity={max_perplexity} for {n_samples} samples")

        tsne = TSNE(n_components=2, perplexity=max_perplexity, max_iter=n_iter, random_state=42, verbose=1)
        tsne_results = tsne.fit_transform(features_scaled)

        self.tsne_results[feature_type] = tsne_results

        return {
            "feature_type": feature_type,
            "n_episodes": len(tsne_results),
            "feature_dim": features.shape[1],
            "tsne_shape": tsne_results.shape,
            "perplexity": max_perplexity,
        }

    def analyze_dataset(
        self, dataset_name: str, max_samples: int = None, feature_types: List[str] = None, perplexity: int = 30
    ) -> Dict[str, Any]:
        """Analyze a single dataset and store raw features for multi-dataset visualization

        Args:
            dataset_name: Name of the dataset to analyze
            max_samples: Maximum number of samples to load
            feature_types: List of feature types to analyze (default: ['state', 'action'])
            perplexity: t-SNE perplexity parameter (not used here, kept for compatibility)

        Returns:
            Dictionary with analysis results for this dataset
        """
        if feature_types is None:
            feature_types = ["state", "action"]

        # Load dataset
        logger.info(f"Analyzing dataset: {dataset_name}")
        info = self.load_dataset(dataset_name, max_samples=max_samples)

        # Initialize storage for this dataset
        if dataset_name not in self.multi_tsne_results:
            self.multi_tsne_results[dataset_name] = {}

        # Extract and store raw features (not t-SNE yet)
        results = {}
        for feature_type in feature_types:
            try:
                logger.info(f"Extracting features for {dataset_name}/{feature_type}")
                features = self.extract_episode_features(feature_type)

                if len(features) == 0:
                    logger.warning(f"No features extracted for {feature_type}")
                    continue

                # Store raw features and episode indices
                episode_indices = sorted(self.current_data['episode_index'].unique())

                self.multi_tsne_results[dataset_name][feature_type] = {
                    'features': features,
                    'episode_indices': episode_indices[: len(features)],  # Match length
                }

                results[feature_type] = {
                    "n_episodes": len(features),
                    "feature_dim": features.shape[1],
                }
            except Exception as e:
                logger.error(f"Error analyzing {dataset_name}/{feature_type}: {e}")
                results[feature_type] = {"error": str(e)}

        return {
            "dataset_name": dataset_name,
            "samples": info["total_samples"],
            "feature_results": results,
        }

    def create_multi_visualization(
        self, output_path: str = "/tmp/multi_tsne_analysis.html", perplexity: int = 30
    ) -> str:
        """Create 3D interactive HTML visualization with all datasets combined

        Args:
            output_path: Output path for HTML file
            perplexity: t-SNE perplexity parameter

        Returns:
            Path to the generated HTML file
        """
        if not self.multi_tsne_results:
            raise ValueError("No multi-dataset results available. Run analyze_dataset first.")

        dataset_names = sorted(list(self.multi_tsne_results.keys()))
        logger.info(f"Creating 3D visualization for {len(dataset_names)} datasets")

        # Collect all features from all datasets and all feature types
        all_features = []
        metadata = []  # Store (dataset_name, feature_type, episode_index)

        for dataset_name in dataset_names:
            for feature_type, data in self.multi_tsne_results[dataset_name].items():
                features = data['features']
                episode_indices = data['episode_indices']

                all_features.append(features)

                # Record metadata for each sample
                for i, ep_idx in enumerate(episode_indices):
                    metadata.append(
                        {
                            'dataset': dataset_name,
                            'feature_type': feature_type,
                            'episode_index': int(ep_idx),
                            'sample_idx': i,
                        }
                    )

        # Concatenate all features
        combined_features = np.vstack(all_features)
        logger.info(f"Combined features shape: {combined_features.shape}")

        # Standardize
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(combined_features)

        # Compute 3D t-SNE on all data together
        n_samples = len(features_scaled)
        max_perplexity = max(5, min(perplexity, (n_samples - 1) // 3))
        logger.info(f"Computing 3D t-SNE with perplexity={max_perplexity} on {n_samples} samples")

        tsne = TSNE(n_components=3, perplexity=max_perplexity, max_iter=1000, random_state=42, verbose=1)
        tsne_3d = tsne.fit_transform(features_scaled)

        # Color map for different datasets (support at least 10 datasets)
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
            "#393b79",
            "#637939",
            "#8c6d31",
            "#843c39",
            "#7b4173",
        ]

        # Create 3D scatter plot
        fig = go.Figure()

        # Add trace for each dataset
        for idx, dataset_name in enumerate(dataset_names):
            # Filter points belonging to this dataset
            mask = np.array([m['dataset'] == dataset_name for m in metadata])
            dataset_points = tsne_3d[mask]
            dataset_meta = [m for m in metadata if m['dataset'] == dataset_name]

            # Create hover text with links to original data
            hover_texts = []
            customdata = []
            for m in dataset_meta:
                raw_dataset_name = self._map_to_raw_dataset_name(m["dataset"])
                raw_episode_path = self.raw_dataset_base_path / raw_dataset_name / f"episode{m['episode_index']}"
                raw_episode_uri = raw_episode_path.as_uri()
                lerobot_dataset_uri = (self.dataset_base_path / m["dataset"]).as_uri()
                hover_text = (
                    f"<b>Dataset:</b> {m['dataset']}<br>"
                    f"<b>Feature:</b> {m['feature_type']}<br>"
                    f"<b>Episode:</b> {m['episode_index']}<br>"
                    f"<b>Sample:</b> {m['sample_idx']}<br>"
                    f"<b>LeRobot Path:</b> <a href='{lerobot_dataset_uri}'>open</a><br>"
                    f"<b>Episode Path:</b> <a href='{raw_episode_uri}'>open</a><br>"
                    f"<b>Episode Dir:</b> {raw_episode_path}"
                )
                hover_texts.append(hover_text)
                customdata.append([m['dataset'], m['episode_index'], raw_episode_uri, lerobot_dataset_uri])

            color = colors[idx % len(colors)]

            scatter_3d = go.Scatter3d(
                x=dataset_points[:, 0],
                y=dataset_points[:, 1],
                z=dataset_points[:, 2],
                mode='markers',
                marker=dict(size=4, color=color, opacity=0.8, line=dict(width=0.5, color='white')),
                text=hover_texts,
                customdata=customdata,
                hovertemplate="%{text}<extra></extra>",
                name=dataset_name,
            )

            fig.add_trace(scatter_3d)

        # Update layout for 3D
        fig.update_layout(
            title=dict(
                text=f"3D t-SNE Analysis: {len(dataset_names)} Datasets Combined<br>"
                f"<sub>Total {n_samples} episodes, perplexity={max_perplexity}</sub>",
                x=0.5,
                xanchor='center',
            ),
            scene=dict(
                xaxis=dict(title='t-SNE Dimension 1', backgroundcolor="rgb(230, 230,230)"),
                yaxis=dict(title='t-SNE Dimension 2', backgroundcolor="rgb(230, 230,230)"),
                zaxis=dict(title='t-SNE Dimension 3', backgroundcolor="rgb(230, 230,230)"),
            ),
            width=1200,
            height=900,
            showlegend=True,
            legend=dict(x=1.02, y=0.5, xanchor='left', yanchor='middle'),
            hovermode='closest',
        )

        # Save to HTML file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        post_script = """
        var plot = document.getElementsByClassName('plotly-graph-div')[0];
        if (plot) {
            plot.on('plotly_click', function(data) {
                if (!data || !data.points || !data.points.length) { return; }
                var uri = data.points[0].customdata[2];
                if (uri) {
                    window.open(uri);
                }
            });
        }
        """
        fig.write_html(str(output_path), include_plotlyjs='cdn', post_script=post_script)

        logger.info(f"3D Visualization saved to {output_path}")
        return str(output_path)

    def create_visualization(self, output_path: str = "/tmp/tsne_analysis.html") -> str:
        """Create interactive HTML visualization of all t-SNE results"""
        if not self.tsne_results:
            raise ValueError("No t-SNE results available. Run compute_tsne first.")

        n_plots = len(self.tsne_results)
        fig = make_subplots(
            rows=1,
            cols=n_plots,
            subplot_titles=[f"{ft.upper()} Features" for ft in self.tsne_results.keys()],
            horizontal_spacing=0.1,
        )

        # colors = px.colors.qualitative.Set2

        for idx, (feature_type, tsne_data) in enumerate(self.tsne_results.items(), 1):
            # Create scatter plot
            scatter = go.Scatter(
                x=tsne_data[:, 0],
                y=tsne_data[:, 1],
                mode='markers',
                marker=dict(
                    size=5,
                    color=np.arange(len(tsne_data)),
                    colorscale='Viridis',
                    showscale=(idx == n_plots),  # Only show colorbar for last plot
                    colorbar=dict(title="Sample Index"),
                ),
                text=[f"Sample {i}" for i in range(len(tsne_data))],
                hovertemplate="<b>%{text}</b><br>x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>",
                name=feature_type,
            )

            fig.add_trace(scatter, row=1, col=idx)

            # Update axes
            fig.update_xaxes(title_text="t-SNE Dimension 1", row=1, col=idx)
            fig.update_yaxes(title_text="t-SNE Dimension 2", row=1, col=idx)

        # Update layout
        fig.update_layout(
            title_text="LeRobot Dataset t-SNE Analysis", height=600, showlegend=False, hovermode='closest'
        )

        # Save to HTML file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))

        logger.info(f"Visualization saved to {output_path}")
        return str(output_path)


# Initialize analyzer
analyzer = LeRobotDatasetAnalyzer()

# Create MCP server
app = Server("data-analyst-mcp-server")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="discover_datasets",
            description="Discover all available LeRobot datasets in the dataset directory",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        Tool(
            name="analyze_multi_datasets",
            description="Analyze all datasets and prepare for combined 3D t-SNE visualization",
            inputSchema={
                "type": "object",
                "properties": {
                    "feature_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["state", "action", "image"]},
                        "description": "Types of features to analyze (default: ['state', 'action'])",
                        "default": ["state", "action"],
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="generate_3d_visualization",
            description="Generate 3D interactive t-SNE visualization with all analyzed datasets combined in one plot",
            inputSchema={
                "type": "object",
                "properties": {
                    "output_path": {
                        "type": "string",
                        "description": "Output path for HTML file (default: /tmp/multi_tsne_3d.html)",
                        "default": "/tmp/multi_tsne_3d.html",
                    },
                    "perplexity": {
                        "type": "integer",
                        "description": "t-SNE perplexity parameter (default: 30)",
                        "default": 30,
                    },
                },
                "required": [],
            },
        ),
        Tool(
            name="full_analysis",
            description="Complete workflow: analyze all datasets and generate combined 3D visualization",
            inputSchema={
                "type": "object",
                "properties": {
                    "feature_types": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["state", "action", "image"]},
                        "description": "Types of features to analyze",
                        "default": ["state", "action"],
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Output path for 3D HTML visualization",
                        "default": "/tmp/multi_tsne_3d.html",
                    },
                },
                "required": [],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""
    try:
        if name == "discover_datasets":
            datasets = analyzer.discover_datasets()
            result = {"datasets": datasets, "count": len(datasets), "base_path": str(analyzer.dataset_base_path)}
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "analyze_multi_datasets":
            feature_types = arguments.get("feature_types", ["state", "action"])

            if not analyzer.datasets:
                analyzer.discover_datasets()

            dataset_names = analyzer.datasets

            results = {}
            for dataset_name in dataset_names:
                try:
                    logger.info(f"Analyzing dataset: {dataset_name}")
                    result = analyzer.analyze_dataset(dataset_name, max_samples=None, feature_types=feature_types)
                    results[dataset_name] = result
                except Exception as e:
                    logger.error(f"Error analyzing {dataset_name}: {e}")
                    results[dataset_name] = {"error": str(e)}

            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "analyzed_datasets": list(results.keys()),
                            "results": results,
                            "message": (
                                f"Analyzed {len(results)} datasets. Use generate_3d_visualization to create combined 3D plot."
                            ),
                        },
                        indent=2,
                    ),
                )
            ]

        elif name == "generate_3d_visualization":
            output_path = arguments.get("output_path", "/tmp/multi_tsne_3d.html")
            perplexity = arguments.get("perplexity", 30)
            html_path = analyzer.create_multi_visualization(output_path, perplexity=perplexity)
            result = {
                "html_path": html_path,
                "message": f"3D visualization saved to {html_path}. Open in browser to view and interact.",
            }
            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "full_analysis":
            if not analyzer.datasets:
                analyzer.discover_datasets()

            if not analyzer.datasets:
                raise ValueError("No datasets available")

            feature_types = arguments.get("feature_types", ["state", "action"])
            output_path = arguments.get("output_path", "/tmp/multi_tsne_3d.html")

            results = {}
            for dataset_name in analyzer.datasets:
                try:
                    logger.info(f"Analyzing dataset: {dataset_name}")
                    result = analyzer.analyze_dataset(dataset_name, max_samples=None, feature_types=feature_types)
                    results[dataset_name] = result
                except Exception as e:
                    logger.error(f"Error analyzing {dataset_name}: {e}")
                    results[dataset_name] = {"error": str(e)}

            html_path = analyzer.create_multi_visualization(output_path, perplexity=30)

            result = {
                "analyzed_datasets": list(results.keys()),
                "results": results,
                "visualization": html_path,
                "message": f"Analysis complete! Open {html_path} in your browser to view the 3D results.",
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        else:
            raise ValueError(f"Unknown tool: {name}")

    except Exception as e:
        logger.error(f"Error executing tool {name}: {e}", exc_info=True)
        return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]


async def main():
    """Run the MCP server"""
    async with stdio_server() as streams:
        await app.run(streams[0], streams[1], app.create_initialization_options())


if __name__ == "__main__":
    sys.exit(anyio.run(main))
