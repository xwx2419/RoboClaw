# Data Analyst MCP Server

An MCP server for analyzing LeRobot datasets using t-SNE dimensionality reduction and visualization.

## Features

- Load and analyze LeRobot parquet datasets
- t-SNE visualization for:
  - State vectors
  - Action sequences
  - Image features
- Interactive HTML visualization that opens in browser
- Pre-computed analysis for fast results

## Installation

```bash
cd src/mcp_server_demo/data_analyst_mcp_server
uv venv
source .venv/bin/activate  # On Linux/Mac
uv pip install -e .
```

## Usage

The server provides tools for analyzing LeRobot datasets stored in parquet format.

Dataset path pattern: `/home/agiuser/datasets_lerobot/*/train/data/chunk-000/file-000.parquet`
