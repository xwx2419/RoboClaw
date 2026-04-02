#!/usr/bin/env python3
"""Test script for data analyst MCP server"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from server import LeRobotDatasetAnalyzer


def main():
    analyzer = LeRobotDatasetAnalyzer()

    print("\n" + "=" * 60)
    print("Discovering datasets...")
    print("=" * 60)

    dataset_names = analyzer.discover_datasets()
    if not dataset_names:
        print("❌ No datasets found!")
        return

    print(f"\n✓ Found {len(dataset_names)} datasets:\n")
    for idx, ds in enumerate(dataset_names, 1):
        print(f"  {idx}. {ds}")

    print("\n" + "=" * 60)
    print("LeRobot Multi-Dataset Analysis - Full Analysis")
    print(f"Datasets: {len(dataset_names)} total")
    print("=" * 60)

    # Analyze each dataset (full data)
    for idx, dataset_name in enumerate(dataset_names, 1):
        print(f"\n[{idx}/{len(dataset_names)}] Processing {dataset_name}...", end=" ", flush=True)
        try:
            result = analyzer.analyze_dataset(dataset_name, max_samples=None)
            print(f"✓ ({result['samples']} samples)")
        except Exception as e:
            print(f"✗ Error: {e}")

    # Generate combined visualization
    print("\n[Generating] Combined 3D visualization...", end=" ", flush=True)
    try:
        output_file = "/tmp/multi_tsne_3d_full.html"
        html_path = analyzer.create_multi_visualization(output_file, perplexity=30)
        print("✓")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return

    # Summary
    print("\n" + "=" * 60)
    print("✅ Analysis completed successfully!")
    print("\nAll datasets combined in a single 3D visualization:")
    print(f"  → {html_path}")
    print("\n📊 Features:")
    print("  • All datasets merged in one 3D t-SNE plot")
    print("  • Different colors for each dataset")
    print("  • Drag to rotate, scroll to zoom")
    print("  • Hover to see data source and episode info")
    print("  • Click legend to toggle datasets")
    print("\nOpen this file in your browser to explore the data!")
    print("=" * 60)


if __name__ == "__main__":
    main()
