"""CLI entry point for raw-data preparation."""

from __future__ import annotations

import argparse
from pathlib import Path

from scrt_agent.preprocess import prepare_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare raw scRNA + scTCR files for scMAA."
    )
    parser.add_argument(
        "--raw-input-path",
        required=True,
        help="Path to a directory of extracted raw files or a GEO RAW.tar archive.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for the prepared outputs.",
    )
    parser.add_argument(
        "--annotation-model",
        default="gpt-4o",
        help="Model used for cluster annotation from marker genes.",
    )
    parser.add_argument(
        "--annotation-notes-path",
        default=None,
        help="Optional text file with short notes to help cluster annotation.",
    )
    parser.add_argument("--min-genes", type=int, default=200, help="Minimum genes per cell.")
    parser.add_argument("--min-cells", type=int, default=3, help="Minimum cells per gene.")
    parser.add_argument("--max-pct-mt", type=float, default=15.0, help="Maximum mitochondrial percentage per cell.")
    parser.add_argument("--n-top-genes", type=int, default=3000, help="Number of highly variable genes.")
    parser.add_argument("--n-pcs", type=int, default=30, help="Number of PCs for neighbors and UMAP.")
    parser.add_argument("--n-neighbors", type=int, default=15, help="Neighbors for graph building.")
    parser.add_argument("--leiden-resolution", type=float, default=0.8, help="Leiden resolution.")
    parser.add_argument(
        "--marker-top-n",
        type=int,
        default=100,
        help="How many DE genes to export per cluster.",
    )
    parser.add_argument(
        "--annotation-marker-top-n",
        type=int,
        default=50,
        help="How many non-lincRNA genes per cluster to send to the annotation model.",
    )
    parser.add_argument("--log-prompts", action="store_true", help="Save annotation prompts in the log directory.")
    args = parser.parse_args()

    result = prepare_dataset(
        raw_input_path=args.raw_input_path,
        output_dir=args.output_dir,
        annotation_model=args.annotation_model,
        annotation_notes_path=args.annotation_notes_path,
        min_genes=args.min_genes,
        min_cells=args.min_cells,
        max_pct_mt=args.max_pct_mt,
        n_top_genes=args.n_top_genes,
        n_pcs=args.n_pcs,
        n_neighbors=args.n_neighbors,
        leiden_resolution=args.leiden_resolution,
        marker_top_n=args.marker_top_n,
        annotation_marker_top_n=args.annotation_marker_top_n,
        log_prompts=args.log_prompts,
    )

    print(f"Prepared output directory: {result.output_dir}")
    print(f"RNA h5ad: {result.rna_h5ad_path}")
    print(f"TCR table: {result.tcr_table_path}")
    print(f"Cluster markers: {result.cluster_markers_path}")
    print(f"Cluster annotations: {result.cluster_annotations_path}")
    print(f"QC summary: {result.qc_summary_path}")
    print(f"Manifest: {result.manifest_path}")
    print(f"UMAP by cluster: {result.umap_cluster_path}")
    print(f"UMAP by annotation: {result.umap_annotation_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

