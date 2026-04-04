"""CLI entry point for scRNA + scATAC raw-data preparation."""

from __future__ import annotations

import argparse

from scrat_agent.preprocess import prepare_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare raw scRNA + scATAC files for the integrated agent.")
    parser.add_argument("--raw-input-path", required=True, help="Path to a 10x multiome raw directory or tar archive.")
    parser.add_argument("--output-dir", required=True, help="Directory for the prepared outputs.")
    parser.add_argument("--annotation-model", default="gpt-4o", help="Model used for RNA cluster annotation from marker genes.")
    parser.add_argument("--annotation-notes-path", default=None, help="Optional text file with short notes to help cluster annotation.")
    parser.add_argument("--min-genes", type=int, default=200, help="Minimum RNA genes per cell.")
    parser.add_argument("--min-cells", type=int, default=3, help="Minimum cells per RNA gene or ATAC feature.")
    parser.add_argument("--max-pct-mt", type=float, default=15.0, help="Maximum mitochondrial percentage for RNA cells.")
    parser.add_argument("--min-atac-features", type=int, default=1000, help="Minimum detected ATAC features per cell.")
    parser.add_argument("--n-top-genes", type=int, default=3000, help="Number of highly variable RNA genes.")
    parser.add_argument("--n-pcs", type=int, default=30, help="Number of RNA PCs and approximate ATAC LSI components.")
    parser.add_argument("--n-neighbors", type=int, default=15, help="Neighbors for graph building.")
    parser.add_argument("--leiden-resolution", type=float, default=0.8, help="Leiden resolution.")
    parser.add_argument("--marker-top-n", type=int, default=100, help="How many marker features to export per cluster.")
    parser.add_argument("--annotation-marker-top-n", type=int, default=50, help="How many non-lincRNA genes per RNA cluster to send to the annotation model.")
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
        min_atac_features=args.min_atac_features,
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
    print(f"ATAC h5ad: {result.atac_h5ad_path}")
    print(f"RNA markers: {result.rna_markers_path}")
    print(f"ATAC markers: {result.atac_markers_path}")
    print(f"RNA annotations: {result.rna_annotations_path}")
    print(f"ATAC annotations: {result.atac_annotations_path}")
    print(f"QC summary: {result.qc_summary_path}")
    print(f"Manifest: {result.manifest_path}")
    print(f"RNA UMAP: {result.rna_umap_path}")
    print(f"ATAC UMAP: {result.atac_umap_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
