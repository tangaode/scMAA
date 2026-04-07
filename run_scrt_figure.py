"""CLI entry point for scMAA figure mode."""

from __future__ import annotations

import argparse
from pathlib import Path

from scrt_agent.agent import refresh_run_summary_from_artifacts, write_figure_status_file
from scrt_agent.figure_mode import build_publication_figure


def main() -> int:
    parser = argparse.ArgumentParser(
        description="scMAA figure mode: generate a publication-style multi-panel figure."
    )
    parser.add_argument("--rna-h5ad-path", required=True, help="Path to the RNA .h5ad file.")
    parser.add_argument("--tcr-path", required=True, help="Path to the TCR annotation table.")
    parser.add_argument("--output-dir", required=True, help="Output directory for figure artifacts.")
    parser.add_argument("--figure-name", default="scrt_publication_figure", help="Base name for figure outputs.")
    parser.add_argument("--hypothesis-model", default="gpt-4o", help="Planning model used for dynamic hypothesis figure generation.")
    args = parser.parse_args()

    result = build_publication_figure(
        rna_h5ad_path=args.rna_h5ad_path,
        tcr_path=args.tcr_path,
        output_dir=args.output_dir,
        figure_name=args.figure_name,
        hypothesis_model=args.hypothesis_model,
    )
    output_dir = Path(args.output_dir).resolve()
    run_dir = output_dir.parent if output_dir.name.lower() == "figure" else output_dir
    if (run_dir / "run_summary.txt").exists():
        write_figure_status_file(
            run_dir,
            figure_result=result,
            note="Figure generated with run_scrt_figure.py",
        )
        refresh_run_summary_from_artifacts(run_dir)
    print(f"PNG: {result.png_path}")
    print(f"PDF: {result.pdf_path}")
    print(f"Summary: {result.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

