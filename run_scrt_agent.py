"""CLI entry point for scMAA."""

from __future__ import annotations

import argparse
import os

from scrt_agent.agent import ScRTAgent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="scMAA: research-oriented agent for integrated scRNA + scTCR analysis."
    )
    parser.add_argument("--rna-h5ad-path", required=True, help="Path to the RNA .h5ad file.")
    parser.add_argument("--tcr-path", required=True, help="Path to the TCR annotation table.")
    parser.add_argument(
        "--research-brief-path",
        default=None,
        help="Path to a freeform research brief text file describing the question, background, and priorities.",
    )
    parser.add_argument(
        "--context-path",
        default=None,
        help="Deprecated alias for --research-brief-path.",
    )
    parser.add_argument(
        "--literature-path",
        action="append",
        default=[],
        help="Optional path to a local literature file or directory. Can be provided multiple times.",
    )
    parser.add_argument("--analysis-name", default="scrt_run", help="Name for the analysis run.")
    parser.add_argument("--model-name", default="gpt-4o", help="Default model fallback.")
    parser.add_argument(
        "--hypothesis-model",
        default=None,
        help="Model for analysis planning and re-planning. Defaults to --model-name.",
    )
    parser.add_argument(
        "--execution-model",
        default=None,
        help="Model for code fixing and text-only execution support. Defaults to --model-name.",
    )
    parser.add_argument(
        "--vision-model",
        default=os.environ.get("SCRT_VISION_MODEL", "gpt-4o"),
        help="Vision model used to interpret figures when enabled.",
    )
    parser.add_argument("--num-analyses", type=int, default=3, help="Number of analyses to run.")
    parser.add_argument("--max-iterations", type=int, default=6, help="Maximum notebook steps per analysis.")
    parser.add_argument("--output-home", default=".", help="Base output directory.")
    parser.add_argument("--prompt-dir", default=None, help="Optional custom prompt directory.")
    parser.add_argument("--max-fix-attempts", type=int, default=3, help="Maximum code-fix retries.")
    parser.add_argument("--deepresearch", action="store_true", help="Enable Deep Research background generation.")
    parser.add_argument("--with-figure", action="store_true", help="Generate a publication-style figure after the run.")
    parser.add_argument("--figure-name", default=None, help="Optional base name for publication figure outputs.")
    parser.add_argument("--no-self-critique", action="store_true", help="Disable self-critique.")
    parser.add_argument("--no-documentation", action="store_true", help="Disable documentation lookup.")
    parser.add_argument("--no-vlm", action="store_true", help="Disable image interpretation.")
    parser.add_argument("--log-prompts", action="store_true", help="Save prompt logs.")
    parser.add_argument(
        "--seed-hypothesis",
        action="append",
        default=[],
        help="Optional seeded hypothesis. Can be provided multiple times.",
    )
    args = parser.parse_args()
    research_brief_path = args.research_brief_path or args.context_path
    if not research_brief_path:
        parser.error("one of --research-brief-path or --context-path is required")

    agent = ScRTAgent(
        rna_h5ad_path=args.rna_h5ad_path,
        tcr_path=args.tcr_path,
        research_brief_path=research_brief_path,
        literature_paths=args.literature_path or None,
        analysis_name=args.analysis_name,
        model_name=args.model_name,
        hypothesis_model=args.hypothesis_model or args.model_name,
        execution_model=args.execution_model or args.model_name,
        vision_model=args.vision_model,
        num_analyses=args.num_analyses,
        max_iterations=args.max_iterations,
        output_home=args.output_home,
        prompt_dir=args.prompt_dir,
        use_self_critique=not args.no_self_critique,
        use_documentation=not args.no_documentation,
        use_VLM=not args.no_vlm,
        use_deepresearch=args.deepresearch,
        generate_publication_figure=args.with_figure,
        publication_figure_name=args.figure_name,
        log_prompts=args.log_prompts,
        max_fix_attempts=args.max_fix_attempts,
    )
    agent.run(seeded_hypotheses=args.seed_hypothesis or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

