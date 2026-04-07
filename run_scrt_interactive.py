"""CLI entry point for interactive scMAA sessions."""

from __future__ import annotations

import argparse
from pathlib import Path

from scrt_agent.agent import ScRTAgent
from scrt_agent.interactive import (
    format_analysis_plan_markdown,
    format_candidate_menu_markdown,
    read_json,
    write_json,
)


def _build_agent_from_args(args, *, analysis_name: str, output_home: str) -> ScRTAgent:
    return ScRTAgent(
        rna_h5ad_path=args.rna_h5ad_path,
        tcr_path=args.tcr_path,
        research_brief_path=args.research_brief_path,
        literature_paths=args.literature_path or None,
        analysis_name=analysis_name,
        model_name=args.model_name,
        hypothesis_model=args.hypothesis_model or args.model_name,
        execution_model=args.execution_model or args.model_name,
        vision_model=args.vision_model,
        num_analyses=1,
        max_iterations=args.max_iterations,
        output_home=output_home,
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


def _load_agent_from_session(session_dir: Path, args) -> tuple[ScRTAgent, dict]:
    config = read_json(session_dir / "session_config.json")
    namespace = argparse.Namespace(**config)
    defaults = {
        "tcr_path": config.get("tcr_path") or config.get("modality_input_path"),
        "model_name": config.get("model_name", "gpt-4o"),
        "hypothesis_model": config.get("hypothesis_model"),
        "execution_model": config.get("execution_model"),
        "vision_model": config.get("vision_model", "gpt-4o"),
        "max_iterations": config.get("max_iterations", 6),
        "prompt_dir": config.get("prompt_dir"),
        "max_fix_attempts": config.get("max_fix_attempts", 3),
        "deepresearch": config.get("deepresearch", False),
        "no_self_critique": config.get("no_self_critique", False),
        "no_documentation": config.get("no_documentation", False),
        "no_vlm": config.get("no_vlm", False),
        "log_prompts": config.get("log_prompts", False),
        "with_figure": config.get("with_figure", False),
        "figure_name": config.get("figure_name"),
    }
    for key, value in defaults.items():
        if not hasattr(namespace, key):
            setattr(namespace, key, value)
    if getattr(args, "with_figure", None) is None:
        namespace.with_figure = config.get("with_figure", False)
    else:
        namespace.with_figure = args.with_figure
    namespace.figure_name = getattr(args, "figure_name", config.get("figure_name"))
    agent = _build_agent_from_args(namespace, analysis_name=session_dir.name, output_home=str(session_dir.parent))
    return agent, config


def _common_agent_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rna-h5ad-path", required=True, help="Path to the RNA .h5ad file.")
    parser.add_argument("--tcr-path", required=True, help="Path to the TCR annotation table.")
    parser.add_argument("--research-brief-path", required=True, help="Path to a freeform research brief text file.")
    parser.add_argument(
        "--literature-path",
        action="append",
        default=[],
        help="Optional path to a local literature file or directory. Can be provided multiple times.",
    )
    parser.add_argument("--session-name", required=True, help="Name of the interactive session.")
    parser.add_argument("--output-home", default=".", help="Base directory where the session folder will be created.")
    parser.add_argument("--model-name", default="gpt-4o", help="Default model fallback.")
    parser.add_argument("--hypothesis-model", default=None, help="Planning model. Defaults to --model-name.")
    parser.add_argument("--execution-model", default=None, help="Execution support model. Defaults to --model-name.")
    parser.add_argument("--vision-model", default="gpt-4o", help="Vision model used when enabled.")
    parser.add_argument("--max-iterations", type=int, default=6, help="Maximum notebook steps per analysis.")
    parser.add_argument("--prompt-dir", default=None, help="Optional custom prompt directory.")
    parser.add_argument("--max-fix-attempts", type=int, default=3, help="Maximum code-fix retries.")
    parser.add_argument("--deepresearch", action="store_true", help="Enable Deep Research background generation.")
    parser.add_argument("--no-self-critique", action="store_true", help="Disable self-critique.")
    parser.add_argument("--no-documentation", action="store_true", help="Disable documentation lookup.")
    parser.add_argument("--no-vlm", action="store_true", help="Disable image interpretation.")
    parser.add_argument("--log-prompts", action="store_true", help="Save prompt logs.")
    parser.add_argument("--with-figure", action="store_true", help="Generate a publication-style figure after run.")
    parser.add_argument("--figure-name", default=None, help="Optional base name for publication figure outputs.")


def cmd_prepare(args) -> int:
    session_root = Path(args.output_home).resolve()
    session_dir = session_root / args.session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    agent = _build_agent_from_args(args, analysis_name=args.session_name, output_home=str(session_root))
    menu = agent.prepare_candidate_hypotheses()

    candidate_json = session_dir / "candidate_hypotheses.json"
    candidate_md = session_dir / "candidate_hypotheses.md"
    write_json(candidate_json, menu.model_dump())
    candidate_md.write_text(format_candidate_menu_markdown(menu), encoding="utf-8")

    config = {
        "rna_h5ad_path": args.rna_h5ad_path,
        "tcr_path": args.tcr_path,
        "research_brief_path": args.research_brief_path,
        "literature_path": args.literature_path or [],
        "model_name": args.model_name,
        "hypothesis_model": args.hypothesis_model,
        "execution_model": args.execution_model,
        "vision_model": args.vision_model,
        "max_iterations": args.max_iterations,
        "prompt_dir": args.prompt_dir,
        "max_fix_attempts": args.max_fix_attempts,
        "deepresearch": args.deepresearch,
        "no_self_critique": args.no_self_critique,
        "no_documentation": args.no_documentation,
        "no_vlm": args.no_vlm,
        "log_prompts": args.log_prompts,
        "with_figure": args.with_figure,
        "figure_name": args.figure_name,
    }
    write_json(session_dir / "session_config.json", config)
    print(f"Session prepared: {session_dir}")
    print(f"Candidates: {candidate_json}")
    print(f"Review guide: {candidate_md}")
    return 0


def _resolve_hypothesis_text(session_dir: Path, args) -> str:
    if args.hypothesis_text:
        return args.hypothesis_text.strip()
    if args.hypothesis_file:
        return Path(args.hypothesis_file).read_text(encoding="utf-8").strip()
    candidate_data = read_json(session_dir / "candidate_hypotheses.json")
    candidates = candidate_data.get("candidates", [])
    if not candidates:
        raise ValueError("No candidate hypotheses found. Run the prepare step first.")
    if args.candidate_index is None:
        raise ValueError("Provide --candidate-index, --hypothesis-text, or --hypothesis-file.")
    idx = int(args.candidate_index) - 1
    if idx < 0 or idx >= len(candidates):
        raise ValueError(f"candidate index {args.candidate_index} is out of range.")
    return str(candidates[idx]["hypothesis"]).strip()


def cmd_review(args) -> int:
    session_dir = Path(args.session_dir).resolve()
    agent, _ = _load_agent_from_session(session_dir, args)
    hypothesis = _resolve_hypothesis_text(session_dir, args)

    feedback_parts: list[str] = []
    if args.feedback_text:
        feedback_parts.append(args.feedback_text.strip())
    if args.feedback_file:
        feedback_parts.append(Path(args.feedback_file).read_text(encoding="utf-8").strip())
    feedback_text = "\n\n".join(part for part in feedback_parts if part).strip()
    revised_hypothesis = agent.revise_hypothesis(hypothesis=hypothesis, user_feedback=feedback_text) if feedback_text else hypothesis
    approved_plan = agent.build_plan_from_hypothesis(revised_hypothesis)

    (session_dir / "approved_hypothesis.txt").write_text(revised_hypothesis + "\n", encoding="utf-8")
    write_json(session_dir / "approved_plan.json", approved_plan.model_dump())
    (session_dir / "approved_plan.md").write_text(format_analysis_plan_markdown(approved_plan), encoding="utf-8")
    if feedback_text:
        (session_dir / "user_feedback.txt").write_text(feedback_text + "\n", encoding="utf-8")

    print(f"Approved hypothesis: {session_dir / 'approved_hypothesis.txt'}")
    print(f"Approved plan: {session_dir / 'approved_plan.json'}")
    print(f"Readable plan: {session_dir / 'approved_plan.md'}")
    return 0


def cmd_run(args) -> int:
    session_dir = Path(args.session_dir).resolve()
    agent, _ = _load_agent_from_session(session_dir, args)
    approved_plan_path = session_dir / "approved_plan.json"
    if not approved_plan_path.exists():
        raise FileNotFoundError("No approved_plan.json found. Run the review step first.")
    approved_plan = read_json(approved_plan_path)
    summary_path = agent.run_approved_plan(approved_plan)
    print(f"Run summary: {summary_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interactive session CLI for scMAA."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Prepare a session and generate candidate hypotheses.")
    _common_agent_arguments(prepare_parser)
    prepare_parser.set_defaults(func=cmd_prepare)

    review_parser = subparsers.add_parser("review", help="Approve or revise a hypothesis before execution.")
    review_parser.add_argument("--session-dir", required=True, help="Path to the prepared session directory.")
    review_parser.add_argument("--candidate-index", type=int, default=None, help="1-based candidate index to approve.")
    review_parser.add_argument("--hypothesis-text", default=None, help="Custom hypothesis text.")
    review_parser.add_argument("--hypothesis-file", default=None, help="Path to a file containing custom hypothesis text.")
    review_parser.add_argument("--feedback-text", default=None, help="Optional freeform user feedback to revise the hypothesis.")
    review_parser.add_argument("--feedback-file", default=None, help="Optional path to freeform user feedback text.")
    review_parser.add_argument("--with-figure", action="store_true", default=None, help="Enable figure generation when this session later runs.")
    review_parser.add_argument("--figure-name", default=None, help="Optional figure base name override.")
    review_parser.set_defaults(func=cmd_review)

    run_parser = subparsers.add_parser("run", help="Run the approved interactive session.")
    run_parser.add_argument("--session-dir", required=True, help="Path to the prepared session directory.")
    run_parser.add_argument("--with-figure", action="store_true", default=None, help="Generate a publication-style figure after run.")
    run_parser.add_argument("--figure-name", default=None, help="Optional figure base name override.")
    run_parser.set_defaults(func=cmd_run)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

