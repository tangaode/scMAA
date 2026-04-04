"""Hypothesis generation for scMAA."""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Optional

import instructor
import litellm
from pydantic import BaseModel, Field

from .research import ResearchStepUpdate
from .utils import get_documentation, summarize_notebook_cells

litellm.drop_params = True


_MODEL_ALIASES = {
    "gpt-5.3": "openai/gpt-5.3-chat-latest",
    "gpt-5.2": "openai/gpt-5.2-chat-latest",
}


def _normalize_model_name(model: str) -> str:
    if model in _MODEL_ALIASES:
        return _MODEL_ALIASES[model]
    if "/" in model:
        return model
    if model.startswith(("gpt-", "o1", "o3", "o4")):
        return model
    if model.startswith("claude-"):
        return model
    return model


def _strategy_expectations(feedback: str) -> list[tuple[str, tuple[str, ...]]]:
    text = (feedback or "").lower()
    expectations: list[tuple[str, tuple[str, ...]]] = []
    keyword_sets = [
        ("pseudotime", ("拟时序", "伪时序", "pseudotime", "trajectory", "轨迹")),
        ("heatmap", ("热图", "heatmap")),
        ("differential_expression", ("差异基因", "差异表达", "deg", "differential expression")),
        ("t_cell", ("t细胞", "t cell", "t-cell", "cd8", "cd4", "treg")),
    ]
    for label, variants in keyword_sets:
        if any(token in text for token in variants):
            expectations.append((label, variants))
    return expectations


def _plan_mentions_strategy(plan: "AnalysisPlan", feedback: str) -> bool:
    expectations = _strategy_expectations(feedback)
    if not expectations:
        return True
    combined = "\n".join(
        [
            plan.priority_question,
            plan.evidence_goal,
            plan.decision_rationale,
            "\n".join(plan.validation_checks),
            "\n".join(plan.analysis_plan),
            plan.first_step_code,
            plan.code_description,
            plan.summary,
        ]
    ).lower()
    for _label, variants in expectations:
        normalized_variants = tuple(token.lower() for token in variants)
        if not any(token in combined for token in normalized_variants):
            return False
    return True


class AnalysisPlan(BaseModel):
    hypothesis: str = Field(description="Specific and testable hypothesis.")
    analysis_type: str = Field(description="One of: rna_only, tcr_only, joint.")
    priority_question: str = Field(description="Most important question this analysis tries to answer next.")
    evidence_goal: str = Field(description="What concrete evidence the next step must produce.")
    decision_rationale: str = Field(description="Why this next step is preferred over alternatives.")
    validation_checks: list[str] = Field(description="Checks or guardrails to validate the next claim.")
    analysis_plan: list[str] = Field(description="Ordered remaining analysis steps.")
    first_step_code: str = Field(description="Executable Python code for the next step only.")
    code_description: str = Field(description="Short description of what the next code cell does.")
    summary: str = Field(description="Brief summary of the full analysis.")


class CandidateHypothesis(BaseModel):
    title: str = Field(description="Short user-facing title for the candidate.")
    hypothesis: str = Field(description="Specific and testable hypothesis.")
    rationale: str = Field(description="Why this hypothesis is worth testing for this dataset.")
    preferred_analysis_type: str = Field(description="One of: rna_only, tcr_only, joint.")
    first_test: str = Field(description="The first practical test the agent would run.")
    cautions: list[str] = Field(description="Important caveats or guardrails.")


class CandidateHypothesisMenu(BaseModel):
    research_focus: str = Field(description="Short synthesis of the current research direction.")
    candidates: list[CandidateHypothesis] = Field(description="Ranked candidate hypotheses.")


class LiteratureHypothesisChoice(BaseModel):
    candidate_title: str = Field(description="Short title of the selected literature candidate.")
    hypothesis: str = Field(description="Selected literature-guided hypothesis.")
    analysis_type: str = Field(description="One of: rna_only, tcr_only, joint.")
    selection_rationale: str = Field(description="Why this candidate is the best next choice for the current dataset.")
    expected_evidence: str = Field(description="What direct evidence should be collected first.")
    guardrail_notes: list[str] = Field(description="Interpretation constraints that should stay attached to the claim.")


class HypothesisRevision(BaseModel):
    revised_hypothesis: str = Field(description="Updated hypothesis after incorporating user feedback.")
    revision_rationale: str = Field(description="Why the hypothesis was revised this way.")
    preferred_analysis_type: str = Field(description="One of: rna_only, tcr_only, joint.")
    retained_constraints: list[str] = Field(description="Constraints or caveats that should remain attached.")


class HypothesisGenerator:
    """Generates and refines integrated scRNA + scTCR analyses."""

    def __init__(
        self,
        *,
        model_name: str,
        prompt_dir: str | os.PathLike[str],
        coding_guidelines: str,
        coding_system_prompt: str,
        rna_summary: str,
        tcr_summary: str,
        joint_summary: str,
        validation_summary: str,
        context_summary: str,
        literature_summary: str,
        literature_candidates_summary: str,
        logger,
        use_self_critique: bool = True,
        use_documentation: bool = True,
        max_iterations: int = 6,
        deepresearch_background: str = "",
        log_prompts: bool = False,
    ) -> None:
        self.model_name = _normalize_model_name(model_name)
        self.prompt_dir = Path(prompt_dir)
        self.coding_guidelines = coding_guidelines
        self.coding_system_prompt = coding_system_prompt
        self.rna_summary = rna_summary
        self.tcr_summary = tcr_summary
        self.joint_summary = joint_summary
        self.validation_summary = validation_summary
        self.context_summary = context_summary
        self.literature_summary = literature_summary
        self.literature_candidates_summary = literature_candidates_summary
        self.logger = logger
        self.use_self_critique = use_self_critique
        self.use_documentation = use_documentation
        self.max_iterations = max_iterations
        self.deepresearch_background = deepresearch_background
        self.log_prompts = log_prompts
        self.client = instructor.from_litellm(litellm.completion)

    def _read_prompt(self, name: str) -> str:
        return (self.prompt_dir / name).read_text(encoding="utf-8")

    def _complete_structured(self, messages: list[dict], response_model: type[BaseModel]) -> BaseModel:
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_model=response_model,
        )

    def _complete_text(self, messages: list[dict]) -> str:
        response = litellm.completion(model=self.model_name, messages=messages)
        return response.choices[0].message.content or ""

    def generate_candidate_hypotheses(
        self,
        research_state_summary: str,
        past_analyses: str = "",
        user_feedback: str = "",
    ) -> CandidateHypothesisMenu:
        prompt = self._read_prompt("candidate_hypotheses.txt").format(
            rna_summary=self.rna_summary,
            tcr_summary=self.tcr_summary,
            joint_summary=self.joint_summary,
            validation_summary=self.validation_summary,
            research_state=research_state_summary or "No research ledger entries yet.",
            literature_summary=self.literature_summary,
            literature_candidates_summary=self.literature_candidates_summary,
            context_summary=self.context_summary,
            past_analyses=past_analyses or "No previous analyses.",
            user_feedback=user_feedback.strip() or "No extra user feedback.",
        )
        if self.log_prompts:
            self.logger.log_prompt("user", prompt, "candidate_hypotheses")
        return self._complete_structured(
            [
                {"role": "system", "content": self.coding_system_prompt},
                {"role": "user", "content": prompt},
            ],
            CandidateHypothesisMenu,
        )

    def revise_hypothesis_with_feedback(
        self,
        *,
        hypothesis: str,
        user_feedback: str,
        research_state_summary: str,
        past_analyses: str = "",
    ) -> HypothesisRevision:
        prompt = self._read_prompt("revise_hypothesis.txt").format(
            hypothesis=hypothesis,
            user_feedback=user_feedback,
            rna_summary=self.rna_summary,
            tcr_summary=self.tcr_summary,
            joint_summary=self.joint_summary,
            validation_summary=self.validation_summary,
            research_state=research_state_summary or "No research ledger entries yet.",
            literature_summary=self.literature_summary,
            literature_candidates_summary=self.literature_candidates_summary,
            context_summary=self.context_summary,
            past_analyses=past_analyses or "No previous analyses.",
        )
        if self.log_prompts:
            self.logger.log_prompt("user", prompt, "revise_hypothesis")
        return self._complete_structured(
            [
                {"role": "system", "content": self.coding_system_prompt},
                {"role": "user", "content": prompt},
            ],
            HypothesisRevision,
        )

    def _generate_initial_analysis_freeform(self, past_analyses: str, research_state_summary: str) -> AnalysisPlan:
        prompt = self._read_prompt("first_draft.txt").format(
            CODING_GUIDELINES=self.coding_guidelines,
            max_iterations=self.max_iterations,
            rna_summary=self.rna_summary,
            tcr_summary=self.tcr_summary,
            joint_summary=self.joint_summary,
            validation_summary=self.validation_summary,
            past_analyses=past_analyses or "No previous analyses.",
            research_state=research_state_summary or "No research ledger entries yet.",
            context_summary=self.context_summary,
            literature_summary=self.literature_summary,
            literature_candidates_summary=self.literature_candidates_summary,
            selected_literature_seed="No literature seed has been selected.",
            deepresearch_background=self.deepresearch_background or "No additional Deep Research background.",
        )
        if self.log_prompts:
            self.logger.log_prompt("user", prompt, "initial_analysis")
        return self._complete_structured(
            [
                {"role": "system", "content": self.coding_system_prompt},
                {"role": "user", "content": prompt},
            ],
            AnalysisPlan,
        )

    def select_literature_hypothesis(
        self,
        past_analyses: str,
        research_state_summary: str,
    ) -> LiteratureHypothesisChoice:
        prompt = self._read_prompt("select_literature_hypothesis.txt").format(
            rna_summary=self.rna_summary,
            tcr_summary=self.tcr_summary,
            joint_summary=self.joint_summary,
            validation_summary=self.validation_summary,
            research_state=research_state_summary or "No research ledger entries yet.",
            literature_summary=self.literature_summary,
            literature_candidates_summary=self.literature_candidates_summary,
            context_summary=self.context_summary,
            past_analyses=past_analyses or "No previous analyses.",
        )
        if self.log_prompts:
            self.logger.log_prompt("user", prompt, "select_literature_hypothesis")
        return self._complete_structured(
            [
                {"role": "system", "content": self.coding_system_prompt},
                {"role": "user", "content": prompt},
            ],
            LiteratureHypothesisChoice,
        )

    def _format_literature_choice(self, choice: LiteratureHypothesisChoice | None) -> str:
        if choice is None:
            return "No literature seed has been selected."
        return (
            f"Selected title: {choice.candidate_title}\n"
            f"Selected hypothesis: {choice.hypothesis}\n"
            f"Preferred analysis type: {choice.analysis_type}\n"
            f"Selection rationale: {choice.selection_rationale}\n"
            f"Expected evidence: {choice.expected_evidence}\n"
            f"Guardrails:\n" + "\n".join(f"- {item}" for item in choice.guardrail_notes)
        )

    def generate_initial_analysis(self, past_analyses: str, research_state_summary: str) -> AnalysisPlan:
        literature_choice: LiteratureHypothesisChoice | None = None
        if self.literature_candidates_summary != "No literature-derived hypothesis candidates.":
            try:
                literature_choice = self.select_literature_hypothesis(
                    past_analyses=past_analyses,
                    research_state_summary=research_state_summary,
                )
                self.logger.log_response(
                    self._format_literature_choice(literature_choice),
                    "selected_literature_hypothesis",
                )
            except Exception as exc:
                self.logger.warning(f"Literature hypothesis selection failed; falling back to freeform planning: {exc}")

        if literature_choice is not None and literature_choice.hypothesis.strip():
            return self.generate_analysis_from_hypothesis(
                literature_choice.hypothesis,
                past_analyses,
                research_state_summary,
                seed_context=self._format_literature_choice(literature_choice),
            )
        return self._generate_initial_analysis_freeform(past_analyses, research_state_summary)

    def generate_analysis_from_hypothesis(
        self,
        seeded_hypothesis: str,
        past_analyses: str,
        research_state_summary: str,
        seed_context: str = "No literature seed has been selected.",
        user_strategy_feedback: str = "",
    ) -> AnalysisPlan:
        def _make_prompt(strategy_feedback: str) -> str:
            return self._read_prompt("analysis_from_hypothesis.txt").format(
                hypothesis=seeded_hypothesis,
                CODING_GUIDELINES=self.coding_guidelines,
                max_iterations=self.max_iterations,
                rna_summary=self.rna_summary,
                tcr_summary=self.tcr_summary,
                joint_summary=self.joint_summary,
                validation_summary=self.validation_summary,
                past_analyses=past_analyses or "No previous analyses.",
                research_state=research_state_summary or "No research ledger entries yet.",
                context_summary=self.context_summary,
                literature_summary=self.literature_summary,
                literature_candidates_summary=self.literature_candidates_summary,
                selected_literature_seed=seed_context,
                user_strategy_feedback=strategy_feedback.strip() or "No extra strategy feedback.",
            )

        normalized_feedback = user_strategy_feedback.strip()
        prompt = _make_prompt(normalized_feedback)
        if self.log_prompts:
            self.logger.log_prompt("user", prompt, "seeded_hypothesis")
        result = self._complete_structured(
            [
                {"role": "system", "content": self.coding_system_prompt},
                {"role": "user", "content": prompt},
            ],
            AnalysisPlan,
        )
        if normalized_feedback and not _plan_mentions_strategy(result, normalized_feedback):
            retry_feedback = (
                normalized_feedback
                + "\n\nThe previous draft plan did not visibly satisfy the user strategy feedback. "
                  "Revise the plan so the requested analysis elements are explicit in the priority question, "
                  "remaining plan, and summary."
            )
            retry_prompt = _make_prompt(retry_feedback)
            if self.log_prompts:
                self.logger.log_prompt("user", retry_prompt, "seeded_hypothesis_retry")
            result = self._complete_structured(
                [
                    {"role": "system", "content": self.coding_system_prompt},
                    {"role": "user", "content": retry_prompt},
                ],
                AnalysisPlan,
            )
        result.hypothesis = seeded_hypothesis
        self.logger.log_response(
            (
                f"Seeded hypothesis: {seeded_hypothesis}\n"
                f"User strategy feedback: {normalized_feedback or 'No extra strategy feedback.'}\n"
                f"Plan analysis type: {result.analysis_type}\n"
                f"Priority question: {result.priority_question}\n"
                f"Evidence goal: {result.evidence_goal}\n"
                f"Plan steps:\n" + "\n".join(f"- {step}" for step in result.analysis_plan)
            ),
            "analysis_from_hypothesis",
        )
        return result

    def revise_analysis_plan(
        self,
        current_plan: AnalysisPlan,
        *,
        research_state_summary: str,
        past_analyses: str = "",
        user_strategy_feedback: str = "",
    ) -> AnalysisPlan:
        normalized_feedback = user_strategy_feedback.strip()

        def _make_prompt(strategy_feedback: str) -> str:
            return self._read_prompt("revise_analysis_plan.txt").format(
                current_plan_json=current_plan.model_dump_json(indent=2),
                CODING_GUIDELINES=self.coding_guidelines,
                max_iterations=self.max_iterations,
                rna_summary=self.rna_summary,
                tcr_summary=self.tcr_summary,
                joint_summary=self.joint_summary,
                validation_summary=self.validation_summary,
                past_analyses=past_analyses or "No previous analyses.",
                research_state=research_state_summary or "No research ledger entries yet.",
                context_summary=self.context_summary,
                literature_summary=self.literature_summary,
                literature_candidates_summary=self.literature_candidates_summary,
                user_strategy_feedback=strategy_feedback or "No extra strategy feedback.",
            )

        prompt = _make_prompt(normalized_feedback)
        if self.log_prompts:
            self.logger.log_prompt("user", prompt, "revise_analysis_plan")
        result = self._complete_structured(
            [
                {"role": "system", "content": self.coding_system_prompt},
                {"role": "user", "content": prompt},
            ],
            AnalysisPlan,
        )
        if normalized_feedback and not _plan_mentions_strategy(result, normalized_feedback):
            retry_feedback = (
                normalized_feedback
                + "\n\nThe revised plan still does not visibly satisfy the user strategy feedback. "
                  "Keep all untouched steps stable, but explicitly add the requested analysis elements."
            )
            retry_prompt = _make_prompt(retry_feedback)
            if self.log_prompts:
                self.logger.log_prompt("user", retry_prompt, "revise_analysis_plan_retry")
            result = self._complete_structured(
                [
                    {"role": "system", "content": self.coding_system_prompt},
                    {"role": "user", "content": retry_prompt},
                ],
                AnalysisPlan,
            )
        self.logger.log_response(
            (
                f"Current plan hypothesis: {current_plan.hypothesis}\n"
                f"User strategy feedback: {normalized_feedback or 'No extra strategy feedback.'}\n"
                f"Revised plan analysis type: {result.analysis_type}\n"
                f"Priority question: {result.priority_question}\n"
                f"Evidence goal: {result.evidence_goal}\n"
                f"Plan steps:\n" + "\n".join(f"- {step}" for step in result.analysis_plan)
            ),
            "revise_analysis_plan",
        )
        return result

    def critique_step(
        self,
        analysis: AnalysisPlan,
        past_analyses: str,
        notebook_cells: Optional[list],
        num_steps_left: int,
        research_state_summary: str,
    ) -> str:
        notebook_summary = summarize_notebook_cells(notebook_cells or [])
        documentation = ""
        if self.use_documentation:
            try:
                documentation = get_documentation(analysis.first_step_code)
            except Exception as exc:
                documentation = f"<documentation lookup failed: {exc}>"

        prompt = self._read_prompt("critic.txt").format(
            hypothesis=analysis.hypothesis,
            analysis_type=analysis.analysis_type,
            priority_question=analysis.priority_question,
            evidence_goal=analysis.evidence_goal,
            decision_rationale=analysis.decision_rationale,
            validation_checks="\n".join(f"- {item}" for item in analysis.validation_checks),
            analysis_plan="\n".join(f"- {step}" for step in analysis.analysis_plan),
            first_step_code=analysis.first_step_code,
            code_description=analysis.code_description,
            summary=analysis.summary,
            CODING_GUIDELINES=self.coding_guidelines,
            rna_summary=self.rna_summary,
            tcr_summary=self.tcr_summary,
            joint_summary=self.joint_summary,
            validation_summary=self.validation_summary,
            context_summary=self.context_summary,
            literature_summary=self.literature_summary,
            literature_candidates_summary=self.literature_candidates_summary,
            selected_literature_seed="Use the selected literature seed if one exists; otherwise critique the plan directly.",
            research_state=research_state_summary or "No research ledger entries yet.",
            past_analyses=past_analyses or "No previous analyses.",
            notebook_summary=notebook_summary or "Notebook is currently empty.",
            documentation=documentation or "No documentation available.",
            num_steps_left=num_steps_left,
        )
        return self._complete_text(
            [
                {"role": "system", "content": "You are a rigorous reviewer of scRNA + scTCR notebook analyses."},
                {"role": "user", "content": prompt},
            ]
        )

    def incorporate_critique(
        self,
        analysis: AnalysisPlan,
        critique_text: str,
        notebook_cells: Optional[list],
        num_steps_left: int,
        research_state_summary: str,
    ) -> AnalysisPlan:
        notebook_summary = summarize_notebook_cells(notebook_cells or [])
        prompt = self._read_prompt("incorporate_critique.txt").format(
            original_hypothesis=analysis.hypothesis,
            original_analysis_type=analysis.analysis_type,
            original_priority_question=analysis.priority_question,
            original_evidence_goal=analysis.evidence_goal,
            original_decision_rationale=analysis.decision_rationale,
            original_validation_checks="\n".join(f"- {item}" for item in analysis.validation_checks),
            original_plan="\n".join(f"- {step}" for step in analysis.analysis_plan),
            original_code=analysis.first_step_code,
            original_code_description=analysis.code_description,
            original_summary=analysis.summary,
            critique=critique_text,
            CODING_GUIDELINES=self.coding_guidelines,
            validation_summary=self.validation_summary,
            research_state=research_state_summary or "No research ledger entries yet.",
            notebook_summary=notebook_summary or "Notebook is currently empty.",
            literature_candidates_summary=self.literature_candidates_summary,
            num_steps_left=num_steps_left,
        )
        if self.log_prompts:
            self.logger.log_prompt("user", prompt, "incorporate_critique")
        return self._complete_structured(
            [
                {"role": "system", "content": self.coding_system_prompt},
                {"role": "user", "content": prompt},
            ],
            AnalysisPlan,
        )

    def get_feedback(
        self,
        analysis: AnalysisPlan,
        past_analyses: str,
        notebook_cells: Optional[list],
        num_steps_left: int,
        research_state_summary: str,
        rounds: int = 1,
    ) -> AnalysisPlan:
        current = analysis
        for _ in range(rounds):
            critique_text = self.critique_step(
                current,
                past_analyses,
                notebook_cells,
                num_steps_left,
                research_state_summary,
            )
            if not critique_text.strip():
                return current
            current = self.incorporate_critique(
                current,
                critique_text,
                notebook_cells,
                num_steps_left,
                research_state_summary,
            )
        return current

    def generate_idea(
        self,
        past_analyses: str,
        research_state_summary: str,
        analysis_idx: int | None = None,
        seeded_hypothesis: str | None = None,
    ) -> AnalysisPlan:
        if seeded_hypothesis:
            analysis = self.generate_analysis_from_hypothesis(
                seeded_hypothesis,
                past_analyses,
                research_state_summary,
            )
        else:
            analysis = self.generate_initial_analysis(past_analyses, research_state_summary)
        if self.use_self_critique:
            analysis = self.get_feedback(
                analysis,
                past_analyses,
                None,
                self.max_iterations,
                research_state_summary,
                rounds=1,
            )
        if analysis_idx is not None:
            self.logger.log_response(
                f"Hypothesis: {analysis.hypothesis}\n"
                f"Type: {analysis.analysis_type}\n"
                f"Priority question: {analysis.priority_question}\n"
                f"Evidence goal: {analysis.evidence_goal}\n"
                f"Decision rationale: {analysis.decision_rationale}\n"
                f"Validation checks:\n" + "\n".join(f"- {item}" for item in analysis.validation_checks) + "\n"
                f"Plan:\n" + "\n".join(f"{idx + 1}. {step}" for idx, step in enumerate(analysis.analysis_plan)),
                f"analysis_{analysis_idx + 1}_initial_plan",
            )
        return analysis

    def generate_next_step(
        self,
        current_analysis: AnalysisPlan,
        past_analyses: str,
        notebook_cells: list,
        num_steps_left: int,
        research_state_summary: str,
        step_validation_summary: str,
    ) -> AnalysisPlan:
        notebook_summary = summarize_notebook_cells(notebook_cells)
        prompt = self._read_prompt("next_step.txt").format(
            hypothesis=current_analysis.hypothesis,
            analysis_type=current_analysis.analysis_type,
            priority_question=current_analysis.priority_question,
            evidence_goal=current_analysis.evidence_goal,
            decision_rationale=current_analysis.decision_rationale,
            validation_checks="\n".join(f"- {item}" for item in current_analysis.validation_checks),
            analysis_plan="\n".join(f"- {step}" for step in current_analysis.analysis_plan),
            current_code=current_analysis.first_step_code,
            current_code_description=current_analysis.code_description,
            summary=current_analysis.summary,
            CODING_GUIDELINES=self.coding_guidelines,
            rna_summary=self.rna_summary,
            tcr_summary=self.tcr_summary,
            joint_summary=self.joint_summary,
            validation_summary=self.validation_summary,
            context_summary=self.context_summary,
            literature_summary=self.literature_summary,
            literature_candidates_summary=self.literature_candidates_summary,
            selected_literature_seed="Keep following the literature-guided direction if it still fits the evidence.",
            research_state=research_state_summary or "No research ledger entries yet.",
            step_validation_summary=step_validation_summary or "No step validation notes.",
            past_analyses=past_analyses or "No previous analyses.",
            notebook_summary=notebook_summary or "Notebook is currently empty.",
            num_steps_left=num_steps_left,
        )
        if self.log_prompts:
            self.logger.log_prompt("user", prompt, "next_step")
        next_analysis = self._complete_structured(
            [
                {"role": "system", "content": self.coding_system_prompt},
                {"role": "user", "content": prompt},
            ],
            AnalysisPlan,
        )
        if self.use_self_critique:
            next_analysis = self.get_feedback(
                next_analysis,
                past_analyses,
                notebook_cells,
                num_steps_left,
                research_state_summary,
                rounds=1,
            )
        return next_analysis

    def summarize_step_research(
        self,
        current_analysis: AnalysisPlan,
        notebook_cells: list,
        text_output: str,
        research_state_summary: str,
        step_validation_summary: str,
    ) -> ResearchStepUpdate:
        notebook_summary = summarize_notebook_cells(notebook_cells)
        prompt = self._read_prompt("step_research_update.txt").format(
            hypothesis=current_analysis.hypothesis,
            analysis_type=current_analysis.analysis_type,
            priority_question=current_analysis.priority_question,
            evidence_goal=current_analysis.evidence_goal,
            decision_rationale=current_analysis.decision_rationale,
            validation_checks="\n".join(f"- {item}" for item in current_analysis.validation_checks),
            code=current_analysis.first_step_code,
            code_description=current_analysis.code_description,
            notebook_summary=notebook_summary or "Notebook is currently empty.",
            text_output=text_output or "No text output.",
            validation_summary=self.validation_summary,
            research_state=research_state_summary or "No research ledger entries yet.",
            step_validation_summary=step_validation_summary or "No step validation notes.",
        )
        if self.log_prompts:
            self.logger.log_prompt("user", prompt, "step_research_update")
        return self._complete_structured(
            [
                {"role": "system", "content": "You update the evidence ledger for a scRNA + scTCR research agent."},
                {"role": "user", "content": prompt},
            ],
            ResearchStepUpdate,
        )

