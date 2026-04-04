"""Desktop GUI for integrated scRNA-TCR, scRNA-ST, and scRNA-ATAC analysis."""

from __future__ import annotations

import datetime as dt
import os
import queue
import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from scrat_agent.agent import ScRATAgent
from scrat_agent.preprocess import prepare_dataset as prepare_atac_dataset
from scrst_agent.agent import ScRSTAgent
from scrst_agent.preprocess import prepare_dataset as prepare_spatial_dataset

from .agent import ScRTAgent
from .hypothesis import CandidateHypothesis, CandidateHypothesisMenu
from .interactive import format_candidate_menu_markdown, read_json, write_json
from .preprocess import prepare_dataset as prepare_tcr_dataset

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


ANALYSIS_MODES = ("scRNA-TCR", "scRNA-ST", "scRNA-ATAC")


def load_local_env_files(project_root: Path) -> None:
    if load_dotenv is None:
        return
    for directory in (project_root, project_root.parent, Path.cwd()):
        for name in (".env", "OPENAI.env", "deepseek.env"):
            env_path = directory / name
            if env_path.exists():
                load_dotenv(env_path, override=False)


class GuiTextHandler:
    def __init__(self, callback):
        self.callback = callback

    def write(self, text: str) -> None:
        if text:
            self.callback(text)

    def flush(self) -> None:  # pragma: no cover
        return


class ScRTDesktopApp(tk.Tk):
    """Desktop UI for integrated single-cell workflows."""

    def __init__(self, project_root: str | Path) -> None:
        super().__init__()
        self.project_root = Path(project_root).resolve()
        load_local_env_files(self.project_root)

        self.title("scMAA")
        self.geometry("1560x980")
        self.minsize(1280, 840)

        self.message_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.current_session_dir: Path | None = None
        self.current_candidates: list[dict] = []
        self.current_task: threading.Thread | None = None
        self.hypothesis_header_var = tk.StringVar(value="Selected hypothesis")
        self.plan_header_var = tk.StringVar(value="Draft or approved plan")

        self._build_variables()
        self._build_layout()
        self._poll_queue()
        self._set_defaults()
        self._update_mode_ui()
        self.after(50, self._maximize_window)

    def _maximize_window(self) -> None:
        try:
            self.state("zoomed")
        except Exception:
            try:
                self.attributes("-zoomed", True)
            except Exception:
                return

    def _build_variables(self) -> None:
        sessions_home = self.project_root / "sessions"
        prepared_home = self.project_root / "prepared"
        self.analysis_mode_var = tk.StringVar(value=ANALYSIS_MODES[0])

        self.raw_input_var = tk.StringVar()
        self.secondary_raw_input_var = tk.StringVar()
        self.prep_output_var = tk.StringVar(value=str(prepared_home))
        self.annotation_model_var = tk.StringVar(value="gpt-4o")
        self.annotation_notes_var = tk.StringVar()
        self.min_genes_var = tk.StringVar(value="200")
        self.min_cells_var = tk.StringVar(value="3")
        self.max_pct_mt_var = tk.StringVar(value="15")
        self.leiden_resolution_var = tk.StringVar(value="0.8")

        self.rna_h5ad_var = tk.StringVar()
        self.modality_input_var = tk.StringVar()
        self.research_brief_var = tk.StringVar()
        self.literature_path_var = tk.StringVar()
        self.session_name_var = tk.StringVar()
        self.output_home_var = tk.StringVar(value=str(sessions_home))
        self.model_name_var = tk.StringVar(value="gpt-4o")
        self.spatial_mapping_method_var = tk.StringVar(value="auto")
        self.with_figure_var = tk.BooleanVar(value=True)
        self.log_prompts_var = tk.BooleanVar(value=False)

    def _set_defaults(self) -> None:
        self.session_name_var.set(f"session_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}")

    def _build_layout(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=0)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        left = ttk.Frame(root)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 10))
        right = ttk.Frame(root)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        self._build_prepare_panel(left)
        self._build_agent_panel(left)
        self._build_action_panel(left)
        self._build_candidates_panel(right)
        self._build_log_panel(right)

    def _build_prepare_panel(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="1. Raw Data Preparation", padding=8)
        frame.pack(fill="x", pady=(0, 10))
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Analysis mode").grid(row=0, column=0, sticky="w", pady=4)
        mode_box = ttk.Combobox(
            frame,
            textvariable=self.analysis_mode_var,
            values=ANALYSIS_MODES,
            state="readonly",
            width=18,
        )
        mode_box.grid(row=0, column=1, sticky="w", pady=4)
        mode_box.bind("<<ComboboxSelected>>", lambda _event: self._update_mode_ui())

        self.prepare_hint_label = ttk.Label(frame, text="")
        self.prepare_hint_label.grid(row=1, column=0, columnspan=4, sticky="w", pady=(0, 4))

        self.raw_primary_label = ttk.Label(frame, text="Raw input folder")
        self.raw_primary_label.grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.raw_input_var, width=42).grid(row=2, column=1, sticky="ew", pady=4)
        ttk.Button(frame, text="Folder", command=self._browse_raw_input_folder).grid(row=2, column=2, sticky="ew", padx=(6, 0), pady=4)
        ttk.Button(frame, text="RAW.tar", command=self._browse_raw_input_tar).grid(row=2, column=3, sticky="ew", padx=(6, 0), pady=4)

        self.raw_secondary_label = ttk.Label(frame, text="Spatial raw folder")
        self.raw_secondary_entry = ttk.Entry(frame, textvariable=self.secondary_raw_input_var, width=42)
        self.raw_secondary_folder_button = ttk.Button(frame, text="Folder", command=self._browse_secondary_raw_input_folder)
        self.raw_secondary_tar_button = ttk.Button(frame, text="RAW.tar", command=self._browse_secondary_raw_input_tar)

        self._path_row(frame, "Output dir", self.prep_output_var, self._browse_prep_output_dir, row=4)
        self._path_row(frame, "Annotation notes", self.annotation_notes_var, self._browse_annotation_notes, row=5, required=False)

        ttk.Label(frame, text="Annotation model").grid(row=6, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.annotation_model_var, width=28).grid(row=6, column=1, sticky="ew", pady=4)
        ttk.Label(frame, text="Min genes").grid(row=7, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.min_genes_var, width=10).grid(row=7, column=1, sticky="w", pady=4)
        ttk.Label(frame, text="Min cells").grid(row=8, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.min_cells_var, width=10).grid(row=8, column=1, sticky="w", pady=4)
        ttk.Label(frame, text="Max pct mt").grid(row=9, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.max_pct_mt_var, width=10).grid(row=9, column=1, sticky="w", pady=4)
        ttk.Label(frame, text="Leiden resolution").grid(row=10, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.leiden_resolution_var, width=10).grid(row=10, column=1, sticky="w", pady=4)
        ttk.Button(frame, text="Prepare Raw Data", command=self.prepare_raw_data).grid(row=11, column=0, columnspan=4, sticky="ew", pady=(8, 0))

    def _build_agent_panel(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="2. Interactive Analysis", padding=8)
        frame.pack(fill="x", pady=(0, 10))
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Analysis mode").grid(row=0, column=0, sticky="w", pady=4)
        mode_box = ttk.Combobox(
            frame,
            textvariable=self.analysis_mode_var,
            values=ANALYSIS_MODES,
            state="readonly",
            width=18,
        )
        mode_box.grid(row=0, column=1, sticky="w", pady=4)
        mode_box.bind("<<ComboboxSelected>>", lambda _event: self._update_mode_ui())

        self.agent_hint_label = ttk.Label(frame, text="")
        self.agent_hint_label.grid(row=1, column=0, columnspan=4, sticky="w", pady=(0, 4))

        self._path_row(frame, "RNA h5ad", self.rna_h5ad_var, self._browse_rna_h5ad, row=2)
        self.modality_label = ttk.Label(frame, text="TCR table")
        self.modality_label.grid(row=3, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.modality_input_var, width=42).grid(row=3, column=1, sticky="ew", pady=4)
        ttk.Button(frame, text="Browse", command=self._browse_modality_input).grid(row=3, column=2, sticky="ew", padx=(6, 0), pady=4)
        self._path_row(frame, "Research brief", self.research_brief_var, self._browse_brief, row=4)
        self._path_row(frame, "Literature", self.literature_path_var, self._browse_literature, row=5, required=False)
        self._path_row(frame, "Sessions dir", self.output_home_var, self._browse_sessions_dir, row=6)

        ttk.Label(frame, text="Session name").grid(row=7, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.session_name_var).grid(row=7, column=1, sticky="ew", pady=4)
        ttk.Label(frame, text="Model").grid(row=8, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.model_name_var, width=28).grid(row=8, column=1, sticky="ew", pady=4)
        self.mapping_method_label = ttk.Label(frame, text="Spatial mapping")
        self.mapping_method_combo = ttk.Combobox(
            frame,
            textvariable=self.spatial_mapping_method_var,
            values=("auto", "marker_transfer", "cell2location"),
            state="readonly",
            width=18,
        )
        self.mapping_method_label.grid(row=9, column=0, sticky="w", pady=4)
        self.mapping_method_combo.grid(row=9, column=1, sticky="w", pady=4)
        ttk.Checkbutton(frame, text="Generate figure", variable=self.with_figure_var).grid(row=10, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Checkbutton(frame, text="Save prompts", variable=self.log_prompts_var).grid(row=11, column=0, columnspan=2, sticky="w", pady=2)

    def _build_action_panel(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="3. Actions", padding=6)
        frame.pack(fill="x")
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        buttons = [
            ("Generate Hypotheses", self.generate_hypotheses, 0, 0),
            ("Regenerate Hypotheses", self.regenerate_hypotheses, 0, 1),
            ("Approve Hypothesis", self.approve_selected_hypothesis, 1, 0),
            ("Generate Plan", self.generate_analysis_plan, 1, 1),
            ("Regenerate Plan", self.regenerate_analysis_plan, 2, 0),
            ("Approve Plan", self.approve_plan, 2, 1),
            ("Run Analysis", self.run_analysis, 3, 0),
            ("Open Session Folder", self.open_session_folder, 3, 1),
        ]
        for text, command, row, column in buttons:
            ttk.Button(frame, text=text, command=command).grid(row=row, column=column, sticky="ew", padx=2, pady=1)

    def _build_candidates_panel(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="Candidates And Plan Review", padding=10)
        frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(1, weight=1)

        ttk.Label(frame, text="Candidate hypotheses").grid(row=0, column=0, sticky="w")
        self.candidate_list = tk.Listbox(frame, height=12, exportselection=False)
        self.candidate_list.grid(row=1, column=0, sticky="nsw", padx=(0, 10))
        self.candidate_list.bind("<<ListboxSelect>>", self._on_candidate_selected)

        right = ttk.Frame(frame)
        right.grid(row=1, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        right.rowconfigure(4, weight=1)
        right.rowconfigure(7, weight=1)
        right.rowconfigure(10, weight=1)

        ttk.Label(right, textvariable=self.hypothesis_header_var).grid(row=0, column=0, sticky="w")
        self.hypothesis_detail = tk.Text(right, wrap="word", width=74, height=12)
        self.hypothesis_detail.grid(row=1, column=0, sticky="nsew")
        ttk.Label(right, text="Hypothesis feedback").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Label(
            right,
            text="Use this text to ask for more candidates, narrow the question, or revise the selected hypothesis.",
        ).grid(row=3, column=0, sticky="w")
        self.hypothesis_feedback_text = tk.Text(right, wrap="word", height=5)
        self.hypothesis_feedback_text.grid(row=4, column=0, sticky="nsew")

        ttk.Label(right, textvariable=self.plan_header_var).grid(row=5, column=0, sticky="w", pady=(10, 0))
        self.plan_detail = tk.Text(right, wrap="word", width=74, height=12)
        self.plan_detail.grid(row=6, column=0, sticky="nsew")
        ttk.Label(right, text="Plan feedback").grid(row=7, column=0, sticky="w", pady=(8, 0))
        ttk.Label(
            right,
            text="Use this text only to change the analysis plan for the approved hypothesis.",
        ).grid(row=8, column=0, sticky="w")
        self.plan_feedback_text = tk.Text(right, wrap="word", height=5)
        self.plan_feedback_text.grid(row=9, column=0, sticky="nsew")

    def _build_log_panel(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="Run Log", padding=10)
        frame.grid(row=1, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        self.log_text = tk.Text(frame, wrap="word", height=16)
        self.log_text.grid(row=0, column=0, sticky="nsew")

    def _path_row(self, parent, label: str, variable: tk.StringVar, browse_command, *, row: int, required: bool = True) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=variable, width=42).grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Button(parent, text="Browse", command=browse_command).grid(row=row, column=2, sticky="ew", padx=(6, 0), pady=4)
        if not required:
            ttk.Label(parent, text="optional").grid(row=row, column=3, sticky="w", padx=(6, 0))

    def _append_log(self, text: str) -> None:
        self.log_text.insert("end", text)
        self.log_text.see("end")

    def _set_hypothesis_text(self, text: str, heading: str) -> None:
        self.hypothesis_header_var.set(heading)
        self.hypothesis_detail.delete("1.0", "end")
        self.hypothesis_detail.insert("1.0", text)
        self.hypothesis_detail.yview_moveto(0.0)

    def _set_plan_text(self, text: str, heading: str) -> None:
        self.plan_header_var.set(heading)
        self.plan_detail.delete("1.0", "end")
        self.plan_detail.insert("1.0", text)
        self.plan_detail.yview_moveto(0.0)

    def _queue_log(self, text: str) -> None:
        self.message_queue.put(("log", text))

    def _poll_queue(self) -> None:
        while True:
            try:
                kind, payload = self.message_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self._append_log(str(payload))
            elif kind == "done":
                callback = payload
                callback()
            elif kind == "error":
                title, message = payload
                messagebox.showerror(title, message)
        self.after(200, self._poll_queue)

    def _run_background(self, title: str, func) -> None:
        if self.current_task is not None and self.current_task.is_alive():
            messagebox.showwarning("Busy", "A task is already running.")
            return

        def runner():
            try:
                self._queue_log(f"\n[{title}] started\n")
                func()
                self.message_queue.put(("done", lambda: self._append_log(f"[{title}] finished\n")))
            except Exception as exc:  # pragma: no cover
                details = "".join(traceback.format_exception(exc))
                self._queue_log(details + "\n")
                self.message_queue.put(("error", (title, str(exc))))

        self.current_task = threading.Thread(target=runner, daemon=True)
        self.current_task.start()

    def _analysis_mode(self) -> str:
        mode = self.analysis_mode_var.get().strip()
        return mode if mode in ANALYSIS_MODES else ANALYSIS_MODES[0]

    def _mode_is_spatial(self) -> bool:
        return self._analysis_mode() == "scRNA-ST"

    def _mode_is_atac(self) -> bool:
        return self._analysis_mode() == "scRNA-ATAC"

    def _update_mode_ui(self) -> None:
        mode = self._analysis_mode()
        if self._mode_is_spatial():
            self.raw_primary_label.config(text="scRNA raw folder")
            self.raw_secondary_label.grid(row=3, column=0, sticky="w", pady=4)
            self.raw_secondary_entry.grid(row=3, column=1, sticky="ew", pady=4)
            self.raw_secondary_folder_button.grid(row=3, column=2, sticky="ew", padx=(6, 0), pady=4)
            self.raw_secondary_tar_button.grid(row=3, column=3, sticky="ew", padx=(6, 0), pady=4)
            self.prepare_hint_label.config(
                text="Mode: scRNA-ST. Provide one raw input for scRNA and one raw input for spatial transcriptomics."
            )
            self.modality_label.config(text="Spatial h5ad")
            self.agent_hint_label.config(
                text="Mode: scRNA-ST. Provide a processed RNA h5ad together with a processed spatial h5ad."
            )
            self.mapping_method_label.grid()
            self.mapping_method_combo.grid()
        elif self._mode_is_atac():
            self.raw_primary_label.config(text="Raw input folder")
            self.raw_secondary_label.grid_remove()
            self.raw_secondary_entry.grid_remove()
            self.raw_secondary_folder_button.grid_remove()
            self.raw_secondary_tar_button.grid_remove()
            self.prepare_hint_label.config(
                text="Mode: scRNA-ATAC. Provide one 10x multiome raw folder or RAW.tar. The preparation step will generate processed RNA and ATAC h5ad files."
            )
            self.modality_label.config(text="ATAC h5ad")
            self.agent_hint_label.config(
                text="Mode: scRNA-ATAC. Provide a processed RNA h5ad together with a processed ATAC h5ad."
            )
            self.mapping_method_label.grid_remove()
            self.mapping_method_combo.grid_remove()
        else:
            self.raw_primary_label.config(text="Raw input folder")
            self.raw_secondary_label.grid_remove()
            self.raw_secondary_entry.grid_remove()
            self.raw_secondary_folder_button.grid_remove()
            self.raw_secondary_tar_button.grid_remove()
            self.prepare_hint_label.config(
                text="Mode: scRNA-TCR. Use one raw folder that contains extracted GEO RNA and TCR files."
            )
            self.modality_label.config(text="TCR table")
            self.agent_hint_label.config(
                text="Mode: scRNA-TCR. Provide a processed RNA h5ad together with a merged TCR table."
            )
            self.mapping_method_label.grid_remove()
            self.mapping_method_combo.grid_remove()
        self._append_log(f"Analysis mode set to {mode}\n")

    def _build_agent(self, *, analysis_name: str, output_home: str):
        literature_paths = [self.literature_path_var.get().strip()] if self.literature_path_var.get().strip() else []
        model_name = self.model_name_var.get().strip() or "gpt-4o"
        common_kwargs = {
            "rna_h5ad_path": self.rna_h5ad_var.get().strip(),
            "research_brief_path": self.research_brief_var.get().strip(),
            "literature_paths": literature_paths or None,
            "analysis_name": analysis_name,
            "model_name": model_name,
            "hypothesis_model": model_name,
            "execution_model": model_name,
            "vision_model": model_name,
            "num_analyses": 1,
            "max_iterations": 6,
            "output_home": output_home,
            "generate_publication_figure": self.with_figure_var.get(),
            "log_prompts": self.log_prompts_var.get(),
        }
        if self._mode_is_spatial():
            return ScRSTAgent(
                spatial_h5ad_path=self.modality_input_var.get().strip(),
                spatial_mapping_method=self.spatial_mapping_method_var.get().strip() or "auto",
                **common_kwargs,
            )
        if self._mode_is_atac():
            return ScRATAgent(
                atac_h5ad_path=self.modality_input_var.get().strip(),
                **common_kwargs,
            )
        return ScRTAgent(
            tcr_path=self.modality_input_var.get().strip(),
            **common_kwargs,
        )

    def _session_dir(self) -> Path:
        name = self.session_name_var.get().strip() or f"session_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_name_var.set(name)
        base = Path(self.output_home_var.get().strip() or str(self.project_root / "sessions")).resolve()
        base.mkdir(parents=True, exist_ok=True)
        return base / name

    def _write_session_config(self, session_dir: Path, hypothesis_feedback: str = "", plan_feedback: str = "") -> None:
        config = {
            "analysis_mode": self._analysis_mode(),
            "rna_h5ad_path": self.rna_h5ad_var.get().strip(),
            "modality_input_path": self.modality_input_var.get().strip(),
            "research_brief_path": self.research_brief_var.get().strip(),
            "literature_path": [self.literature_path_var.get().strip()] if self.literature_path_var.get().strip() else [],
            "model_name": self.model_name_var.get().strip() or "gpt-4o",
            "with_figure": self.with_figure_var.get(),
            "log_prompts": self.log_prompts_var.get(),
            "hypothesis_feedback": hypothesis_feedback,
            "plan_feedback": plan_feedback,
            "spatial_mapping_method": self.spatial_mapping_method_var.get().strip() or "auto",
        }
        if self._mode_is_spatial():
            config["spatial_h5ad_path"] = self.modality_input_var.get().strip()
        elif self._mode_is_atac():
            config["atac_h5ad_path"] = self.modality_input_var.get().strip()
        else:
            config["tcr_path"] = self.modality_input_var.get().strip()
        write_json(session_dir / "session_config.json", config)

    def _load_session_config(self, session_dir: Path) -> None:
        config_path = session_dir / "session_config.json"
        if not config_path.exists():
            return
        config = read_json(config_path)
        self.analysis_mode_var.set(config.get("analysis_mode", "scRNA-TCR"))
        self._update_mode_ui()
        self.rna_h5ad_var.set(config.get("rna_h5ad_path", ""))
        modality_path = (
            config.get("modality_input_path")
            or config.get("tcr_path")
            or config.get("spatial_h5ad_path")
            or config.get("atac_h5ad_path")
            or ""
        )
        self.modality_input_var.set(modality_path)
        self.research_brief_var.set(config.get("research_brief_path", ""))
        literature_paths = config.get("literature_path", [])
        self.literature_path_var.set(literature_paths[0] if literature_paths else "")
        self.model_name_var.set(config.get("model_name", "gpt-4o"))
        self.spatial_mapping_method_var.set(config.get("spatial_mapping_method", "auto"))
        self.with_figure_var.set(bool(config.get("with_figure", True)))
        self.log_prompts_var.set(bool(config.get("log_prompts", False)))
        hypothesis_feedback = config.get("hypothesis_feedback", "")
        if hypothesis_feedback:
            self.hypothesis_feedback_text.delete("1.0", "end")
            self.hypothesis_feedback_text.insert("1.0", hypothesis_feedback)
        plan_feedback = config.get("plan_feedback", "")
        if plan_feedback:
            self.plan_feedback_text.delete("1.0", "end")
            self.plan_feedback_text.insert("1.0", plan_feedback)

    def prepare_raw_data(self) -> None:
        raw_input = self.raw_input_var.get().strip()
        output_dir = self.prep_output_var.get().strip()
        if not raw_input or not output_dir:
            messagebox.showwarning("Missing input", "Please provide the required raw input path and output directory.")
            return
        if self._mode_is_spatial() and not self.secondary_raw_input_var.get().strip():
            messagebox.showwarning("Missing input", "Please provide the spatial raw input path.")
            return

        def task():
            if self._mode_is_spatial():
                result = prepare_spatial_dataset(
                    rna_raw_input_path=raw_input,
                    spatial_raw_input_path=self.secondary_raw_input_var.get().strip(),
                    output_dir=output_dir,
                    annotation_model=self.annotation_model_var.get().strip() or "gpt-4o",
                    annotation_notes_path=self.annotation_notes_var.get().strip() or None,
                    min_genes=int(self.min_genes_var.get().strip() or "200"),
                    min_cells=int(self.min_cells_var.get().strip() or "3"),
                    max_pct_mt=float(self.max_pct_mt_var.get().strip() or "15"),
                    leiden_resolution=float(self.leiden_resolution_var.get().strip() or "0.8"),
                    log_prompts=self.log_prompts_var.get(),
                )
                self.rna_h5ad_var.set(str(result.rna_h5ad_path))
                self.modality_input_var.set(str(result.spatial_h5ad_path))
                self._queue_log(f"Prepared RNA: {result.rna_h5ad_path}\n")
                self._queue_log(f"Prepared spatial: {result.spatial_h5ad_path}\n")
            elif self._mode_is_atac():
                result = prepare_atac_dataset(
                    raw_input_path=raw_input,
                    output_dir=output_dir,
                    annotation_model=self.annotation_model_var.get().strip() or "gpt-4o",
                    annotation_notes_path=self.annotation_notes_var.get().strip() or None,
                    min_genes=int(self.min_genes_var.get().strip() or "200"),
                    min_cells=int(self.min_cells_var.get().strip() or "3"),
                    max_pct_mt=float(self.max_pct_mt_var.get().strip() or "15"),
                    leiden_resolution=float(self.leiden_resolution_var.get().strip() or "0.8"),
                    log_prompts=self.log_prompts_var.get(),
                )
                self.rna_h5ad_var.set(str(result.rna_h5ad_path))
                self.modality_input_var.set(str(result.atac_h5ad_path))
                self._queue_log(f"Prepared RNA: {result.rna_h5ad_path}\n")
                self._queue_log(f"Prepared ATAC: {result.atac_h5ad_path}\n")
            else:
                result = prepare_tcr_dataset(
                    raw_input_path=raw_input,
                    output_dir=output_dir,
                    annotation_model=self.annotation_model_var.get().strip() or "gpt-4o",
                    annotation_notes_path=self.annotation_notes_var.get().strip() or None,
                    min_genes=int(self.min_genes_var.get().strip() or "200"),
                    min_cells=int(self.min_cells_var.get().strip() or "3"),
                    max_pct_mt=float(self.max_pct_mt_var.get().strip() or "15"),
                    leiden_resolution=float(self.leiden_resolution_var.get().strip() or "0.8"),
                    log_prompts=self.log_prompts_var.get(),
                )
                self.rna_h5ad_var.set(str(result.rna_h5ad_path))
                self.modality_input_var.set(str(result.tcr_table_path))
                self._queue_log(f"Prepared RNA: {result.rna_h5ad_path}\n")
                self._queue_log(f"Prepared TCR: {result.tcr_table_path}\n")

        self._run_background("Prepare raw data", task)

    def generate_hypotheses(self) -> None:
        session_dir = self._session_dir()

        def task():
            session_dir.mkdir(parents=True, exist_ok=True)
            agent = self._build_agent(analysis_name=session_dir.name, output_home=str(session_dir.parent))
            menu = agent.prepare_candidate_hypotheses(user_feedback="")
            write_json(session_dir / "candidate_hypotheses.json", menu.model_dump())
            (session_dir / "candidate_hypotheses.md").write_text(format_candidate_menu_markdown(menu), encoding="utf-8")
            for artifact_name in (
                "draft_hypothesis.txt",
                "approved_hypothesis.txt",
                "draft_plan.json",
                "draft_plan.md",
                "approved_plan.json",
                "approved_plan.md",
                "user_feedback.txt",
                "draft_strategy_feedback.txt",
                "approved_strategy_feedback.txt",
            ):
                artifact_path = session_dir / artifact_name
                if artifact_path.exists():
                    artifact_path.unlink()
            feedback_path = session_dir / "candidate_generation_feedback.txt"
            if feedback_path.exists():
                feedback_path.unlink()
            self._write_session_config(
                session_dir,
                hypothesis_feedback=self.hypothesis_feedback_text.get("1.0", "end").strip(),
                plan_feedback=self.plan_feedback_text.get("1.0", "end").strip(),
            )
            self.current_session_dir = session_dir
            self.current_candidates = [item.model_dump() for item in menu.candidates]
            self.message_queue.put(("done", lambda menu_payload=menu.model_dump(): self._show_candidates(menu_payload)))
            self._queue_log(f"Generated {len(menu.candidates)} candidate hypotheses.\n")
            self._queue_log("Select one candidate, then click Approve Hypothesis.\n")

        self._run_background("Generate hypotheses", task)

    def _current_menu_summary(self) -> str:
        if not self.current_candidates:
            return "No current candidates."
        lines = []
        for idx, item in enumerate(self.current_candidates, start=1):
            lines.append(
                f"{idx}. {item.get('title', '')}\n"
                f"   hypothesis: {item.get('hypothesis', '')}\n"
                f"   analysis_type: {item.get('preferred_analysis_type', '')}\n"
                f"   first_test: {item.get('first_test', '')}"
            )
        return "\n".join(lines)

    def regenerate_hypotheses(self) -> None:
        if not self.current_candidates:
            messagebox.showwarning("No candidates", "Generate hypotheses first.")
            return
        if not self.candidate_list.curselection():
            messagebox.showwarning("No selection", "Select one hypothesis to revise.")
            return
        session_dir = self.current_session_dir or self._session_dir()
        feedback_text = self.hypothesis_feedback_text.get("1.0", "end").strip()
        if not feedback_text:
            messagebox.showwarning("No feedback", "Enter hypothesis feedback before regenerating hypotheses.")
            return
        selected_index = self.candidate_list.curselection()[0]
        selected_candidate = dict(self.current_candidates[selected_index])

        def task():
            session_dir.mkdir(parents=True, exist_ok=True)
            agent = self._build_agent(analysis_name=session_dir.name, output_home=str(session_dir.parent))
            revised_hypothesis = agent.revise_hypothesis(
                hypothesis=selected_candidate.get("hypothesis", "").strip(),
                user_feedback=feedback_text,
            )
            revised_candidate_payload = dict(selected_candidate)
            revised_candidate_payload["hypothesis"] = revised_hypothesis
            revised_menu = CandidateHypothesisMenu(
                research_focus="User-revised hypothesis under review.",
                candidates=[CandidateHypothesis(**revised_candidate_payload)],
            )
            write_json(session_dir / "candidate_hypotheses.json", revised_menu.model_dump())
            (session_dir / "candidate_hypotheses.md").write_text(
                format_candidate_menu_markdown(revised_menu),
                encoding="utf-8",
            )
            (session_dir / "candidate_generation_feedback.txt").write_text(feedback_text + "\n", encoding="utf-8")
            for artifact_name in (
                "draft_hypothesis.txt",
                "approved_hypothesis.txt",
                "draft_plan.json",
                "draft_plan.md",
                "approved_plan.json",
                "approved_plan.md",
                "user_feedback.txt",
                "draft_strategy_feedback.txt",
                "approved_strategy_feedback.txt",
            ):
                artifact_path = session_dir / artifact_name
                if artifact_path.exists():
                    artifact_path.unlink()
            self._write_session_config(
                session_dir,
                hypothesis_feedback=feedback_text,
                plan_feedback=self.plan_feedback_text.get("1.0", "end").strip(),
            )
            self.current_session_dir = session_dir
            self.current_candidates = [revised_candidate_payload]
            self.message_queue.put(
                ("done", lambda menu_payload=revised_menu.model_dump(): self._show_candidates(menu_payload))
            )
            self._queue_log("Regenerated 1 revised hypothesis from the selected candidate.\n")
            self._queue_log(f"Hypothesis feedback applied: {feedback_text}\n")
            self._queue_log("Review the revised hypothesis, then click Approve Hypothesis.\n")

        self._run_background("Regenerate hypotheses", task)

    def _show_candidates(self, payload: dict) -> None:
        self.candidate_list.delete(0, "end")
        self.current_candidates = payload.get("candidates", [])
        for idx, item in enumerate(self.current_candidates, start=1):
            self.candidate_list.insert("end", f"{idx}. {item.get('title', 'Untitled')}")
        if self.current_candidates:
            self.candidate_list.selection_set(0)
            self._render_candidate_detail(0)
        self._set_plan_text("No plan has been generated yet.\n\nApprove a hypothesis first, then generate a draft plan.", "Draft or approved plan")

    def _on_candidate_selected(self, _event=None) -> None:
        if not self.candidate_list.curselection():
            return
        self._render_candidate_detail(self.candidate_list.curselection()[0])

    def _render_candidate_detail(self, index: int) -> None:
        if index < 0 or index >= len(self.current_candidates):
            return
        item = self.current_candidates[index]
        lines = [
            f"Title: {item.get('title', '')}",
            "",
            f"Hypothesis: {item.get('hypothesis', '')}",
            "",
            f"Rationale: {item.get('rationale', '')}",
            "",
            f"Preferred analysis type: {item.get('preferred_analysis_type', '')}",
            "",
            f"First test: {item.get('first_test', '')}",
            "",
            "Cautions:",
        ]
        lines.extend(f"- {text}" for text in item.get("cautions", []))
        self._set_hypothesis_text("\n".join(lines), "Selected hypothesis")

    def _plan_lines(self, payload: dict, heading: str = "Draft analysis plan") -> list[str]:
        lines = [
            heading,
            "",
            f"Hypothesis: {payload.get('hypothesis', '')}",
            f"Analysis type: {payload.get('analysis_type', '')}",
            "",
            f"Priority question: {payload.get('priority_question', '')}",
            "",
            f"Evidence goal: {payload.get('evidence_goal', '')}",
            "",
            f"Decision rationale: {payload.get('decision_rationale', '')}",
            "",
            "Validation checks:",
        ]
        lines.extend(f"- {item}" for item in payload.get("validation_checks", []))
        lines.extend(["", "Remaining plan:"])
        lines.extend(f"{idx + 1}. {item}" for idx, item in enumerate(payload.get("analysis_plan", [])))
        lines.extend(
            [
                "",
                f"Code description: {payload.get('code_description', '')}",
                "",
                "Summary:",
                payload.get("summary", ""),
            ]
        )
        return lines

    def _show_plan(self, payload: dict, heading: str) -> None:
        self._set_plan_text("\n".join(self._plan_lines(payload, heading=heading)), heading)

    def load_session(self) -> None:
        session_dir_text = filedialog.askdirectory(initialdir=self.output_home_var.get().strip() or str(self.project_root / "sessions"))
        if not session_dir_text:
            return
        session_dir = Path(session_dir_text)
        candidate_path = session_dir / "candidate_hypotheses.json"
        if not candidate_path.exists():
            messagebox.showwarning("Missing file", "candidate_hypotheses.json was not found in that folder.")
            return
        self.current_session_dir = session_dir
        self.session_name_var.set(session_dir.name)
        self.output_home_var.set(str(session_dir.parent))
        self._load_session_config(session_dir)
        payload = read_json(candidate_path)
        self._show_candidates(payload)
        approved_hypothesis_path = session_dir / "approved_hypothesis.txt"
        if approved_hypothesis_path.exists():
            self._set_hypothesis_text(approved_hypothesis_path.read_text(encoding="utf-8"), "Approved hypothesis")
        draft_plan_path = session_dir / "draft_plan.json"
        approved_plan_path = session_dir / "approved_plan.json"
        if draft_plan_path.exists():
            self._show_plan(read_json(draft_plan_path), "Draft analysis plan")
        elif approved_plan_path.exists():
            self._show_plan(read_json(approved_plan_path), "Approved analysis plan")
        self._append_log(f"Loaded session: {session_dir}\n")

    def regenerate_analysis_plan(self) -> None:
        if self.current_session_dir is None:
            self.current_session_dir = self._session_dir()
        approved_hypothesis_path = self.current_session_dir / "approved_hypothesis.txt"
        if not approved_hypothesis_path.exists():
            messagebox.showwarning("No approved hypothesis", "Approve a hypothesis first.")
            return
        feedback_text = self.plan_feedback_text.get("1.0", "end").strip()

        def task():
            agent = self._build_agent(analysis_name=self.current_session_dir.name, output_home=str(self.current_session_dir.parent))
            hypothesis = approved_hypothesis_path.read_text(encoding="utf-8").strip()
            plan = agent.build_plan_from_hypothesis(hypothesis, user_strategy_feedback=feedback_text)
            (self.current_session_dir / "draft_hypothesis.txt").write_text(hypothesis + "\n", encoding="utf-8")
            write_json(self.current_session_dir / "draft_plan.json", plan.model_dump())
            (self.current_session_dir / "draft_plan.md").write_text(
                "\n".join(self._plan_lines(plan.model_dump(), heading="Draft analysis plan")) + "\n",
                encoding="utf-8",
            )
            if feedback_text:
                (self.current_session_dir / "draft_strategy_feedback.txt").write_text(feedback_text + "\n", encoding="utf-8")
            self._write_session_config(
                self.current_session_dir,
                hypothesis_feedback=self.hypothesis_feedback_text.get("1.0", "end").strip(),
                plan_feedback=feedback_text,
            )
            self.message_queue.put(
                (
                    "done",
                    lambda payload=plan.model_dump(): self._apply_regenerated_plan(payload),
                )
            )
            self._queue_log(f"Draft analysis plan saved in {self.current_session_dir}\n")
            self._queue_log("Review the draft plan on the right. If it looks good, click Approve Plan.\n")

        self._run_background("Regenerate analysis plan", task)

    def generate_analysis_plan(self) -> None:
        if self.current_session_dir is None:
            self.current_session_dir = self._session_dir()
        approved_hypothesis_path = self.current_session_dir / "approved_hypothesis.txt"
        if not approved_hypothesis_path.exists():
            messagebox.showwarning("No approved hypothesis", "Approve a hypothesis first.")
            return

        def task():
            agent = self._build_agent(analysis_name=self.current_session_dir.name, output_home=str(self.current_session_dir.parent))
            hypothesis = approved_hypothesis_path.read_text(encoding="utf-8").strip()
            plan = agent.build_plan_from_hypothesis(hypothesis, user_strategy_feedback="")
            (self.current_session_dir / "draft_hypothesis.txt").write_text(hypothesis + "\n", encoding="utf-8")
            write_json(self.current_session_dir / "draft_plan.json", plan.model_dump())
            (self.current_session_dir / "draft_plan.md").write_text(
                "\n".join(self._plan_lines(plan.model_dump(), heading="Draft analysis plan")) + "\n",
                encoding="utf-8",
            )
            draft_feedback_path = self.current_session_dir / "draft_strategy_feedback.txt"
            if draft_feedback_path.exists():
                draft_feedback_path.unlink()
            self._write_session_config(
                self.current_session_dir,
                hypothesis_feedback=self.hypothesis_feedback_text.get("1.0", "end").strip(),
                plan_feedback=self.plan_feedback_text.get("1.0", "end").strip(),
            )
            self.message_queue.put(("done", lambda payload=plan.model_dump(): self._apply_regenerated_plan(payload)))
            self._queue_log(f"Draft analysis plan saved in {self.current_session_dir}\n")
            self._queue_log("Review the draft plan on the right. If it looks good, click Approve Plan.\n")

        self._run_background("Generate analysis plan", task)

    def _apply_regenerated_plan(self, payload: dict) -> None:
        self._append_log("Updated draft analysis plan is now shown on the right.\n")
        self._show_plan(payload, "Draft analysis plan")

    def approve_selected_hypothesis(self) -> None:
        if not self.current_candidates:
            messagebox.showwarning("No candidates", "Generate or load hypotheses first.")
            return
        if not self.candidate_list.curselection():
            messagebox.showwarning("No selection", "Please select a hypothesis.")
            return
        if self.current_session_dir is None:
            self.current_session_dir = self._session_dir()
        selected_index = self.candidate_list.curselection()[0]
        selected = self.current_candidates[selected_index]
        feedback_text = self.hypothesis_feedback_text.get("1.0", "end").strip()

        agent = self._build_agent(analysis_name=self.current_session_dir.name, output_home=str(self.current_session_dir.parent))
        hypothesis_text = selected.get("hypothesis", "").strip()
        if feedback_text:
            hypothesis_text = agent.revise_hypothesis(hypothesis=hypothesis_text, user_feedback=feedback_text)
            (self.current_session_dir / "user_feedback.txt").write_text(feedback_text + "\n", encoding="utf-8")
        self.current_candidates[selected_index]["hypothesis"] = hypothesis_text

        approved_hypothesis_path = self.current_session_dir / "approved_hypothesis.txt"
        approved_hypothesis_md_path = self.current_session_dir / "approved_hypothesis.md"
        approved_hypothesis_path.write_text(hypothesis_text + "\n", encoding="utf-8")
        approved_hypothesis_md_path.write_text(
            "Approved hypothesis\n\n" + hypothesis_text + "\n",
            encoding="utf-8",
        )
        for artifact_name in ("draft_plan.json", "draft_plan.md", "approved_plan.json", "approved_plan.md"):
            artifact_path = self.current_session_dir / artifact_name
            if artifact_path.exists():
                artifact_path.unlink()
        self._write_session_config(
            self.current_session_dir,
            hypothesis_feedback=feedback_text,
            plan_feedback=self.plan_feedback_text.get("1.0", "end").strip(),
        )
        self._render_candidate_detail(selected_index)
        self._set_hypothesis_text(hypothesis_text, "Approved hypothesis")
        self._set_plan_text("No plan has been generated for the approved hypothesis yet.", "Draft or approved plan")
        self._append_log(f"Approved hypothesis saved in {self.current_session_dir}\n")

    def approve_plan(self) -> None:
        if self.current_session_dir is None:
            self.current_session_dir = self._session_dir()
        draft_plan_path = self.current_session_dir / "draft_plan.json"
        if not draft_plan_path.exists():
            messagebox.showwarning("No draft plan", "Generate a draft analysis plan first.")
            return
        approved_plan_path = self.current_session_dir / "approved_plan.json"
        approved_plan_md_path = self.current_session_dir / "approved_plan.md"
        approved_feedback_path = self.current_session_dir / "approved_strategy_feedback.txt"
        approved_payload = read_json(draft_plan_path)
        write_json(approved_plan_path, approved_payload)
        approved_plan_md_path.write_text(
            "\n".join(self._plan_lines(approved_payload, heading="Approved analysis plan")) + "\n",
            encoding="utf-8",
        )
        draft_feedback_path = self.current_session_dir / "draft_strategy_feedback.txt"
        if draft_feedback_path.exists():
            approved_feedback_path.write_text(draft_feedback_path.read_text(encoding="utf-8"), encoding="utf-8")
        self._show_plan(approved_payload, "Approved analysis plan")
        self._append_log(f"Approved plan saved in {self.current_session_dir}\n")

    def run_analysis(self) -> None:
        if self.current_session_dir is None:
            self.current_session_dir = self._session_dir()
        approved_plan_path = self.current_session_dir / "approved_plan.json"
        if not approved_plan_path.exists():
            messagebox.showwarning("No approved plan", "Approve a plan first.")
            return

        def task():
            agent = self._build_agent(analysis_name=self.current_session_dir.name, output_home=str(self.current_session_dir.parent))
            approved_plan = read_json(approved_plan_path)
            summary_path = agent.run_approved_plan(approved_plan)
            self._queue_log(f"Run summary: {summary_path}\n")
            figure_status_path = self.current_session_dir / "figure_status.txt"
            if figure_status_path.exists():
                figure_status = figure_status_path.read_text(encoding="utf-8").strip()
                self._queue_log(f"Figure status:\n{figure_status}\n")

        self._run_background("Run analysis", task)

    def open_session_folder(self) -> None:
        if self.current_session_dir is None:
            self.current_session_dir = self._session_dir()
        self.current_session_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(self.current_session_dir))
        except Exception as exc:  # pragma: no cover
            messagebox.showerror("Open folder failed", str(exc))

    def _browse_raw_input_folder(self) -> None:
        title = "Select scRNA raw directory" if self._mode_is_spatial() else "Select extracted raw data directory"
        if self._mode_is_atac():
            title = "Select 10x multiome raw directory"
        directory = filedialog.askdirectory(title=title)
        if directory:
            self.raw_input_var.set(directory)

    def _browse_raw_input_tar(self) -> None:
        title = "Select scRNA RAW.tar archive" if self._mode_is_spatial() else "Select GEO RAW.tar archive"
        if self._mode_is_atac():
            title = "Select 10x multiome RAW.tar archive"
        path = filedialog.askopenfilename(title=title, filetypes=[("TAR archives", "*.tar"), ("All files", "*.*")])
        if path:
            self.raw_input_var.set(path)

    def _browse_secondary_raw_input_folder(self) -> None:
        directory = filedialog.askdirectory(title="Select spatial raw directory")
        if directory:
            self.secondary_raw_input_var.set(directory)

    def _browse_secondary_raw_input_tar(self) -> None:
        path = filedialog.askopenfilename(title="Select spatial RAW.tar archive", filetypes=[("TAR archives", "*.tar"), ("All files", "*.*")])
        if path:
            self.secondary_raw_input_var.set(path)

    def _browse_prep_output_dir(self) -> None:
        directory = filedialog.askdirectory(title="Select preparation output directory")
        if directory:
            self.prep_output_var.set(directory)

    def _browse_annotation_notes(self) -> None:
        path = filedialog.askopenfilename(title="Select annotation notes", filetypes=[("Text files", "*.txt *.md"), ("All files", "*.*")])
        if path:
            self.annotation_notes_var.set(path)

    def _browse_rna_h5ad(self) -> None:
        path = filedialog.askopenfilename(title="Select RNA h5ad", filetypes=[("AnnData", "*.h5ad"), ("All files", "*.*")])
        if path:
            self.rna_h5ad_var.set(path)

    def _browse_modality_input(self) -> None:
        if self._mode_is_spatial():
            path = filedialog.askopenfilename(title="Select spatial h5ad", filetypes=[("AnnData", "*.h5ad"), ("All files", "*.*")])
        elif self._mode_is_atac():
            path = filedialog.askopenfilename(title="Select ATAC h5ad", filetypes=[("AnnData", "*.h5ad"), ("All files", "*.*")])
        else:
            path = filedialog.askopenfilename(title="Select TCR table", filetypes=[("Tables", "*.tsv *.csv *.txt *.gz"), ("All files", "*.*")])
        if path:
            self.modality_input_var.set(path)

    def _browse_brief(self) -> None:
        path = filedialog.askopenfilename(title="Select research brief", filetypes=[("Text files", "*.txt *.md"), ("All files", "*.*")])
        if path:
            self.research_brief_var.set(path)

    def _browse_literature(self) -> None:
        path = filedialog.askdirectory(title="Select literature folder")
        if path:
            self.literature_path_var.set(path)

    def _browse_sessions_dir(self) -> None:
        directory = filedialog.askdirectory(title="Select sessions directory")
        if directory:
            self.output_home_var.set(directory)


def run_desktop_app(project_root: str | Path) -> None:
    app = ScRTDesktopApp(project_root)
    app.mainloop()

