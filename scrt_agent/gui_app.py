"""Desktop GUI for scRT-agent."""

from __future__ import annotations

import datetime as dt
import os
import queue
import threading
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .agent import ScRTAgent
from .interactive import (
    format_analysis_plan_markdown,
    format_candidate_menu_markdown,
    read_json,
    write_json,
)
from .preprocess import prepare_dataset

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


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
    """Simple desktop UI for preprocessing and interactive analysis."""

    def __init__(self, project_root: str | Path) -> None:
        super().__init__()
        self.project_root = Path(project_root).resolve()
        load_local_env_files(self.project_root)

        self.title("scRT-agent")
        self.geometry("1300x900")
        self.minsize(1120, 760)

        self.message_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.current_session_dir: Path | None = None
        self.current_candidates: list[dict] = []
        self.current_task: threading.Thread | None = None

        self._build_variables()
        self._build_layout()
        self._poll_queue()
        self._set_defaults()

    def _build_variables(self) -> None:
        sessions_home = self.project_root / "sessions"
        prepared_home = self.project_root / "prepared"
        self.raw_input_var = tk.StringVar()
        self.prep_output_var = tk.StringVar(value=str(prepared_home))
        self.annotation_model_var = tk.StringVar(value="gpt-4o")
        self.annotation_notes_var = tk.StringVar()
        self.min_genes_var = tk.StringVar(value="200")
        self.min_cells_var = tk.StringVar(value="3")
        self.max_pct_mt_var = tk.StringVar(value="15")
        self.leiden_resolution_var = tk.StringVar(value="0.8")

        self.rna_h5ad_var = tk.StringVar()
        self.tcr_path_var = tk.StringVar()
        self.research_brief_var = tk.StringVar()
        self.literature_path_var = tk.StringVar()
        self.session_name_var = tk.StringVar()
        self.output_home_var = tk.StringVar(value=str(sessions_home))
        self.model_name_var = tk.StringVar(value="gpt-4o")
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
        frame = ttk.LabelFrame(parent, text="1. Raw Data Preparation", padding=10)
        frame.pack(fill="x", pady=(0, 10))
        ttk.Label(frame, text="Raw input folder").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.raw_input_var, width=42).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Button(frame, text="Folder", command=self._browse_raw_input_folder).grid(row=0, column=2, sticky="ew", padx=(6, 0), pady=4)
        ttk.Button(frame, text="RAW.tar", command=self._browse_raw_input_tar).grid(row=0, column=3, sticky="ew", padx=(6, 0), pady=4)
        ttk.Label(frame, text="Use the folder that contains the extracted GEO files.").grid(
            row=1, column=1, columnspan=3, sticky="w"
        )

        self._path_row(frame, "Output dir", self.prep_output_var, self._browse_prep_output_dir, row=2)
        self._path_row(frame, "Annotation notes", self.annotation_notes_var, self._browse_annotation_notes, row=3, required=False)

        ttk.Label(frame, text="Annotation model").grid(row=4, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.annotation_model_var, width=28).grid(row=4, column=1, sticky="ew", pady=4)
        ttk.Label(frame, text="Min genes").grid(row=5, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.min_genes_var, width=10).grid(row=5, column=1, sticky="w", pady=4)
        ttk.Label(frame, text="Min cells").grid(row=6, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.min_cells_var, width=10).grid(row=6, column=1, sticky="w", pady=4)
        ttk.Label(frame, text="Max pct mt").grid(row=7, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.max_pct_mt_var, width=10).grid(row=7, column=1, sticky="w", pady=4)
        ttk.Label(frame, text="Leiden resolution").grid(row=8, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.leiden_resolution_var, width=10).grid(row=8, column=1, sticky="w", pady=4)
        ttk.Button(frame, text="Prepare Raw Data", command=self.prepare_raw_data).grid(row=9, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        frame.columnconfigure(1, weight=1)

    def _build_agent_panel(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="2. Interactive Analysis", padding=10)
        frame.pack(fill="x", pady=(0, 10))
        self._path_row(frame, "RNA h5ad", self.rna_h5ad_var, self._browse_rna_h5ad)
        self._path_row(frame, "TCR table", self.tcr_path_var, self._browse_tcr_path)
        self._path_row(frame, "Research brief", self.research_brief_var, self._browse_brief)
        self._path_row(frame, "Literature", self.literature_path_var, self._browse_literature, required=False)
        self._path_row(frame, "Sessions dir", self.output_home_var, self._browse_sessions_dir)

        ttk.Label(frame, text="Session name").grid(row=5, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.session_name_var).grid(row=5, column=1, sticky="ew", pady=4)
        ttk.Label(frame, text="Model").grid(row=6, column=0, sticky="w", pady=4)
        ttk.Entry(frame, textvariable=self.model_name_var, width=28).grid(row=6, column=1, sticky="ew", pady=4)
        ttk.Checkbutton(frame, text="Generate figure", variable=self.with_figure_var).grid(row=7, column=0, columnspan=2, sticky="w", pady=2)
        ttk.Checkbutton(frame, text="Save prompts", variable=self.log_prompts_var).grid(row=8, column=0, columnspan=2, sticky="w", pady=2)
        frame.columnconfigure(1, weight=1)

    def _build_action_panel(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="3. Actions", padding=10)
        frame.pack(fill="x")
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        buttons = [
            ("Generate Hypotheses", self.generate_hypotheses, 0, 0),
            ("Load Session", self.load_session, 0, 1),
            ("Approve Selected Hypothesis", self.approve_selected_hypothesis, 1, 0),
            ("Run Approved Analysis", self.run_analysis, 1, 1),
            ("Open Final Hypothesis", self.open_final_hypothesis, 2, 0),
            ("Open Figure Status", self.open_figure_status, 2, 1),
        ]
        for text, command, row, column in buttons:
            ttk.Button(frame, text=text, command=command).grid(
                row=row,
                column=column,
                sticky="ew",
                padx=3,
                pady=3,
            )
        ttk.Button(frame, text="Open Session Folder", command=self.open_session_folder).grid(
            row=3,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=3,
            pady=3,
        )

    def _build_candidates_panel(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="Candidates And Review", padding=10)
        frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(1, weight=1)
        frame.rowconfigure(3, weight=1)

        ttk.Label(frame, text="Candidate hypotheses").grid(row=0, column=0, sticky="w")
        self.candidate_list = tk.Listbox(frame, height=12, exportselection=False)
        self.candidate_list.grid(row=1, column=0, sticky="nsw", padx=(0, 10))
        self.candidate_list.bind("<<ListboxSelect>>", self._on_candidate_selected)

        self.candidate_detail = tk.Text(frame, wrap="word", width=70, height=18)
        self.candidate_detail.grid(row=1, column=1, sticky="nsew")

        ttk.Label(frame, text="Feedback for revision").grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 0))
        self.feedback_text = tk.Text(frame, wrap="word", height=8)
        self.feedback_text.grid(row=3, column=0, columnspan=2, sticky="nsew")

    def _build_log_panel(self, parent) -> None:
        frame = ttk.LabelFrame(parent, text="Run Log", padding=10)
        frame.grid(row=1, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        self.log_text = tk.Text(frame, wrap="word", height=16)
        self.log_text.grid(row=0, column=0, sticky="nsew")

    def _path_row(self, parent, label: str, variable: tk.StringVar, browse_command, *, row: int | None = None, required: bool = True) -> None:
        if row is None:
            row = parent.grid_size()[1]
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=variable, width=42).grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Button(parent, text="Browse", command=browse_command).grid(row=row, column=2, sticky="ew", padx=(6, 0), pady=4)
        if not required:
            ttk.Label(parent, text="optional").grid(row=row, column=3, sticky="w", padx=(6, 0))

    def _append_log(self, text: str) -> None:
        self.log_text.insert("end", text)
        self.log_text.see("end")

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

    def _build_agent(self, *, analysis_name: str, output_home: str) -> ScRTAgent:
        literature_paths = [self.literature_path_var.get().strip()] if self.literature_path_var.get().strip() else []
        return ScRTAgent(
            rna_h5ad_path=self.rna_h5ad_var.get().strip(),
            tcr_path=self.tcr_path_var.get().strip(),
            research_brief_path=self.research_brief_var.get().strip(),
            literature_paths=literature_paths or None,
            analysis_name=analysis_name,
            model_name=self.model_name_var.get().strip() or "gpt-4o",
            hypothesis_model=self.model_name_var.get().strip() or "gpt-4o",
            execution_model=self.model_name_var.get().strip() or "gpt-4o",
            vision_model=self.model_name_var.get().strip() or "gpt-4o",
            num_analyses=1,
            max_iterations=6,
            output_home=output_home,
            generate_publication_figure=self.with_figure_var.get(),
            log_prompts=self.log_prompts_var.get(),
        )

    def _session_dir(self) -> Path:
        name = self.session_name_var.get().strip() or f"session_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_name_var.set(name)
        base = Path(self.output_home_var.get().strip() or str(self.project_root / "sessions")).resolve()
        base.mkdir(parents=True, exist_ok=True)
        return base / name

    def prepare_raw_data(self) -> None:
        raw_input = self.raw_input_var.get().strip()
        output_dir = self.prep_output_var.get().strip()
        if not raw_input or not output_dir:
            messagebox.showwarning("Missing input", "Please provide raw input and output directory.")
            return

        def task():
            result = prepare_dataset(
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
            self.tcr_path_var.set(str(result.tcr_table_path))
            self._queue_log(f"Prepared RNA: {result.rna_h5ad_path}\n")
            self._queue_log(f"Prepared TCR: {result.tcr_table_path}\n")

        self._run_background("Prepare raw data", task)

    def generate_hypotheses(self) -> None:
        session_dir = self._session_dir()

        def task():
            session_dir.mkdir(parents=True, exist_ok=True)
            agent = self._build_agent(analysis_name=session_dir.name, output_home=str(session_dir.parent))
            menu = agent.prepare_candidate_hypotheses()
            write_json(session_dir / "candidate_hypotheses.json", menu.model_dump())
            (session_dir / "candidate_hypotheses.md").write_text(format_candidate_menu_markdown(menu), encoding="utf-8")
            config = {
                "rna_h5ad_path": self.rna_h5ad_var.get().strip(),
                "tcr_path": self.tcr_path_var.get().strip(),
                "research_brief_path": self.research_brief_var.get().strip(),
                "literature_path": [self.literature_path_var.get().strip()] if self.literature_path_var.get().strip() else [],
                "model_name": self.model_name_var.get().strip() or "gpt-4o",
                "with_figure": self.with_figure_var.get(),
                "log_prompts": self.log_prompts_var.get(),
            }
            write_json(session_dir / "session_config.json", config)
            self.current_session_dir = session_dir
            self.current_candidates = [item.model_dump() for item in menu.candidates]
            self.message_queue.put(("done", lambda: self._show_candidates(menu.model_dump())))

        self._run_background("Generate hypotheses", task)

    def _show_candidates(self, payload: dict) -> None:
        self.candidate_list.delete(0, "end")
        self.current_candidates = payload.get("candidates", [])
        for idx, item in enumerate(self.current_candidates, start=1):
            self.candidate_list.insert("end", f"{idx}. {item.get('title', 'Untitled')}")
        if self.current_candidates:
            self.candidate_list.selection_set(0)
            self._render_candidate_detail(0)

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
        self.candidate_detail.delete("1.0", "end")
        self.candidate_detail.insert("1.0", "\n".join(lines))

    def load_session(self) -> None:
        session_dir_text = filedialog.askdirectory(initialdir=self.output_home_var.get().strip() or str(self.project_root / "sessions"))
        if not session_dir_text:
            return
        session_dir = Path(session_dir_text)
        candidate_path = session_dir / "candidate_hypotheses.json"
        if not candidate_path.exists():
            messagebox.showwarning("Missing file", "candidate_hypotheses.json was not found in that folder.")
            return
        payload = read_json(candidate_path)
        self.current_session_dir = session_dir
        self.session_name_var.set(session_dir.name)
        self.output_home_var.set(str(session_dir.parent))
        self._show_candidates(payload)
        self._append_log(f"Loaded session: {session_dir}\n")

    def approve_selected_hypothesis(self) -> None:
        if not self.current_candidates:
            messagebox.showwarning("No candidates", "Generate or load a session first.")
            return
        if not self.candidate_list.curselection():
            messagebox.showwarning("No selection", "Please select a hypothesis.")
            return
        if self.current_session_dir is None:
            self.current_session_dir = self._session_dir()
        selected = self.current_candidates[self.candidate_list.curselection()[0]]
        feedback_text = self.feedback_text.get("1.0", "end").strip()

        def task():
            agent = self._build_agent(analysis_name=self.current_session_dir.name, output_home=str(self.current_session_dir.parent))
            hypothesis = selected.get("hypothesis", "").strip()
            if feedback_text:
                hypothesis = agent.revise_hypothesis(hypothesis=hypothesis, user_feedback=feedback_text)
                (self.current_session_dir / "user_feedback.txt").write_text(feedback_text + "\n", encoding="utf-8")
            plan = agent.build_plan_from_hypothesis(hypothesis)
            (self.current_session_dir / "approved_hypothesis.txt").write_text(hypothesis + "\n", encoding="utf-8")
            write_json(self.current_session_dir / "approved_plan.json", plan.model_dump())
            (self.current_session_dir / "approved_plan.md").write_text(format_analysis_plan_markdown(plan), encoding="utf-8")
            self._queue_log(f"Approved hypothesis saved in {self.current_session_dir}\n")

        self._run_background("Approve hypothesis", task)

    def run_analysis(self) -> None:
        if self.current_session_dir is None:
            self.current_session_dir = self._session_dir()
        approved_path = self.current_session_dir / "approved_hypothesis.txt"
        if not approved_path.exists():
            messagebox.showwarning("No approved hypothesis", "Approve a hypothesis first.")
            return

        def task():
            agent = self._build_agent(analysis_name=self.current_session_dir.name, output_home=str(self.current_session_dir.parent))
            hypothesis = approved_path.read_text(encoding="utf-8").strip()
            summary_path = agent.run(seeded_hypotheses=[hypothesis])
            self._queue_log(f"Run summary: {summary_path}\n")
            executed_path = self.current_session_dir / "executed_hypotheses.txt"
            if executed_path.exists():
                self._queue_log(f"Executed hypotheses: {executed_path}\n")
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

    def open_final_hypothesis(self) -> None:
        if self.current_session_dir is None:
            messagebox.showwarning("No session", "Generate, load, or run a session first.")
            return
        executed_path = self.current_session_dir / "executed_hypotheses.txt"
        approved_path = self.current_session_dir / "approved_hypothesis.txt"
        target = executed_path if executed_path.exists() else approved_path
        if not target.exists():
            messagebox.showwarning(
                "Missing file",
                "No final hypothesis file was found yet. Run the session first or approve a hypothesis.",
            )
            return
        try:
            os.startfile(str(target))
        except Exception as exc:  # pragma: no cover
            messagebox.showerror("Open file failed", str(exc))

    def open_figure_status(self) -> None:
        if self.current_session_dir is None:
            messagebox.showwarning("No session", "Generate, load, or run a session first.")
            return
        status_path = self.current_session_dir / "figure_status.txt"
        if not status_path.exists():
            messagebox.showwarning(
                "Missing file",
                "No figure status file was found. Run the analysis with figure generation enabled first.",
            )
            return
        try:
            os.startfile(str(status_path))
        except Exception as exc:  # pragma: no cover
            messagebox.showerror("Open file failed", str(exc))

    def _browse_raw_input_folder(self) -> None:
        directory = filedialog.askdirectory(title="Select extracted raw data directory")
        if directory:
            self.raw_input_var.set(directory)

    def _browse_raw_input_tar(self) -> None:
        path = filedialog.askopenfilename(
            title="Select GEO RAW.tar archive",
            filetypes=[("TAR archives", "*.tar"), ("All files", "*.*")],
        )
        if path:
            self.raw_input_var.set(path)

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

    def _browse_tcr_path(self) -> None:
        path = filedialog.askopenfilename(title="Select TCR table", filetypes=[("Tables", "*.tsv *.csv *.txt *.gz"), ("All files", "*.*")])
        if path:
            self.tcr_path_var.set(path)

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
