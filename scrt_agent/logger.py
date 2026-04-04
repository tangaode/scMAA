"""Logging utilities for scMAA."""

from __future__ import annotations

import datetime as _dt
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


class AgentLogger:
    """Per-run logger with prompt/response helpers."""

    def __init__(self, analysis_name: str, log_dir: str | os.PathLike[str], log_prompts: bool = False) -> None:
        self.analysis_name = analysis_name
        self.log_prompts_enabled = log_prompts
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{analysis_name}_{timestamp}.log"
        self.prompt_dir = self.log_dir / "prompts"
        if self.log_prompts_enabled:
            self.prompt_dir.mkdir(parents=True, exist_ok=True)

        logger_name = f"scrt_agent.{analysis_name}.{timestamp}"
        self._logger = logging.getLogger(logger_name)
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

        formatter = logging.Formatter(
            "\n\n" + "=" * 88 + "\n%(asctime)s - %(levelname)s\n" + "=" * 88 + "\n%(message)s"
        )
        handler = RotatingFileHandler(self.log_file, maxBytes=25 * 1024 * 1024, backupCount=3, encoding="utf-8")
        handler.setFormatter(formatter)
        self._logger.addHandler(handler)
        self._logger.info("Logging started.")

    def info(self, message: str) -> None:
        self._logger.info(message)

    def warning(self, message: str) -> None:
        self._logger.warning(message)

    def error(self, message: str) -> None:
        self._logger.error(message)

    def exception(self, message: str) -> None:
        self._logger.exception(message)

    def log_prompt(self, role: str, prompt_text: str, prompt_name: str) -> None:
        self._logger.info(f"PROMPT: {prompt_name} ({role})\n\n{prompt_text}")
        if self.log_prompts_enabled:
            safe_name = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in prompt_name.lower())
            out = self.prompt_dir / f"{safe_name}_{role}.txt"
            out.write_text(prompt_text, encoding="utf-8")

    def log_response(self, response_text: str, source: str) -> None:
        self._logger.info(f"RESPONSE/OUTPUT: {source}\n\n{response_text}")

