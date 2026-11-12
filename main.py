"""AI Education Agent — local stakeholder demo with optional Meta Model API.

Run:
    python main.py

Without MODEL_API_KEY, responses are simulated locally. When configured, Meta's
OpenAI-compatible Model API streams real model responses in a worker thread.
"""

from __future__ import annotations

import os
import io
import queue
import random
import re
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from mimetypes import guess_type
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Iterator, Optional

from learning_core import (
    EducationTurn,
    MockEducationService,
    ProblemSolvingSession,
    StudentProfile,
    WorkflowStage,
    is_direct_answer_request,
)
from meta_education_service import (
    DEFAULT_META_BASE_URL,
    DEFAULT_META_MODEL,
    MetaEducationService,
    filter_provider_urls,
    select_provider,
)
from resource_catalog import RESOURCES, format_recommendations, recommend_resources
from markdown_renderer import RenderToken, parse_markdown
from knowledge_base import approved_source_urls
from backend.math_verification.agent_pipeline import (
    MathAgentPipeline,
    OpenAIResponsesProvider,
    PipelineError,
)
from backend.math_verification.examples import (
    RLC_PROBLEM,
    render_reviewed_rlc_answer,
    solve_reviewed_rlc_example,
)


try:
    from dotenv import load_dotenv

    # Source checkouts may use a local, gitignored .env. Packaged builds use a
    # user-owned file outside the application bundle. Existing environment
    # variables always take precedence.
    load_dotenv(Path(__file__).with_name(".env"), override=False)
    load_dotenv(
        Path.home()
        / "Library"
        / "Application Support"
        / "AI Education Agent"
        / "config.env",
        override=False,
    )
except ImportError:
    pass


APP_TITLE = "AI Education Agent"
AGENTS = ("Tutor Agent", "Problem-Solving Agent", "Verified Math Solver")
WORKFLOW_STAGES = ("Diagnose", "Design Hint", "Gather Report", "Iterate")
MAX_ATTACHMENTS = 10
MAX_ATTACHMENT_BYTES = 1024 * 1024 * 1024
MAX_EXTRACTED_CHARS = 120_000
TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".jsonl", ".xml",
    ".yaml", ".yml", ".html", ".htm", ".css", ".py", ".ipynb", ".js", ".ts",
    ".java", ".c", ".cc", ".cpp", ".h", ".hpp", ".sql", ".tex", ".log",
}
NATIVE_META_EXTENSIONS = {
    ".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff",
    ".mp4", ".mov", ".m4v", ".avi", ".webm", ".mp3", ".wav", ".m4a", ".aac",
}


LIGHT = {
    "bg": "#F4F7FB",
    "surface": "#FFFFFF",
    "surface_alt": "#EEF2F7",
    "text": "#172033",
    "muted": "#667085",
    "border": "#D9E0EA",
    "accent": "#5B5CE2",
    "accent_hover": "#4B4CCB",
    "accent_soft": "#E9E9FF",
    "user_bubble": "#5B5CE2",
    "user_text": "#FFFFFF",
    "ai_bubble": "#FFFFFF",
    "success": "#168A5B",
    "warning": "#C77A00",
    "danger": "#C43D4B",
}

DARK = {
    "bg": "#0E1420",
    "surface": "#171F2E",
    "surface_alt": "#212B3D",
    "text": "#F2F4F7",
    "muted": "#A7B0C0",
    "border": "#303C50",
    "accent": "#818CF8",
    "accent_hover": "#6975E8",
    "accent_soft": "#2C315F",
    "user_bubble": "#6366F1",
    "user_text": "#FFFFFF",
    "ai_bubble": "#212B3D",
    "success": "#48C78E",
    "warning": "#F2B84B",
    "danger": "#F47682",
}


@dataclass(frozen=True)
class Attachment:
    """A local file selected by the student; paths are never stored in logs."""

    path: Path
    name: str
    size: int
    mime_type: str


def _truncate_text(text: str, limit: int = MAX_EXTRACTED_CHARS) -> str:
    text = text.replace("\x00", "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n[Content truncated locally to control token usage.]"


def _extract_attachment_text(attachment: Attachment) -> str:
    """Extract common document formats locally; return an empty string for binary files."""
    path = attachment.path
    suffix = path.suffix.lower()
    try:
        if suffix == ".docx":
            from docx import Document

            document = Document(path)
            paragraphs = [paragraph.text for paragraph in document.paragraphs]
            for table in document.tables:
                for row in table.rows:
                    paragraphs.append("\t".join(cell.text for cell in row.cells))
            return _truncate_text("\n".join(paragraphs))
        if suffix == ".rtf":
            try:
                from striprtf.striprtf import rtf_to_text

                return _truncate_text(rtf_to_text(path.read_text(encoding="utf-8", errors="replace")))
            except ImportError:
                pass
        if suffix == ".doc":
            command: list[str] | None = None
            if shutil.which("textutil"):
                command = ["textutil", "-convert", "txt", "-stdout", str(path)]
            elif shutil.which("antiword"):
                command = ["antiword", str(path)]
            if command:
                result = subprocess.run(command, capture_output=True, timeout=20, check=False)
                if result.returncode == 0:
                    return _truncate_text(result.stdout.decode("utf-8", errors="replace"))
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader

                reader = PdfReader(path)
                pages: list[str] = []
                for page in reader.pages[:100]:
                    pages.append(page.extract_text() or "")
                    if sum(map(len, pages)) >= MAX_EXTRACTED_CHARS:
                        break
                return _truncate_text("\n\n".join(pages))
            except Exception:
                return ""
        if suffix in TEXT_EXTENSIONS or attachment.mime_type.startswith("text/"):
            raw = path.read_bytes()[: MAX_EXTRACTED_CHARS * 4]
            for encoding in ("utf-8-sig", "utf-16", "latin-1"):
                try:
                    return _truncate_text(raw.decode(encoding))
                except UnicodeDecodeError:
                    continue
    except (OSError, ValueError, subprocess.SubprocessError):
        return ""
    return ""


class MetaModelService:
    """Text-only transport for Meta Model API's official Responses API."""

    def __init__(self, api_key: str, model: str, base_url: str) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Meta mode requires the openai package. "
                "Install dependencies with: python -m pip install -r requirements.txt"
            ) from exc
        # Disable automatic retries so one UI action cannot silently become
        # multiple paid inference calls. The user can retry explicitly.
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=90.0, max_retries=0)
        self.model = model

    def stream_reply(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
    ) -> Iterator[str]:
        # The official API documents client.responses.stream() and
        # response.output_text.delta events for streamed visible output.
        with self.client.responses.stream(
            model=self.model,
            input=[dict(message) for message in messages],
            max_output_tokens=max_tokens,
            reasoning={"effort": "low"},
        ) as stream:
            produced_text = False
            for event in stream:
                if getattr(event, "type", "") == "response.output_text.delta":
                    text = getattr(event, "delta", "")
                    if text:
                        produced_text = True
                        yield text
            if not produced_text:
                raise RuntimeError(
                    "Meta completed without visible text. Increase META_MAX_COMPLETION_TOKENS."
                )

    def close(self) -> None:
        self.client.close()


class ScrollableFrame(tk.Frame):
    """A vertically scrollable frame suitable for chat bubbles and dashboards."""

    def __init__(self, parent: tk.Widget, palette: dict[str, str], **kwargs) -> None:
        super().__init__(parent, bg=palette["bg"], **kwargs)
        self.palette = palette
        self.canvas = tk.Canvas(self, highlightthickness=0, bg=palette["bg"])
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.content = tk.Frame(self.canvas, bg=palette["bg"])
        self.window_id = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.content.bind("<Configure>", self._sync_region)
        self.canvas.bind("<Configure>", self._sync_width)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        self.canvas.bind_all("<Button-4>", self._on_mousewheel, add="+")
        self.canvas.bind_all("<Button-5>", self._on_mousewheel, add="+")

    def _sync_region(self, _event=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _sync_width(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.window_id, width=event.width)

    def _on_mousewheel(self, event: tk.Event) -> None:
        if self.winfo_exists() and self.winfo_ismapped():
            # macOS trackpads often report tiny deltas (for example, +/-1),
            # while Windows commonly reports multiples of 120.
            if getattr(event, "num", None) == 4:
                units = -3
            elif getattr(event, "num", None) == 5:
                units = 3
            else:
                delta = getattr(event, "delta", 0)
                if not delta:
                    return
                units = -int(delta / 120) if abs(delta) >= 120 else (-1 if delta > 0 else 1)
            self.canvas.yview_scroll(units, "units")

    def scroll_to_bottom(self) -> None:
        self.update_idletasks()
        self.canvas.yview_moveto(1.0)


class RichMessage(tk.Text):
    """Read-only chat bubble with Markdown styles, math images, and word wrapping."""

    def __init__(
        self,
        parent: tk.Widget,
        palette: dict[str, str],
        *,
        is_user: bool = False,
    ) -> None:
        self.palette = palette
        self.is_user = is_user
        background = palette["user_bubble"] if is_user else palette["ai_bubble"]
        foreground = palette["user_text"] if is_user else palette["text"]
        super().__init__(
            parent,
            height=1,
            width=1,
            wrap="word",
            relief="flat",
            borderwidth=0,
            padx=14,
            pady=11,
            cursor="arrow",
            bg=background,
            fg=foreground,
            insertbackground=foreground,
            highlightthickness=1,
            highlightbackground=palette["user_bubble"] if is_user else palette["border"],
            selectbackground=palette["accent_soft"],
            selectforeground=palette["text"],
            takefocus=False,
        )
        self._images: list[Any] = []
        self._math_cache: dict[tuple[str, bool], Any] = {}
        self._resize_job: Optional[str] = None
        self._configure_tags()
        self.bind("<Configure>", self._schedule_height_update, add="+")

    def _configure_tags(self) -> None:
        foreground = self.palette["user_text"] if self.is_user else self.palette["text"]
        muted = foreground if self.is_user else self.palette["muted"]
        self.tag_configure("body", font=("Helvetica", 10), foreground=foreground, spacing3=3)
        self.tag_configure("bold", font=("Helvetica", 10, "bold"), foreground=foreground)
        self.tag_configure("italic", font=("Helvetica", 10, "italic"), foreground=foreground)
        self.tag_configure("heading1", font=("Helvetica", 16, "bold"), foreground=foreground,
                           spacing1=8, spacing3=5)
        self.tag_configure("heading2", font=("Helvetica", 14, "bold"), foreground=foreground,
                           spacing1=7, spacing3=4)
        self.tag_configure("heading3", font=("Helvetica", 12, "bold"), foreground=foreground,
                           spacing1=6, spacing3=3)
        self.tag_configure("list_marker", font=("Helvetica", 10, "bold"), foreground=foreground,
                           lmargin1=8, lmargin2=25)
        self.tag_configure("quote", font=("Helvetica", 10, "italic"), foreground=muted,
                           lmargin1=14, lmargin2=14)
        self.tag_configure("code", font=("Menlo", 9), foreground=foreground,
                           background=self.palette["surface_alt"])
        self.tag_configure("code_block", font=("Menlo", 9), foreground=foreground,
                           background=self.palette["surface_alt"], lmargin1=8, lmargin2=8,
                           spacing1=6, spacing3=6)
        self.tag_configure("link", font=("Helvetica", 10, "underline"),
                           foreground=foreground if self.is_user else self.palette["accent"])
        self.tag_configure("cursor", font=("Helvetica", 10, "bold"), foreground=muted)

    def set_content(self, text: str, *, markdown: bool, streaming: bool = False) -> None:
        self.configure(state="normal")
        self.delete("1.0", "end")
        self._images.clear()
        if markdown:
            for token in parse_markdown(text):
                self._insert_token(token)
        else:
            self.insert("end", text, ("body",))
        if streaming:
            self.insert("end", " ▌", ("cursor",))
        if self.get("end-2c", "end-1c") == "\n":
            self.delete("end-2c", "end-1c")
        self.configure(state="disabled")
        self._schedule_height_update()

    def _insert_token(self, token: RenderToken) -> None:
        if token.kind in {"math_inline", "math_display"}:
            image = self._math_image(token.text, display=token.kind == "math_display")
            if image is not None:
                if token.kind == "math_display" and self.index("end-1c") != "1.0":
                    self.insert("end", "\n")
                self.image_create("end", image=image, padx=4, pady=5, align="center")
                self._images.append(image)
                if token.kind == "math_display":
                    self.insert("end", "\n")
                return
            # A malformed expression remains readable without exposing delimiters.
            self.insert("end", token.text, ("code",))
            return
        if token.kind == "newline":
            self.insert("end", "\n")
        elif token.text:
            self.insert("end", token.text, (token.style,))

    def _math_image(self, expression: str, *, display: bool) -> Any | None:
        key = (expression, display)
        if key in self._math_cache:
            return self._math_cache[key]
        try:
            from PIL import Image, ImageTk
            from matplotlib.font_manager import FontProperties
            from matplotlib.mathtext import math_to_image

            buffer = io.BytesIO()
            color = self.palette["user_text"] if self.is_user else self.palette["text"]
            math_to_image(
                f"${expression}$",
                buffer,
                prop=FontProperties(size=15 if display else 11),
                dpi=150,
                format="png",
                color=color,
            )
            buffer.seek(0)
            source = Image.open(buffer).convert("RGBA")
            max_width = max(240, self.winfo_width() - 42)
            if source.width > max_width:
                ratio = max_width / source.width
                source = source.resize(
                    (max_width, max(1, int(source.height * ratio))), Image.Resampling.LANCZOS
                )
            image = ImageTk.PhotoImage(source, master=self)
        except Exception:
            return None
        self._math_cache[key] = image
        return image

    def _schedule_height_update(self, _event=None) -> None:
        if self._resize_job:
            try:
                self.after_cancel(self._resize_job)
            except tk.TclError:
                pass
        self._resize_job = self.after_idle(self._update_height)

    def _update_height(self) -> None:
        self._resize_job = None
        if not self.winfo_exists():
            return
        try:
            display_lines = self.count("1.0", "end-1c", "displaylines")
            lines = display_lines[0] if display_lines else 1
            image_height = sum(image.height() for image in self._images)
            image_lines = (image_height + 17) // 18
            self.configure(height=max(1, lines + image_lines))
        except tk.TclError:
            pass


class EducationAgentApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1280x820")
        self.minsize(1050, 680)
        self.option_add("*Font", ("Helvetica", 11))

        self.dark_mode = False
        self.palette = LIGHT.copy()
        self.active_agent = "Tutor Agent"
        self.workflow_index = 0
        self.streaming_job: Optional[str] = None
        self.stream_generation = 0
        self.session_messages = {agent: [] for agent in AGENTS}
        self.pending_attachments: dict[str, list[Attachment]] = {agent: [] for agent in AGENTS}
        self.education_service = MockEducationService()
        self.student_profile = StudentProfile()
        self.problem_session = ProblemSolvingSession()

        self.meta_service: Optional[MetaModelService] = None
        self.meta_education_service: Optional[MetaEducationService] = None
        self.math_pipeline: Optional[MathAgentPipeline] = None
        self.meta_model = os.getenv(
            "META_MODEL", os.getenv("LLAMA_MODEL", DEFAULT_META_MODEL)
        ).strip()
        self.meta_base_url = os.getenv("META_BASE_URL", DEFAULT_META_BASE_URL).strip().rstrip("/")
        self.max_completion_tokens = self._bounded_env_int(
            "META_MAX_COMPLETION_TOKENS", default=2048, minimum=256, maximum=4096
        )
        self.provider_error = ""
        self._configure_provider()

        self.status_var = tk.StringVar(value=self._base_status())
        self.profile_major_var = tk.StringVar(value=self.student_profile.engineering_major)
        self.profile_level_var = tk.StringVar(value=self.student_profile.academic_level)
        self.profile_goal_var = tk.StringVar(value=self.student_profile.learning_goal)
        self.profile_preference_var = tk.StringVar(value=self.student_profile.explanation_preference)
        self.math_sig_figs_var = tk.IntVar(value=4)
        for variable in (
            self.profile_major_var,
            self.profile_level_var,
            self.profile_goal_var,
            self.profile_preference_var,
        ):
            variable.trace_add("write", self._sync_student_profile)

        self._configure_styles()
        self._build_shell()
        self._build_student_view()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    @staticmethod
    def _bounded_env_int(name: str, *, default: int, minimum: int, maximum: int) -> int:
        try:
            value = int(os.getenv(name, str(default)))
        except ValueError:
            value = default
        return min(max(value, minimum), maximum)

    @staticmethod
    def _env_enabled(name: str, *, default: bool = False) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def _configure_provider(self) -> None:
        mode = os.getenv("AI_AGENT_MODE", "auto").strip().lower()
        api_key = os.getenv("MODEL_API_KEY", os.getenv("LLAMA_API_KEY", "")).strip()
        selection = select_provider(mode, api_key)
        self.provider_error = selection.error
        if selection.name == "mock":
            return
        try:
            self.meta_service = MetaModelService(api_key, self.meta_model, self.meta_base_url)
            self.meta_education_service = MetaEducationService(
                self.meta_service,
                max_tokens=self.max_completion_tokens,
                enable_web_research=self._env_enabled(
                    "ENABLE_WEB_RESEARCH", default=True
                ),
            )
            math_provider = OpenAIResponsesProvider(
                client=self.meta_service.client,
                model=self.meta_model,
            )
            self.math_pipeline = MathAgentPipeline(
                provider=math_provider,
                provider_name="Meta Responses API",
                model=self.meta_model,
                max_tool_attempts=3,
            )
        except RuntimeError as exc:
            self.provider_error = str(exc)

    def _base_status(self) -> str:
        if self.meta_education_service:
            return f"Meta Model API connected • {self.meta_model} • streamed responses"
        if self.provider_error:
            return f"Mock mode • {self.provider_error}"
        return "Local simulation • Set AI_AGENT_MODE=meta and MODEL_API_KEY for Meta • 0 external tokens"

    def _sync_student_profile(self, *_args) -> None:
        """Keep the in-memory profile aligned with the sidebar controls."""
        if not hasattr(self, "profile_major_var"):
            return
        self.student_profile.engineering_major = self.profile_major_var.get().strip() or "General Engineering"
        self.student_profile.academic_level = self.profile_level_var.get().strip() or "Undergraduate"
        self.student_profile.learning_goal = self.profile_goal_var.get().strip() or "Build conceptual understanding"
        self.student_profile.explanation_preference = (
            self.profile_preference_var.get().strip() or "Step-by-step"
        )

    def _sync_workflow_index(self) -> None:
        stage_order = list(WorkflowStage)
        self.workflow_index = stage_order.index(self.problem_session.stage)

    def _configure_styles(self) -> None:
        p = self.palette
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TScrollbar", background=p["surface_alt"], troughcolor=p["bg"],
                        bordercolor=p["bg"], arrowcolor=p["muted"])
        style.configure("TCombobox", fieldbackground=p["surface"], background=p["surface"],
                        foreground=p["text"], bordercolor=p["border"], arrowsize=15)
        style.map("TCombobox", fieldbackground=[("readonly", p["surface"])],
                  foreground=[("readonly", p["text"])])
        style.configure("Horizontal.TProgressbar", troughcolor=p["surface_alt"],
                        background=p["accent"], bordercolor=p["surface_alt"], thickness=8)

    def _build_shell(self) -> None:
        p = self.palette
        self.configure(bg=p["bg"])
        self.header = tk.Frame(self, bg=p["surface"], height=72,
                               highlightthickness=1, highlightbackground=p["border"])
        self.header.pack(fill="x")
        self.header.pack_propagate(False)

        brand = tk.Frame(self.header, bg=p["surface"])
        brand.pack(side="left", padx=26, pady=13)
        tk.Label(brand, text="◈", font=("Helvetica", 25, "bold"), fg=p["accent"],
                 bg=p["surface"]).pack(side="left", padx=(0, 10))
        brand_text = tk.Frame(brand, bg=p["surface"])
        brand_text.pack(side="left")
        tk.Label(brand_text, text=APP_TITLE, font=("Helvetica", 16, "bold"),
                 fg=p["text"], bg=p["surface"]).pack(anchor="w")
        tk.Label(brand_text, text="Instructor-governed learning support",
                 font=("Helvetica", 9), fg=p["muted"], bg=p["surface"]).pack(anchor="w")

        self.theme_button = tk.Button(self.header, text="☾  Dark", bd=0, cursor="hand2",
                                      font=("Helvetica", 10, "bold"), padx=14, pady=9,
                                      bg=p["surface_alt"], fg=p["text"],
                                      activebackground=p["border"],
                                      command=self.toggle_theme)
        self.theme_button.pack(side="right", padx=24, pady=16)

        self.main = tk.Frame(self, bg=p["bg"])
        self.main.pack(fill="both", expand=True)

        self.footer = tk.Frame(self, bg=p["surface"], height=34,
                               highlightthickness=1, highlightbackground=p["border"])
        self.footer.pack(fill="x", side="bottom")
        self.footer.pack_propagate(False)
        self.status_label = tk.Label(self.footer, textvariable=self.status_var,
                                     font=("Helvetica", 9), fg=p["muted"], bg=p["surface"])
        self.status_label.pack(side="left", padx=20, pady=8)
        tk.Label(self.footer, text="Prototype data • FERPA-safe synthetic records",
                 font=("Helvetica", 9), fg=p["muted"], bg=p["surface"]).pack(
                     side="right", padx=20, pady=8)

    def _page_heading(self, parent: tk.Widget, title: str, subtitle: str) -> tk.Frame:
        p = self.palette
        area = tk.Frame(parent, bg=p["bg"])
        area.pack(fill="x", padx=28, pady=(24, 16))
        tk.Label(area, text=title, font=("Helvetica", 22, "bold"),
                 fg=p["text"], bg=p["bg"]).pack(anchor="w")
        tk.Label(area, text=subtitle, font=("Helvetica", 10),
                 fg=p["muted"], bg=p["bg"]).pack(anchor="w", pady=(4, 0))
        return area

    # --------------------------- Student view ---------------------------

    def _build_student_view(self) -> None:
        p = self.palette
        self._page_heading(self.main, "Student learning workspace",
                           "Ask questions, practice concepts, and work through problems with guardrails.")

        body = tk.Frame(self.main, bg=p["bg"])
        body.pack(fill="both", expand=True, padx=28, pady=(0, 22))

        sidebar = tk.Frame(body, bg=p["surface"], width=270,
                           highlightthickness=1, highlightbackground=p["border"])
        sidebar.pack(side="left", fill="y", padx=(0, 16))
        sidebar.pack_propagate(False)
        tk.Label(sidebar, text="CHOOSE AN AGENT", font=("Helvetica", 9, "bold"),
                 fg=p["muted"], bg=p["surface"]).pack(anchor="w", padx=18, pady=(20, 10))

        self.agent_buttons: dict[str, tk.Button] = {}
        agent_details = {
            "Tutor Agent": ("◎  Tutor Agent", "Concepts • review • questions"),
            "Problem-Solving Agent": ("◇  Problem Solver", "Hint-first guided workflow"),
            "Verified Math Solver": ("∑  Verified Math", "SymPy • Pint • unit checks"),
        }
        for agent, (label, detail) in agent_details.items():
            wrap = tk.Frame(sidebar, bg=p["surface"])
            wrap.pack(fill="x", padx=10, pady=4)
            button = tk.Button(wrap, text=label, anchor="w", bd=0, cursor="hand2",
                               font=("Helvetica", 11, "bold"), padx=12, pady=10,
                               command=lambda a=agent: self.switch_agent(a))
            button.pack(fill="x")
            tk.Label(wrap, text=detail, anchor="w", font=("Helvetica", 8),
                     fg=p["muted"], bg=p["surface"]).pack(fill="x", padx=13, pady=(0, 7))
            self.agent_buttons[agent] = button

        divider = tk.Frame(sidebar, bg=p["border"], height=1)
        divider.pack(fill="x", padx=18, pady=16)
        tk.Label(sidebar, text="STUDENT PROFILE", font=("Helvetica", 9, "bold"),
                 fg=p["muted"], bg=p["surface"]).pack(anchor="w", padx=18)

        profile_fields = (
            ("Major", self.profile_major_var,
             ("General Engineering", "Mechanical Engineering", "Electrical Engineering",
              "Civil Engineering", "Aerospace Engineering", "Chemical Engineering",
              "Computer Engineering", "Data Science")),
            ("Academic level", self.profile_level_var,
             ("First-year Undergraduate", "Undergraduate", "Advanced Undergraduate", "Graduate")),
            ("Explanation", self.profile_preference_var,
             ("Step-by-step", "Concise", "Analogy-first")),
        )
        for label, variable, values in profile_fields:
            tk.Label(sidebar, text=label, font=("Helvetica", 8), fg=p["muted"],
                     bg=p["surface"]).pack(anchor="w", padx=18, pady=(6, 2))
            box = ttk.Combobox(sidebar, state="readonly", textvariable=variable, values=values)
            box.pack(fill="x", padx=18)

        tk.Label(sidebar, text="Learning goal", font=("Helvetica", 8), fg=p["muted"],
                 bg=p["surface"]).pack(anchor="w", padx=18, pady=(6, 2))
        goal = tk.Entry(sidebar, textvariable=self.profile_goal_var, relief="flat",
                        bg=p["surface_alt"], fg=p["text"], insertbackground=p["text"],
                        highlightthickness=1, highlightbackground=p["border"])
        goal.pack(fill="x", padx=18, ipady=5)

        tk.Label(sidebar, text="✓ Local personalized mock\n✓ Verified university links\n✓ Deterministic unit checks",
                 justify="left", font=("Helvetica", 8), fg=p["success"],
                 bg=p["surface"]).pack(anchor="w", padx=18, pady=(12, 6))

        self.chat_card = tk.Frame(body, bg=p["surface"],
                                  highlightthickness=1, highlightbackground=p["border"])
        self.chat_card.pack(side="left", fill="both", expand=True)
        self._render_agent_panel()
        self._update_agent_buttons()

    def switch_agent(self, agent: str) -> None:
        if agent == self.active_agent or agent not in AGENTS:
            return
        self._cancel_stream()
        self.active_agent = agent
        if agent == "Problem-Solving Agent":
            self._sync_workflow_index()
        self._render_agent_panel()
        self._update_agent_buttons()

    def _update_agent_buttons(self) -> None:
        if not hasattr(self, "agent_buttons"):
            return
        p = self.palette
        for agent, button in self.agent_buttons.items():
            active = agent == self.active_agent
            button.configure(bg=p["accent_soft"] if active else p["surface"],
                             fg=p["accent"] if active else p["text"],
                             activebackground=p["accent_soft"], activeforeground=p["accent"])

    def _render_agent_panel(self) -> None:
        p = self.palette
        for widget in self.chat_card.winfo_children():
            widget.destroy()

        top = tk.Frame(self.chat_card, bg=p["surface"])
        top.pack(fill="x", padx=20, pady=(16, 12))
        icon = {
            "Tutor Agent": "◎",
            "Problem-Solving Agent": "◇",
            "Verified Math Solver": "∑",
        }[self.active_agent]
        title = self.active_agent
        subtitle = {
            "Tutor Agent": "Explains concepts and can provide complete worked solutions",
            "Problem-Solving Agent": "Guides your reasoning without revealing graded answers",
            "Verified Math Solver": (
                "Executes symbolic math and unit conversions with SymPy and Pint"
            ),
        }[self.active_agent]
        tk.Label(top, text=f"{icon}  {title}", font=("Helvetica", 15, "bold"),
                 fg=p["text"], bg=p["surface"]).pack(side="left")
        tk.Button(top, text="New session", bd=0, cursor="hand2", padx=12, pady=7,
                  font=("Helvetica", 9, "bold"), bg=p["surface_alt"], fg=p["text"],
                  activebackground=p["border"], command=self._new_session).pack(side="right")
        tk.Label(self.chat_card, text=subtitle, font=("Helvetica", 9),
                 fg=p["muted"], bg=p["surface"]).pack(anchor="w", padx=22, pady=(0, 10))

        if self.active_agent == "Verified Math Solver":
            math_controls = tk.Frame(self.chat_card, bg=p["surface_alt"])
            math_controls.pack(fill="x", padx=20, pady=(0, 10))
            tk.Label(
                math_controls,
                text="Significant figures",
                font=("Helvetica", 9, "bold"),
                fg=p["text"],
                bg=p["surface_alt"],
            ).pack(side="left", padx=(12, 8), pady=8)
            tk.Spinbox(
                math_controls,
                from_=2,
                to=8,
                width=4,
                textvariable=self.math_sig_figs_var,
                bg=p["surface"],
                fg=p["text"],
                buttonbackground=p["surface_alt"],
            ).pack(side="left", pady=8)
            tk.Button(
                math_controls,
                text="Load RLC example",
                bd=0,
                cursor="hand2",
                padx=11,
                pady=6,
                font=("Helvetica", 9, "bold"),
                bg=p["accent_soft"],
                fg=p["accent"],
                command=self._load_rlc_example,
            ).pack(side="right", padx=10, pady=6)

        if self.active_agent == "Problem-Solving Agent":
            self.workflow_bar = tk.Frame(self.chat_card, bg=p["surface_alt"])
            self.workflow_bar.pack(fill="x", padx=20, pady=(0, 10))
            self._draw_workflow()

        divider = tk.Frame(self.chat_card, bg=p["border"], height=1)
        divider.pack(fill="x")
        self.chat_scroll = ScrollableFrame(self.chat_card, p)
        self.chat_scroll.pack(fill="both", expand=True, padx=1, pady=1)
        self.chat_scroll.canvas.configure(bg=p["bg"])
        self.chat_scroll.content.configure(bg=p["bg"])

        messages = self.session_messages[self.active_agent]
        if not messages:
            welcome_messages = {
                "Tutor Agent": (
                    "Hi! I’m your Tutor Agent. Ask a question or upload a PDF, TXT, Word document, "
                    "image, or other course file. In mock mode I provide clearly labeled, profile-aware "
                    "engineering explanations and worked practice examples."
                ),
                "Problem-Solving Agent": (
                    "Hi! We begin at Diagnose. Describe the problem, known information, unknown variable, "
                    "and anything you have tried. I will then design a progressive hint, gather your "
                    "report, and iterate without giving the final answer."
                ),
                "Verified Math Solver": (
                    "Enter an engineering calculation in words. I will parse the knowns, unknowns, "
                    "governing equations, and units; execute approved SymPy/Pint code; then release only "
                    "the verified result. Mock Mode supports the reviewed RLC demonstration."
                ),
            }
            welcome = welcome_messages[self.active_agent]
            self.session_messages[self.active_agent].append(("ai", welcome))
            messages = self.session_messages[self.active_agent]
        for sender, text in messages:
            self._add_bubble(sender, text)

        composer = tk.Frame(self.chat_card, bg=p["surface"])
        composer.pack(fill="x", padx=18, pady=16)
        self.attachment_bar = tk.Frame(composer, bg=p["surface"])
        self.attachment_bar.pack(fill="x", pady=(0, 8))
        self._render_attachment_bar()
        input_row = tk.Frame(composer, bg=p["surface"])
        input_row.pack(fill="x")
        self.upload_button = tk.Button(
            input_row, text="＋ Attach", bd=0, cursor="hand2", padx=13, pady=14,
            font=("Helvetica", 10, "bold"), bg=p["surface_alt"], fg=p["accent"],
            activebackground=p["border"], activeforeground=p["accent_hover"],
            command=self._choose_files,
        )
        self.upload_button.pack(side="left", fill="y", padx=(0, 8))
        self.message_entry = tk.Text(input_row, height=3, wrap="word", relief="flat",
                                     padx=12, pady=10, font=("Helvetica", 11),
                                     bg=p["surface_alt"], fg=p["text"],
                                     insertbackground=p["text"], highlightthickness=1,
                                     highlightbackground=p["border"], highlightcolor=p["accent"])
        self.message_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.message_entry.bind("<Return>", self._handle_enter)
        self.message_entry.focus_set()
        self.send_button = tk.Button(input_row, text="Send  ➜", bd=0, cursor="hand2",
                                     font=("Helvetica", 10, "bold"), padx=18, pady=14,
                                     bg=p["surface_alt"], fg=p["accent"],
                                     activebackground=p["border"], activeforeground=p["accent_hover"],
                                     disabledforeground=p["muted"], highlightbackground=p["accent"],
                                     command=self.send_message)
        self.send_button.pack(side="right", fill="y")

    def _draw_workflow(self) -> None:
        p = self.palette
        for widget in self.workflow_bar.winfo_children():
            widget.destroy()
        for i, stage in enumerate(WORKFLOW_STAGES):
            active = i == self.workflow_index
            complete = i < self.workflow_index
            color = p["success"] if complete else (p["accent"] if active else p["muted"])
            prefix = "✓" if complete else str(i + 1)
            label = tk.Label(self.workflow_bar, text=f" {prefix}  {stage} ",
                             font=("Helvetica", 9, "bold" if active else "normal"),
                             fg=color, bg=p["surface_alt"], pady=10)
            label.pack(side="left", expand=True)
            if i < len(WORKFLOW_STAGES) - 1:
                tk.Label(self.workflow_bar, text="→", fg=p["muted"],
                         bg=p["surface_alt"]).pack(side="left")

    def _new_session(self) -> None:
        self._cancel_stream()
        self.session_messages[self.active_agent] = []
        self.pending_attachments[self.active_agent] = []
        if self.active_agent == "Problem-Solving Agent":
            self.problem_session = ProblemSolvingSession()
            self._sync_workflow_index()
        else:
            self.workflow_index = 0
        self._render_agent_panel()

    def _load_rlc_example(self) -> None:
        if self.active_agent != "Verified Math Solver":
            return
        self.message_entry.delete("1.0", "end")
        self.message_entry.insert("1.0", RLC_PROBLEM)
        self.message_entry.focus_set()

    @staticmethod
    def _format_bytes(size: int) -> str:
        value = float(size)
        for unit in ("B", "KB", "MB", "GB"):
            if value < 1024 or unit == "GB":
                return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
            value /= 1024
        return f"{size} B"

    def _choose_files(self) -> None:
        available = MAX_ATTACHMENTS - len(self.pending_attachments[self.active_agent])
        if available <= 0:
            messagebox.showinfo("Local attachment limit", f"You can select up to {MAX_ATTACHMENTS} files per message.")
            return
        selected = filedialog.askopenfilenames(
            title="Attach course files",
            filetypes=(
                ("All files", "*.*"),
                ("Documents", "*.pdf *.txt *.doc *.docx *.rtf *.md"),
                ("Images", "*.png *.jpg *.jpeg *.gif *.webp *.tif *.tiff"),
            ),
        )
        existing = {item.path for item in self.pending_attachments[self.active_agent]}
        added = 0
        oversized: list[str] = []
        for raw_path in selected:
            if added >= available:
                break
            path = Path(raw_path)
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > MAX_ATTACHMENT_BYTES:
                oversized.append(path.name)
                continue
            resolved = path.resolve()
            if resolved in existing:
                continue
            mime_type = guess_type(path.name)[0] or "application/octet-stream"
            self.pending_attachments[self.active_agent].append(
                Attachment(resolved, path.name, size, mime_type)
            )
            existing.add(resolved)
            added += 1
        self._render_attachment_bar()
        if oversized:
            messagebox.showwarning(
                "File too large",
                "The local picker limit is 1 GB. These files were not selected:\n" + "\n".join(oversized),
            )

    def _remove_attachment(self, path: Path) -> None:
        self.pending_attachments[self.active_agent] = [
            item for item in self.pending_attachments[self.active_agent] if item.path != path
        ]
        self._render_attachment_bar()

    def _render_attachment_bar(self) -> None:
        if not hasattr(self, "attachment_bar") or not self.attachment_bar.winfo_exists():
            return
        p = self.palette
        for widget in self.attachment_bar.winfo_children():
            widget.destroy()
        attachments = self.pending_attachments[self.active_agent]
        if not attachments:
            tk.Label(
                self.attachment_bar,
                text="Files stay local • external analysis is disabled pending privacy approval",
                font=("Helvetica", 8), fg=p["muted"], bg=p["surface"],
            ).pack(side="left")
            return
        for attachment in attachments:
            chip = tk.Frame(self.attachment_bar, bg=p["accent_soft"], highlightthickness=1,
                            highlightbackground=p["border"])
            chip.pack(side="left", padx=(0, 6))
            tk.Label(
                chip, text=f"📎 {attachment.name}  {self._format_bytes(attachment.size)}",
                font=("Helvetica", 8), fg=p["text"], bg=p["accent_soft"], padx=7, pady=5,
            ).pack(side="left")
            tk.Button(
                chip, text="×", bd=0, cursor="hand2", font=("Helvetica", 10, "bold"),
                fg=p["muted"], bg=p["accent_soft"], activebackground=p["border"],
                command=lambda selected_path=attachment.path: self._remove_attachment(selected_path),
            ).pack(side="left", padx=(0, 5))

    def _handle_enter(self, event: tk.Event):
        if event.state & 0x1:  # Shift+Enter inserts a new line.
            return None
        self.send_message()
        return "break"

    def send_message(self) -> None:
        if self.streaming_job:
            return
        message = self.message_entry.get("1.0", "end").strip()
        attachments = list(self.pending_attachments[self.active_agent])
        if not message and not attachments:
            return
        if not message:
            message = "Please analyze the attached file(s) and solve or explain the relevant problem."
        self.message_entry.delete("1.0", "end")
        self.pending_attachments[self.active_agent] = []
        self._render_attachment_bar()
        self._sync_student_profile()
        attachment_summary = "\n".join(f"📎 {item.name}" for item in attachments)
        display_message = f"{message}\n{attachment_summary}" if attachment_summary else message
        self.session_messages[self.active_agent].append(("user", display_message))
        self._add_bubble("user", display_message)

        flagged, reason = self._detect_flag(message)
        if self.active_agent == "Verified Math Solver" and not attachments:
            if self.math_pipeline:
                self._stream_verified_math_response(message)
            else:
                normalized = message.lower()
                if "rlc" in normalized or all(
                    term in normalized for term in ("resistor", "inductor", "capacitor")
                ):
                    response = render_reviewed_rlc_answer(
                        solve_reviewed_rlc_example(self._math_significant_figures())
                    )
                else:
                    response = (
                        "### Mock Mode limitation\n"
                        "The deterministic engine will not guess a symbolic translation without a "
                        "configured model. In Mock Mode, select **Load RLC example** to run the "
                        "reviewed local SymPy/Pint demonstration. Connect Meta Mode to parse other "
                        "engineering problems."
                    )
                self._stream_response(response)
        elif self.meta_education_service and not attachments:
            directive: EducationTurn | None = None
            if self.active_agent == "Problem-Solving Agent":
                directive = self.meta_education_service.prepare_problem_turn(
                    self.student_profile, self.problem_session, message
                )
                flagged = flagged or directive.flagged
                reason = reason or directive.reason
                self._sync_workflow_index()
                self._draw_workflow()
            self._stream_meta_education_response(message, directive)
        else:
            if attachments:
                response = (
                    "The selected files stayed on this device and were not sent to Meta. External "
                    "file analysis is disabled until retention and deletion guarantees are approved. "
                    "Paste a short, non-sensitive excerpt as text if you want help with it."
                )
            elif self.active_agent == "Tutor Agent":
                turn = self.education_service.tutor_reply(self.student_profile, message)
                flagged = flagged or turn.flagged
                reason = reason or turn.reason
                response = turn.text
            else:
                turn = self.education_service.problem_reply(
                    self.student_profile, self.problem_session, message
                )
                flagged = flagged or turn.flagged
                reason = reason or turn.reason
                self._sync_workflow_index()
                self._draw_workflow()
                response = turn.text
            if self.active_agent != "Verified Math Solver":
                recommendations = recommend_resources(message, self.student_profile, limit=3)
                if recommendations:
                    response += "\n\n" + format_recommendations(recommendations)
            self._stream_response(response)

    def _math_significant_figures(self) -> int:
        try:
            value = int(self.math_sig_figs_var.get())
        except (TypeError, ValueError, tk.TclError):
            value = 4
        return min(max(value, 2), 8)

    @staticmethod
    def _detect_flag(message: str) -> tuple[bool, str]:
        if is_direct_answer_request(message):
            return True, "Potential answer leakage or policy bypass request"
        return False, ""

    def _mock_response(self, message: str) -> str:
        topic = self._extract_topic(message)
        if self.active_agent == "Tutor Agent":
            responses = [
                f"Let’s build an intuition for {topic}. Start by naming the system boundary and the "
                "quantity that must be conserved. Then connect each term in the equation to a physical "
                "effect. What changes—and what stays constant—in your example?",
                f"A useful way to study {topic} is: definition → visual model → equation → edge case. "
                "In plain language, explain what the governing relationship predicts. Then I’ll help "
                "you test it with a short exam-style variation.",
                f"Good question. For {topic}, avoid memorizing the formula in isolation. Identify the "
                "assumptions first, then ask which inputs control the output. Quick check: if one input "
                "doubles while the others stay fixed, what do you expect to happen?",
            ]
            return random.choice(responses)

        normalized = message.lower()
        if any(term in normalized for term in ("answer", "solve it", "do it for me", "final number")):
            return ("I can’t provide the direct or final answer to assessed work. I can help you reach "
                    "it yourself. First hint: draw the system boundary, list every known and unknown "
                    "quantity, and identify the governing principle. Tell me which step becomes unclear.")
        stage_responses = (
            f"Diagnosis: before calculating {topic}, what is the unknown, what information is given, "
            "and which assumption are you least confident about? I’ll use that to choose the smallest useful hint.",
            "Hint design: sketch a free-body or system diagram and label directions before writing an "
            "equation. Which interaction is missing from your current diagram? I won’t give the final answer.",
            "Progress report: show me the equation you chose and one sentence explaining why it applies. "
            "Even an incomplete attempt helps me identify the next misconception without revealing the solution.",
            "Iteration: your approach is moving in the right direction. Recheck the sign convention and "
            "units, then substitute symbolically before using numbers. What result do you get after that revision?",
        )
        return stage_responses[self.workflow_index]

    @staticmethod
    def _extract_topic(message: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9+\- ]", "", message).strip()
        words = cleaned.split()
        if len(words) > 8:
            cleaned = " ".join(words[-8:])
        return cleaned or "this concept"

    def _add_bubble(self, sender: str, text: str) -> RichMessage:
        p = self.palette
        row = tk.Frame(self.chat_scroll.content, bg=p["bg"])
        row.pack(fill="x", padx=18, pady=7)
        is_user = sender == "user"
        bubble = RichMessage(row, p, is_user=is_user)
        bubble.pack(
            side="right" if is_user else "left",
            fill="x",
            expand=True,
            padx=(90, 0) if is_user else (0, 90),
        )
        bubble.set_content(text, markdown=not is_user)
        self.chat_scroll.scroll_to_bottom()
        return bubble

    def _stream_response(self, response: str) -> None:
        self.stream_generation += 1
        generation = self.stream_generation
        target_agent = self.active_agent
        bubble = self._add_bubble("ai", "Thinking…")
        self.send_button.configure(state="disabled", text="Responding…")
        self.upload_button.configure(state="disabled")
        self.status_var.set("Local agent is composing a response…")
        words = response.split()
        chunks: list[str] = []
        step = 3
        for index in range(0, len(words), step):
            chunks.append(" ".join(words[index:index + step]))

        def write_chunk(index: int = 0, assembled: str = "") -> None:
            if generation != self.stream_generation or not bubble.winfo_exists():
                return
            if index >= len(chunks):
                self.streaming_job = None
                bubble.set_content(response, markdown=True)
                self.session_messages[target_agent].append(("ai", response))
                self.send_button.configure(state="normal", text="Send  ➜")
                self.upload_button.configure(state="normal")
                self.status_var.set(self._base_status())
                return
            assembled = f"{assembled} {chunks[index]}".strip()
            bubble.set_content(assembled, markdown=True, streaming=True)
            self.chat_scroll.scroll_to_bottom()
            self.streaming_job = self.after(65, write_chunk, index + 1, assembled)

        self.streaming_job = self.after(350, write_chunk)

    def _stream_verified_math_response(self, message: str) -> None:
        if not self.math_pipeline:
            return
        self.stream_generation += 1
        generation = self.stream_generation
        target_agent = self.active_agent
        bubble = self._add_bubble(
            "ai",
            "### Verification activity\n"
            "- Parsing knowns, unknowns, governing equations, and units\n"
            "- Waiting for the deterministic SymPy/Pint tool call",
        )
        self.send_button.configure(state="disabled", text="Verifying…")
        self.upload_button.configure(state="disabled")
        self.status_var.set("Verified Math Solver • symbolic solve and dimensional checks")
        events: queue.Queue[tuple[str, Any]] = queue.Queue()
        pipeline = self.math_pipeline
        significant_figures = self._math_significant_figures()
        discipline = self.student_profile.engineering_major or "General Engineering"

        def worker() -> None:
            try:
                outcome = pipeline.solve(
                    problem=message,
                    discipline=discipline,
                    significant_figures=significant_figures,
                )
                events.put(("done", outcome))
            except Exception as exc:
                events.put(("error", exc))

        threading.Thread(
            target=worker,
            daemon=True,
            name="verified-math-solver",
        ).start()

        def poll() -> None:
            if generation != self.stream_generation or not bubble.winfo_exists():
                return
            try:
                kind, value = events.get_nowait()
            except queue.Empty:
                self.streaming_job = self.after(50, poll)
                return

            if kind == "error":
                exc = value
                if isinstance(exc, PipelineError):
                    response = (
                        "### Verification failed safely\n"
                        f"{exc}\n\nNo unverified numerical answer was released."
                    )
                else:
                    response = (
                        "### Verification failed safely\n"
                        f"The deterministic solver stopped ({type(exc).__name__}). "
                        "No unverified numerical answer was released."
                    )
                bubble.set_content(response, markdown=True)
                self.session_messages[target_agent].append(("ai", response))
                self.status_var.set("Verified Math Solver stopped • no answer released")
            else:
                outcome = value
                verification = outcome.verification
                checks = "\n".join(
                    f"- {item}" for item in verification.verification
                ) or "- Sandbox completed its required checks."
                activity = "\n".join(f"- {item}" for item in outcome.trace)
                response = (
                    f"{outcome.answer}\n\n"
                    "### Deterministic verification report\n"
                    f"- Status: {verification.status}\n"
                    f"- Symbolic solution: `{verification.symbolic_solution}`\n"
                    f"- Numerical result: {verification.numerical_result} "
                    f"{verification.units or ''}\n"
                    f"- Tool attempts: {outcome.tool_attempts}\n"
                    f"{checks}\n\n"
                    "### Agent activity\n"
                    f"{activity}\n\n"
                    "The activity log shows observable workflow events, not private chain-of-thought."
                )
                bubble.set_content(response, markdown=True)
                self.session_messages[target_agent].append(("ai", response))
                self.status_var.set(
                    f"Verified Math released • {verification.numerical_result} "
                    f"{verification.units or ''} • {outcome.tool_attempts} tool attempt(s)"
                )

            self.streaming_job = None
            self.send_button.configure(state="normal", text="Send  ➜")
            self.upload_button.configure(state="normal")

        self.streaming_job = self.after(50, poll)

    def _stream_meta_education_response(
        self,
        message: str,
        directive: EducationTurn | None,
    ) -> None:
        if not self.meta_education_service:
            return
        self.stream_generation += 1
        generation = self.stream_generation
        target_agent = self.active_agent
        bubble = self._add_bubble("ai", "Contacting Meta Model API…")
        self.send_button.configure(state="disabled", text="Responding…")
        self.upload_button.configure(state="disabled")
        activity = (
            "Generating and checking a hint"
            if target_agent == "Problem-Solving Agent"
            else "Harness: analyze → ground → draft → verify → release"
        )
        self.status_var.set(
            f"{activity} • {self.meta_model} • limit {self.max_completion_tokens} tokens"
        )
        events: queue.Queue[tuple[str, str]] = queue.Queue()
        service = self.meta_education_service
        # The current user turn was appended before this method was called.
        history = tuple(self.session_messages[target_agent][:-1])
        # Both workflows are release-gated: no unchecked provider text is shown.
        buffer_before_display = True
        allowed_resource_urls = {
            *(resource.url for resource in RESOURCES),
            *approved_source_urls(),
        }
        activity_lines: list[str] = []

        def worker() -> None:
            try:
                if target_agent == "Tutor Agent":
                    chunks = service.stream_harnessed_tutor_reply(
                        self.student_profile,
                        message,
                        history,
                        progress=lambda event: events.put(("trace", event)),
                    )
                else:
                    if directive is None:
                        raise RuntimeError("Missing deterministic problem-solving directive")
                    chunks = service.stream_problem_language(
                        self.student_profile,
                        self.problem_session,
                        message,
                        directive,
                        history,
                    )
                for text in chunks:
                    events.put(("delta", text))
                events.put(("done", ""))
            except Exception as exc:  # Provider failures must not crash Tk's main loop.
                events.put(("error", self._safe_provider_error(exc)))

        threading.Thread(target=worker, daemon=True, name="meta-llama-stream").start()
        assembled = ""

        def poll() -> None:
            nonlocal assembled
            if generation != self.stream_generation or not bubble.winfo_exists():
                return
            finished = False
            while True:
                try:
                    kind, value = events.get_nowait()
                except queue.Empty:
                    break
                if kind == "delta":
                    assembled += value
                    if not buffer_before_display:
                        visible, _ = filter_provider_urls(assembled, allowed_resource_urls)
                        bubble.set_content(visible, markdown=True, streaming=True)
                        self.chat_scroll.scroll_to_bottom()
                elif kind == "trace":
                    activity_lines.append(value)
                    visible_activity = "### Agent activity\n" + "\n".join(
                        f"- {line}" for line in activity_lines
                    )
                    bubble.set_content(visible_activity, markdown=True, streaming=True)
                    self.chat_scroll.scroll_to_bottom()
                elif kind == "error":
                    assembled = value
                    bubble.set_content(assembled, markdown=True)
                    self.session_messages[target_agent].append(("ai", assembled))
                    self.status_var.set("Meta request failed • check key, access, network, and model")
                    finished = True
                elif kind == "done":
                    if buffer_before_display and directive is not None:
                        checked = service.finalize_problem_text(assembled, directive)
                        assembled = checked.text
                    else:
                        assembled = assembled.strip() or "Meta returned an empty response. Please try again."
                    trace = service.last_tutor_trace if target_agent == "Tutor Agent" else None
                    if trace:
                        allowed_resource_urls.update(trace.source_urls)
                    assembled, links_removed = filter_provider_urls(
                        assembled, allowed_resource_urls
                    )
                    if links_removed:
                        assembled += (
                            "\n\nA provider-generated link was removed because it was not in "
                            "the verified local resource catalog."
                        )
                    recommendations = recommend_resources(
                        message, self.student_profile, limit=3
                    )
                    if recommendations:
                        assembled += "\n\n" + format_recommendations(recommendations)
                    if trace and trace.quality_scores:
                        assembled += (
                            "\n\n### Automated quality checks\n"
                            "These are harness checks, not official benchmark or ground-truth scores.\n"
                            + "\n".join(
                                f"- {name.title()}: {score}/100"
                                for name, score in trace.quality_scores.items()
                            )
                        )
                    if activity_lines:
                        assembled += "\n\n### Agent activity\n" + "\n".join(
                            f"- {line}" for line in activity_lines
                        )
                    bubble.set_content(assembled, markdown=True)
                    self.session_messages[target_agent].append(("ai", assembled))
                    if trace:
                        revision = "revised once" if trace.revised else "draft passed"
                        reliability = trace.quality_scores.get("reliability", 0)
                        grounding = trace.quality_scores.get("grounding", 0)
                        self.status_var.set(
                            f"Tutor released • reliability {reliability}/100 • "
                            f"grounding {grounding}/100 • {revision}"
                        )
                    else:
                        self.status_var.set(self._base_status())
                    finished = True
            if finished:
                self.streaming_job = None
                self.send_button.configure(state="normal", text="Send  ➜")
                self.upload_button.configure(state="normal")
                return
            self.streaming_job = self.after(40, poll)

        self.streaming_job = self.after(40, poll)

    @staticmethod
    def _safe_provider_error(exc: Exception) -> str:
        """Return a useful error without exposing credentials or raw response bodies."""
        name = type(exc).__name__
        if name in {"AuthenticationError", "PermissionDeniedError"}:
            return "Meta rejected the credential or this model is not enabled for the account."
        if name in {"RateLimitError"}:
            return "Meta rate-limited the request. Wait briefly and try again."
        if name in {"APITimeoutError", "APIConnectionError", "TimeoutError", "ConnectionError"}:
            return "The Meta API could not be reached. Check the network and try again."
        return f"The Meta request failed ({name}). Check the local setup and try again."

    def _cancel_stream(self) -> None:
        self.stream_generation += 1
        if self.streaming_job:
            try:
                self.after_cancel(self.streaming_job)
            except tk.TclError:
                pass
        self.streaming_job = None

    # ---------------------------- App shell -----------------------------

    def toggle_theme(self) -> None:
        self.dark_mode = not self.dark_mode
        self.palette = DARK.copy() if self.dark_mode else LIGHT.copy()
        self._cancel_stream()
        for widget in self.winfo_children():
            widget.destroy()
        self._configure_styles()
        self._build_shell()
        self.theme_button.configure(text="☀  Light" if self.dark_mode else "☾  Dark")
        self._build_student_view()

    def _on_close(self) -> None:
        self._cancel_stream()
        if self.meta_service:
            try:
                self.meta_service.close()
            except Exception:
                pass
        self.destroy()


def _run_packaged_self_test() -> None:
    """Exercise frozen runtime dependencies without network or credentials."""

    from PIL import Image
    from docx import Document
    from matplotlib.font_manager import FontProperties
    from matplotlib.mathtext import math_to_image
    from openai import OpenAI
    from pypdf import PdfReader
    from striprtf.striprtf import rtf_to_text

    from education_harness import REQUIRED_SECTIONS, build_question_contract
    from knowledge_base import find_concept_cards

    assert Document and OpenAI and PdfReader and rtf_to_text
    tokens = parse_markdown("## Check\n- **Force**: $F=ma$")
    assert any(token.kind == "math_inline" for token in tokens)
    buffer = io.BytesIO()
    math_to_image(
        r"$F=ma$",
        buffer,
        prop=FontProperties(size=12),
        dpi=120,
        format="png",
    )
    buffer.seek(0)
    with Image.open(buffer) as rendered:
        rendered.verify()
    profile = StudentProfile()
    cards = find_concept_cards("Explain the throttling process")
    contract = build_question_contract(profile, "Explain the throttling process", cards)
    answer = MockEducationService().tutor_reply(
        profile, "Explain the throttling process"
    ).text
    assert cards and contract.required_sections == REQUIRED_SECTIONS
    assert all(section in answer for section in REQUIRED_SECTIONS)
    assert recommend_resources("statics", profile, limit=1)
    math_result = solve_reviewed_rlc_example(4)
    assert math_result.status == "success"
    assert math_result.units == "C"
    assert math_result.numerical_result == "-0.005984"


def main() -> None:
    if "--math-sandbox-worker" in sys.argv:
        from backend.math_verification.math_sandbox import _worker_main

        raise SystemExit(_worker_main())
    if "--package-self-test" in sys.argv:
        _run_packaged_self_test()
        return
    app = EducationAgentApp()
    app.mainloop()


if __name__ == "__main__":
    main()
