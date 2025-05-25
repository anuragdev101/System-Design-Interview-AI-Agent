# modules/gui.py

import json
import logging
import os
import queue
import time
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from tkinter.font import Font
from typing import Any, Dict, List, Optional

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

logger = logging.getLogger(__name__)


class AgentGUI:
    """Main GUI class for the System Design Interview AI Agent."""

    def __init__(self, root: tk.Tk, agent: Optional[Any] = None):
        self.root = root
        self.agent = agent
        self.is_running = False
        self.update_queue: queue.Queue = queue.Queue(maxsize=200)

        self.current_diagram_photo: Optional[ImageTk.PhotoImage] = None
        self.current_diagram_path: Optional[str] = None
        self.current_diagram_title: Optional[str] = None
        self._resize_job: Optional[str] = None

        self.gui_active = True
        self._after_id_gui_updates: Optional[str] = None

        self.root.title("System Design Interview AI Agent")
        self.root.geometry("1250x850")

        # --- Fonts ---
        self.base_font_family = "Segoe UI" if os.name == "nt" else "Arial"
        self.title_font = Font(family=self.base_font_family, size=16, weight="bold")
        self.header_font = Font(family=self.base_font_family, size=11, weight="bold")
        self.text_font = Font(family=self.base_font_family, size=10)
        self.mono_font = Font(
            family="Consolas" if os.name == "nt" else "Courier", size=10
        )

        # --- Colors ---
        self.COLOR_BG_MAIN = "#F0F2F5"
        self.COLOR_BG_FRAME = "#FFFFFF"
        self.COLOR_BG_ACCENT_LIGHT = "#E0EAFC"  # Lightened blue for hover
        self.COLOR_TEXT_PRIMARY = "#1F1F1F"
        self.COLOR_TEXT_SECONDARY = "#595959"
        self.COLOR_ACCENT_PRIMARY = "#0078D4"
        self.COLOR_ACCENT_INTERVIEWER = "#005A9E"
        self.COLOR_ACCENT_CANDIDATE = "#107C10"
        self.COLOR_ACCENT_AGENT = "#5C2D91"
        self.COLOR_ACCENT_ERROR = "#A80000"  # Darker red for error
        self.COLOR_BORDER = "#D1D1D1"
        self.COLOR_PLACEHOLDER_TEXT = "#A0A0A0"

        self.root.configure(bg=self.COLOR_BG_MAIN)

        # --- TTK Styling ---
        self.style = ttk.Style()
        available_themes = self.style.theme_names()
        logger.info(f"Available ttk themes: {available_themes}")
        # Try to use ttkthemes if available for more modern looks
        # Order of preference: arc (ttkthemes), plastik (ttkthemes), then OS-native
        chosen_theme_name = "default"
        try:
            from ttkthemes import ThemedStyle

            themed_style = ThemedStyle(root)
            if "arc" in themed_style.get_themes():
                themed_style.set_theme("arc")
                chosen_theme_name = "arc (ttkthemes)"
            elif "plastik" in themed_style.get_themes():
                themed_style.set_theme("plastik")
                chosen_theme_name = "plastik (ttkthemes)"
            else:  # Fallback to standard ttk themes if ttkthemes preferred aren't there
                preferred_ttk_themes = ["vista", "xpnative", "clam", "alt"]
                for theme in preferred_ttk_themes:
                    if theme in available_themes:
                        self.style.theme_use(theme)
                        chosen_theme_name = f"{theme} (standard ttk)"
                        break
            logger.info(f"Using theme: {chosen_theme_name}")
        except ImportError:
            logger.info("ttkthemes not installed. Using standard ttk themes.")
            preferred_ttk_themes = ["vista", "xpnative", "clam", "alt", "default"]
            for theme in preferred_ttk_themes:
                if theme in available_themes:
                    try:
                        self.style.theme_use(theme)
                        chosen_theme_name = f"{theme} (standard ttk)"
                        logger.info(f"Using ttk theme: {chosen_theme_name}")
                        break
                    except tk.TclError:
                        pass  # Theme might exist but fail to apply
        except tk.TclError as e:
            logger.warning(
                f"TclError setting theme with ttkthemes: {e}. Will use standard ttk."
            )
            # Fallback for standard ttk if ttkthemes failed
            preferred_ttk_themes = ["vista", "xpnative", "clam", "alt", "default"]
            for theme in preferred_ttk_themes:
                if theme in available_themes:
                    try:
                        self.style.theme_use(theme)
                        chosen_theme_name = f"{theme} (standard ttk)"
                        logger.info(f"Using ttk theme: {chosen_theme_name}")
                        break
                    except tk.TclError:
                        pass

        self.style.configure(
            ".",
            font=self.text_font,
            background=self.COLOR_BG_MAIN,
            foreground=self.COLOR_TEXT_PRIMARY,
        )
        self.style.configure("TFrame", background=self.COLOR_BG_MAIN)
        self.style.configure(
            "Content.TFrame", background=self.COLOR_BG_FRAME
        )  # For inner content areas

        self.style.configure(
            "TLabel", background=self.COLOR_BG_MAIN, foreground=self.COLOR_TEXT_PRIMARY
        )
        self.style.configure(
            "Header.TLabel",
            font=self.header_font,
            foreground=self.COLOR_TEXT_PRIMARY,
            background=self.COLOR_BG_MAIN,
        )
        self.style.configure(
            "Title.TLabel",
            font=self.title_font,
            foreground=self.COLOR_ACCENT_PRIMARY,
            background=self.COLOR_BG_MAIN,
        )
        self.style.configure(
            "Muted.TLabel",
            foreground=self.COLOR_TEXT_SECONDARY,
            background=self.COLOR_BG_MAIN,
        )  # For less important labels

        self.style.configure(
            "TButton",
            font=Font(family=self.base_font_family, size=9),
            padding=(10, 5),
            relief=tk.FLAT,
            borderwidth=1,
        )
        self.style.map(
            "TButton",
            background=[
                ("active", self.COLOR_BG_ACCENT_LIGHT),
                ("pressed", self.COLOR_BG_ACCENT_LIGHT),
                ("!disabled", self.COLOR_BG_FRAME),
            ],
            foreground=[("!disabled", self.COLOR_ACCENT_PRIMARY)],
            bordercolor=[
                ("focus", self.COLOR_ACCENT_PRIMARY),
                ("!disabled", self.COLOR_BORDER),
            ],
            relief=[("pressed", tk.SUNKEN)],
        )

        self.style.configure(
            "Primary.TButton", foreground=self.COLOR_BG_FRAME, borderwidth=0
        )  # Primary button
        self.style.map(
            "Primary.TButton",
            background=[
                ("active", "#005a9e"),
                ("pressed", "#004578"),
                ("!disabled", self.COLOR_ACCENT_PRIMARY),
            ],  # Darker shades on press/active
            relief=[("pressed", tk.FLAT), ("!pressed", tk.FLAT)],
        )

        self.style.configure(
            "Refresh.TButton",
            foreground=self.COLOR_TEXT_SECONDARY,
            padding=(6, 4),
            font=Font(family=self.base_font_family, size=9),
        )
        self.style.map(
            "Refresh.TButton", bordercolor=[("!disabled", self.COLOR_BORDER)]
        )

        self.style.configure(
            "TLabelframe",
            background=self.COLOR_BG_MAIN,
            padding=(10, 5, 10, 10),
            relief=tk.SOLID,
            borderwidth=1,
            bordercolor=self.COLOR_BORDER,
        )
        self.style.configure(
            "TLabelframe.Label",
            font=self.header_font,
            foreground=self.COLOR_ACCENT_PRIMARY,
            background=self.COLOR_BG_MAIN,
            padding=(0, 0, 0, 5),
        )

        self.style.configure(
            "Content.TLabelframe",
            background=self.COLOR_BG_FRAME,
            padding=5,
            relief=tk.SOLID,
            borderwidth=1,
            bordercolor=self.COLOR_BORDER,
        )
        self.style.configure(
            "Content.TLabelframe.Label",
            font=self.header_font,
            foreground=self.COLOR_TEXT_PRIMARY,
            background=self.COLOR_BG_FRAME,
            padding=(5, 0, 0, 5),
        )

        self.style.configure("TPanedwindow", background=self.COLOR_BG_MAIN)
        # Attempt to style PanedWindow sash (highly theme dependent)
        self.style.configure(
            "TPanedwindow.Sash",
            background=self.COLOR_BORDER,
            sashthickness=6,
            gripcount=0,
            relief=tk.FLAT,
        )

        self.style.configure("TCombobox", font=self.text_font, padding=4)
        self.style.map("TCombobox", fieldbackground=[("readonly", self.COLOR_BG_FRAME)])
        self.style.map(
            "TCombobox", selectbackground=[("readonly", self.COLOR_BG_FRAME)]
        )
        self.style.map(
            "TCombobox", selectforeground=[("readonly", self.COLOR_TEXT_PRIMARY)]
        )
        self.root.option_add(
            "*TCombobox*Listbox.font", self.text_font
        )  # Font for dropdown list
        self.root.option_add("*TCombobox*Listbox.background", self.COLOR_BG_FRAME)
        self.root.option_add("*TCombobox*Listbox.foreground", self.COLOR_TEXT_PRIMARY)
        self.root.option_add(
            "*TCombobox*Listbox.selectBackground", self.COLOR_ACCENT_PRIMARY
        )
        self.root.option_add("*TCombobox*Listbox.selectForeground", self.COLOR_BG_FRAME)

        self.main_frame = ttk.Frame(self.root, padding="15")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        self._create_header()
        self._create_controls()
        self._create_main_content()
        self._create_status_bar()

        if hasattr(self, "details_text"):
            self.details_text.tag_configure(
                "bold_header",
                font=(
                    self.header_font.cget("family"),
                    self.header_font.cget("size"),
                    "bold",
                ),
                foreground=self.COLOR_ACCENT_PRIMARY,
                spacing1=7,
                spacing3=5,
            )
            self.details_text.tag_configure(
                "placeholder",
                foreground=self.COLOR_PLACEHOLDER_TEXT,
                font=(
                    self.mono_font.cget("family"),
                    self.mono_font.cget("size"),
                    "italic",
                ),
            )
        if hasattr(self, "chat_text"):
            self.chat_text.tag_configure(
                "interviewer",
                foreground=self.COLOR_ACCENT_INTERVIEWER,
                font=(
                    self.text_font.cget("family"),
                    self.text_font.cget("size"),
                    "bold",
                ),
            )
            self.chat_text.tag_configure(
                "candidate",
                foreground=self.COLOR_ACCENT_CANDIDATE,
                font=(
                    self.text_font.cget("family"),
                    self.text_font.cget("size"),
                    "bold",
                ),
            )
            self.chat_text.tag_configure(
                "agent",
                foreground=self.COLOR_ACCENT_AGENT,
                font=(
                    self.text_font.cget("family"),
                    self.text_font.cget("size"),
                    "bold",
                ),
            )
            self.chat_text.tag_configure(
                "system",
                foreground=self.COLOR_TEXT_SECONDARY,
                font=(
                    self.mono_font.cget("family"),
                    self.mono_font.cget("size") - 1,
                    "italic",
                ),
            )
            self.chat_text.tag_configure(
                "error",
                foreground=self.COLOR_ACCENT_ERROR,
                font=(
                    self.text_font.cget("family"),
                    self.text_font.cget("size"),
                    "bold",
                ),
            )

        if self.gui_active and self.root.winfo_exists():
            self._after_id_gui_updates = self.root.after(50, self._process_updates)

        self._update_running_status_display(False)
        self.update_status("Ready")
        logger.info("AgentGUI initialized with enhanced styling.")

    def shutdown_gui_updates(self):
        logger.info("AgentGUI.shutdown_gui_updates() called.")
        self.gui_active = False
        if self._after_id_gui_updates:
            current_id = self._after_id_gui_updates
            self._after_id_gui_updates = None
            try:
                if self.root.winfo_exists():
                    self.root.after_cancel(current_id)
            except Exception:
                pass

    def destroy_matplotlib_resources(self):
        logger.debug("Attempting to destroy Matplotlib resources.")
        if hasattr(self, "canvas") and self.canvas:
            try:
                if hasattr(self.canvas, "figure") and self.canvas.figure:
                    self.canvas.figure.clear()
                    plt.close(self.canvas.figure)
                if (
                    hasattr(self.canvas, "get_tk_widget")
                    and self.canvas.get_tk_widget().winfo_exists()
                ):
                    self.canvas.get_tk_widget().destroy()
                logger.info("Matplotlib resources released.")
            except Exception as e:
                logger.error(f"Error destroying Matplotlib: {e}", exc_info=True)
        self.canvas = None
        self.fig = None
        self.ax = None
        self.line = None

    def _process_updates(self):
        # (Same as before)
        if not self.gui_active:
            self._after_id_gui_updates = None
            return
        try:
            for _ in range(self.update_queue.qsize()):
                if self.update_queue.empty():
                    break
                try:
                    update_type, data = self.update_queue.get_nowait()
                    if not self.root.winfo_exists():
                        break
                    if update_type == "chat":
                        self._add_to_chat(data["speaker"], data["text"])
                    elif update_type == "audio":
                        self._update_audio_viz_data(data.get("data"))
                    elif update_type == "status":
                        self.status_bar_text_var.set(data.get("text", "N/A"))
                    elif update_type == "diagram_and_details":
                        self._display_diagram_and_details(data)
                    elif update_type == "speaker":
                        self._update_speaker_indicators(data)
                    elif update_type == "devices":
                        self._update_device_list_in_gui(data)
                    elif update_type == "error_message_box":
                        messagebox.showerror(
                            "Agent Error",
                            data.get("message", "Error"),
                            parent=self.root,
                        )
                    elif update_type == "set_stopped_ui_state":
                        self.is_running = False
                        self.status_label_text_var.set("Stopped")
                        self._update_running_status_display(False)
                        if hasattr(self, "start_button"):
                            self.start_button.config(state=tk.NORMAL)
                        current_status = self.status_bar_text_var.get()
                        if (
                            "stopping" in current_status.lower()
                            or "error" not in current_status.lower()
                        ):
                            self.status_bar_text_var.set("Agent stopped.")
                    elif update_type == "clear_chat":
                        self._clear_chat_display()
                    elif update_type == "clear_diagram":
                        self._clear_diagram_display()
                except queue.Empty:
                    break
                except Exception as e:
                    logger.error(
                        f"Error processing GUI update '{update_type}': {e}",
                        exc_info=True,
                    )
        finally:
            if self.gui_active and self.root.winfo_exists():
                self._after_id_gui_updates = self.root.after(50, self._process_updates)
            else:
                self._after_id_gui_updates = None

    def set_agent(self, agent_instance):
        self.agent = agent_instance

    def start(self):
        logger.info("Starting GUI mainloop.")
        self.root.mainloop()

    def update_chat(self, speaker: str, text: str):
        try:
            self.update_queue.put_nowait(("chat", {"speaker": speaker, "text": text}))
        except queue.Full:
            logger.warning("GUI Q full, drop chat.")

    def update_audio_visualization(self, audio_data_np_array: Optional[np.ndarray]):
        try:
            data = (
                audio_data_np_array
                if isinstance(audio_data_np_array, np.ndarray)
                else np.zeros(100)
            )
            self.update_queue.put_nowait(("audio", {"data": data}))
        except queue.Full:
            logger.warning("GUI Q full, drop audio_viz.")

    def update_status(self, status_text: str):
        try:
            self.update_queue.put_nowait(("status", {"text": status_text}))
        except queue.Full:
            logger.warning("GUI Q full, drop status.")

    def display_error(self, error_message_text: str):
        try:
            self.update_queue.put_nowait(
                ("error_message_box", {"message": error_message_text})
            )
        except queue.Full:
            logger.warning("GUI Q full, drop error.")

    def _update_running_status_display(self, is_now_running: bool):
        if hasattr(self, "status_indicator") and self.status_indicator.winfo_exists():
            self.status_indicator.config(
                foreground=(
                    self.COLOR_ACCENT_CANDIDATE
                    if is_now_running
                    else self.COLOR_ACCENT_ERROR
                )
            )  # Candidate Green for running
        if hasattr(self, "start_button") and self.start_button.winfo_exists():
            self.start_button.config(
                text="Stop Interview" if is_now_running else "Start Interview"
            )
            self.start_button.configure(
                style="Primary.TButton" if is_now_running else "TButton"
            )

    def _create_header(self):
        header_frame = ttk.Frame(
            self.main_frame, padding=(0, 0, 0, 5)
        )  # Less bottom padding for header frame itself
        header_frame.pack(fill=tk.X, pady=(0, 10))
        title_label = ttk.Label(
            header_frame, text="System Design Interview AI Agent", style="Title.TLabel"
        )
        title_label.pack(
            side=tk.LEFT, padx=5, pady=(5, 10)
        )  # Add some vertical padding
        self.status_indicator = ttk.Label(
            header_frame, text="●", font=Font(family=self.base_font_family, size=20)
        )
        self.status_indicator.pack(side=tk.RIGHT, padx=(0, 10), pady=(5, 10))
        self.status_label_text_var = tk.StringVar(value="Stopped")
        status_label = ttk.Label(
            header_frame, textvariable=self.status_label_text_var, style="Header.TLabel"
        )
        status_label.pack(side=tk.RIGHT, padx=(0, 10), pady=(5, 10))

    def _create_controls(self):
        control_frame = ttk.LabelFrame(
            self.main_frame, text="Controls", style="TLabelframe"
        )
        control_frame.pack(fill=tk.X, pady=(0, 10), padx=0)
        device_frame = ttk.Frame(
            control_frame, padding=10
        )  # This frame will use control_frame's BG
        device_frame.pack(fill=tk.X)

        ttk.Label(device_frame, text="Candidate Device:").grid(
            row=0, column=0, padx=(0, 5), pady=5, sticky="w"
        )
        self.candidate_device_var = tk.StringVar()
        self.candidate_device_combo = ttk.Combobox(
            device_frame,
            textvariable=self.candidate_device_var,
            width=40,
            state="readonly",
            style="TCombobox",
        )
        self.candidate_device_combo.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        ttk.Label(device_frame, text="Interviewer Device:").grid(
            row=0, column=2, padx=(15, 5), pady=5, sticky="w"
        )
        self.interviewer_device_var = tk.StringVar()
        self.interviewer_device_combo = ttk.Combobox(
            device_frame,
            textvariable=self.interviewer_device_var,
            width=40,
            state="readonly",
            style="TCombobox",
        )
        self.interviewer_device_combo.grid(row=0, column=3, padx=5, pady=5, sticky="ew")

        self.refresh_button = ttk.Button(
            device_frame,
            text="Refresh",
            command=self._refresh_devices,
            style="Refresh.TButton",
            width=8,
        )  # Shorter text
        self.refresh_button.grid(
            row=0, column=4, padx=(10, 0), pady=5
        )  # Adjusted padding

        device_frame.grid_columnconfigure(1, weight=1)
        device_frame.grid_columnconfigure(3, weight=1)

        action_frame = ttk.Frame(
            control_frame, padding=(10, 10, 10, 10)
        )  # Added top padding
        action_frame.pack(fill=tk.X)
        self.start_button = ttk.Button(
            action_frame,
            text="Start Interview",
            command=self._toggle_agent,
            width=18,
            style="Primary.TButton",
        )  # Use Primary style
        self.start_button.pack(side=tk.LEFT, padx=0)

    def _refresh_devices(self):
        if self.agent and hasattr(self.agent, "_refresh_and_populate_devices"):
            self.agent._refresh_and_populate_devices()
            self.update_status("Refreshing audio devices...")
        elif self.root.winfo_exists():
            messagebox.showwarning(
                "Agent Offline", "Agent not available.", parent=self.root
            )

    def _update_device_list_in_gui(self, devices_from_agent: List[Dict[str, Any]]):
        # (Implementation is mostly the same, ensure it uses self.text_font for combobox if not covered by style)
        if not (
            hasattr(self, "candidate_device_combo")
            and self.candidate_device_combo.winfo_exists()
        ):
            return
        try:
            candidate_values = [
                f"{d['name']} (idx: {d['index']})"
                for d in devices_from_agent
                if d.get("max_input_channels", 0) > 0
            ]
            interviewer_values = list(candidate_values)
            candidate_values.sort(
                key=lambda x: not any(k in x.lower() for k in ["microphone", "mic"])
            )
            interviewer_values.sort(
                key=lambda x: not any(
                    k in x.lower()
                    for k in ["stereo mix", "loopback", "what u hear", "cable"]
                )
            )
            self.candidate_device_combo["values"] = candidate_values
            self.interviewer_device_combo["values"] = interviewer_values
            current_cand = self.candidate_device_var.get()
            current_int = self.interviewer_device_var.get()
            if candidate_values and (
                not current_cand or current_cand not in candidate_values
            ):
                self.candidate_device_var.set(candidate_values[0])
            elif not candidate_values:
                self.candidate_device_var.set("")
            if interviewer_values and (
                not current_int or current_int not in interviewer_values
            ):
                pref = next(
                    (
                        v
                        for v in interviewer_values
                        if any(k in v.lower() for k in ["stereo", "loop", "mix"])
                    ),
                    interviewer_values[0] if interviewer_values else "",
                )
                self.interviewer_device_var.set(pref)
            elif not interviewer_values:
                self.interviewer_device_var.set("")
            logger.info("Device list updated.")
            self.update_status("Audio devices refreshed.")
        except Exception as e:
            logger.error(f"Error updating GUI devices: {e}", exc_info=True)

    def _create_main_content(self):
        content_frame = ttk.Frame(self.main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, pady=0)
        self.paned_window = ttk.PanedWindow(
            content_frame, orient=tk.HORIZONTAL, style="TPanedwindow"
        )
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        left_panel_outer = ttk.Frame(self.paned_window)
        self.paned_window.add(left_panel_outer, weight=2)
        left_panel = ttk.Frame(left_panel_outer, style="Content.TFrame", padding=10)
        left_panel.pack(fill=tk.BOTH, expand=True)

        audio_frame = ttk.LabelFrame(
            left_panel, text="Audio Visualization", style="Content.TLabelframe"
        )  # Use Content style for LFs inside white BG
        audio_frame.pack(fill=tk.X, pady=(0, 10))
        self.fig = Figure(figsize=(5, 1.0), dpi=100)
        self.fig.patch.set_facecolor(self.COLOR_BG_FRAME)  # Reduced height
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(self.COLOR_BG_ACCENT_LIGHT)
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_xlim(0, 100)
        self.ax.set_yticks([])
        self.ax.set_xticks([])
        for spine in ["top", "right", "bottom", "left"]:
            self.ax.spines[spine].set_color(self.COLOR_BORDER)
        (self.line,) = self.ax.plot(
            np.zeros(100), color=self.COLOR_ACCENT_CANDIDATE, linewidth=1.5
        )
        self.fig.tight_layout(pad=0.1)  # Tighter padding
        self.canvas = FigureCanvasTkAgg(self.fig, master=audio_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.configure(bg=self.COLOR_BG_FRAME)
        self.canvas_widget.pack(fill=tk.X, expand=False, pady=(2, 2))  # Reduced pady
        self.canvas.draw()

        indicators_frame = ttk.Frame(left_panel, style="Content.TFrame")
        indicators_frame.pack(fill=tk.X, pady=(0, 5), padx=5)  # Reduced bottom pady
        self.interviewer_indicator = ttk.Label(
            indicators_frame,
            text="● Interviewer",
            style="Muted.TLabel",
            background=self.COLOR_BG_FRAME,
        )
        self.interviewer_indicator.pack(side=tk.LEFT, padx=(0, 20))
        self.candidate_indicator = ttk.Label(
            indicators_frame,
            text="● Candidate",
            style="Muted.TLabel",
            background=self.COLOR_BG_FRAME,
        )
        self.candidate_indicator.pack(side=tk.LEFT)

        chat_frame = ttk.LabelFrame(
            left_panel, text="Interview Conversation", style="Content.TLabelframe"
        )
        chat_frame.pack(fill=tk.BOTH, expand=True)
        self.chat_text = scrolledtext.ScrolledText(
            chat_frame,
            wrap=tk.WORD,
            font=self.text_font,
            relief=tk.FLAT,
            borderwidth=1,
            bg=self.COLOR_BG_FRAME,
            fg=self.COLOR_TEXT_PRIMARY,
            padx=8,
            pady=8,
            highlightthickness=1,
            highlightbackground=self.COLOR_BORDER,
            insertbackground=self.COLOR_TEXT_PRIMARY,
        )
        self.chat_text.pack(
            fill=tk.BOTH, expand=True, padx=0, pady=0
        )  # No extra padding if LF has it
        self.chat_text.config(state=tk.DISABLED)

        right_panel_outer = ttk.Frame(self.paned_window)
        self.paned_window.add(right_panel_outer, weight=3)
        right_panel = ttk.Frame(right_panel_outer, style="Content.TFrame", padding=10)
        right_panel.pack(fill=tk.BOTH, expand=True)

        design_frame = ttk.LabelFrame(
            right_panel, text="System Design Diagram", style="Content.TLabelframe"
        )
        design_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.design_canvas = tk.Canvas(
            design_frame,
            bg=self.COLOR_BG_FRAME,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
        )
        self.design_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=0, pady=0)
        self.design_canvas.bind("<Configure>", self._on_diagram_canvas_resize)

        details_frame = ttk.LabelFrame(
            right_panel, text="Design Details / Notes", style="Content.TLabelframe"
        )
        details_frame.pack(fill=tk.BOTH, expand=True)
        self.details_text = scrolledtext.ScrolledText(
            details_frame,
            wrap=tk.WORD,
            font=self.mono_font,
            height=8,
            relief=tk.FLAT,
            borderwidth=1,
            bg=self.COLOR_BG_FRAME,
            fg=self.COLOR_TEXT_PRIMARY,
            padx=8,
            pady=8,
            highlightthickness=1,
            highlightbackground=self.COLOR_BORDER,
            insertbackground=self.COLOR_TEXT_PRIMARY,
        )
        self.details_text.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        self.details_text.config(state=tk.DISABLED)

    def _create_status_bar(self):
        ttk.Separator(self.main_frame, orient=tk.HORIZONTAL).pack(
            fill=tk.X, pady=(10, 2), padx=0
        )
        status_frame = ttk.Frame(self.main_frame, padding=(5, 3))  # No sunken relief
        status_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(0, 0))
        self.status_bar_text_var = tk.StringVar(value="Ready")
        status_bar_label = ttk.Label(
            status_frame,
            textvariable=self.status_bar_text_var,
            anchor=tk.W,
            style="Muted.TLabel",
        )
        status_bar_label.pack(fill=tk.X, side=tk.LEFT, expand=True)

    def _add_to_chat(self, speaker: str, text: str):
        # (Same as previously provided version that uses defined tags)
        if not (hasattr(self, "chat_text") and self.chat_text.winfo_exists()):
            return
        if not text or not text.strip():
            return
        self.chat_text.config(state=tk.NORMAL)
        timestamp = time.strftime("[%H:%M:%S] ")
        self.chat_text.insert(tk.END, timestamp, "system")
        speaker_display = str(speaker).capitalize()
        tag = str(speaker).lower()
        prefix_map = {
            "interviewer": f"{speaker_display}: ",
            "candidate": f"{speaker_display}: ",
            "agent": "AI Agent: ",
        }
        final_tag = (
            tag
            if tag in ["interviewer", "candidate", "agent", "system", "error"]
            else "system"
        )
        prefix = prefix_map.get(tag, f"{speaker_display}: ")

        self.chat_text.insert(tk.END, prefix, final_tag)
        if tag not in ["system", "error"]:
            self.chat_text.insert(tk.END, f"{text}\n")
        else:
            self.chat_text.insert(
                tk.END, f"{text}\n", final_tag
            )  # Ensure system/error messages also get their tag

        self.chat_text.config(state=tk.DISABLED)
        self.chat_text.see(tk.END)

    def _update_audio_viz_data(self, audio_data_np: Optional[np.ndarray]):
        # (Same as before)
        try:
            if not (
                hasattr(self, "line")
                and self.line
                and hasattr(self, "canvas")
                and self.canvas
            ):
                return
            data = (
                audio_data_np.flatten()
                if isinstance(audio_data_np, np.ndarray)
                else np.zeros(100)
            )
            if len(data) > 100:
                data = data[-100:]
            elif len(data) < 100:
                data = np.pad(data, (0, 100 - len(data)), "constant", constant_values=0)
            self.line.set_ydata(data)
            if self.canvas.get_tk_widget().winfo_exists():
                self.canvas.draw_idle()
        except Exception:
            pass

    def _update_speaker_indicators(self, speaker_data: Dict[str, Any]):
        # (Using defined colors)
        speaker = speaker_data.get("speaker", "unknown")
        interviewer_color = (
            self.COLOR_ACCENT_INTERVIEWER
            if speaker == "interviewer"
            else self.COLOR_TEXT_SECONDARY
        )
        candidate_color = (
            self.COLOR_ACCENT_CANDIDATE
            if speaker == "candidate"
            else self.COLOR_TEXT_SECONDARY
        )
        if (
            hasattr(self, "interviewer_indicator")
            and self.interviewer_indicator.winfo_exists()
        ):
            self.interviewer_indicator.config(foreground=interviewer_color)
        if (
            hasattr(self, "candidate_indicator")
            and self.candidate_indicator.winfo_exists()
        ):
            self.candidate_indicator.config(foreground=candidate_color)

    def _clear_chat_display(self):
        # (Same as before)
        if hasattr(self, "chat_text") and self.chat_text.winfo_exists():
            self.chat_text.config(state=tk.NORMAL)
            self.chat_text.delete(1.0, tk.END)
            self._add_to_chat("system", "New interview session started.")
            self.chat_text.config(state=tk.DISABLED)

    def _clear_diagram_display(self):
        # (Same as before)
        if hasattr(self, "design_canvas") and self.design_canvas.winfo_exists():
            self.design_canvas.delete("all")
        self.current_diagram_photo = None
        self.current_diagram_path = None
        self.current_diagram_title = None
        if hasattr(self, "details_text") and self.details_text.winfo_exists():
            self.details_text.config(state=tk.NORMAL)
            self.details_text.delete(1.0, tk.END)
            self.details_text.config(state=tk.DISABLED)

    def _on_diagram_canvas_resize(self, event: Any):
        # (Same as before)
        if not (
            hasattr(self, "design_canvas")
            and self.design_canvas
            and self.design_canvas.winfo_exists()
        ):
            return
        if hasattr(self, "_resize_job") and self._resize_job:
            try:
                self.design_canvas.after_cancel(self._resize_job)
            except tk.TclError:
                pass
        self._resize_job = self.design_canvas.after(
            250, self._redisplay_current_diagram
        )

    def _redisplay_current_diagram(self):
        # (Same as before)
        if (
            self.current_diagram_path
            and self.current_diagram_title
            and os.path.exists(self.current_diagram_path)
            and hasattr(self, "design_canvas")
            and self.design_canvas
            and self.design_canvas.winfo_exists()
        ):
            self._render_diagram_image(
                self.current_diagram_path, self.current_diagram_title
            )

    def _render_diagram_image(
        self, image_path_str: Optional[str], diagram_title_str: Optional[str]
    ):
        # (Same as before, ensures canvas exists checks and fallback dimensions)
        if not (
            hasattr(self, "design_canvas")
            and self.design_canvas
            and self.design_canvas.winfo_exists()
        ):
            return
        if (
            not image_path_str
            or not diagram_title_str
            or not os.path.exists(image_path_str)
        ):
            self.design_canvas.delete("all")
            self.current_diagram_photo = None
            return
        try:
            self.design_canvas.update_idletasks()
            canvas_width = self.design_canvas.winfo_width()
            canvas_height = self.design_canvas.winfo_height()
            if canvas_width <= 1:
                canvas_width = (
                    self.design_canvas.cget("width")
                    if self.design_canvas.cget("width") > 1
                    else 500
                )
            if canvas_height <= 1:
                canvas_height = (
                    self.design_canvas.cget("height")
                    if self.design_canvas.cget("height") > 1
                    else 400
                )
            image = Image.open(image_path_str)
            img_width, img_height = image.size
            if img_width == 0 or img_height == 0:
                return
            img_aspect = img_width / img_height
            canvas_aspect = canvas_width / canvas_height
            new_width = (
                canvas_width
                if img_aspect > canvas_aspect
                else int(canvas_height * img_aspect)
            )
            new_height = (
                int(new_width / img_aspect)
                if img_aspect > canvas_aspect
                else canvas_height
            )  # Recalculate one based on the other
            new_width = max(1, int(new_width))
            new_height = max(1, int(new_height))  # Ensure int and positive
            resized_image = image.resize(
                (new_width, new_height), Image.Resampling.LANCZOS
            )
            self.current_diagram_photo = ImageTk.PhotoImage(resized_image)
            self.design_canvas.delete("all")
            x_pos = (canvas_width - new_width) // 2
            y_pos = (canvas_height - new_height) // 2
            self.design_canvas.create_image(
                x_pos, y_pos, anchor=tk.NW, image=self.current_diagram_photo
            )
        except Exception as e:
            logger.error(
                f"Failed to render diagram '{diagram_title_str}': {e}", exc_info=True
            )
            if hasattr(self, "design_canvas") and self.design_canvas.winfo_exists():
                self.design_canvas.delete("all")
            self.current_diagram_photo = None

    def _display_diagram_and_details(self, details_data: Dict[str, Any]):
        # (Same as before, uses "placeholder" tag for "Not specified...")
        image_path = details_data.get("image_path")
        title = details_data.get("title", "System Design")
        self.current_diagram_path = image_path
        self.current_diagram_title = title
        self._render_diagram_image(image_path, title)

        if not (hasattr(self, "details_text") and self.details_text.winfo_exists()):
            return
        self.details_text.config(state=tk.NORMAL)
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(tk.END, f"Title: {title}\n", "bold_header")
        img_display_path = (
            os.path.basename(str(image_path))
            if image_path and os.path.exists(str(image_path))
            else "Not available"
        )
        self.details_text.insert(tk.END, f"Image: {img_display_path}\n\n")
        sections = {
            "main_explanation": "--- System Overview ---",
            "diagram_content_components": "--- System Components (from Diagram Data) ---",
            "api_specifications": "--- Key API Specifications ---",
            "data_models": "--- Core Data Models ---",
            "tradeoff_analysis_summary": "--- Key Trade-offs ---",
            "failure_analysis_summary": "--- Key Failure Analysis ---",
            "quantitative_analysis": "--- Quantitative Analysis (Estimates) ---",
            "security_architecture": "--- Security Highlights ---",
        }
        for key, header in sections.items():
            content = details_data.get(key)
            self.details_text.insert(tk.END, f"{header}\n", "bold_header")
            if content:
                if isinstance(content, str):
                    self.details_text.insert(tk.END, f"{content}\n")
                elif isinstance(content, list) and content:
                    for i, item in enumerate(content):
                        if key == "api_specifications":
                            comp = item.get("c", "N/A")
                            p = item.get("p", "N/A")
                            m = item.get("m", "N/A")
                            d = item.get("d", "N/A")  # Shorter keys if prompt changes
                            comp = item.get("component", comp)
                            p = item.get("path", p)
                            m = item.get("method", m)
                            d = item.get("description", d)
                            self.details_text.insert(
                                tk.END, f"  {i+1}. [{comp}] {m} {p}\n     Desc: {d}\n"
                            )
                            if item.get("request_parameters"):
                                self.details_text.insert(
                                    tk.END,
                                    f"     Req: {json.dumps(item.get('request_parameters'), indent=2)}\n",
                                )
                            if item.get("response_schema"):
                                self.details_text.insert(
                                    tk.END,
                                    f"     Res: {json.dumps(item.get('response_schema'), indent=2)}\n",
                                )
                        elif key == "data_models":
                            comp = item.get("c", "N/A")
                            n = item.get("n", "N/A")
                            flds = item.get("f", {})
                            d = item.get("d", "N/A")
                            comp = item.get("component", comp)
                            n = item.get("name", n)
                            flds = item.get("fields", flds)
                            d = item.get("description", d)
                            self.details_text.insert(
                                tk.END,
                                f"  {i+1}. {n} (in {comp})\n     Desc: {d}\n     Fields: {json.dumps(flds, indent=2)}\n",
                            )
                        elif key == "diagram_content_components":
                            name = item.get("name", "N/A")
                            ctype = item.get("type", "N/A")
                            desc = item.get("description", "N/A")
                            tech = item.get("technologies", [])
                            self.details_text.insert(
                                tk.END,
                                f"  {i+1}. Name: {name}\n     Type: {ctype}\n     Diagram Desc: {desc}\n",
                            )
                            self.details_text.insert(
                                tk.END,
                                f"     Technologies: {', '.join(tech) if tech else '(Not specified)'}\n",
                            )
                        else:
                            self.details_text.insert(
                                tk.END,
                                f"  - {json.dumps(item, indent=2) if isinstance(item, (dict,list)) else item}\n",
                            )
                        self.details_text.insert(tk.END, "\n")
                elif isinstance(content, dict) and content:
                    for sub_key, value in content.items():
                        self.details_text.insert(
                            tk.END,
                            f"  {sub_key.replace('_', ' ').capitalize()}: {json.dumps(value, indent=2) if isinstance(value, (dict,list)) else value}\n",
                        )
                else:
                    self.details_text.insert(
                        tk.END,
                        "  (Not specified by AI for this query)\n",
                        "placeholder",
                    )
            else:
                self.details_text.insert(
                    tk.END, "  (Not specified by AI for this query)\n", "placeholder"
                )
            self.details_text.insert(tk.END, "\n")
        self.details_text.config(state=tk.DISABLED)
        self.details_text.yview_moveto(0.0)
        if self.root.winfo_exists():
            self.update_status(f"Displayed: {title}")

    def _toggle_agent(self):
        # (Same as previous version, with parent=self.root for messageboxes)
        if self.is_running:
            logger.info("GUI: Stop Interview clicked.")
            self.status_label_text_var.set("Stopping...")
            self.update_status("Agent stopping...")
            if hasattr(self, "start_button"):
                self.start_button.config(state=tk.DISABLED)
            if self.agent and hasattr(self.agent, "_execute_stop_async"):
                self.agent._execute_stop_async()
            else:
                logger.error("Agent/stop method missing.")
                self.update_queue.put(("set_stopped_ui_state", {}))
        else:
            logger.info("GUI: Start Interview clicked.")
            cand_dev = self.candidate_device_var.get()
            int_dev = self.interviewer_device_var.get()
            if not cand_dev or not int_dev:
                if self.root.winfo_exists():
                    messagebox.showerror(
                        "Device Error", "Please select devices.", parent=self.root
                    )
                    return
            if not self.agent or not hasattr(self.agent, "start_agent_processing"):
                if self.root.winfo_exists():
                    messagebox.showerror(
                        "Agent Error", "Agent not initialized.", parent=self.root
                    )
                    return
            try:
                cand_idx = int(cand_dev.split("idx: ")[-1].replace(")", ""))
                int_idx = int(int_dev.split("idx: ")[-1].replace(")", ""))
                self.status_label_text_var.set("Starting...")
                if hasattr(self, "start_button"):
                    self.start_button.config(state=tk.DISABLED)
                agent_started = self.agent.start_agent_processing(cand_idx, int_idx)
                if agent_started:
                    self.is_running = True
                    self.status_label_text_var.set("Running")
                    self._update_running_status_display(True)
                    self._add_to_chat("system", "Agent started.")
                else:
                    self.is_running = False
                    self.status_label_text_var.set("Start Failed")
                    self._update_running_status_display(False)
                if hasattr(self, "start_button"):
                    self.start_button.config(state=tk.NORMAL)
            except ValueError:
                if self.root.winfo_exists():
                    messagebox.showerror(
                        "Device Error", "Invalid device index.", parent=self.root
                    )
                self._update_running_status_display(False)
                self.status_label_text_var.set("Device Error")
                if hasattr(self, "start_button"):
                    self.start_button.config(state=tk.NORMAL)
            except Exception as e:
                logger.error(f"Error during agent start: {e}", exc_info=True)
                if self.root.winfo_exists():
                    messagebox.showerror(
                        "Start Error", f"Unexpected error: {e}", parent=self.root
                    )
                self._update_running_status_display(False)
                self.status_label_text_var.set("Error")
                if hasattr(self, "start_button"):
                    self.start_button.config(state=tk.NORMAL)
