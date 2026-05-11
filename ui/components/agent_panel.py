"""
XAI Studio — AI Agent Panel (VS Code Copilot-style)
=====================================================
Togglable right-side chat panel with Ask / Agent / Plan modes.
"""

import tkinter as tk
from tkinter import ttk
import time

from ui.widgets import C, F, scaled, scaled_font
from services.i18n import _


class AgentPanel(tk.Frame):
    """Copilot-style AI assistant panel that sits on the right side of the app."""

    WIDTH = 400

    MODES = [
        ("ask", "Ask"),
        ("agent", "Agent"),
        ("plan", "Plan"),
    ]

    def __init__(self, parent, agent_service, on_close=None):
        super().__init__(parent, bg=C.BG_CARD, width=self.WIDTH)
        self.pack_propagate(False)
        self._agent = agent_service
        self._on_close = on_close
        self._messages: list[dict] = []
        self._build()

    def _build(self):
        # ── Separator on left edge ───────────────────────────────────
        sep = tk.Frame(self, bg=C.BORDER, width=1)
        sep.pack(side="left", fill="y")

        # ── Main content ─────────────────────────────────────────────
        main = tk.Frame(self, bg=C.BG_CARD)
        main.pack(side="left", fill="both", expand=True)

        # ── Header ───────────────────────────────────────────────────
        header = tk.Frame(main, bg=C.BG_CARD)
        header.pack(fill="x", padx=16, pady=(12, 0))

        tk.Label(
            header, text=_("agent_header"), font=scaled_font(13, "bold"),
            bg=C.BG_CARD, fg=C.TEXT,
        ).pack(side="left")

        # Provider selector container
        self._provider_container = tk.Frame(header, bg=C.BG_CARD)
        self._provider_container.pack(side="left", padx=(8, 0))
        self._provider_badge = None
        self._provider_combo = None
        self._build_provider_selector()

        # Settings button (rightmost)
        settings_btn = tk.Label(
            header, text="⚙", font=scaled_font(14),
            bg=C.BG_CARD, fg=C.TEXT_MUTED, cursor="hand2",
        )
        settings_btn.pack(side="right", padx=(4, 0))
        settings_btn.bind("<Button-1>", lambda e: self._open_settings())
        settings_btn.bind("<Enter>", lambda e: settings_btn.configure(fg=C.ACCENT))
        settings_btn.bind("<Leave>", lambda e: settings_btn.configure(fg=C.TEXT_MUTED))

        # History button
        history_btn = tk.Label(
            header, text="🕒", font=scaled_font(13),
            bg=C.BG_CARD, fg=C.TEXT_MUTED, cursor="hand2",
        )
        history_btn.pack(side="right", padx=(4, 0))
        history_btn.bind("<Button-1>", lambda e: self._open_history())
        history_btn.bind("<Enter>", lambda e: history_btn.configure(fg=C.ACCENT))
        history_btn.bind("<Leave>", lambda e: history_btn.configure(fg=C.TEXT_MUTED))

        # New chat button
        new_btn = tk.Label(
            header, text="＋", font=scaled_font(14, "bold"),
            bg=C.BG_CARD, fg=C.TEXT_MUTED, cursor="hand2",
        )
        new_btn.pack(side="right", padx=(4, 0))
        new_btn.bind("<Button-1>", lambda e: self._new_chat())
        new_btn.bind("<Enter>", lambda e: new_btn.configure(fg=C.ACCENT))
        new_btn.bind("<Leave>", lambda e: new_btn.configure(fg=C.TEXT_MUTED))

        # Header separator
        tk.Frame(main, bg=C.BORDER, height=1).pack(fill="x", padx=0, pady=(10, 0))

        # ── Chat Messages Area ───────────────────────────────────────
        chat_container = tk.Frame(main, bg=C.BG_CARD)
        chat_container.pack(fill="both", expand=True, padx=0, pady=0)

        self._chat_canvas = tk.Canvas(
            chat_container, bg=C.BG_CARD, highlightthickness=0, bd=0,
        )
        self._chat_scrollbar = ttk.Scrollbar(
            chat_container, orient="vertical", command=self._chat_canvas.yview,
        )
        self._chat_frame = tk.Frame(self._chat_canvas, bg=C.BG_CARD)

        self._chat_frame.bind("<Configure>", self._on_chat_configure)
        self._chat_canvas_window = self._chat_canvas.create_window(
            (0, 0), window=self._chat_frame, anchor="nw",
        )
        self._chat_canvas.configure(yscrollcommand=self._chat_scrollbar.set)

        self._chat_scrollbar.pack(side="right", fill="y")
        self._chat_canvas.pack(side="left", fill="both", expand=True)

        self._chat_canvas.bind("<Configure>", self._on_canvas_resize)

        # Mouse wheel scrolling
        self._chat_canvas.bind("<Enter>", self._bind_mousewheel)
        self._chat_canvas.bind("<Leave>", self._unbind_mousewheel)

        # ── Welcome message ──────────────────────────────────────────
        self._show_welcome()

        # ── Typing indicator ─────────────────────────────────────────
        self._typing_frame = tk.Frame(main, bg=C.BG_CARD)
        self._typing_label = tk.Label(
            self._typing_frame, text="● ● ●", font=(F.FAM, 10),
            bg=C.BG_CARD, fg=C.ACCENT,
        )
        self._typing_label.pack(side="left", padx=16, pady=4)
        self._typing_anim_id = None

        # ── Input Area ───────────────────────────────────────────────
        input_container = tk.Frame(main, bg=C.BG_CARD)
        input_container.pack(fill="x", padx=12, pady=(4, 12))

        # Input border frame
        input_border = tk.Frame(
            input_container, bg=C.INPUT_BORDER,
            highlightbackground=C.INPUT_BORDER, highlightthickness=1,
        )
        input_border.pack(fill="x")

        input_inner = tk.Frame(input_border, bg=C.BG_INPUT)
        input_inner.pack(fill="x", padx=1, pady=1)

        self._input_text = tk.Text(
            input_inner, height=3, bg=C.BG_INPUT, fg=C.TEXT,
            font=(F.FAM, 10), relief="flat", wrap="word",
            insertbackground=C.TEXT, padx=10, pady=8,
            selectbackground=C.TREE_SELECT,
        )
        self._input_text.pack(fill="x", expand=True)

        # Placeholder text
        self._placeholder = _("agent_placeholder")
        self._show_placeholder()
        self._input_text.bind("<FocusIn>", self._on_focus_in)
        self._input_text.bind("<FocusOut>", self._on_focus_out)
        self._input_text.bind("<Return>", self._on_enter)
        self._input_text.bind("<Shift-Return>", lambda e: None)

        # ── Bottom bar: Mode dropdown + Send button ──────────────────
        btn_row = tk.Frame(input_container, bg=C.BG_CARD)
        btn_row.pack(fill="x", pady=(6, 0))

        # Mode dropdown (left side of send button)
        self._mode_var = tk.StringVar(value="Ask")
        self._mode_dropdown = ttk.Combobox(
            btn_row,
            textvariable=self._mode_var,
            values=[label for _, label in self.MODES],
            state="readonly",
            font=(F.FAM, 9),
            width=12,
        )
        self._mode_dropdown.pack(side="left")
        self._mode_dropdown.bind("<<ComboboxSelected>>", self._on_mode_dropdown_change)

        # Send button (right side)
        self._send_btn = tk.Label(
            btn_row, text=_("agent_send"), font=scaled_font(9, "bold"),
            bg=C.ACCENT, fg="#ffffff", padx=14, pady=5, cursor="hand2",
        )
        self._send_btn.pack(side="right")
        self._send_btn.bind("<Button-1>", lambda e: self._send_message())
        self._send_btn.bind("<Enter>", lambda e: self._send_btn.configure(bg=C.ACCENT_LIGHT))
        self._send_btn.bind("<Leave>", lambda e: self._send_btn.configure(bg=C.ACCENT))

        # Set initial mode
        self._set_mode("ask")

    # ── Mode switching ───────────────────────────────────────────────

    def _on_mode_dropdown_change(self, event):
        """Handle mode change from the dropdown."""
        selected_label = self._mode_var.get()
        for mode_id, label in self.MODES:
            if label == selected_label:
                self._set_mode(mode_id)
                break

    def _set_mode(self, mode: str):
        self._agent.mode = mode
        # Sync dropdown display
        for mode_id, label in self.MODES:
            if mode_id == mode:
                self._mode_var.set(label)
                break

    # ── Chat Messages ────────────────────────────────────────────────

    def _show_welcome(self):
        self._add_message("assistant", _("agent_welcome"))

    def _add_message(self, role: str, content: str, msg_type: str = "text"):
        """Add a message bubble to the chat area."""
        self._messages.append({"role": role, "content": content, "type": msg_type})

        # Container for the message
        msg_frame = tk.Frame(self._chat_frame, bg=C.BG_CARD)
        msg_frame.pack(fill="x", padx=12, pady=4)

        if role == "user":
            self._render_user_message(msg_frame, content)
        elif role == "tool":
            self._render_tool_message(msg_frame, content)
        else:
            self._render_assistant_message(msg_frame, content)

        # Scroll to bottom
        self._chat_frame.update_idletasks()
        self._chat_canvas.configure(scrollregion=self._chat_canvas.bbox("all"))
        self._chat_canvas.yview_moveto(1.0)

    def _render_user_message(self, parent: tk.Frame, content: str):
        """Render a user message (right-aligned, accent bg, framed)."""
        row = tk.Frame(parent, bg=C.BG_CARD)
        row.pack(fill="x")

        # Right-aligned spacer
        tk.Frame(row, bg=C.BG_CARD).pack(side="left", fill="x", expand=True)

        # Outer frame border — matches BG_CARD in light mode (invisible)
        border_color = C.BG_CARD
        border_frame = tk.Frame(row, bg=border_color, padx=1, pady=1)
        border_frame.pack(side="right")

        bubble = tk.Frame(border_frame, bg=C.ACCENT)
        bubble.pack(fill="both", expand=True)

        inner = tk.Frame(bubble, bg=C.ACCENT)
        inner.pack(padx=10, pady=6)

        lbl = tk.Label(
            inner, text=content, font=(F.FAM, 10), bg=C.ACCENT, fg="#ffffff",
            wraplength=280, justify="left", anchor="w",
        )
        lbl.pack(side="top", anchor="w")

        # Actions frame (Edit / Copy)
        actions_frame = tk.Frame(inner, bg=C.ACCENT)
        actions_frame.pack(side="right", anchor="e", pady=(4, 0))

        # Copy button
        copy_lbl = tk.Label(
            actions_frame, text="Copy", font=(F.FAM, 7, "bold"), bg=C.ACCENT, fg="#e0e0e0",
            cursor="hand2", pady=0,
        )
        copy_lbl.pack(side="right", padx=(6, 0))
        copy_lbl.bind("<Button-1>", lambda e: self._copy_prompt(content))
        copy_lbl.bind("<Enter>", lambda e: copy_lbl.configure(fg="#ffffff"))
        copy_lbl.bind("<Leave>", lambda e: copy_lbl.configure(fg="#e0e0e0"))

        # Edit button
        edit_lbl = tk.Label(
            actions_frame, text=_("agent_edit"), font=(F.FAM, 7, "bold"), bg=C.ACCENT, fg="#e0e0e0",
            cursor="hand2", pady=0,
        )
        edit_lbl.pack(side="right")
        edit_lbl.bind("<Button-1>", lambda e: self._edit_prompt(content))
        edit_lbl.bind("<Enter>", lambda e: edit_lbl.configure(fg="#ffffff"))
        edit_lbl.bind("<Leave>", lambda e: edit_lbl.configure(fg="#e0e0e0"))

    def _render_assistant_message(self, parent: tk.Frame, content: str):
        """Render an assistant message (left-aligned, subtle bg, framed)."""
        row = tk.Frame(parent, bg=C.BG_CARD)
        row.pack(fill="x")

        bubble_bg = C.BG_SURFACE

        # Outer frame border — matches BG_CARD in light mode (invisible)
        border_color = C.BG_CARD
        border_frame = tk.Frame(row, bg=border_color, padx=1, pady=1)
        border_frame.pack(side="left", fill="x", expand=True)

        bubble = tk.Frame(border_frame, bg=bubble_bg)
        bubble.pack(fill="both", expand=True)

        inner = tk.Frame(bubble, bg=bubble_bg)
        inner.pack(fill="x", padx=10, pady=6)

        # Render with markdown formatting
        self._render_formatted_text(inner, content, bubble_bg)

    def _render_tool_message(self, parent: tk.Frame, content: str):
        """Render a tool-call indicator."""
        row = tk.Frame(parent, bg=C.BG_CARD)
        row.pack(fill="x")

        badge = tk.Frame(row, bg=C.SUCCESS_DIM)
        badge.pack(side="left")

        tk.Label(
            badge, text=content, font=(F.FAM, 8, "bold"),
            bg=C.SUCCESS_DIM, fg=C.ACCENT, padx=8, pady=3,
        ).pack()

    def _render_formatted_text(self, parent: tk.Frame, text: str, bg: str):
        """Render text with proper markdown support (bold, bullets, code)."""
        txt = tk.Text(
            parent, bg=bg, fg=C.TEXT, font=(F.FAM, 10),
            relief="flat", wrap="word", padx=0, pady=0,
            highlightthickness=0, bd=0, cursor="arrow",
        )

        # Configure tags for formatting
        txt.tag_configure("bold", font=(F.FAM, 10, "bold"))
        txt.tag_configure("code", font=(F.MONO, 9), background=C.BG_CARD_ALT, foreground=C.ACCENT)
        txt.tag_configure("bullet", lmargin1=8, lmargin2=20)

        lines = text.split("\n")
        for i, line in enumerate(lines):
            if i > 0:
                txt.insert("end", "\n")

            stripped = line.strip()
            if stripped.startswith("• ") or stripped.startswith("- "):
                # Parse markdown inside bullet lines too
                prefix = line[:len(line) - len(line.lstrip())]
                bullet_char = stripped[:2]
                remainder = stripped[2:]
                txt.insert("end", prefix + bullet_char, "bullet")
                self._insert_formatted_line(txt, remainder, tag_prefix="bullet")
            else:
                self._insert_formatted_line(txt, line)

        # Auto-size height based on content
        txt.configure(state="disabled")
        txt.update_idletasks()
        line_count = int(txt.index("end-1c").split(".")[0])
        txt.configure(height=line_count, width=38)
        txt.pack(fill="x")

    def _insert_formatted_line(self, txt: tk.Text, line: str, tag_prefix: str = ""):
        """Insert a line with **bold** and `code` inline formatting."""
        i = 0
        n = len(line)
        buf = []  # Buffer for plain characters

        def flush_buf():
            if buf:
                plain = "".join(buf)
                if tag_prefix:
                    txt.insert("end", plain, tag_prefix)
                else:
                    txt.insert("end", plain)
                buf.clear()

        while i < n:
            # ── Check for **bold** ───────────────────────────────
            if i + 1 < n and line[i] == "*" and line[i + 1] == "*":
                close = line.find("**", i + 2)
                if close != -1:
                    flush_buf()
                    bold_text = line[i + 2:close]
                    if tag_prefix:
                        # Combine tags: we need both bullet + bold
                        combined_tag = f"{tag_prefix}_bold"
                        try:
                            txt.tag_configure(combined_tag,
                                              font=(F.FAM, 10, "bold"),
                                              lmargin1=8, lmargin2=20)
                        except Exception:
                            pass
                        txt.insert("end", bold_text, combined_tag)
                    else:
                        txt.insert("end", bold_text, "bold")
                    i = close + 2
                    continue

            # ── Check for `code` ─────────────────────────────────
            if line[i] == "`":
                close = line.find("`", i + 1)
                if close != -1:
                    flush_buf()
                    txt.insert("end", line[i + 1:close], "code")
                    i = close + 1
                    continue

            buf.append(line[i])
            i += 1

        flush_buf()

    # ── Typing indicator ─────────────────────────────────────────────

    def _show_typing(self):
        self._typing_frame.pack(fill="x", before=self._input_text.master.master)
        self._animate_typing()

    def _hide_typing(self):
        if self._typing_anim_id:
            self.after_cancel(self._typing_anim_id)
            self._typing_anim_id = None
        self._typing_frame.pack_forget()

    def _animate_typing(self):
        labels = ["o . .", ". o .", ". . o", ". o ."]
        current = getattr(self, "_typing_step", 0) % len(labels)
        self._typing_label.configure(text=labels[current])
        self._typing_step = current + 1
        self._typing_anim_id = self.after(400, self._animate_typing)

    # ── Input Handling ───────────────────────────────────────────────

    def _show_placeholder(self):
        self._input_text.insert("1.0", self._placeholder)
        self._input_text.configure(fg=C.TEXT_DIM)
        self._has_placeholder = True

    def _on_focus_in(self, e):
        if self._has_placeholder:
            self._input_text.delete("1.0", "end")
            self._input_text.configure(fg=C.TEXT)
            self._has_placeholder = False

    def _on_focus_out(self, e):
        text = self._input_text.get("1.0", "end").strip()
        if not text:
            self._show_placeholder()

    def _on_enter(self, event):
        if event.state & 0x1:  # Shift is held
            return None
        self._send_message()
        return "break"

    def _send_message(self):
        if self._has_placeholder:
            return

        text = self._input_text.get("1.0", "end").strip()
        if not text:
            return

        if self._agent.is_busy:
            return

        # Clear input
        self._input_text.delete("1.0", "end")
        self._has_placeholder = False

        # Add user message to chat
        self._add_message("user", text)

        # Show typing indicator
        self._show_typing()

        # Disable send button
        self._send_btn.configure(bg=C.TEXT_DIM, cursor="arrow")

        # Send to agent (async)
        self._agent.send_message(
            text,
            on_token=lambda t: self.after(0, self._on_agent_token, t),
            on_tool_call=lambda name, result: self.after(0, self._on_tool_call, name, result),
            on_done=lambda r: self.after(0, self._on_agent_done, r),
            on_error=lambda e: self.after(0, self._on_agent_error, e),
        )

    def _on_agent_token(self, text: str):
        pass  # Full response displayed in on_done

    def _on_tool_call(self, name: str, result: str):
        self._add_message("tool", f"[Tool] {name}")

    def _on_agent_done(self, response: str):
        self._hide_typing()
        self._send_btn.configure(bg=C.ACCENT, cursor="hand2")
        if response:
            self._add_message("assistant", response)

    def _on_agent_error(self, error: str):
        self._hide_typing()
        self._send_btn.configure(bg=C.ACCENT, cursor="hand2")
        self._add_message("assistant", f"Error: {error}")

    # ── Actions ──────────────────────────────────────────────────────

    def _edit_prompt(self, content: str):
        """Put the prompt back into the input box."""
        if self._has_placeholder:
            self._input_text.delete("1.0", "end")
            self._input_text.configure(fg=C.TEXT)
            self._has_placeholder = False
        self._input_text.delete("1.0", "end")
        self._input_text.insert("1.0", content)
        self._input_text.focus_set()

    def _copy_prompt(self, content: str):
        """Copy the prompt to clipboard."""
        self.clipboard_clear()
        self.clipboard_append(content)

    def _new_chat(self):
        self._agent.clear_conversation()
        self._messages.clear()
        for widget in self._chat_frame.winfo_children():
            widget.destroy()
        self._show_welcome()

    def _open_settings(self):
        from ui.components.agent_settings import AgentSettingsDialog
        AgentSettingsDialog(self.winfo_toplevel(), self._agent, self._on_settings_saved)

    def _open_history(self):
        from ui.components.agent_history import HistoryDialog
        HistoryDialog(self.winfo_toplevel(), self._agent, self._load_chat)

    def _load_chat(self, session_id: str):
        """Load a past chat into the UI."""
        messages = self._agent.load_conversation(session_id)
        
        self._messages.clear()
        for widget in self._chat_frame.winfo_children():
            widget.destroy()
            
        if not messages:
            self._show_welcome()
            return
            
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            
            if role == "system":
                continue
                
            if role == "tool":
                self._add_message("tool", f"[Tool] {msg.get('name', 'unknown')}")
                continue
                
            if role == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    self._add_message("tool", f"[Tool] {tc.get('function', {}).get('name', 'unknown')}")
                
            if content:
                self._add_message(role, content)

    def _on_settings_saved(self):
        """Update provider display after settings are saved."""
        self._build_provider_selector()

    def _build_provider_selector(self):
        """Dynamically build the provider badge or dropdown based on config."""
        for widget in self._provider_container.winfo_children():
            widget.destroy()

        self._provider_badge = None
        self._provider_combo = None

        if self._agent.has_multiple_providers:
            # Dropdown selector when both providers are available
            self._provider_var = tk.StringVar(value=self._agent.provider_name)
            self._provider_combo = ttk.Combobox(
                self._provider_container, textvariable=self._provider_var,
                values=self._agent.available_providers,
                state="readonly", font=(F.FAM, 8), width=8,
            )
            self._provider_combo.pack(side="left")
            self._provider_combo.bind("<<ComboboxSelected>>", self._on_provider_change)
        else:
            # Static badge for single provider
            provider = self._agent.provider_name if self._agent.is_configured else "—"
            self._provider_badge = tk.Label(
                self._provider_container, text=provider, font=(F.FAM, 8, "bold"),
                bg=C.ACCENT, fg="#ffffff", padx=6, pady=1,
            )
            self._provider_badge.pack(side="left")

    def _on_provider_change(self, event=None):
        """Handle provider switch from the dropdown."""
        selected = self._provider_var.get()
        if self._agent.set_provider(selected):
            self._add_message("tool", f"Switched to {selected}")

    # ── Canvas helpers ───────────────────────────────────────────────

    def _on_chat_configure(self, event):
        self._chat_canvas.configure(scrollregion=self._chat_canvas.bbox("all"))

    def _on_canvas_resize(self, event):
        self._chat_canvas.itemconfig(self._chat_canvas_window, width=event.width)

    def _bind_mousewheel(self, event):
        self._chat_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, event):
        self._chat_canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        self._chat_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
