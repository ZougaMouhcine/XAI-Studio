import tkinter as tk
from tkinter import ttk
from datetime import datetime

from ui.widgets import C, F, scaled, scaled_font

class HistoryDialog(tk.Toplevel):
    """Dialog to list, load, and delete previous Copilot conversations."""

    def __init__(self, parent, agent_service, on_load_callback):
        super().__init__(parent)
        self.title("Chat History")
        self.geometry(f"{scaled(450)}x{scaled(500)}")
        self.configure(bg=C.BG_MAIN)
        self.transient(parent)
        self.grab_set()

        self._agent = agent_service
        self._on_load = on_load_callback

        self._build()
        self._load_list()

        # Center on screen
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() // 2) - (self.winfo_width() // 2)
        y = parent.winfo_y() + (parent.winfo_height() // 2) - (self.winfo_height() // 2)
        self.geometry(f"+{x}+{y}")

    def _build(self):
        # Header
        header = tk.Frame(self, bg=C.BG_MAIN)
        header.pack(fill="x", padx=20, pady=20)

        tk.Label(
            header, text="Chat History", font=scaled_font(14, "bold"),
            bg=C.BG_MAIN, fg=C.TEXT,
        ).pack(side="left")

        # Scrollable list area
        list_container = tk.Frame(self, bg=C.BG_MAIN)
        list_container.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        self._canvas = tk.Canvas(
            list_container, bg=C.BG_CARD, highlightthickness=1, 
            highlightbackground=C.BORDER, bd=0
        )
        self._scrollbar = ttk.Scrollbar(
            list_container, orient="vertical", command=self._canvas.yview
        )
        self._list_frame = tk.Frame(self._canvas, bg=C.BG_CARD)

        self._list_frame.bind("<Configure>", lambda e: self._canvas.configure(
            scrollregion=self._canvas.bbox("all")
        ))
        self._canvas_window = self._canvas.create_window(
            (0, 0), window=self._list_frame, anchor="nw"
        )
        self._canvas.configure(yscrollcommand=self._scrollbar.set)
        
        self._canvas.bind("<Configure>", lambda e: self._canvas.itemconfig(
            self._canvas_window, width=e.width
        ))

        self._scrollbar.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        # Mouse wheel
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _load_list(self):
        # Clear existing items
        for widget in self._list_frame.winfo_children():
            widget.destroy()

        chats = self._agent.list_conversations()
        
        if not chats:
            tk.Label(
                self._list_frame, text="No conversation history found.",
                font=(F.FAM, 10), bg=C.BG_CARD, fg=C.TEXT_DIM, pady=20
            ).pack(fill="x")
            return

        for chat in chats:
            self._create_chat_item(chat)

    def _create_chat_item(self, chat: dict):
        item = tk.Frame(self._list_frame, bg=C.BG_CARD)
        item.pack(fill="x", padx=1, pady=1)

        # Add a bottom border line
        border = tk.Frame(self._list_frame, bg=C.BORDER, height=1)
        border.pack(fill="x")

        # Hover effect
        def on_enter(e, widget=item):
            widget.configure(bg=C.BG_SURFACE)
            for child in widget.winfo_children():
                if isinstance(child, tk.Frame) and child.cget("bg") == C.BG_CARD:
                    child.configure(bg=C.BG_SURFACE)
                elif isinstance(child, tk.Label) and child.cget("bg") == C.BG_CARD:
                    child.configure(bg=C.BG_SURFACE)

        def on_leave(e, widget=item):
            widget.configure(bg=C.BG_CARD)
            for child in widget.winfo_children():
                if isinstance(child, tk.Frame) and child.cget("bg") == C.BG_SURFACE:
                    child.configure(bg=C.BG_CARD)
                elif isinstance(child, tk.Label) and child.cget("bg") == C.BG_SURFACE:
                    child.configure(bg=C.BG_CARD)

        item.bind("<Enter>", on_enter)
        item.bind("<Leave>", on_leave)

        # Info side
        info_frame = tk.Frame(item, bg=C.BG_CARD)
        info_frame.pack(side="left", fill="both", expand=True, padx=12, pady=12)

        # Format date
        try:
            dt = datetime.fromisoformat(chat.get("updated_at", ""))
            date_str = dt.strftime("%Y-%m-%d %H:%M")
        except:
            date_str = ""

        title_lbl = tk.Label(
            info_frame, text=chat.get("title", "Unknown"), 
            font=scaled_font(10, "bold"), bg=C.BG_CARD, fg=C.TEXT,
            anchor="w"
        )
        title_lbl.pack(fill="x")
        
        date_lbl = tk.Label(
            info_frame, text=date_str, 
            font=(F.FAM, 8), bg=C.BG_CARD, fg=C.TEXT_DIM,
            anchor="w"
        )
        date_lbl.pack(fill="x", pady=(2, 0))

        # Actions side
        actions_frame = tk.Frame(item, bg=C.BG_CARD)
        actions_frame.pack(side="right", padx=12, pady=12)

        load_btn = tk.Label(
            actions_frame, text="Load", font=scaled_font(9, "bold"),
            bg=C.ACCENT, fg="#ffffff", padx=10, pady=4, cursor="hand2"
        )
        load_btn.pack(side="left", padx=(0, 6))
        load_btn.bind("<Button-1>", lambda e, sid=chat["session_id"]: self._handle_load(sid))

        del_btn = tk.Label(
            actions_frame, text="🗑", font=scaled_font(12),
            bg=C.BG_CARD, fg=C.DANGER, cursor="hand2"
        )
        del_btn.pack(side="left")
        del_btn.bind("<Button-1>", lambda e, sid=chat["session_id"]: self._handle_delete(sid))

        # Make sure hover events bubble up
        for widget in [info_frame, title_lbl, date_lbl, actions_frame]:
            widget.bind("<Enter>", on_enter, add="+")
            widget.bind("<Leave>", on_leave, add="+")

    def _handle_load(self, session_id: str):
        self._on_load(session_id)
        self.destroy()

    def _handle_delete(self, session_id: str):
        if self._agent.delete_conversation(session_id):
            self._load_list()

    def destroy(self):
        self._canvas.unbind_all("<MouseWheel>")
        super().destroy()
