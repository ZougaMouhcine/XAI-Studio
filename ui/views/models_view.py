"""
XAI Studio — Models View (v2 — Dashboard Design)
===================================================
Manage saved models with a polished card-based layout.
"""

import tkinter as tk
from tkinter import ttk

from ui.widgets import C, F, Card, ModernButton, SectionHeader, StyledTreeview, Badge, bind_mousewheel_to
from ui.components.dialogs import show_error, show_info, ask_confirm
from services.pipeline_service import PipelineService


class ModelsView(ttk.Frame):
    """Model management dashboard."""

    def __init__(self, parent):
        super().__init__(parent, style="TFrame")
        self._service = PipelineService()
        self._build()

    def _build(self):
        canvas = tk.Canvas(self, bg=C.BG_MAIN, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self._scroll = tk.Frame(canvas, bg=C.BG_MAIN)
        self._scroll.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        window_id = canvas.create_window((0, 0), window=self._scroll, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window_id, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        bind_mousewheel_to(canvas, self._scroll)

        ct = self._scroll
        px = 24

        # ── Header ───────────────────────────────────────────────
        header = tk.Frame(ct, bg=C.BG_MAIN)
        header.pack(fill="x", padx=px, pady=(24, 0))
        SectionHeader(header, icon="", title="Modèles",
                      subtitle="Sauvegardez, chargez et gérez vos modèles entraînés").pack(side="left")

        btn_row = tk.Frame(header, bg=C.BG_MAIN)
        btn_row.pack(side="right")
        ModernButton(btn_row, text="Prediction", icon="",
                 style="secondary", command=lambda: self._navigate_to("prediction"),
                 bg=C.BG_MAIN).pack(side="left", padx=(0, 8), pady=6)
        ModernButton(btn_row, text="Sauvegarder tout", icon="",
                     style="primary", command=self._on_save_all,
                     bg=C.BG_MAIN).pack(side="left", padx=(0, 8), pady=6)
        ModernButton(btn_row, text="Rafraîchir", icon="",
                     style="secondary", command=self._refresh,
                     bg=C.BG_MAIN).pack(side="left", pady=6)

        # ── In-memory models card ────────────────────────────────
        mem_card = Card(ct, accent_color=C.INFO, pad=16)
        mem_card.pack(fill="x", padx=px, pady=(16, 0))

        tk.Label(mem_card.inner, text="Modèles en mémoire", font=F.H3,
                 bg=C.BG_CARD, fg=C.TEXT).pack(anchor="w", pady=(0, 8))

        self._memory_frame = tk.Frame(mem_card.inner, bg=C.BG_CARD)
        self._memory_frame.pack(fill="x")

        # ── Saved models table ───────────────────────────────────
        tk.Label(ct, text="Modèles sur disque", font=F.H2,
                 bg=C.BG_MAIN, fg=C.TEXT).pack(anchor="w", padx=px, pady=(16, 16))

        self._table_container = tk.Frame(ct, bg=C.BG_MAIN)
        self._table_container.pack(fill="both", expand=True, padx=px, pady=(0, 16))

        cols = ("Fichier", "Classe", "Tâche", "Cible", "Taille (KB)", "Date")
        widths = {"Fichier": 220, "Classe": 180, "Tâche": 100,
                  "Cible": 120, "Taille (KB)": 90, "Date": 160}
        self._stv = StyledTreeview(self._table_container, columns=cols,
                                    col_widths=widths, height=8)
        self._stv.pack(fill="both", expand=True)

        # Action bar
        actions = tk.Frame(ct, bg=C.BG_MAIN)
        actions.pack(fill="x", padx=px, pady=(16, 24))

        ModernButton(actions, text="Supprimer le modèle sélectionné", icon="",
                     style="danger", command=self._on_delete,
                     bg=C.BG_MAIN).pack(side="right")

    # ──────────────────────────────────────────────────────────────
    def on_enter(self):
        self._refresh_memory()
        self._refresh()

    def _refresh_memory(self):
        for w in self._memory_frame.winfo_children():
            w.destroy()

        models = self._service.trained_models
        if not models:
            tk.Label(self._memory_frame,
                     text="Aucun modèle en mémoire — entraînez des modèles d'abord",
                     font=F.BODY, bg=C.BG_CARD, fg=C.TEXT_MUTED).pack(anchor="w", pady=4)
            return

        for name, entry in models.items():
            if entry.get("model") is None:
                continue

            row = tk.Frame(self._memory_frame, bg=C.BG_CARD)
            row.pack(fill="x", pady=3)

            tk.Label(row, text="●", font=F.ICON_S, bg=C.BG_CARD,
                     fg=C.SUCCESS).pack(side="left", padx=(0, 8))
            tk.Label(row, text=name, font=F.H4, bg=C.BG_CARD,
                     fg=C.TEXT).pack(side="left")
            tk.Label(row, text=type(entry["model"]).__name__, font=F.SMALL,
                     bg=C.BG_CARD, fg=C.TEXT_MUTED).pack(side="left", padx=(12, 0))

            ModernButton(row, text="Sauvegarder", style="secondary",
                         command=lambda n=name: self._on_save_single(n),
                         bg=C.BG_CARD, width=120).pack(side="right")

    def _refresh(self):
        self._stv.tree.delete(*self._stv.tree.get_children())
        models = self._service.get_saved_models()

        for m in models:
            self._stv.tree.insert("", "end", values=(
                m.get("filename", "?"),
                m.get("model_class", "?"),
                m.get("task_type", "?"),
                m.get("target_column", "?"),
                m.get("size_kb", "?"),
                m.get("saved_at", "?")[:19] if m.get("saved_at") else "?",
            ), tags=(m.get("filepath", ""),))

    def _on_save_single(self, model_name):
        try:
            path = self._service.save_trained_model(model_name)
            show_info("Succès", f"Modèle '{model_name}' sauvegardé :\n{path}")
            self._refresh()
        except Exception as exc:
            show_error("Erreur", str(exc))

    def _on_save_all(self):
        if not self._service.trained_models:
            show_error("Erreur", "Aucun modèle à sauvegarder.")
            return
        try:
            paths = self._service.save_all_trained_models()
            show_info("Succès", f"{len(paths)} modèle(s) sauvegardé(s).")
            self._refresh()
        except Exception as exc:
            show_error("Erreur", str(exc))

    def _on_delete(self):
        sel = self._stv.tree.selection()
        if not sel:
            show_error("Erreur", "Sélectionnez un modèle à supprimer.")
            return

        item = self._stv.tree.item(sel[0])
        filename = item["values"][0]
        filepath = item["tags"][0] if item["tags"] else None

        if not filepath:
            show_error("Erreur", "Chemin introuvable.")
            return
        if not ask_confirm("Confirmation", f"Supprimer '{filename}' ?"):
            return
        if self._service.delete_saved_model(filepath):
            show_info("Succès", f"'{filename}' supprimé.")
            self._refresh()
        else:
            show_error("Erreur", "Échec de la suppression.")

    def _navigate_to(self, view_name: str) -> None:
        root = self.winfo_toplevel()
        navigate = getattr(root, "navigate_to", None)
        if callable(navigate):
            navigate(view_name)
