"""
Neural Network Builder — MVP
Provides a simple list-based visual designer for feedforward networks.
Exports Keras Sequential code.
"""

import tkinter as tk
from tkinter import ttk
from ui.widgets import C, F, Card, ModernButton, SectionHeader
from ui.components.dialogs import ask_export_python_file, show_info


class NNBuilderView(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent, style="TFrame")
        self._layers: list[dict] = []
        self._build()

    def _build(self):
        px = 20
        header = tk.Frame(self, bg=C.BG_MAIN)
        header.pack(fill="x", padx=px, pady=(24, 0))
        SectionHeader(header, title="Neural Network Builder", subtitle="MVP: construire et exporter un réseau Keras").pack(side="left")

        layout = tk.Frame(self, bg=C.BG_MAIN)
        layout.pack(fill="both", expand=True, padx=px, pady=(12, 24))
        layout.columnconfigure(0, weight=1)
        layout.columnconfigure(1, weight=1)

        left = tk.Frame(layout, bg=C.BG_MAIN)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right = tk.Frame(layout, bg=C.BG_MAIN)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        # Left: layer list and controls
        card = Card(left, accent_color=C.ACCENT, pad=12)
        card.pack(fill="both", expand=True)

        row = tk.Frame(card.inner, bg=C.BG_CARD)
        row.pack(fill="x")
        ModernButton(row, text="Ajouter Dense", style="primary", command=lambda: self._add_layer("Dense"), bg=C.BG_CARD).pack(side="left", padx=(0, 8))
        ModernButton(row, text="Ajouter Dropout", style="secondary", command=lambda: self._add_layer("Dropout"), bg=C.BG_CARD).pack(side="left", padx=(0, 8))
        ModernButton(row, text="Ajouter Flatten", style="ghost", command=lambda: self._add_layer("Flatten"), bg=C.BG_CARD).pack(side="left", padx=(0, 8))

        self._layers_frame = tk.Frame(card.inner, bg=C.BG_CARD)
        self._layers_frame.pack(fill="both", expand=True, pady=(12, 0))

        # Right: visual + export
        card2 = Card(right, accent_color=C.INFO, pad=12)
        card2.pack(fill="both", expand=True)

        tk.Label(card2.inner, text="Aperçu du réseau", font=F.H3, bg=C.BG_CARD, fg=C.TEXT).pack(anchor="w")
        self._canvas = tk.Canvas(card2.inner, height=400, bg=C.BG_CARD, bd=0, highlightthickness=0)
        self._canvas.pack(fill="both", expand=True, pady=(8, 8))

        btn_row = tk.Frame(card2.inner, bg=C.BG_CARD)
        btn_row.pack(fill="x")
        ModernButton(btn_row, text="Exporter Keras", style="primary", command=self._export_code, bg=C.BG_CARD).pack(side="right")

        self._render_layers()

    def _add_layer(self, layer_type: str):
        if layer_type == "Dense":
            layer = {"type": "Dense", "units": 64, "activation": "relu"}
        elif layer_type == "Dropout":
            layer = {"type": "Dropout", "rate": 0.5}
        else:
            layer = {"type": layer_type}
        self._layers.append(layer)
        self._render_layers()

    def _remove_layer(self, idx: int):
        if 0 <= idx < len(self._layers):
            self._layers.pop(idx)
            self._render_layers()

    def _move_layer(self, idx: int, direction: str):
        if direction == "up" and idx > 0:
            self._layers[idx - 1], self._layers[idx] = self._layers[idx], self._layers[idx - 1]
        if direction == "down" and idx < len(self._layers) - 1:
            self._layers[idx + 1], self._layers[idx] = self._layers[idx], self._layers[idx + 1]
        self._render_layers()

    def _render_layers(self):
        for w in self._layers_frame.winfo_children():
            w.destroy()

        for i, layer in enumerate(self._layers):
            row = tk.Frame(self._layers_frame, bg=C.BG_CARD)
            row.pack(fill="x", pady=(4, 4))

            left = tk.Frame(row, bg=C.BG_CARD)
            left.pack(side="left", fill="x", expand=True)

            tk.Label(left, text=f"{i+1}. {layer['type']}", bg=C.BG_CARD, fg=C.TEXT, font=F.BODY).pack(anchor="w")

            props = tk.Frame(row, bg=C.BG_CARD)
            props.pack(side="right")

            ModernButton(props, text="▲", style="ghost", command=lambda i=i: self._move_layer(i, "up"), bg=C.BG_CARD).pack(side="left", padx=4)
            ModernButton(props, text="▼", style="ghost", command=lambda i=i: self._move_layer(i, "down"), bg=C.BG_CARD).pack(side="left", padx=4)
            ModernButton(props, text="Suppr", style="danger", command=lambda i=i: self._remove_layer(i), bg=C.BG_CARD).pack(side="left", padx=4)

            # Parameters inline
            if layer["type"] == "Dense":
                pframe = tk.Frame(self._layers_frame, bg=C.BG_CARD)
                pframe.pack(fill="x")
                tk.Label(pframe, text="Neurones:", bg=C.BG_CARD, fg=C.TEXT_MUTED, font=F.TINY).pack(side="left")
                e = ttk.Entry(pframe, width=6)
                e.insert(0, str(layer.get("units", 64)))
                e.pack(side="left", padx=(6, 12))
                e.bind("<FocusOut>", lambda ev, i=i, ent=e: self._update_layer_param(i, "units", ent.get()))

                tk.Label(pframe, text="Activation:", bg=C.BG_CARD, fg=C.TEXT_MUTED, font=F.TINY).pack(side="left")
                act = ttk.Combobox(pframe, values=["relu", "sigmoid", "tanh", "softmax", "linear"], width=10)
                act.set(layer.get("activation", "relu"))
                act.pack(side="left", padx=(6, 12))
                act.bind("<<ComboboxSelected>>", lambda ev, i=i, cb=act: self._update_layer_param(i, "activation", cb.get()))

            if layer["type"] == "Dropout":
                pframe = tk.Frame(self._layers_frame, bg=C.BG_CARD)
                pframe.pack(fill="x")
                tk.Label(pframe, text="Rate:", bg=C.BG_CARD, fg=C.TEXT_MUTED, font=F.TINY).pack(side="left")
                e = ttk.Entry(pframe, width=6)
                e.insert(0, str(layer.get("rate", 0.5)))
                e.pack(side="left", padx=(6, 12))
                e.bind("<FocusOut>", lambda ev, i=i, ent=e: self._update_layer_param(i, "rate", ent.get()))

        self._draw_canvas()

    def _update_layer_param(self, idx: int, key: str, value: str):
        try:
            if key in ("units",):
                self._layers[idx][key] = int(value)
            elif key in ("rate",):
                self._layers[idx][key] = float(value)
            else:
                self._layers[idx][key] = value
        except Exception:
            pass
        self._draw_canvas()

    def _draw_canvas(self):
        self._canvas.delete("all")
        w = self._canvas.winfo_width() or 400
        h = self._canvas.winfo_height() or 300
        x = 20
        gap = max(40, (w - 40) // max(1, len(self._layers)))
        for i, layer in enumerate(self._layers):
            cx = x + i * gap
            self._canvas.create_rectangle(cx, h//2 - 40, cx + 80, h//2 + 40, fill=C.BG_CARD_ALT, outline=C.BORDER)
            self._canvas.create_text(cx + 40, h//2, text=layer["type"], fill=C.TEXT)

    def _export_code(self):
        path = ask_export_python_file()
        if not path:
            return
        code = self._generate_keras_code()
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(code)
            show_info("Exporté", f"Code Keras exporté vers: {path}")
        except Exception as exc:
            show_info("Erreur", str(exc))

    def _generate_keras_code(self) -> str:
        lines = [
            "from tensorflow import keras",
            "from tensorflow.keras import layers",
            "",
            "def build_model(input_shape):",
            "    model = keras.Sequential()",
        ]
        for layer in self._layers:
            if layer["type"] == "Dense":
                lines.append(f"    model.add(layers.Dense({layer.get('units',64)}, activation='{layer.get('activation','relu')}'))")
            elif layer["type"] == "Dropout":
                lines.append(f"    model.add(layers.Dropout({layer.get('rate',0.5)}))")
            elif layer["type"] == "Flatten":
                lines.append("    model.add(layers.Flatten())")
            else:
                lines.append(f"    # Unsupported layer: {layer['type']}")
        lines.append("    return model")
        return "\n".join(lines)