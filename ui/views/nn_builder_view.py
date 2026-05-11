"""
Neural Network Builder — Training & Deep Learning Studio
Advanced visual builder with configurable layers and training options.
"""

import json
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np

from ui.widgets import C, F, Card, ModernButton, SectionHeader, LogPanel, Tooltip
from ui.components.dialogs import ask_export_python_file, show_error, show_info
from ui.components.plot_canvas import PlotCanvas, create_styled_figure
from services.pipeline_service import PipelineService
from services.i18n import _


ACTIVATIONS = {
    "ReLU": "relu",
    "Sigmoid": "sigmoid",
    "Tanh": "tanh",
    "Softmax": "softmax",
    "LeakyReLU": "leaky_relu",
    "ELU": "elu",
    "GELU": "gelu",
}

OPTIMIZERS = {
    "Adam": "adam",
    "SGD": "sgd",
    "RMSprop": "rmsprop",
    "Adagrad": "adagrad",
    "AdamW": "adamw",
}

LOSSES_CLASSIFICATION = ["binary_crossentropy", "categorical_crossentropy", "focal_loss"]
LOSSES_REGRESSION = ["mse", "mae", "huber"]

LAYER_SCHEMAS = {
    "Dense": {
        "units": {"type": "int", "default": 64, "min": 1, "max": 2048, "desc": "nn_layer_units_desc"},
        "activation": {"type": "choice", "choices": list(ACTIVATIONS.keys()), "default": "ReLU", "desc": "nn_layer_act_desc"},
    },
    "Dropout": {
        "rate": {"type": "float", "default": 0.5, "min": 0.0, "max": 0.9, "desc": "nn_layer_drop_desc"},
    },
    "BatchNormalization": {},
    "Conv2D": {
        "filters": {"type": "int", "default": 32, "min": 1, "max": 256, "desc": "nn_layer_filters_desc"},
        "kernel_size": {"type": "string", "default": "3,3", "desc": "nn_layer_kernel_desc"},
        "activation": {"type": "choice", "choices": list(ACTIVATIONS.keys()), "default": "ReLU", "desc": "nn_layer_act"},
        "padding": {"type": "choice", "choices": ["valid", "same"], "default": "same", "desc": "nn_layer_pad_desc"},
    },
    "MaxPooling2D": {
        "pool_size": {"type": "string", "default": "2,2", "desc": "nn_layer_pool_desc"},
    },
    "Flatten": {},
    "LSTM": {
        "units": {"type": "int", "default": 64, "min": 1, "max": 512, "desc": "nn_layer_lstm_desc"},
        "return_sequences": {"type": "bool", "default": False, "desc": "nn_layer_lstm_ret"},
    },
    "GRU": {
        "units": {"type": "int", "default": 64, "min": 1, "max": 512, "desc": "nn_layer_gru_desc"},
        "return_sequences": {"type": "bool", "default": False, "desc": "nn_layer_gru_ret"},
    },
    "Embedding": {
        "input_dim": {"type": "int", "default": 1000, "min": 2, "max": 50000, "desc": "nn_layer_emb_in"},
        "output_dim": {"type": "int", "default": 64, "min": 2, "max": 512, "desc": "nn_layer_emb_out"},
    },
}


class NNBuilderView(ttk.Frame):
    """Visual neural network builder with training controls."""

    def __init__(self, parent, service: PipelineService | None = None, show_header: bool = True):
        super().__init__(parent, style="TFrame")
        self._service = service or PipelineService()
        self._layers: list[dict] = []
        self._selected_index: int | None = None
        self._layer_vars: dict[str, tk.Variable] = {}
        self._show_header = show_header

        self._epochs_var = tk.StringVar(value="20")
        self._batch_var = tk.StringVar(value="32")
        self._lr_var = tk.StringVar(value="0.001")
        self._val_split_var = tk.StringVar(value="0.2")
        self._shuffle_var = tk.BooleanVar(value=True)
        self._seed_var = tk.StringVar(value="42")
        self._optimizer_var = tk.StringVar(value="Adam")
        self._loss_var = tk.StringVar(value=LOSSES_CLASSIFICATION[0])
        self._device_var = tk.StringVar(value="auto")
        self._auto_output_var = tk.BooleanVar(value=True)
        self._input_shape_var = tk.StringVar(value="auto")

        self._train_status_var = tk.StringVar(value=_("train_status_ready"))
        self._train_progress_var = tk.DoubleVar(value=0.0)
        self._build()

    def _build(self):
        px = 16

        if self._show_header:
            header = tk.Frame(self, bg=C.BG_MAIN)
            header.pack(fill="x", padx=px, pady=(24, 0))
            SectionHeader(header, title=_("nn_builder_title"), subtitle=_("nn_builder_subtitle")).pack(side="left")

        layout = tk.Frame(self, bg=C.BG_MAIN)
        layout.pack(fill="both", expand=True, padx=px, pady=(16, 16))
        layout.columnconfigure(0, weight=1)
        layout.columnconfigure(1, weight=1)

        left = tk.Frame(layout, bg=C.BG_MAIN)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right = tk.Frame(layout, bg=C.BG_MAIN)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        self._layer_card = Card(left, accent_color=C.ACCENT, pad=12)
        self._layer_card.pack(fill="both", expand=True)
        tk.Label(self._layer_card.inner, text=_("nn_arch"), font=F.H3, bg=C.BG_CARD, fg=C.TEXT).pack(anchor="w")

        add_row = tk.Frame(self._layer_card.inner, bg=C.BG_CARD)
        add_row.pack(fill="x", pady=(8, 8))
        self._layer_type_var = tk.StringVar(value=list(LAYER_SCHEMAS.keys())[0])
        ttk.Combobox(add_row, textvariable=self._layer_type_var, values=list(LAYER_SCHEMAS.keys()), state="readonly", width=16).pack(side="left")
        ModernButton(add_row, text=_("nn_btn_add"), style="primary", command=self._add_layer, bg=C.BG_CARD).pack(side="left", padx=(8, 0))

        control_row = tk.Frame(self._layer_card.inner, bg=C.BG_CARD)
        control_row.pack(fill="x", pady=(0, 8))
        ModernButton(control_row, text=_("nn_btn_up"), style="ghost", command=lambda: self._move_layer(-1), bg=C.BG_CARD).pack(side="left", padx=(0, 8))
        ModernButton(control_row, text=_("nn_btn_down"), style="ghost", command=lambda: self._move_layer(1), bg=C.BG_CARD).pack(side="left", padx=(0, 8))
        ModernButton(control_row, text=_("nn_btn_del"), style="danger", command=self._remove_layer, bg=C.BG_CARD).pack(side="left")

        list_frame = tk.Frame(self._layer_card.inner, bg=C.BG_CARD)
        list_frame.pack(fill="both", expand=True)
        self._layers_list = tk.Listbox(list_frame, height=10)
        self._layers_list.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(list_frame, orient="vertical", command=self._layers_list.yview)
        sb.pack(side="right", fill="y")
        self._layers_list.configure(yscrollcommand=sb.set)
        self._layers_list.bind("<<ListboxSelect>>", self._on_select_layer)

        self._preview_card = Card(right, accent_color=C.INFO, pad=12)
        self._preview_card.pack(fill="both", expand=True)
        tk.Label(self._preview_card.inner, text=_("nn_preview"), font=F.H3, bg=C.BG_CARD, fg=C.TEXT).pack(anchor="w")

        self._canvas = tk.Canvas(self._preview_card.inner, height=240, bg=C.BG_CARD, highlightthickness=0)
        self._canvas.pack(fill="both", expand=True, pady=(8, 8))

        tk.Label(self._preview_card.inner, text=_("nn_inspector"), font=F.H4, bg=C.BG_CARD, fg=C.TEXT).pack(anchor="w")
        self._inspector = tk.Frame(self._preview_card.inner, bg=C.BG_CARD)
        self._inspector.pack(fill="x", pady=(6, 0))

        self._train_card = Card(self, accent_color=C.SUCCESS, pad=12)
        self._train_card.pack(fill="x", padx=px, pady=(0, 16))
        tk.Label(self._train_card.inner, text=_("nn_train_config"), font=F.H3, bg=C.BG_CARD, fg=C.TEXT).pack(anchor="w")

        cfg = tk.Frame(self._train_card.inner, bg=C.BG_CARD)
        cfg.pack(fill="x", pady=(8, 8))

        self._add_cfg_row(cfg, "Epochs", self._epochs_var, 0)
        self._add_cfg_row(cfg, "Batch size", self._batch_var, 1)
        self._add_cfg_row(cfg, "Learning rate", self._lr_var, 2)
        self._add_cfg_row(cfg, "Validation split", self._val_split_var, 3)
        self._add_cfg_row(cfg, "Seed", self._seed_var, 4)

        tk.Label(cfg, text="Optimizer", font=F.TINY, bg=C.BG_CARD, fg=C.TEXT_SEC).grid(row=0, column=6, sticky="w")
        ttk.Combobox(cfg, textvariable=self._optimizer_var, values=list(OPTIMIZERS.keys()), state="readonly", width=16).grid(row=1, column=6, sticky="w")

        tk.Label(cfg, text="Loss", font=F.TINY, bg=C.BG_CARD, fg=C.TEXT_SEC).grid(row=0, column=7, sticky="w", padx=(12, 0))
        self._loss_combo = ttk.Combobox(cfg, textvariable=self._loss_var, values=LOSSES_CLASSIFICATION, state="readonly", width=20)
        self._loss_combo.grid(row=1, column=7, sticky="w", padx=(12, 0))

        tk.Label(cfg, text="Device", font=F.TINY, bg=C.BG_CARD, fg=C.TEXT_SEC).grid(row=0, column=8, sticky="w", padx=(12, 0))
        ttk.Combobox(cfg, textvariable=self._device_var, values=["auto", "CPU", "GPU"], state="readonly", width=10).grid(row=1, column=8, sticky="w", padx=(12, 0))

        ttk.Checkbutton(cfg, text="Shuffle", variable=self._shuffle_var, style="Card.TCheckbutton").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Checkbutton(cfg, text=_("nn_auto_output"), variable=self._auto_output_var, style="Card.TCheckbutton").grid(row=2, column=1, sticky="w", pady=(6, 0))

        tk.Label(cfg, text="Input shape", font=F.TINY, bg=C.BG_CARD, fg=C.TEXT_SEC).grid(row=2, column=2, sticky="w", padx=(12, 0))
        ttk.Entry(cfg, textvariable=self._input_shape_var, width=16).grid(row=2, column=3, sticky="w", padx=(8, 0))

        btn_row = tk.Frame(self._train_card.inner, bg=C.BG_CARD)
        btn_row.pack(fill="x", pady=(8, 0))
        ModernButton(btn_row, text=_("nn_btn_export"), style="secondary", command=self._export_code, bg=C.BG_CARD).pack(side="right")
        ModernButton(btn_row, text=_("nn_btn_train"), style="primary", command=self._train_model, bg=C.BG_CARD).pack(side="right", padx=(0, 8))

        monitor = Card(self, accent_color=C.ACCENT, pad=12)
        monitor.pack(fill="both", expand=True, padx=px, pady=(0, 16))
        tk.Label(monitor.inner, text=_("nn_monitor_title"), font=F.H3, bg=C.BG_CARD, fg=C.TEXT).pack(anchor="w")

        row = tk.Frame(monitor.inner, bg=C.BG_CARD)
        row.pack(fill="x", pady=(8, 8))
        self._nn_progress = ttk.Progressbar(row, variable=self._train_progress_var, maximum=1.0)
        self._nn_progress.pack(side="left", fill="x", expand=True, padx=(0, 12))
        tk.Label(row, textvariable=self._train_status_var, font=F.SMALL, bg=C.BG_CARD, fg=C.TEXT_SEC).pack(side="left")

        self._nn_log = LogPanel(monitor.inner, height=6, label=_("nn_monitor_logs"), bg_outer=C.BG_CARD, scrollbar=True)
        self._nn_log.pack(fill="both", expand=True, pady=(0, 8))

        self._nn_plot = PlotCanvas(monitor.inner, bg=C.BG_CARD)
        self._nn_plot.pack(fill="both", expand=True)

    def _add_cfg_row(self, parent: tk.Frame, label: str, var: tk.StringVar, col: int):
        tk.Label(parent, text=label, font=F.TINY, bg=C.BG_CARD, fg=C.TEXT_SEC).grid(row=0, column=col, sticky="w", padx=(0, 12))
        ttk.Entry(parent, textvariable=var, width=10).grid(row=1, column=col, sticky="w", padx=(0, 12))

    def _add_layer(self):
        layer_type = self._layer_type_var.get()
        params = {}
        for k, meta in LAYER_SCHEMAS.get(layer_type, {}).items():
            params[k] = meta.get("default")
        self._layers.append({"type": layer_type, "params": params})
        self._refresh_layers()

    def _remove_layer(self):
        if self._selected_index is None:
            return
        if 0 <= self._selected_index < len(self._layers):
            self._layers.pop(self._selected_index)
            self._selected_index = None
        self._refresh_layers()

    def _move_layer(self, delta: int):
        if self._selected_index is None:
            return
        new_idx = self._selected_index + delta
        if new_idx < 0 or new_idx >= len(self._layers):
            return
        self._layers[self._selected_index], self._layers[new_idx] = self._layers[new_idx], self._layers[self._selected_index]
        self._selected_index = new_idx
        self._refresh_layers()

    def _refresh_layers(self):
        self._layers_list.delete(0, "end")
        for i, layer in enumerate(self._layers, start=1):
            self._layers_list.insert("end", f"{i}. {layer['type']}")
        self._draw_canvas()
        self._render_inspector()

    def _on_select_layer(self, _):
        selection = self._layers_list.curselection()
        self._selected_index = int(selection[0]) if selection else None
        self._render_inspector()

    def _render_inspector(self):
        for w in self._inspector.winfo_children():
            w.destroy()
        if self._selected_index is None or self._selected_index >= len(self._layers):
            tk.Label(self._inspector, text=_("nn_msg_sel_layer"), bg=C.BG_CARD, fg=C.TEXT_DIM, font=F.SMALL).pack(anchor="w")
            return

        layer = self._layers[self._selected_index]
        schema = LAYER_SCHEMAS.get(layer["type"], {})
        self._layer_vars = {}

        for r, (name, meta) in enumerate(schema.items()):
            name_label = tk.Label(self._inspector, text=name, bg=C.BG_CARD, fg=C.TEXT, font=F.SMALL)
            name_label.grid(row=r, column=0, sticky="w", pady=2, padx=(0, 8))
            if meta.get("desc"):
                Tooltip(name_label, _(meta.get("desc")))
            control = self._build_layer_control(layer, name, meta)
            control.grid(row=r, column=1, sticky="w", pady=2)

    def _build_layer_control(self, layer: dict, name: str, meta: dict) -> tk.Widget:
        ptype = meta.get("type")
        default = layer["params"].get(name, meta.get("default"))

        if ptype == "bool":
            var = tk.BooleanVar(value=bool(default))
            self._layer_vars[name] = var
            var.trace_add("write", lambda *a: self._update_layer_param(name, var.get()))
            return ttk.Checkbutton(self._inspector, variable=var, style="Card.TCheckbutton")

        if ptype == "choice":
            var = tk.StringVar(value=str(default))
            self._layer_vars[name] = var
            cb = ttk.Combobox(self._inspector, textvariable=var, values=[str(c) for c in meta.get("choices", [])], state="readonly", width=14)
            cb.bind("<<ComboboxSelected>>", lambda _e: self._update_layer_param(name, var.get()))
            return cb

        var = tk.StringVar(value=str(default) if default is not None else "")
        self._layer_vars[name] = var
        ent = ttk.Entry(self._inspector, textvariable=var, width=16)
        ent.bind("<FocusOut>", lambda _e: self._update_layer_param(name, var.get()))
        return ent

    def _update_layer_param(self, name: str, value):
        if self._selected_index is None:
            return
        layer = self._layers[self._selected_index]
        layer["params"][name] = value
        self._draw_canvas()

    def _draw_canvas(self):
        self._canvas.delete("all")
        w = self._canvas.winfo_width() or 400
        h = self._canvas.winfo_height() or 240
        gap = max(40, (w - 40) // max(1, len(self._layers)))
        x = 20
        for i, layer in enumerate(self._layers):
            cx = x + i * gap
            color = C.BG_CARD_ALT if i != self._selected_index else C.ACCENT
            self._canvas.create_rectangle(cx, h // 2 - 32, cx + 90, h // 2 + 32, fill=color, outline=C.BORDER)
            self._canvas.create_text(cx + 45, h // 2, text=layer["type"], fill=C.TEXT)

    def _export_code(self):
        path = ask_export_python_file()
        if not path:
            return
        code = self._generate_keras_code()
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(code)
            show_info(_("train_success_title"), _("nn_msg_exported").format(path))
        except Exception as exc:
            show_error(_("train_err_title"), str(exc))

    def _generate_keras_code(self) -> str:
        lines = [
            "from tensorflow import keras",
            "from tensorflow.keras import layers",
            "",
            "def build_model(input_shape, num_classes=None):",
            "    model = keras.Sequential()",
            "    model.add(layers.Input(shape=input_shape))",
        ]
        for layer in self._layers:
            lines.extend(self._layer_to_code(layer))
        lines.append("    return model")
        return "\n".join(lines)

    def _layer_to_code(self, layer: dict) -> list[str]:
        lt = layer["type"]
        p = layer.get("params", {})
        if lt == "Dense":
            act = ACTIVATIONS.get(p.get("activation", "ReLU"), "relu")
            return [f"    model.add(layers.Dense({p.get('units', 64)}, activation='{act}'))"]
        if lt == "Dropout":
            return [f"    model.add(layers.Dropout({p.get('rate', 0.5)}))"]
        if lt == "BatchNormalization":
            return ["    model.add(layers.BatchNormalization())"]
        if lt == "Flatten":
            return ["    model.add(layers.Flatten())"]
        if lt == "Conv2D":
            kernel = p.get("kernel_size", "3,3")
            act = ACTIVATIONS.get(p.get("activation", "ReLU"), "relu")
            padding = p.get("padding", "same")
            return [f"    model.add(layers.Conv2D({p.get('filters', 32)}, ({kernel}), activation='{act}', padding='{padding}'))"]
        if lt == "MaxPooling2D":
            pool = p.get("pool_size", "2,2")
            return [f"    model.add(layers.MaxPooling2D(pool_size=({pool})))"]
        if lt == "LSTM":
            return [f"    model.add(layers.LSTM({p.get('units', 64)}, return_sequences={p.get('return_sequences', False)}))"]
        if lt == "GRU":
            return [f"    model.add(layers.GRU({p.get('units', 64)}, return_sequences={p.get('return_sequences', False)}))"]
        if lt == "Embedding":
            return [f"    model.add(layers.Embedding({p.get('input_dim', 1000)}, {p.get('output_dim', 64)}))"]
        return [f"    # Unsupported layer: {lt}"]

    def _train_model(self):
        if self._service.preprocessing_result is None:
            show_error(_("train_err_title"), _("nn_err_prep_first"))
            return

        try:
            import tensorflow as tf
        except Exception as exc:
            show_error(_("train_err_title"), _("nn_err_tf_req").format(exc))
            return

        pr = self._service.preprocessing_result
        X_train = pr.X_train
        y_train = pr.y_train
        task_type = pr.task_type

        input_shape = self._resolve_input_shape(X_train)
        if input_shape is None:
            show_error(_("train_err_title"), _("nn_err_input_shape"))
            return

        epochs = int(self._epochs_var.get() or 20)
        batch_size = int(self._batch_var.get() or 32)
        val_split = float(self._val_split_var.get() or 0.2)
        lr = float(self._lr_var.get() or 0.001)
        seed = int(self._seed_var.get() or 42)

        np.random.seed(seed)
        tf.random.set_seed(seed)

        device = self._device_var.get()
        device_ctx = self._get_device_context(tf, device)

        self._train_status_var.set(_("nn_status_running"))
        self._train_progress_var.set(0)
        self._nn_log.set_content(_("nn_log_start"))
        self._nn_progress.configure(maximum=epochs)

        self._sync_loss_options(task_type)

        def thread():
            try:
                with device_ctx:
                    model, prepared = self._build_keras_model(tf, input_shape, task_type, X_train, y_train)
                    X, y = prepared
                    cb = self._make_callback()
                    history = model.fit(
                        X,
                        y,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=val_split,
                        shuffle=self._shuffle_var.get(),
                        callbacks=[cb] if cb else [],
                        verbose=0,
                    )
                self.after(0, lambda: self._on_train_finished(history))
            except Exception as exc:
                self.after(0, lambda: show_error(_("train_err_title"), str(exc)))

        threading.Thread(target=thread, daemon=True).start()

    def _make_callback(self):
        try:
            import tensorflow as tf
        except Exception:
            return None

        view = self

        class _CB(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                view.after(0, lambda: view._update_epoch(epoch + 1, logs))

        return _CB()

    def _update_epoch(self, epoch: int, logs: dict):
        self._train_progress_var.set(epoch)
        msg = f"Epoch {epoch}: " + ", ".join(f"{k}={v:.4f}" for k, v in logs.items() if isinstance(v, float))
        self._nn_log.append(msg)

    def _on_train_finished(self, history):
        self._train_status_var.set(_("nn_status_done"))
        self._render_history_plot(history.history)
        show_info(_("train_success_title"), _("nn_success_done"))

    def _render_history_plot(self, history: dict):
        if not history:
            return
        fig = create_styled_figure(figsize=(5, 3.2))
        ax = fig.add_subplot(111)
        if "loss" in history:
            ax.plot(history["loss"], label="loss")
        if "accuracy" in history:
            ax.plot(history["accuracy"], label="accuracy")
        if "val_loss" in history:
            ax.plot(history["val_loss"], label="val_loss")
        ax.legend()
        ax.set_title("Training metrics")
        self._nn_plot.update_figure(fig)

    def _resolve_input_shape(self, X_train: np.ndarray) -> tuple | None:
        raw = self._input_shape_var.get().strip().lower()
        if raw in ("", "auto"):
            return (X_train.shape[1],)
        try:
            parts = [int(p.strip()) for p in raw.split(",") if p.strip()]
            return tuple(parts)
        except Exception:
            return None

    def _build_keras_model(self, tf, input_shape, task_type: str, X, y):
        from tensorflow import keras
        from tensorflow.keras import layers

        model = keras.Sequential()
        model.add(layers.Input(shape=input_shape))

        for layer in self._layers:
            self._add_layer_to_model(model, layers, layer)

        num_classes = len(np.unique(y)) if task_type == "classification" else 1
        y_prepared = y
        if self._auto_output_var.get():
            if task_type == "classification":
                if num_classes <= 2:
                    model.add(layers.Dense(1, activation="sigmoid"))
                    if self._loss_var.get() not in LOSSES_CLASSIFICATION:
                        self._loss_var.set("binary_crossentropy")
                else:
                    model.add(layers.Dense(num_classes, activation="softmax"))
                    y_prepared = tf.keras.utils.to_categorical(y, num_classes)
                    if self._loss_var.get() not in LOSSES_CLASSIFICATION:
                        self._loss_var.set("categorical_crossentropy")
            else:
                model.add(layers.Dense(1, activation="linear"))

        optimizer = self._build_optimizer(tf)
        loss = self._build_loss(tf, task_type)
        metrics = ["accuracy"] if task_type == "classification" else ["mae"]
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model, (X, y_prepared)

    def _sync_loss_options(self, task_type: str):
        if task_type == "regression":
            self._loss_combo.configure(values=LOSSES_REGRESSION)
            if self._loss_var.get() not in LOSSES_REGRESSION:
                self._loss_var.set("mse")
        else:
            self._loss_combo.configure(values=LOSSES_CLASSIFICATION)
            if self._loss_var.get() not in LOSSES_CLASSIFICATION:
                self._loss_var.set(LOSSES_CLASSIFICATION[0])

    def _add_layer_to_model(self, model, layers, layer: dict):
        lt = layer["type"]
        p = layer.get("params", {})
        if lt == "Dense":
            act = ACTIVATIONS.get(p.get("activation", "ReLU"), "relu")
            if act in ("leaky_relu", "elu", "gelu"):
                model.add(layers.Dense(int(p.get("units", 64))))
                model.add(self._activation_layer(layers, act))
            else:
                model.add(layers.Dense(int(p.get("units", 64)), activation=act))
        elif lt == "Dropout":
            model.add(layers.Dropout(float(p.get("rate", 0.5))))
        elif lt == "BatchNormalization":
            model.add(layers.BatchNormalization())
        elif lt == "Flatten":
            model.add(layers.Flatten())
        elif lt == "Conv2D":
            kernel = self._parse_tuple(p.get("kernel_size", "3,3"), default=(3, 3))
            act = ACTIVATIONS.get(p.get("activation", "ReLU"), "relu")
            padding = p.get("padding", "same")
            if act in ("leaky_relu", "elu", "gelu"):
                model.add(layers.Conv2D(int(p.get("filters", 32)), kernel_size=kernel, padding=padding))
                model.add(self._activation_layer(layers, act))
            else:
                model.add(layers.Conv2D(int(p.get("filters", 32)), kernel_size=kernel, activation=act, padding=padding))
        elif lt == "MaxPooling2D":
            pool = self._parse_tuple(p.get("pool_size", "2,2"), default=(2, 2))
            model.add(layers.MaxPooling2D(pool_size=pool))
        elif lt == "LSTM":
            model.add(layers.LSTM(int(p.get("units", 64)), return_sequences=bool(p.get("return_sequences", False))))
        elif lt == "GRU":
            model.add(layers.GRU(int(p.get("units", 64)), return_sequences=bool(p.get("return_sequences", False))))
        elif lt == "Embedding":
            model.add(layers.Embedding(int(p.get("input_dim", 1000)), int(p.get("output_dim", 64))))

    def _activation_layer(self, layers, act: str):
        if act == "leaky_relu":
            return layers.LeakyReLU()
        if act == "elu":
            return layers.ELU()
        if act == "gelu":
            return layers.Activation("gelu")
        return layers.Activation("relu")

    def _build_optimizer(self, tf):
        name = OPTIMIZERS.get(self._optimizer_var.get(), "adam")
        lr = float(self._lr_var.get() or 0.001)
        opts = {
            "adam": tf.keras.optimizers.Adam,
            "sgd": tf.keras.optimizers.SGD,
            "rmsprop": tf.keras.optimizers.RMSprop,
            "adagrad": tf.keras.optimizers.Adagrad,
            "adamw": tf.keras.optimizers.AdamW,
        }
        return opts.get(name, tf.keras.optimizers.Adam)(learning_rate=lr)

    def _build_loss(self, tf, task_type: str):
        loss_name = self._loss_var.get()
        if loss_name == "focal_loss":
            return self._focal_loss(tf)
        if task_type == "regression" and loss_name not in LOSSES_REGRESSION:
            return "mse"
        return loss_name

    def _focal_loss(self, tf, gamma: float = 2.0, alpha: float = 0.25):
        def loss(y_true, y_pred):
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
            ce = -y_true * tf.math.log(y_pred)
            weight = alpha * tf.math.pow(1 - y_pred, gamma)
            return tf.reduce_mean(weight * ce)

        return loss

    def _parse_tuple(self, raw: str, default=(1, 1)):
        try:
            parts = [int(p.strip()) for p in str(raw).split(",") if p.strip()]
            if len(parts) >= 2:
                return tuple(parts[:2])
            if len(parts) == 1:
                return (parts[0], parts[0])
        except Exception:
            pass
        return default

    def _get_device_context(self, tf, device: str):
        if device == "CPU":
            return tf.device("/CPU:0")
        if device == "GPU":
            return tf.device("/GPU:0")
        return tf.device("/device:CPU:0") if not tf.config.list_physical_devices("GPU") else tf.device("/device:GPU:0")