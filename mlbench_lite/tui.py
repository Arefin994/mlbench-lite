"""
Interactive TUI for mlbench-lite, built with Textual.

This is a *separate* entry point (`mlbench-lite-tui`) from the existing
argparse CLI (`mlbench-lite`). It calls the same `benchmark()` /
`recommend_scaling()` functions underneath -- no logic is duplicated -- so
anything that works from the Python API or the flag-based CLI works here
too, just picked with the keyboard or mouse instead of typed as flags.

Supports either a built-in demo dataset, or your own CSV/TSV file with
pickable target (y) and feature (X) columns.

Optional install (this whole module is inert without it):
    pip install mlbench-lite[tui]

Run:
    mlbench-lite-tui
"""

from __future__ import annotations

import sys
import threading

try:
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, VerticalScroll
    from textual.widgets import (
        Button,
        DataTable,
        Footer,
        Header,
        Input,
        Label,
        ProgressBar,
        RadioButton,
        RadioSet,
        SelectionList,
        Static,
    )
    from textual.widgets.selection_list import Selection

    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False


if TEXTUAL_AVAILABLE:
    _DATASET_MAP = {
        "ds_clover": "clover",
        "ds_iris": "iris",
        "ds_wine": "wine",
        "ds_bc": "breast_cancer",
    }
    _SCALE_MAP = {
        "scale_off": None,
        "scale_standard": "standard",
        "scale_minmax": "minmax",
        "scale_robust": "robust",
        "scale_auto": "auto",
    }
    _DEVICE_MAP = {"dev_auto": "auto", "dev_cpu": "cpu", "dev_gpu": "gpu"}

    class MLBenchApp(App):
        """A guided, keyboard-and-mouse-friendly way to run mlbench-lite."""

        TITLE = "mlbench-lite"
        SUB_TITLE = "interactive benchmark runner"
        BINDINGS = [
            ("q", "quit", "Quit"),
            ("r", "run_benchmark", "Run"),
            ("c", "cancel_benchmark", "Cancel"),
        ]

        CSS = """
        #main { padding: 1 2; }
        #title { text-style: bold; padding-bottom: 1; }
        Label { padding-top: 1; }
        #model_select { height: 12; border: solid $accent; }
        #feature_col_select { height: 8; border: solid $accent; }
        #results_table { height: 14; margin-top: 1; }
        #reco_table { height: 8; margin-top: 1; }
        #buttons { padding-top: 1; height: auto; }
        #csv_row { height: auto; padding-top: 1; }
        #csv_path_input { width: 3fr; }
        #load_csv_btn { width: 1fr; margin-left: 1; }
        #status { padding-top: 1; color: $text-muted; }
        #csv_status { padding-top: 1; color: $text-muted; }
        #progress_bar { margin-top: 1; }
        """

        def __init__(self):
            super().__init__()
            self.results = None
            self._custom_df = None
            self._csv_columns: list[str] = []
            self._cancel_event = threading.Event()

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with VerticalScroll(id="main"):
                yield Static(
                    "mlbench-lite -- pick your dataset, models, and options below",
                    id="title",
                )

                yield Label("Dataset source")
                yield RadioSet(
                    RadioButton("Built-in demo dataset", value=True, id="src_builtin"),
                    RadioButton("My own CSV/TSV file", id="src_custom"),
                    id="dataset_source_set",
                )

                yield Label(
                    "Built-in dataset (used when source is 'Built-in demo dataset')"
                )
                yield RadioSet(
                    RadioButton("clover (built-in demo)", value=True, id="ds_clover"),
                    RadioButton("iris", id="ds_iris"),
                    RadioButton("wine", id="ds_wine"),
                    RadioButton("breast_cancer", id="ds_bc"),
                    id="dataset_set",
                )

                yield Label("CSV/TSV path (used when source is 'My own CSV/TSV file')")
                with Horizontal(id="csv_row"):
                    yield Input(
                        placeholder="e.g. C:\\Users\\you\\data\\my_dataset.csv",
                        id="csv_path_input",
                    )
                    yield Button("Load columns", id="load_csv_btn")
                yield Static("", id="csv_status")

                yield Label("Target column (y) -- appears after loading a CSV")
                yield RadioSet(id="target_col_set")

                yield Label(
                    "Feature columns (X) -- appears after loading a CSV, all selected by default"
                )
                yield SelectionList(id="feature_col_select")

                yield Label("Task")
                yield RadioSet(
                    RadioButton("auto-detect", value=True, id="task_auto"),
                    RadioButton("classification", id="task_clf"),
                    RadioButton("regression", id="task_reg"),
                    id="task_set",
                )

                yield Label(
                    "Models to benchmark (space to toggle, all selected by default)"
                )
                yield SelectionList(id="model_select")

                yield Label("Cross-validation folds (blank = single train/test split)")
                yield Input(placeholder="e.g. 5", id="cv_input")

                yield Label("Scaling mode")
                yield RadioSet(
                    RadioButton("off", value=True, id="scale_off"),
                    RadioButton("standard", id="scale_standard"),
                    RadioButton("minmax", id="scale_minmax"),
                    RadioButton("robust", id="scale_robust"),
                    RadioButton("auto (per-column)", id="scale_auto"),
                    id="scale_set",
                )
                yield Button(
                    "Show scaling recommendation for this dataset", id="reco_btn"
                )
                yield DataTable(id="reco_table")

                yield Label("Device")
                yield RadioSet(
                    RadioButton(
                        "auto (use GPU if available)", value=True, id="dev_auto"
                    ),
                    RadioButton("cpu", id="dev_cpu"),
                    RadioButton("gpu", id="dev_gpu"),
                    id="device_set",
                )

                yield Label("Random seed")
                yield Input(value="42", id="seed_input")

                with Horizontal(id="buttons"):
                    yield Button("Run Benchmark (r)", id="run_btn", variant="success")
                    yield Button(
                        "Cancel", id="cancel_btn", variant="error", disabled=True
                    )
                    yield Button("Save results to CSV", id="save_btn", disabled=True)

                yield Static("", id="status")
                yield ProgressBar(id="progress_bar", show_eta=False)
                yield DataTable(id="results_table")
            yield Footer()

        def on_mount(self) -> None:
            self._populate_models()

        # -- helpers ---------------------------------------------------
        def _current_task(self):
            rs = self.query_one("#task_set", RadioSet)
            pressed = rs.pressed_button
            if pressed is None or pressed.id == "task_auto":
                return None
            return "classification" if pressed.id == "task_clf" else "regression"

        def _populate_models(self) -> None:
            from mlbench_lite.benchmark import list_available_models

            task = self._current_task() or "classification"
            model_select = self.query_one("#model_select", SelectionList)
            model_select.clear_options()
            for category, names in list_available_models(task=task).items():
                for name in names:
                    model_select.add_option(
                        Selection(f"[{category}] {name}", name, True)
                    )

        def _current_target_column(self):
            if not self._csv_columns:
                return None
            target_set = self.query_one("#target_col_set", RadioSet)
            pressed = target_set.pressed_button
            if pressed is None:
                return self._csv_columns[-1]
            idx = int(pressed.id.rsplit("_", 1)[-1])
            return self._csv_columns[idx]

        def _rebuild_feature_select(self) -> None:
            target_col = self._current_target_column()
            feature_select = self.query_one("#feature_col_select", SelectionList)
            feature_select.clear_options()
            for col in self._csv_columns:
                if col == target_col:
                    continue
                feature_select.add_option(Selection(str(col), col, True))

        def _load_csv(self) -> None:
            path = self.query_one("#csv_path_input", Input).value.strip()
            status = self.query_one("#csv_status", Static)
            if not path:
                status.update("Enter a CSV/TSV file path first.")
                return
            try:
                import pandas as pd

                if path.lower().endswith((".xlsx", ".xls")):
                    df = pd.read_excel(path)
                else:
                    sep = "\t" if path.lower().endswith((".tsv", ".tab")) else ","
                    df = pd.read_csv(path, sep=sep)
            except Exception as exc:
                status.update(f"Could not load '{path}': {exc}")
                return
            if df.shape[1] < 2:
                status.update(
                    f"'{path}' needs at least 2 columns (features + target); found {df.shape[1]}."
                )
                return
            self._custom_df = df
            self._csv_columns = [str(c) for c in df.columns]

            target_set = self.query_one("#target_col_set", RadioSet)
            target_set.remove_children()
            last_idx = len(self._csv_columns) - 1
            for i, col in enumerate(self._csv_columns):
                target_set.mount(
                    RadioButton(col, value=(i == last_idx), id=f"target_col_{i}")
                )

            self._rebuild_feature_select()
            status.update(
                f"Loaded '{path}' -- {df.shape[0]} rows, {df.shape[1]} columns. "
                f"Target defaults to the last column ('{self._csv_columns[-1]}') -- "
                f"change it below if needed. Remember to also set Dataset source to "
                f"'My own CSV/TSV file' above."
            )

        def _resolve_dataset(self):
            """(X, y) from whichever source is currently selected. Raises ValueError
            with a user-facing message if the state isn't ready yet."""
            src_set = self.query_one("#dataset_source_set", RadioSet)
            src_id = (
                src_set.pressed_button.id if src_set.pressed_button else "src_builtin"
            )
            if src_id == "src_custom":
                if self._custom_df is None:
                    raise ValueError(
                        "Dataset source is set to 'My own CSV/TSV file' but none is "
                        "loaded yet -- enter a path and click 'Load columns' first."
                    )
                target_col = self._current_target_column()
                feature_select = self.query_one("#feature_col_select", SelectionList)
                feature_cols = list(feature_select.selected)
                if not feature_cols:
                    raise ValueError("Select at least one feature column (X).")
                return self._custom_df[feature_cols], self._custom_df[target_col]
            ds_set = self.query_one("#dataset_set", RadioSet)
            dataset_id = (
                ds_set.pressed_button.id if ds_set.pressed_button else "ds_clover"
            )
            dataset_name = _DATASET_MAP.get(dataset_id, "clover")
            task = self._current_task()
            from mlbench_lite.cli import _load_demo_dataset

            return _load_demo_dataset(dataset_name, task or "classification")

        def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
            if event.radio_set.id == "task_set":
                self._populate_models()
            elif event.radio_set.id == "target_col_set":
                self._rebuild_feature_select()

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "run_btn":
                self.action_run_benchmark()
            elif event.button.id == "cancel_btn":
                self.action_cancel_benchmark()
            elif event.button.id == "save_btn":
                self._save_results()
            elif event.button.id == "load_csv_btn":
                self._load_csv()
            elif event.button.id == "reco_btn":
                self._show_recommendation()

        def action_run_benchmark(self) -> None:
            self._cancel_event.clear()
            self.query_one("#run_btn", Button).disabled = True
            self.query_one("#cancel_btn", Button).disabled = False
            self.query_one("#save_btn", Button).disabled = True
            self.query_one("#status", Static).update("Starting benchmark...")
            bar = self.query_one("#progress_bar", ProgressBar)
            bar.update(total=100, progress=0)
            self.run_worker(self._run_benchmark_worker, thread=True)

        def action_cancel_benchmark(self) -> None:
            if not self._cancel_event.is_set():
                self._cancel_event.set()
                self.query_one("#status", Static).update(
                    "Cancelling... finishing the current model, then stopping (partial results will still be shown)."
                )
                self.query_one("#cancel_btn", Button).disabled = True

        def _expected_total(self, selected: list) -> int:
            if selected:
                return len(selected)
            from mlbench_lite.benchmark import list_available_models

            task = self._current_task() or "classification"
            return sum(len(v) for v in list_available_models(task=task).values())

        def _run_benchmark_worker(self) -> None:
            from mlbench_lite.benchmark import benchmark

            try:
                X, y = self._resolve_dataset()
                task = self._current_task()

                cv_input = self.query_one("#cv_input", Input).value.strip()
                cv = int(cv_input) if cv_input else None

                scale_set = self.query_one("#scale_set", RadioSet)
                scale_id = (
                    scale_set.pressed_button.id
                    if scale_set.pressed_button
                    else "scale_off"
                )
                scaling_mode = _SCALE_MAP.get(scale_id)

                dev_set = self.query_one("#device_set", RadioSet)
                dev_id = (
                    dev_set.pressed_button.id if dev_set.pressed_button else "dev_auto"
                )
                device = _DEVICE_MAP.get(dev_id, "auto")

                seed_str = self.query_one("#seed_input", Input).value.strip()
                seed = int(seed_str) if seed_str else 42

                model_select = self.query_one("#model_select", SelectionList)
                selected = list(model_select.selected)

                total = self._expected_total(selected)
                self.call_from_thread(self._reset_progress, total)

                def progress_cb(completed, total_n, name, status):
                    self.call_from_thread(
                        self._update_progress, completed, total_n, name, status
                    )

                results = benchmark(
                    X,
                    y,
                    cv=cv,
                    task=task,
                    scaling=scaling_mode,
                    device=device,
                    random_state=seed,
                    models=selected if selected else None,
                    progress_callback=progress_cb,
                    cancel_check=self._cancel_event.is_set,
                )
            except (
                Exception
            ) as exc:  # surface errors in the UI instead of crashing the app
                self.call_from_thread(self._show_error, str(exc))
                self.call_from_thread(self._finish_run)
                return
            self.call_from_thread(self._show_results, results)
            self.call_from_thread(self._finish_run)

        def _reset_progress(self, total: int) -> None:
            bar = self.query_one("#progress_bar", ProgressBar)
            bar.update(total=total, progress=0)
            self.query_one("#status", Static).update(
                f"Running benchmark... 0/{total} models"
            )

        def _update_progress(
            self, completed: int, total: int, name: str, status: str
        ) -> None:
            bar = self.query_one("#progress_bar", ProgressBar)
            bar.update(total=total, progress=completed)
            icon = {"done": "\u2713", "error": "\u2717"}.get(status, "?")
            self.query_one("#status", Static).update(
                f"[{completed}/{total}] {icon} {name} ({status})"
            )

        def _finish_run(self) -> None:
            self.query_one("#run_btn", Button).disabled = False
            self.query_one("#cancel_btn", Button).disabled = True

        def _show_recommendation(self) -> None:
            self.query_one("#status", Static).update(
                "Computing scaling recommendation..."
            )
            self.run_worker(self._show_recommendation_worker, thread=True)

        def _show_recommendation_worker(self) -> None:
            from mlbench_lite.preprocessing import recommend_scaling

            try:
                X, _y = self._resolve_dataset()
                report = recommend_scaling(X)
            except Exception as exc:
                self.call_from_thread(self._show_error, str(exc))
                return
            self.call_from_thread(self._display_recommendation, report)

        def _display_recommendation(self, report) -> None:
            table = self.query_one("#reco_table", DataTable)
            table.clear(columns=True)
            table.add_columns(*[str(c) for c in report.columns])
            for _, row in report.iterrows():
                table.add_row(*[str(v) for v in row.values])
            self.query_one("#status", Static).update(
                "Scaling recommendation ready below -- informational only, doesn't change "
                "anything by itself. Pick a Scaling mode above to actually apply it."
            )

        def _show_error(self, message: str) -> None:
            self.query_one("#status", Static).update(f"Error: {message}")

        def _show_results(self, results) -> None:
            self.results = results
            table = self.query_one("#results_table", DataTable)
            table.clear(columns=True)
            if results.empty:
                msg = "No results produced -- check your model selection."
                if results.attrs.get("cancelled"):
                    msg = "Cancelled before any model finished -- no results to show."
                self.query_one("#status", Static).update(msg)
                return
            table.add_columns(*[str(c) for c in results.columns])
            for _, row in results.iterrows():
                table.add_row(*[str(v) for v in row.values])
            if results.attrs.get("cancelled"):
                self.query_one("#status", Static).update(
                    f"Cancelled -- showing partial results for the {len(results)} model(s) that finished first."
                )
            else:
                self.query_one("#status", Static).update(
                    f"Done -- {len(results)} model(s) benchmarked."
                )
            self.query_one("#save_btn", Button).disabled = False

        def _save_results(self) -> None:
            if self.results is None:
                return
            path = "mlbench_tui_results.csv"
            self.results.to_csv(path, index=False)
            self.query_one("#status", Static).update(f"Saved to {path}")


def main() -> None:
    if not TEXTUAL_AVAILABLE:
        print(
            "The interactive TUI needs the 'textual' package, which isn't installed.\n"
            "Install it with:\n\n    pip install mlbench-lite[tui]\n",
            file=sys.stderr,
        )
        sys.exit(1)
    MLBenchApp().run()


if __name__ == "__main__":
    main()
