"""
Desktop front end of the preprocessing (python -m BehaviorScreen) and of the
overlay videos (python -m BehaviorScreen.overlay). The forms are built from the
two command-line parsers; a run is those commands in a subprocess, with their
output in the log. Settings are kept between sessions.

    python -m BehaviorScreen.gui     (from the folder you run the pipeline from)
"""
import argparse
import inspect
import json
import re
import shlex
import sys
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from qtpy.QtCore import QProcess, QProcessEnvironment, QSettings, Qt, QTimer, QUrl
from qtpy.QtGui import QCloseEvent, QColor, QDesktopServices, QTextCursor
from qtpy.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QPlainTextEdit, QPushButton, QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from BehaviorScreen.__main__ import STEPS
from BehaviorScreen.__main__ import build_parser as pipeline_parser
from BehaviorScreen.overlay import build_parser as overlay_parser
from BehaviorScreen.status import tab_status

Value = Union[str, bool]
TABS = {
    "Export": ["export"], "Pose estimation": ["pose", "eye_crop", "eye_pose"], "Quality control": ["qc"],
    "Bouts": ["bouts"], "Saccades": ["saccades"], "Heatmaps": ["heatmaps"],
}
DOWNLOADS = {"full_model_dir": "full", "eyes_model_dir": "eyes", "saccade_model": "saccades"}  # model folders and their download_model name; unpacked here
# What each tab writes; its other paths are inputs, its flags and numbers options.
OUTPUTS = {
    "Export": {"results"}, "Pose estimation": {"lightning_pose", "eyes_video_dir"},
    "Quality control": {"qc_csv", "valid_trials_csv"}, "Bouts": {"bouts_csv"},
    "Heatmaps": {"plots", "bouts_png", "saccades_png"}, "Saccades": {"plots"}, "Overlay": {"output_dir"},
}


def form_actions(parser: argparse.ArgumentParser, skip: set) -> List[argparse.Action]:
    return [a for a in parser._actions if a.option_strings and a.dest not in skip and not isinstance(a, argparse._HelpAction)]


def tab_options(parser: argparse.ArgumentParser) -> Dict[str, List[argparse.Action]]:
    """Each tab's options: those its steps read (args.<dest> in their code)."""
    actions = form_actions(parser, {"root", "steps"})
    used = {tab: set(re.findall(r"args\.(\w+)", "".join(inspect.getsource(STEPS[s]) for s in steps))) for tab, steps in TABS.items()}
    return {tab: [a for a in actions if a.dest in u] for tab, u in used.items()}


def grouped(tab: str, actions: Sequence[argparse.Action]) -> Dict[str, List[argparse.Action]]:
    groups: Dict[str, List[argparse.Action]] = {"Inputs": [], "Outputs": [], "Options": []}
    for a in actions:
        option = isinstance(a, argparse._StoreTrueAction) or a.type in (int, float)
        groups["Outputs" if a.dest in OUTPUTS[tab] else "Options" if option else "Inputs"].append(a)
    return groups


def to_argv(action: argparse.Action, value: Value) -> List[str]:
    """Command-line tokens of a form value; none when it is the parser default."""
    flag = action.option_strings[-1]
    if isinstance(action, argparse._StoreTrueAction):
        return [flag] if value else []
    return [] if value == ("" if action.default is None else str(action.default)) else [flag, value]


def is_folder(action: argparse.Action) -> bool:
    """Path options whose default has no file extension."""
    option = isinstance(action, argparse._StoreTrueAction) or action.type in (int, float)
    return not option and bool(action.default) and not Path(str(action.default)).suffix


def label(action: argparse.Action) -> str:
    return action.option_strings[-1].lstrip("-") + (" (folder)" if is_folder(action) else "")


def browsable(action: argparse.Action) -> bool:
    return action.type is Path or action.dest.endswith("dir")


class ArgumentForm(QWidget):
    def __init__(self, groups: Dict[str, List[argparse.Action]], buttons: Dict[str, Tuple[str, Callable[[], None]]] = {}) -> None:
        super().__init__()
        self.actions = [a for actions in groups.values() for a in actions]
        self.fields: Dict[str, Union[QCheckBox, QLineEdit]] = {}
        layout = QVBoxLayout(self)
        for title, actions in groups.items():
            if not actions:
                continue
            box = QGroupBox(title)
            rows = QFormLayout(box)
            for a in actions:
                field = QCheckBox() if isinstance(a, argparse._StoreTrueAction) else QLineEdit("" if a.default is None else str(a.default))
                field.setToolTip(a.help or "")
                self.fields[a.dest] = field
                widget = with_browse(field, not is_folder(a)) if browsable(a) else field
                rows.addRow(label(a), with_button(widget, *buttons[a.dest]) if a.dest in buttons else widget)
            layout.addWidget(box)
        layout.addStretch()

    def values(self) -> Dict[str, Value]:
        return {dest: f.isChecked() if isinstance(f, QCheckBox) else f.text() for dest, f in self.fields.items()}

    def set_values(self, values: Dict[str, Value]) -> None:
        for dest, value in values.items():
            if dest in self.fields:
                field = self.fields[dest]
                if isinstance(field, QCheckBox):
                    field.setChecked(bool(value))
                else:
                    field.setText(str(value))

    def tokens(self) -> Dict[str, List[str]]:
        """Command-line tokens per option."""
        values = self.values()
        return {a.dest: to_argv(a, values[a.dest]) for a in self.actions}


def with_button(widget: QWidget, text: str, callback: Callable[[], None]) -> QWidget:
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    button = QPushButton(text)
    layout.addWidget(widget)
    layout.addWidget(button)
    button.clicked.connect(lambda: callback())
    return row


def with_browse(edit: QLineEdit, pick_file: bool) -> QWidget:
    def browse() -> None:
        path = QFileDialog.getOpenFileName(edit, "", edit.text())[0] if pick_file else QFileDialog.getExistingDirectory(edit, "", edit.text())
        if path:
            edit.setText(path)
    return with_button(edit, "…", browse)


def link(forms: Sequence[ArgumentForm]) -> None:
    """Fields of the same option in several forms show one value."""
    for dest in {d for form in forms for d in form.fields}:
        fields = [form.fields[dest] for form in forms if dest in form.fields]
        for field in fields:
            others = [o for o in fields if o is not field]
            if isinstance(field, QCheckBox):
                field.toggled.connect(lambda on, others=others: [o.setChecked(on) for o in others if o.isChecked() != on])
            else:
                field.textChanged.connect(lambda text, others=others: [o.setText(text) for o in others if o.text() != text])


STATE_COLORS = {"done": QColor(0, 170, 0, 50), "ready": QColor(230, 170, 0, 70), "waiting": QColor(0, 0, 0, 0)}


def fill(widget: QTableWidget, table: pd.DataFrame, colors: List[QColor]) -> None:
    widget.setSortingEnabled(False)
    widget.clear()
    widget.setRowCount(len(table))
    widget.setColumnCount(len(table.columns))
    widget.setHorizontalHeaderLabels([str(c) for c in table.columns])
    for i, (row, color) in enumerate(zip(table.itertuples(index=False), colors)):
        for j, value in enumerate(row):
            item = QTableWidgetItem(("✓" if value else "–") if isinstance(value, (bool, np.bool_)) else str(value))
            item.setBackground(color)
            widget.setItem(i, j, item)
    widget.resizeColumnsToContents()
    widget.setSortingEnabled(True)


def fit_height(table: QTableWidget) -> None:
    """Exactly as tall as its rows."""
    rows = sum(table.rowHeight(i) for i in range(table.rowCount()))
    scrollbar = table.horizontalScrollBar().sizeHint().height() if table.horizontalScrollBar().maximum() > 0 else 0
    table.setFixedHeight(table.horizontalHeader().sizeHint().height() + rows + scrollbar + 2 * table.frameWidth())


def boxed(title: str, widget: QWidget) -> QGroupBox:
    box = QGroupBox(title)
    QVBoxLayout(box).addWidget(widget)
    return box


class StatusPanel(QWidget):
    """What a step reads, the recordings it can work on, and what it already wrote."""

    def __init__(self, tab: str) -> None:
        super().__init__()
        self.tab = tab
        self.summary = QLabel()
        self.inputs, self.recordings, self.outputs = QTableWidget(), QTableWidget(), QTableWidget()
        self.refresh = QPushButton("Refresh")
        for table in (self.inputs, self.recordings, self.outputs):
            table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            table.verticalHeader().setVisible(False)
            table.horizontalHeader().setStretchLastSection(True)
        self.boxes = [boxed("Input files", self.inputs), boxed("Recordings", self.recordings), boxed("Output files", self.outputs)]
        layout = QVBoxLayout(self)
        layout.addWidget(self.summary)
        for box, stretch in zip(self.boxes, (0, 1, 0)):  # the recordings take the free height
            layout.addWidget(box, stretch=stretch)
        layout.addStretch(0)  # ... or nothing does, when they are hidden
        layout.addWidget(self.refresh)

    def show_status(self, root: Path, values: Dict[str, Value]) -> None:
        try:
            table, files = tab_status(self.tab, root, values)
        except (OSError, ValueError, KeyError) as error:  # unreadable or foreign tables
            self.summary.setText(f"Status unavailable: {error}")
            return
        counts = table["state"].value_counts()
        self.summary.setText(" · ".join(f"{counts.get(s, 0)} {s}" for s in STATE_COLORS) if len(table) else "")
        parts = [files[files["role"] == "input"].drop(columns="role"), table.reset_index(), files[files["role"] == "output"].drop(columns="role")]
        colors = [[STATE_COLORS["done" if p else "waiting"] for p in parts[0]["present"]], [STATE_COLORS[s] for s in table["state"]],
                  [STATE_COLORS["done" if p else "waiting"] for p in parts[2]["present"]]]
        for widget, box, part, color in zip((self.inputs, self.recordings, self.outputs), self.boxes, parts, colors):
            fill(widget, part, color)
            box.setVisible(len(part) > 0)
        for files in (self.inputs, self.outputs):
            fit_height(files)


def scrolled(widget: QWidget) -> QScrollArea:
    area = QScrollArea()
    area.setWidget(widget)
    area.setWidgetResizable(True)
    return area


class Window(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("BehaviorScreen")
        self.queue: List[List[str]] = []

        self.root = QLineEdit()
        self.root.setPlaceholderText("Root experiment folder")

        downloads = {dest: ("Download", lambda model=model: self.run([[sys.executable, "-m", "BehaviorScreen.download_model", model]]))
                     for dest, model in DOWNLOADS.items()}
        self.forms = {tab: ArgumentForm(grouped(tab, actions), downloads) for tab, actions in tab_options(pipeline_parser()).items()}
        link(self.forms.values())
        self.substeps = {step: QCheckBox(step) for steps in TABS.values() if len(steps) > 1 for step in steps}
        self.run_buttons: List[QPushButton] = []
        tabs = self.tabs = QTabWidget()
        self.splitters: List[QSplitter] = []
        self.panels: Dict[str, StatusPanel] = {}
        for tab, form in self.forms.items():
            page = QWidget()
            layout = QVBoxLayout(page)
            boxes = QWidget()
            box_layout = QHBoxLayout(boxes)
            box_layout.setContentsMargins(0, 0, 0, 0)
            for step in TABS.get(tab, []):
                if step in self.substeps:
                    self.substeps[step].setChecked(True)
                    box_layout.addWidget(self.substeps[step])
            if box_layout.count():
                layout.addWidget(boxes)
            panel = self.panels[tab] = StatusPanel(tab)
            panel.refresh.clicked.connect(self.refresh_status)
            layout.addWidget(self.side_by_side(panel, form), stretch=1)
            if tab in TABS:
                layout.addWidget(self.run_button(f"Run {tab.lower()}", lambda tab=tab: [self.pipeline_command(self.tab_steps(tab))]))
            tabs.addTab(page, tab)

        self.overlay = ArgumentForm(grouped("Overlay", form_actions(overlay_parser(), {"root", "fish"})))
        panel = self.panels["Overlay"] = StatusPanel("Overlay")
        panel.refresh.clicked.connect(self.refresh_status)
        panel.recordings.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        panel.recordings.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        panel.recordings.setToolTip("Select the recordings to render")
        open_folder = QPushButton("Open output folder")
        open_folder.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.root.text()) / self.overlay.values()["output_dir"]))))
        overlay_tab = QWidget()
        layout = QVBoxLayout(overlay_tab)
        layout.addWidget(self.side_by_side(panel, self.overlay), stretch=1)
        buttons = QWidget()
        button_layout = QHBoxLayout(buttons)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.addWidget(self.run_button("Render selected fish", self.overlay_commands))
        button_layout.addWidget(open_folder)
        layout.addWidget(buttons)

        tabs.addTab(overlay_tab, "Overlay")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(20000)
        self.stop = QPushButton("Stop")
        self.stop.setEnabled(False)
        self.stop.clicked.connect(self.cancel)
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(tabs)
        splitter.addWidget(self.log)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(with_browse(self.root, pick_file=False))
        layout.addWidget(splitter)
        bottom = QWidget()
        bottom_layout = QHBoxLayout(bottom)
        bottom_layout.addWidget(self.run_button("Run all steps", lambda: [self.pipeline_command(list(STEPS))]))
        bottom_layout.addWidget(self.stop)
        layout.addWidget(bottom)
        self.setCentralWidget(central)

        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        self.process.setProcessEnvironment(env)
        self.process.readyReadStandardOutput.connect(lambda: self.write(bytes(self.process.readAllStandardOutput()).decode(errors="replace")))
        self.process.finished.connect(lambda code, status: self.finished(code == 0 and status == QProcess.ExitStatus.NormalExit, f"exit code {code}"))
        self.process.errorOccurred.connect(lambda error: self.finished(False, "failed to start") if error == QProcess.ProcessError.FailedToStart else None)
        self.restore()
        self.tabs.currentChanged.connect(self.refresh_status)
        # Any change of the root or an option refreshes the status, once typing pauses.
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setSingleShot(True)
        self.refresh_timer.setInterval(400)
        self.refresh_timer.timeout.connect(self.refresh_status)
        restart = lambda *_: self.refresh_timer.start()  # signals pass their value, which start() would take as a delay
        self.root.textChanged.connect(restart)
        for form in [*self.forms.values(), self.overlay]:
            for field in form.fields.values():
                (field.toggled if isinstance(field, QCheckBox) else field.textChanged).connect(restart)
        self.refresh_status()

    def run_button(self, label: str, commands: Callable[[], List[List[str]]]) -> QPushButton:
        button = QPushButton(label)
        button.clicked.connect(lambda: self.run(commands()))
        self.run_buttons.append(button)
        return button

    def side_by_side(self, panel: QWidget, form: QWidget) -> QSplitter:
        """Status panel and options, the panel wider; the divider is kept between sessions."""
        splitter = QSplitter()
        splitter.addWidget(panel)
        splitter.addWidget(scrolled(form))
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([700, 350])
        self.splitters.append(splitter)
        return splitter

    def refresh_status(self) -> None:
        panel = self.panels.get(self.tabs.tabText(self.tabs.currentIndex()))
        if panel is not None and self.root.text():
            forms = [self.overlay] if panel.tab == "Overlay" else self.forms.values()
            panel.show_status(Path(self.root.text()), {k: v for form in forms for k, v in form.values().items()})

    def tab_steps(self, tab: str) -> List[str]:
        return [s for s in TABS[tab] if s not in self.substeps or self.substeps[s].isChecked()]

    def pipeline_command(self, steps: List[str]) -> List[str]:
        tokens = {dest: t for form in self.forms.values() for dest, t in form.tokens().items()}
        options = [token for t in tokens.values() for token in t]
        return [sys.executable, "-m", "BehaviorScreen", self.root.text(), "--steps", *steps, *options]

    def overlay_commands(self) -> List[List[str]]:
        options = [token for t in self.overlay.tokens().values() for token in t]
        base = [sys.executable, "-m", "BehaviorScreen.overlay", self.root.text(), *options]
        table = self.panels["Overlay"].recordings
        rows = sorted({index.row() for index in table.selectionModel().selectedRows()})
        return [[*base, "--fish", table.item(row, 0).text()] for row in rows]

    def run(self, commands: List[List[str]]) -> None:
        if not commands:
            return
        self.queue = commands
        for button in self.run_buttons:
            button.setEnabled(False)
        self.stop.setEnabled(True)
        self.next()

    def next(self) -> None:
        command = self.queue.pop(0)
        self.write(f"\n$ {shlex.join(command)}\n")
        self.statusBar().showMessage(f"Running {command[2]}")
        self.process.start(command[0], command[1:])

    def finished(self, ok: bool, message: str) -> None:
        self.write(f"\n[{message}]\n")
        if self.queue and ok:
            return self.next()
        self.queue = []
        for button in self.run_buttons:
            button.setEnabled(True)
        self.stop.setEnabled(False)
        self.statusBar().showMessage("Done" if ok else f"Failed ({message})")
        self.refresh_status()

    def cancel(self) -> None:
        self.queue = []
        self.process.kill()

    def write(self, text: str) -> None:
        """Append process output; a carriage return rewrites the line (progress bars)."""
        cursor = self.log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        for i, line in enumerate(text.replace("\r\n", "\n").split("\n")):
            if i:
                cursor.insertText("\n")
            if "\r" in line:
                cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock, QTextCursor.MoveMode.KeepAnchor)
                line = line.rsplit("\r", 1)[1]
            cursor.insertText(line)
        self.log.setTextCursor(cursor)
        self.log.ensureCursorVisible()

    def restore(self) -> None:
        state = json.loads(QSettings("BehaviorScreen", "gui").value("state", "{}"))
        self.root.setText(state.get("root", ""))
        for name, checked in state.get("steps", {}).items():
            if name in self.substeps:
                self.substeps[name].setChecked(checked)
        for form in self.forms.values():
            form.set_values(state.get("pipeline", {}))
        self.overlay.set_values(state.get("overlay", {}))
        for splitter, sizes in zip(self.splitters, state.get("splitters", [])):
            splitter.setSizes(sizes)

    def closeEvent(self, event: QCloseEvent) -> None:
        state = {"root": self.root.text(), "steps": {n: b.isChecked() for n, b in self.substeps.items()},
                 "pipeline": {k: v for form in self.forms.values() for k, v in form.values().items()}, "overlay": self.overlay.values(),
                 "splitters": [s.sizes() for s in self.splitters]}
        QSettings("BehaviorScreen", "gui").setValue("state", json.dumps(state))
        self.cancel()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    window = Window()
    window.resize(1400, 900)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
