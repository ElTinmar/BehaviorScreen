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
from typing import Callable, Dict, List, Sequence, Union

from qtpy.QtCore import QProcess, QProcessEnvironment, QSettings, Qt, QUrl
from qtpy.QtGui import QCloseEvent, QDesktopServices, QTextCursor
from qtpy.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QFileDialog, QFormLayout, QGridLayout, QHBoxLayout, QLineEdit,
    QListWidget, QMainWindow, QPlainTextEdit, QPushButton, QScrollArea, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from BehaviorScreen.__main__ import STEPS
from BehaviorScreen.__main__ import build_parser as pipeline_parser
from BehaviorScreen.overlay import build_parser as overlay_parser

Value = Union[str, bool]
TABS = {
    "Export": ["export"], "Pose estimation": ["pose", "eye_crop", "eye_pose"], "Quality control": ["qc"],
    "Bouts": ["bouts"], "Heatmaps": ["heatmaps"], "Saccades": ["saccades"],
}
SHARED = "Folders and files"


def form_actions(parser: argparse.ArgumentParser, skip: set) -> List[argparse.Action]:
    return [a for a in parser._actions if a.option_strings and a.dest not in skip and not isinstance(a, argparse._HelpAction)]


def tab_options(parser: argparse.ArgumentParser) -> Dict[str, List[argparse.Action]]:
    """Each tab's options: those its steps read (args.<dest> in their code); options several tabs read go to SHARED."""
    actions = form_actions(parser, {"root", "steps"})
    used = {tab: set(re.findall(r"args\.(\w+)", "".join(inspect.getsource(STEPS[s]) for s in steps))) for tab, steps in TABS.items()}
    shared = [a for a in actions if sum(a.dest in u for u in used.values()) > 1]
    return {SHARED: shared, **{tab: [a for a in actions if a.dest in u and a not in shared] for tab, u in used.items()}}


def to_argv(action: argparse.Action, value: Value) -> List[str]:
    """Command-line tokens of a form value; none when it is the parser default."""
    flag = action.option_strings[-1]
    if isinstance(action, argparse._StoreTrueAction):
        return [flag] if value else []
    return [] if value == ("" if action.default is None else str(action.default)) else [flag, value]


def browsable(action: argparse.Action) -> bool:
    return action.type is Path or action.dest.endswith("dir")


class ArgumentForm(QWidget):
    def __init__(self, actions: Sequence[argparse.Action]) -> None:
        super().__init__()
        self.actions = list(actions)
        self.fields: Dict[str, Union[QCheckBox, QLineEdit]] = {}
        layout = QFormLayout(self)
        for a in self.actions:
            field = QCheckBox() if isinstance(a, argparse._StoreTrueAction) else QLineEdit("" if a.default is None else str(a.default))
            field.setToolTip(a.help or "")
            self.fields[a.dest] = field
            layout.addRow(a.option_strings[-1].lstrip("-"), with_browse(field, bool(Path(str(a.default or "")).suffix)) if browsable(a) else field)

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

    def argv(self) -> List[str]:
        values = self.values()
        return [token for a in self.actions for token in to_argv(a, values[a.dest])]


def with_browse(edit: QLineEdit, pick_file: bool) -> QWidget:
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    button = QPushButton("…")
    button.setFixedWidth(32)
    layout.addWidget(edit)
    layout.addWidget(button)

    def browse() -> None:
        path = QFileDialog.getOpenFileName(edit, "", edit.text())[0] if pick_file else QFileDialog.getExistingDirectory(edit, "", edit.text())
        if path:
            edit.setText(path)
    button.clicked.connect(browse)
    return row


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

        self.forms = {tab: ArgumentForm(actions) for tab, actions in tab_options(pipeline_parser()).items()}
        self.substeps = {step: QCheckBox(step) for steps in TABS.values() if len(steps) > 1 for step in steps}
        self.run_buttons: List[QPushButton] = []
        tabs = QTabWidget()
        for tab, form in self.forms.items():
            page = QWidget()
            layout = QVBoxLayout(page)
            boxes = QWidget()
            box_layout = QHBoxLayout(boxes)
            for step in TABS.get(tab, []):
                if step in self.substeps:
                    self.substeps[step].setChecked(True)
                    box_layout.addWidget(self.substeps[step])
            if box_layout.count():
                layout.addWidget(boxes)
            layout.addWidget(scrolled(form))
            if tab in TABS:
                layout.addWidget(self.run_button(f"Run {tab.lower()}", lambda tab=tab: [self.pipeline_command(self.tab_steps(tab))]))
            tabs.addTab(page, tab)

        self.fish = QListWidget()
        self.fish.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.overlay = ArgumentForm(form_actions(overlay_parser(), {"root", "fish"}))
        refresh, open_folder = QPushButton("Refresh fish"), QPushButton("Open output folder")
        refresh.clicked.connect(self.list_fish)
        open_folder.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.root.text()) / self.overlay.values()["output_dir"]))))
        overlay_tab = QWidget()
        layout = QGridLayout(overlay_tab)
        layout.addWidget(self.fish, 0, 0)
        layout.addWidget(scrolled(self.overlay), 0, 1)
        layout.addWidget(refresh, 1, 0)
        buttons = QWidget()
        button_layout = QHBoxLayout(buttons)
        button_layout.addWidget(self.run_button("Render selected fish", self.overlay_commands))
        button_layout.addWidget(open_folder)
        layout.addWidget(buttons, 1, 1)

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
        self.list_fish()

    def run_button(self, label: str, commands: Callable[[], List[List[str]]]) -> QPushButton:
        button = QPushButton(label)
        button.clicked.connect(lambda: self.run(commands()))
        self.run_buttons.append(button)
        return button

    def tab_steps(self, tab: str) -> List[str]:
        return [s for s in TABS[tab] if s not in self.substeps or self.substeps[s].isChecked()]

    def pipeline_command(self, steps: List[str]) -> List[str]:
        options = [token for form in self.forms.values() for token in form.argv()]
        return [sys.executable, "-m", "BehaviorScreen", self.root.text(), "--steps", *steps, *options]

    def overlay_commands(self) -> List[List[str]]:
        base = [sys.executable, "-m", "BehaviorScreen.overlay", self.root.text(), *self.overlay.argv()]
        return [[*base, "--fish", item.text()] for item in self.fish.selectedItems()]

    def list_fish(self) -> None:
        folder = Path(self.root.text()) / self.overlay.values()["metadata"]
        self.fish.clear()
        self.fish.addItems(sorted(p.stem for p in folder.glob("*.metadata")) if self.root.text() else [])

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
        self.root.editingFinished.connect(self.list_fish)

    def closeEvent(self, event: QCloseEvent) -> None:
        state = {"root": self.root.text(), "steps": {n: b.isChecked() for n, b in self.substeps.items()},
                 "pipeline": {k: v for form in self.forms.values() for k, v in form.values().items()}, "overlay": self.overlay.values()}
        QSettings("BehaviorScreen", "gui").setValue("state", json.dumps(state))
        self.cancel()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    window = Window()
    window.resize(1000, 800)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
