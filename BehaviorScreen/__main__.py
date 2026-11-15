"""
The pipeline's stages, each with its own options (python -m BehaviorScreen STAGE -h):
    python -m BehaviorScreen STAGE [options]
"""
import ast
import importlib
import importlib.util
import sys
from pathlib import Path

STAGES = {
    "preprocess": "BehaviorScreen.preprocessing.submit",
    "merge": "BehaviorScreen.screen.merge_screen",
    "fit-controls": "BehaviorScreen.screen.control_models.submit",
    "test-lines": "BehaviorScreen.screen.line_tests.submit",
}


def summary(module: str) -> str:
    """The first line of the module's docstring, read without importing it."""
    return ast.get_docstring(ast.parse(Path(importlib.util.find_spec(module).origin).read_text())).splitlines()[0].rstrip(":.")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in STAGES:
        print(__doc__.strip() + "\n\n" + "\n".join(f"  {stage:<12}  {summary(module)}" for stage, module in STAGES.items()))
        raise SystemExit(0 if sys.argv[1:] in ([], ["-h"], ["--help"]) else 2)
    stage = sys.argv.pop(1)
    sys.argv[0] = f"python -m BehaviorScreen {stage}"
    importlib.import_module(STAGES[stage]).main()


if __name__ == "__main__":
    main()
