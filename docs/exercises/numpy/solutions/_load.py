import importlib.util
from pathlib import Path


def load(solution_file):
    here = Path(solution_file).resolve()
    spec = importlib.util.spec_from_file_location(f"exercise_{here.stem}", here.parent.parent / here.name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
