import sys
import pathlib
import inspect
import importlib
from typing import Type


def find_importable_classes(folder_route: str, base_class: Type) -> dict[str, Type]:
    candidates = {}
    folder_path = pathlib.Path(folder_route).resolve()
    project_root = folder_path.parents[0]

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    for py_file in folder_path.rglob("*.py"):
        rel_path = py_file.relative_to(project_root).with_suffix("")
        module_name = ".".join(rel_path.parts)
        try:
            module = importlib.import_module(module_name)
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, base_class) and obj is not base_class:
                    candidates[obj.__module__.split(".")[1]] = obj
        except Exception as _:
            continue

    return candidates
