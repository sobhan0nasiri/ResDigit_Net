import pkgutil
import importlib
from pathlib import Path

__path__ = [str(Path(__file__).parent)]

for loader, module_name, is_pkg in pkgutil.iter_modules(__path__):

    module = importlib.import_module(f".{module_name}", package=__name__)
    
    attrs = [attr for attr in dir(module) if not attr.startswith('_')]
    globals().update({attr: getattr(module, attr) for attr in attrs})

    if "__all__" not in globals():
        globals()["__all__"] = []
    globals()["__all__"].extend(attrs)