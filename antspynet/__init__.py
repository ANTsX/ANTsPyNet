# mypackage/__init__.py

import importlib
import pkgutil
import sys

# Initialize an empty list to hold all functions
__all__ = []

def import_submodules(package_name):
    try:
        package = importlib.import_module(package_name)
    except ModuleNotFoundError as e:
        print(f"Error importing package {package_name}: {e}", file=sys.stderr)
        return
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as e:
            print(f"Error importing module {name}: {e}", file=sys.stderr)
            continue
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)
            if callable(attribute):
                globals()[attribute_name] = attribute
                __all__.append(attribute_name)

# Import all submodules and their functions
import_submodules('antspynet.architectures')
import_submodules('antspynet.utilities')
