import importlib
modules = ['architectures', 'utilities']
# Import each module dynamically
for module in modules:
    importlib.import_module(f'.{module}', package=__name__)

# Define the public API
__all__ = ['architectures', 'utilities']
