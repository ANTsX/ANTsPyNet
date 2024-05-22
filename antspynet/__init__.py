import importlib
import pkgutil

modules = ['architectures', 'utilities']

# Initialize an empty list to hold all functions
__all__ = []

def import_submodules(package_name):
    package = importlib.import_module(package_name)
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
        module = importlib.import_module(name)
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)
            if callable(attribute):
                globals()[attribute_name] = attribute
                __all__.append(attribute_name)

# Import all submodules and their functions
for x in range(len(modules)):
    import_submodules(modules[x])
