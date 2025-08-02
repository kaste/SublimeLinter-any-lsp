import sys


# kiss-reloader:
prefix = __spec__.parent + "."  # don't clear the base package
for module_name in [
    module_name
    for module_name in sys.modules
    if module_name.startswith(prefix) and module_name != __name__
]:
    del sys.modules[module_name]


from .core.server import *  # noqa: F403
