from __future__ import annotations

from itertools import count
import os
import signal
import subprocess
import sys
import threading
from typing import Callable, TypeVar

from typing_extensions import ParamSpec

P = ParamSpec('P')
T = TypeVar('T')



def try_kill_proc(proc: subprocess.Popen):
    if proc:
        try:
            kill_proc(proc)
        except ProcessLookupError:
            pass
        proc.was_terminated = True  # type: ignore[attr-defined]


def kill_proc(proc: subprocess.Popen):
    if sys.platform == "win32":
        # terminate would not kill process opened by the shell cmd.exe,
        # it will only kill cmd.exe leaving the child running
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        subprocess.Popen(
            "taskkill /PID %d /T /F" % proc.pid,
            startupinfo=startupinfo)
    else:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.terminate()


def proc_has_been_killed(proc: subprocess.Popen) -> bool:
    return getattr(proc, "was_terminated", False)


def run_on_new_thread(fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> threading.Thread:
    thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
    thread.start()
    return thread


class Counter:
    """Thread-safe, lockless counter.

    Implementation idea from @grantjenks
    https://github.com/jd/fastcounter/issues/2#issue-548504668
    """
    def __init__(self):
        self._incs = count()
        self._decs = count()

    def inc(self):
        next(self._incs)

    def dec(self):
        next(self._decs)

    def count(self) -> int:
        return next(self._incs) - next(self._decs)

    def next(self) -> int:
        self.inc()
        return self.count()


def inflate(d: dict) -> dict:
    """
    Transforms a flat dictionary with dot-separated keys into a nested
    dictionary.

    Example:
        >>> inflate({'a.b.c': 1, 'a.b.d': 2, 'e': 3})
        {'a': {'b': {'c': 1, 'd': 2}}, 'e': 3}
    """
    rv: dict = {}
    for key, value in d.items():
        parts = key.split(".")
        edge = rv
        for part in parts[:-1]:
            edge = edge.setdefault(part, {})
        edge[parts[-1]] = value
    return rv


def read_path(d: dict, path: str, /, default=None) -> object:
    """
    Retrieve a value from a nested dictionary using a dot-separated path.

    Returns the value found at the specified path, or the default value
    if the path is not found.

    Examples:
        >>> read_path({'a': {'b': {'c': 1}}}, "a.b.c")
        1
        >>> read_path({'a': {'b': {'c': 1}}}, "a.b.d")
        None
        >>> read_path({'a': {'b': {'c': 1}}}, "a.b.d", default="default")
        "default"
    """
    parts = path.split(".")
    edge = d
    for part in parts:
        try:
            edge = edge[part]
        except KeyError:
            return default
    return edge

