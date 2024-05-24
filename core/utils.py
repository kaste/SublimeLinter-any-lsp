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



def try_kill_proc(proc):
    if proc:
        try:
            kill_proc(proc)
        except ProcessLookupError:
            pass
        proc.got_killed = True


def kill_proc(proc):
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


def proc_has_been_killed(proc):
    return getattr(proc, "got_killed", False)


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


def unflatten(d: dict) -> dict:
    rv: dict = {}
    for key, value in d.items():
        parts = key.split(".")
        edge = rv
        for part in parts[:-1]:
            edge = edge.setdefault(part, {})
        edge[parts[-1]] = value
    return rv


def read_path(d: dict, path: str) -> object:
    parts = path.split(".")
    edge = d
    for part in parts:
        edge = edge.get(part, None)
        if edge is None:
            return None  # type: ignore[unreachable]
    return edge

