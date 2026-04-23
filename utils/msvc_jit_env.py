"""
ColBERT JIT-compiles ``segmented_maxsim_cpp`` via PyTorch. On Windows, ``cl.exe`` must be
on PATH. Notebooks started from Cursor or plain PowerShell usually do not run ``vcvars64.bat``,
so ``where cl`` fails. This module calls Visual Studio's ``vcvars64.bat`` and merges the
resulting environment into ``os.environ`` for the current Python process.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Optional


def _cl_on_path() -> bool:
    return shutil.which("cl") is not None


def _vcvars64_bat() -> Optional[str]:
    if sys.platform != "win32":
        return None

    vswhere = os.path.expandvars(
        r"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
    )
    if os.path.isfile(vswhere):
        try:
            root = (
                subprocess.check_output(
                    [
                        vswhere,
                        "-latest",
                        "-products",
                        "*",
                        "-requires",
                        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                        "-property",
                        "installationPath",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                .strip()
                .splitlines()
            )
            if root:
                cand = os.path.join(
                    root[0], "VC", "Auxiliary", "Build", "vcvars64.bat"
                )
                if os.path.isfile(cand):
                    return cand
        except (subprocess.CalledProcessError, OSError, IndexError):
            pass

    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    for edition in ("BuildTools", "Community", "Professional", "Enterprise", "Preview"):
        cand = os.path.join(
            program_files,
            "Microsoft Visual Studio",
            "2022",
            edition,
            "VC",
            "Auxiliary",
            "Build",
            "vcvars64.bat",
        )
        if os.path.isfile(cand):
            return cand
    return None


def ensure_msvc_for_torch_jit() -> None:
    """
    No-op on non-Windows. On Windows, if ``cl`` is missing, run ``vcvars64.bat`` and copy
    env vars into ``os.environ``. Raises ``RuntimeError`` if MSVC still cannot be found.
    """
    if sys.platform != "win32":
        return
    if _cl_on_path():
        return

    vcvars = _vcvars64_bat()
    if not vcvars:
        raise RuntimeError(
            "MSVC (cl.exe) not found and vcvars64.bat could not be located. "
            "Install 'Desktop development with C++' in Visual Studio 2022 or Build Tools, "
            "or start Jupyter from 'x64 Native Tools Command Prompt for VS 2022'."
        )

    # Use shell=True: a single argv to ``cmd /c`` with embedded quotes is mangled by
    # ``list2cmdline`` (inner ``"`` become ``\\\"``), so ``call`` breaks and exits 1.
    cmd_exe = os.environ.get("COMSPEC", "cmd.exe")
    proc = subprocess.run(
        f'call "{vcvars}" && set',
        shell=True,
        executable=cmd_exe,
        capture_output=True,
        text=True,
        errors="replace",
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()[:1200]
        raise RuntimeError(
            f"vcvars64.bat failed (exit {proc.returncode}). {err}\n"
            "Install the MSVC x64 toolset in Visual Studio Build Tools (C++ workload)."
        )

    for line in proc.stdout.splitlines():
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if key:
            os.environ[key] = val

    if not _cl_on_path():
        raise RuntimeError(
            "After vcvars64.bat, cl.exe is still not on PATH. "
            "Repair the C++ workload in Visual Studio Installer."
        )
