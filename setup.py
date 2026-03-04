from __future__ import annotations

import os
import sys
from pathlib import Path
import subprocess

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup


def _find_libomp_prefix() -> str | None:
    env_prefix = os.environ.get("LIBOMP_PREFIX")
    if env_prefix:
        p = Path(env_prefix)
        if (p / "include").exists() and (p / "lib").exists():
            return str(p)
    try:
        out = subprocess.check_output(["brew", "--prefix", "libomp"], stderr=subprocess.DEVNULL, text=True)
    except Exception:
        return None
    prefix = out.strip()
    if not prefix:
        return None
    p = Path(prefix)
    if (p / "include").exists() and (p / "lib").exists():
        return str(p)
    return None


def _openmp_config() -> tuple[list[str], list[str], bool]:
    setting = os.environ.get("PYDESEQ2_OPENMP", "auto").strip().lower()
    force_enable = setting in {"1", "true", "on", "yes"}
    force_disable = setting in {"0", "false", "off", "no"}
    if force_disable:
        return [], [], False

    compile_args: list[str] = []
    link_args: list[str] = []

    if sys.platform == "darwin":
        # On macOS, enable OpenMP only when explicitly requested or libomp is discoverable.
        libomp_prefix = _find_libomp_prefix()
        if (not force_enable) and (libomp_prefix is None):
            return [], [], False
        compile_args.extend(["-Xpreprocessor", "-fopenmp"])
        link_args.append("-lomp")
        if libomp_prefix is not None:
            compile_args.append(f"-I{libomp_prefix}/include")
            link_args.append(f"-L{libomp_prefix}/lib")
        return compile_args, link_args, True

    # Linux and other Unix-like systems.
    compile_args.append("-fopenmp")
    link_args.append("-fopenmp")
    return compile_args, link_args, True


def _build_extensions() -> list[Extension]:
    omp_compile_args, omp_link_args, _ = _openmp_config()
    compile_args = ["-O3"] + omp_compile_args
    link_args = list(omp_link_args)
    extensions = [
        Extension(
            "pydeseq2._core_cy",
            sources=["src/pydeseq2/_core_cy.pyx"],
            include_dirs=[np.get_include()],
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            optional=True,
        )
    ]
    return cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "initializedcheck": False,
        },
        force=True,
    )


setup(ext_modules=_build_extensions())
