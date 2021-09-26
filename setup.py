import os
import platform
from pathlib import Path

from Cython.Build import cythonize
from numpy import get_include
from setuptools import Extension, setup

SYSTEM = platform.system()
LINK = "shared" if SYSTEM == "Linux" else "static"
SDK = Path("sdk") / SYSTEM / LINK
SDK_LEGACY = Path("sdk_legacy") / SYSTEM
LIB = SDK / "lib"
INCLUDE = str(Path("sdk") / "include")

# set env CYTHON_TRACE=1 to enable coverage on .pyx files
CYTHON_TRACE = bool(os.getenv("CYTHON_TRACE", "0") not in ("0", "False"))

nd2file = Extension(
    name="nd2._nd2file",
    sources=["nd2/_nd2file.pyx"],
    libraries=[f"nd2readsdk-{LINK}"],
    library_dirs=[str(LIB)],
    include_dirs=[INCLUDE, get_include()],
    extra_objects=[str(x) for x in LIB.glob("*") if not x.name.startswith(".")],
    define_macros=[("LX_STATIC_LINKING", None), ("CYTHON_TRACE", int(CYTHON_TRACE))],
)

nd2file_legacy = Extension(
    name="nd2._nd2file_legacy",
    sources=["nd2/_nd2file_legacy.pyx"],
    libraries=["nd2sdk"],
    library_dirs=[f"{SDK_LEGACY}/shared/lib"],
    include_dirs=[INCLUDE, get_include()],
)


setup(
    use_scm_version={"write_to": "nd2/_version.py"},
    ext_modules=cythonize(
        [nd2file, nd2file_legacy],
        language_level="3",
        compiler_directives={
            "linetrace": CYTHON_TRACE,
            "c_string_type": "unicode",
            "c_string_encoding": "utf-8",
        },
    ),
)
