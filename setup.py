import os
import platform
from pathlib import Path

from Cython.Build import cythonize
from numpy import get_include
from setuptools import Extension, setup

SYSTEM = platform.system()
LINK = "shared" if SYSTEM == "Linux" else "static"
SDK = Path("sdk") / SYSTEM / LINK
LIB = SDK / "lib"
INCLUDE = SDK / "include"

# set env CYTHON_TRACE=1 to enable coverage on .pyx files
CYTHON_TRACE = bool(os.getenv("CYTHON_TRACE", "0") not in ("0", "False"))


nd2file = Extension(
    name="nd2._nd2file",
    sources=["nd2/_nd2file.pyx"],
    libraries=[f"nd2readsdk-{LINK}"],
    library_dirs=[str(LIB)],
    include_dirs=[str(INCLUDE), get_include()],
    extra_objects=[str(x) for x in LIB.glob("*") if not x.name.startswith(".")],
    define_macros=[("LX_STATIC_LINKING", None), ("CYTHON_TRACE", int(CYTHON_TRACE))],
)

EXTENSIONS = [nd2file]
if SYSTEM == "Darwin":
    nd2file_legacy = Extension(
        name="pynd2._nd2file_legacy",
        sources=["pynd2/_nd2file_legacy.pyx", "pynd2/nd2Reader_helper.c"],
        include_dirs=["sdk_legacy/Darwin/", get_include()],
        extra_link_args=["-F", "sdk_legacy/Darwin", "-framework", "nd2sdk"],
    )
    EXTENSIONS.append(nd2file_legacy)


setup(
    use_scm_version={"write_to": "nd2/_version.py"},
    ext_modules=cythonize(
        EXTENSIONS,
        language_level="3",
        compiler_directives={
            "linetrace": CYTHON_TRACE,
            "c_string_type": "unicode",
            "c_string_encoding": "utf-8",
        },
    ),
)
