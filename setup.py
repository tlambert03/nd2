import os
import platform
from pathlib import Path

from Cython.Build import cythonize
from numpy import get_include
from setuptools import Extension, setup

SYSTEM = platform.system()
SDK = Path("sdk") / "latest" / SYSTEM
SDK_LEGACY = Path("sdk") / "v9" / SYSTEM
LIB = SDK / "lib"
LINK = "shared" if SYSTEM == "Linux" else "static"

# set env CYTHON_TRACE=1 to enable coverage on .pyx files
CYTHON_TRACE = bool(os.getenv("CYTHON_TRACE", "0") not in ("0", "False"))

nd2file = Extension(
    name="nd2._nd2file",
    sources=["nd2/_nd2file.pyx"],
    libraries=[f"nd2readsdk-{LINK}"],
    library_dirs=[str(SDK / "lib")],
    include_dirs=[str(SDK / "include"), get_include()],
    extra_objects=[str(x) for x in LIB.glob("*") if not x.name.startswith(".")],
    define_macros=[("LX_STATIC_LINKING", None), ("CYTHON_TRACE", int(CYTHON_TRACE))],
)

nd2file_legacy = Extension(
    name="nd2._nd2file_legacy",
    sources=["nd2/_nd2file_legacy.pyx"],
    libraries=["v6_w32_nd2ReadSDK" if SYSTEM == "Windows" else "nd2ReadSDK"],
    library_dirs=[str(SDK_LEGACY / "lib")],
    include_dirs=[str(SDK_LEGACY / "include"), get_include()],
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
