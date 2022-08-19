import os
import platform
from pathlib import Path

from Cython.Build import cythonize
from numpy import get_include
from setuptools import Extension, setup

SYSTEM = platform.system()
PLATFORM = platform.machine().replace("AMD64", "x86_64")
if "arm64" in os.getenv("_PYTHON_HOST_PLATFORM", ""):
    # e.g. macosx-arm builds
    PLATFORM = "arm64"
SDK = Path("src/sdk") / SYSTEM / PLATFORM
LIB = SDK / "lib"
INCLUDE = SDK / "include"
LINK = "shared" if SYSTEM == "Linux" else "static"
# set env CYTHON_TRACE=1 to enable coverage on .pyx files
CYTHON_TRACE = os.getenv("CYTHON_TRACE", "0") not in ("0", "False")


sdk = Extension(
    name="nd2._sdk.latest",
    sources=["src/nd2/_sdk/latest.pyx"],
    libraries=[f"nd2readsdk-{LINK}"],
    library_dirs=[str(LIB)],
    runtime_library_dirs=[str(LIB)] if SYSTEM == "Linux" else [],
    include_dirs=[str(INCLUDE), get_include()],
    extra_objects=[str(x) for x in LIB.glob("*") if not x.name.startswith(".")],
    define_macros=[("LX_STATIC_LINKING", None), ("CYTHON_TRACE", int(CYTHON_TRACE))],
    # extra_link_args=[
    #     "-ltiff",
    #     "-lz",
    #     "-ljpeg",
    #     "-llzma",
    #     "-ljbig",
    #     "-ltiffxx",
    #     # "-lm",
    #     # "-lstdc++fs",
    # ],
)


setup(
    use_scm_version={"write_to": "src/nd2/_version.py"},
    ext_modules=cythonize(
        [sdk],
        language_level="3",
        compiler_directives={
            "linetrace": CYTHON_TRACE,
            "c_string_type": "unicode",
            "c_string_encoding": "utf-8",
        },
    ),
)
