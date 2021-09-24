import os
import platform
from distutils.command import build_ext
from pathlib import Path

from Cython.Build import cythonize
from numpy import get_include
from setuptools import Extension, setup

SYSTEM = platform.system()
LINK = "static"
SDK = Path("sdk") / SYSTEM / LINK
LIB = SDK / "lib"
INCLUDE = SDK / "include"

# os.environ["MACOSX_DEPLOYMENT_TARGET"] = "11.0"

if os.name == "nt":

    # fix LINK : error LNK2001: unresolved external symbol PyInit___init__
    # Patch from: https://bugs.python.org/issue35893

    def get_export_symbols(self, ext):
        """
        Slightly modified from:
        https://github.com/python/cpython/blob/8849e5962ba481d5d414b3467a256aba2134b4da\
        /Lib/distutils/command/build_ext.py#L686-L703
        """
        parts = ext.name.split(".")
        suffix = parts[-2] if parts[-1] == "__init__" else parts[-1]
        # from here on unchanged
        try:
            # Unicode module name support as defined in PEP-489
            # https://www.python.org/dev/peps/pep-0489/#export-hook-name
            suffix.encode("ascii")
        except UnicodeEncodeError:
            suffix = "U" + suffix.encode("punycode").replace(b"-", b"_").decode("ascii")

        initfunc_name = "PyInit_" + suffix
        if initfunc_name not in ext.export_symbols:
            ext.export_symbols.append(initfunc_name)
        return ext.export_symbols

    build_ext.build_ext.get_export_symbols = get_export_symbols  # type: ignore

nd2file = Extension(
    name="nd2._nd2file",
    sources=["nd2/_nd2file.pyx"],
    libraries=["nd2readsdk-static"],
    library_dirs=[str(LIB)],
    include_dirs=[str(INCLUDE), get_include()],
    extra_objects=[str(x) for x in LIB.glob("*")],
    define_macros=[("LX_STATIC_LINKING", None)],
)


setup(
    use_scm_version={"write_to": "nd2/_version.py"},
    ext_modules=cythonize([nd2file], language_level="3"),
)
