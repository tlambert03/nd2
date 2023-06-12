from setuptools import setup

setup(
    name="nd2",
    use_scm_version={"write_to": "src/nd2/_version.py"},
    install_requires=[
        "resource-backed-dask-array",
        "typing-extensions",
        "numpy>=1.14.5",
    ],
)
