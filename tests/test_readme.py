import os
from pathlib import Path

import pytest

README = Path(__file__).parent.parent / "README.md"
SAMPLE = Path(__file__).parent / "data" / "dims_c2y32x32.nd2"


@pytest.mark.skipif(os.name == "nt", reason="paths annoying on windows")
def test_readme(capsys):
    code = README.read_text().split("```python")[1].split("```")[0]
    code = code.replace("some_file.nd2", str(SAMPLE.absolute()))
    exec(code)
    captured = capsys.readouterr()
    assert captured.out.startswith("Metadata(")  # because of the print statement
