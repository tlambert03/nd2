from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from nd2._chunkmap import iter_chunks


def get_nd2_stats(path: Path):
    print(path)
    with open(path, "rb") as fh:
        if fh.read(4) != b"\xda\xce\xbe\n":
            return (path.name, {})
        fh.seek(0)
        data = {"xml": [], "other": [], "seqCount": 0}
        for name, start, length in iter_chunks(fh):
            if name.startswith("ImageDataSeq"):
                data["seqCount"] += 1  # type:ignore
                continue
            fh.seek(start)
            if name == "ND2 FILE SIGNATURE CHUNK NAME01!":
                data["ver"] = fh.read(length).split(b"\x00", 1)[0].decode()
                continue
            h = fh.read(5)
            if h == b"<?xml":
                data["xml"].append((name, start, length))  # type:ignore
            else:
                data["other"].append(  # type:ignore
                    (name, h.hex().upper(), start, length)
                )

        return (path.name, data)


if __name__ == "__main__":
    DATA = Path(__file__).parent.parent / "tests" / "data"
    with ThreadPoolExecutor() as exc:
        results = dict(exc.map(get_nd2_stats, DATA.glob("*.nd2")))
    with open("ndchunks.json", "w") as fh:
        import json

        json.dump(results, fh)
