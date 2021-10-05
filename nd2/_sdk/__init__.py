from wurlitzer import pipes

with pipes():
    from . import latest, v9  # type: ignore


__all__ = ["latest", "v9"]
