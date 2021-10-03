from wurlitzer import pipes

with pipes():
    from . import latest, v9

__all__ = ["latest", "v9"]
