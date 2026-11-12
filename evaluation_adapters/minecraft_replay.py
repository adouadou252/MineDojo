"""Stream an initial frame and every action result to a bounded-memory GIF."""
from pathlib import Path


class GifReplay:
    def __init__(self, path, duration_ms=100):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = self.path.with_suffix(".gif.part")
        self.stream = self.temporary.open("wb")
        self.duration_ms = duration_ms
        self.frames = 0
        self.previous = None

    def append(self, rgb, duration_ms=None):
        import numpy as np
        from PIL import GifImagePlugin, Image, ImageChops

        pixels = np.asarray(rgb, dtype=np.uint8)
        if pixels.ndim != 3 or pixels.shape[0] != 3:
            raise ValueError("MineDojo returned an unexpected replay RGB frame")
        current = Image.fromarray(np.moveaxis(pixels, 0, -1))
        if self.previous is not None and current.size != self.previous.size:
            raise ValueError("Minecraft replay dimensions changed during a scene")
        if self.previous is None:
            bounds = (0, 0, *current.size)
        else:
            # Keep a real GIF frame even when nothing changed: one action must
            # remain one replay frame, including invalid and stationary actions.
            bounds = ImageChops.difference(current, self.previous).getbbox() or (0, 0, 1, 1)
        frame = current.crop(bounds).quantize(colors=256, method=Image.Quantize.FASTOCTREE)
        if self.frames == 0:
            header, _ = GifImagePlugin.getheader(frame, info={"loop": 0, "optimize": False})
            for block in header:
                self.stream.write(block)
        for block in GifImagePlugin.getdata(frame, offset=bounds[:2], duration=duration_ms or self.duration_ms,
                                          disposal=1, include_color_table=self.frames > 0):
            self.stream.write(block)
        self.previous = current
        self.frames += 1

    def metadata(self):
        return {"frames": self.frames, "frame_duration_ms": self.duration_ms,
                "capture": "initial-and-every-action", "size": list(self.previous.size)}

    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback):
        try:
            if error_type is None and self.frames:
                self.stream.write(b";")
        finally:
            self.stream.close()
        if error_type is None and self.frames:
            self.temporary.replace(self.path)
        else:
            self.temporary.unlink(missing_ok=True)
        self.previous = None
