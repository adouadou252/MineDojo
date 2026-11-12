"""Stream every environment frame to MP4 and a bounded GIF preview."""
import math
from pathlib import Path
import shutil
import subprocess

from .minecraft_replay import GifReplay


class EpisodeRecorder:
    def __init__(self, directory, max_steps, fps=20, *, label="Minecraft", preview_size=(320, 200), maxrate_kbps=1000):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.label = label
        self.preview_size = preview_size
        self.maxrate_kbps = maxrate_kbps
        self.stride = max(1, math.ceil((max_steps + 1) / 120))
        self.count = 0
        self.process = self.log = self.last = None
        self.preview = []
        self.result = None
        self.temporary = self.directory / "replay.mp4.part"

    def append(self, rgb):
        import numpy as np
        from PIL import Image

        pixels = np.asarray(rgb, dtype=np.uint8)
        if pixels.ndim != 3 or pixels.shape[0] != 3:
            raise ValueError(f"{self.label} returned an unexpected replay RGB frame")
        frame = np.ascontiguousarray(np.moveaxis(pixels, 0, -1))
        height, width = frame.shape[:2]
        if not self.count:
            self.size = [width, height]
            encoder = shutil.which("ffmpeg")
            if not encoder:
                raise RuntimeError(f"ffmpeg is required for the complete {self.label} MP4 replay")
            self.log = (self.directory / "replay-encoding.log").open("wb")
            self.process = subprocess.Popen([encoder, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                "-r", str(self.fps), "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2",
                "-preset", "veryfast", "-crf", "23", "-maxrate", f"{self.maxrate_kbps}k", "-bufsize", f"{self.maxrate_kbps * 2}k",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-f", "mp4", str(self.temporary)],
                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log)
        elif self.size != [width, height]:
            raise ValueError(f"{self.label} replay dimensions changed during a scene")
        try:
            self.process.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError(f"{self.label} MP4 encoding failed; see replay-encoding.log") from error
        image = Image.fromarray(frame)
        image.thumbnail(self.preview_size, Image.Resampling.LANCZOS)
        self.last = (self.count, image)
        if self.count % self.stride == 0:
            self.preview.append(self.last)
            # Full worlds have no maximum step count. Keep the preview bounded
            # even when a run exceeds the 36000-step efficiency reference.
            if len(self.preview) > 120:
                self.stride *= 2
                self.preview = [(i, frame) for i, frame in self.preview if i % self.stride == 0]
        self.count += 1

    def finish(self):
        import numpy as np

        if self.result is not None:
            return self.result
        if not self.count:
            raise ValueError(f"{self.label} replay has no frames")
        self.process.stdin.close()
        if self.process.wait(timeout=60):
            raise RuntimeError(f"{self.label} MP4 encoding failed; see replay-encoding.log")
        self.log.close()
        if self.preview[-1][0] != self.last[0]:
            self.preview.append(self.last)
        with GifReplay(self.directory / "replay.gif", duration_ms=50) as gif:
            for index, (frame_index, image) in enumerate(self.preview):
                next_index = self.preview[index + 1][0] if index + 1 < len(self.preview) else self.count
                gif.append(np.moveaxis(np.asarray(image), -1, 0),
                           duration_ms=max(10, round((next_index - frame_index) * 1000 / self.fps)))
        self.temporary.replace(self.directory / "replay.mp4")
        self.result = {"frames": self.count, "size": self.size, "fps": self.fps,
            "capture": "initial-and-every-action", "video": "replay.mp4", "preview": "replay.gif",
            "preview_frame_indices": [index for index, _ in self.preview],
            "timeline": "environment actions; model waiting time is excluded"}
        self.preview.clear()
        self.last = None
        return self.result

    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback):
        try:
            if error_type is None:
                self.finish()
        finally:
            if self.process and self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            if self.process and self.process.stdin and not self.process.stdin.closed:
                try:
                    self.process.stdin.close()
                except OSError:
                    pass  # The encoder may already have closed the pipe.
            if self.log:
                self.log.close()
            if self.result is None:
                self.temporary.unlink(missing_ok=True)
            self.preview.clear()
            self.last = None
