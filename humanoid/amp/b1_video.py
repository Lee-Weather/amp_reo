"""Isaac Gym camera capture for a fixed B1 student, without policy changes."""
import hashlib
from pathlib import Path
import subprocess

import numpy as np


class B1Video:
    def __init__(self, env, output, width=1280, height=720):
        from isaacgym import gymapi
        import cv2
        self.env, self.api, self.cv2 = env, gymapi, cv2
        self.width, self.height = width, height
        self.frames, self.closed = 0, False
        if env.camera_handle is None:
            raise ValueError('B1 camera sensor is unavailable')
        self.encoders, self.paths = [], []
        self.fps = 1/float(env.dt)
        for name, fps in [('play_output.mp4', self.fps), ('play_slow_025x.mp4', self.fps/4)]:
            path = Path(output)/name
            if path.exists(): raise FileExistsError(path)
            process = subprocess.Popen(['ffmpeg', '-n', '-loglevel', 'error', '-f', 'rawvideo',
                '-pix_fmt', 'bgr24', '-s', '%dx%d' % (width, height), '-r', str(fps), '-i', '-', '-an',
                '-c:v', 'libx264', '-threads', '2', '-preset', 'fast', '-crf', '19', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart', str(path)], stdin=subprocess.PIPE)
            self.encoders.append(process); self.paths.append(path)

    def capture(self, step, root, joint_error=0., root_error=0., reason=None):
        env, api = self.env, self.api
        # Follow translation only; camera orientation does not follow robot pitch/roll.
        origin = env.env_origins[0].cpu().numpy()
        position = np.asarray(root[:3])+origin
        eye = position+np.array([2.2, -2.2, 1.0])
        target = position+np.array([0., 0., .10])
        env.gym.set_camera_location(env.camera_handle, env.envs[0], api.Vec3(*eye), api.Vec3(*target))
        if step > 0: env.gym.fetch_results(env.sim, True)
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        rgba = env.gym.get_camera_image(env.sim, env.envs[0], env.camera_handle, api.IMAGE_COLOR)
        pixels = self.cv2.cvtColor(np.asarray(rgba, dtype=np.uint8).reshape(self.height, self.width, 4), self.cv2.COLOR_RGBA2BGR)
        lines = ['X1 | exp2.14 model_200 | student only | seed5',
            'Physics t=%.2fs | step=%d | joint RMSE=%.3frad | root error=%.3fm' %
            (step*float(env.dt), step, joint_error, root_error)]
        if reason: lines.append('STOP: '+reason)
        for speed, encoder in zip(('1x REAL TIME', '0.25x SLOW MOTION'), self.encoders):
            image = pixels.copy()
            self.cv2.rectangle(image, (0, 0), (self.width, 126 if reason else 96), (25, 25, 25), -1)
            for index, line in enumerate([speed]+lines):
                self.cv2.putText(image, line, (18, 26+29*index), self.cv2.FONT_HERSHEY_SIMPLEX,
                    .57, (230, 230, 230) if index < 3 else (70, 100, 255), 1, self.cv2.LINE_AA)
            encoder.stdin.write(image.tobytes())
        self.frames += 1

    def close(self):
        if self.closed: return self.receipts
        for encoder in self.encoders: encoder.stdin.close()
        codes = [encoder.wait(timeout=60) for encoder in self.encoders]
        if any(codes): raise RuntimeError('B1 video encoding failed')
        self.closed = True
        self.receipts = {path.name: dict(bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), frames=self.frames,
            fps=self.fps/(4 if 'slow' in path.name else 1)) for path in self.paths}
        return self.receipts
