"""Ground slow-walk references using the actual URDF foot collision meshes."""
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np
import torch

from .motion import quat_mul, quat_rotate


class FootGrounding:
    def __init__(self, urdf, fk, clearance):
        self.fk, self.clearance = fk, clearance
        if clearance < 0:
            raise ValueError("Foot clearance must be nonnegative")
        root = ET.parse(urdf).getroot()
        self.feet = []
        for side in ("left", "right"):
            name = side+"_ankle_roll_link"
            link = next(link for link in root.findall("link") if link.get("name") == name)
            collision = link.find("collision")
            origin = collision.find("origin")
            if origin is not None and any(float(x) != 0 for x in (origin.get("xyz", "0 0 0")+" "+origin.get("rpy", "0 0 0")).split()):
                raise ValueError("Grounding currently requires foot collision meshes at the link origin")
            mesh = collision.find("geometry/mesh")
            data = (Path(urdf).parent/mesh.get("filename")).read_bytes()
            count = struct.unpack_from("<I", data, 80)[0]
            if len(data) != 84+50*count:
                raise ValueError("Grounding requires binary STL collision meshes")
            dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
            vertices = np.frombuffer(data, dtype=dtype, offset=84)["vertices"].reshape(-1, 3).copy()
            vertices *= np.array([float(x) for x in mesh.get("scale", "1 1 1").split()])
            vertices = np.unique(vertices, axis=0)
            self.feet.append((fk.key_body_names.index(name), torch.tensor(vertices, device=fk.lower.device)))

    def minimum_height(self, position, quaternion, q):
        local, rotations = self.fk.key_transforms(q)
        minima = []
        for index, vertices in self.feet:
            foot_quat = quat_mul(quaternion, rotations[..., index, :])
            x, y, z, w = foot_quat.unbind(-1)
            zrow = torch.stack((2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1)
            origin = position + quat_rotate(quaternion, local[..., index, :])
            minima.append((zrow @ vertices.T).min(-1).values+origin[..., 2])
        return torch.stack(minima, -1).min(-1).values

    def align(self, position, quaternion, q):
        result = position.clone()
        result[..., 2] += self.clearance-self.minimum_height(position, quaternion, q)
        return result
