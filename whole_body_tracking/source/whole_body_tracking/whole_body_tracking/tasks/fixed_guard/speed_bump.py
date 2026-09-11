"""Trapezoidal speed-bump strips for T800 FG72 stage-6 terrain."""

from __future__ import annotations

import numpy as np
import trimesh
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass


def make_trapezoid_speed_bump(
    *,
    bottom: float,
    top: float,
    height: float,
    length: float,
    center: tuple[float, float, float],
    yaw: float = 0.0,
) -> trimesh.Trimesh:
    """Road hump: trapezoid in the travel plane, long along ``length``."""
    hb = 0.5 * float(bottom)
    ht = 0.5 * float(top)
    hl = 0.5 * float(length)
    verts = np.array(
        [
            [-hb, -hl, 0.0],
            [hb, -hl, 0.0],
            [hb, hl, 0.0],
            [-hb, hl, 0.0],
            [-ht, -hl, height],
            [ht, -hl, height],
            [ht, hl, height],
            [-ht, hl, height],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ],
        dtype=np.int64,
    )
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    transform = trimesh.transformations.compose_matrix(
        translate=center, angles=(0.0, 0.0, yaw)
    )
    mesh.apply_transform(transform)
    return mesh


def speed_bump_strips_terrain(difficulty: float, cfg: "SpeedBumpStripsTerrainCfg"):
    """Parallel trapezoidal humps around a flat reset pad."""
    del difficulty
    from isaaclab.terrains.trimesh.utils import make_plane

    meshes = [make_plane(cfg.size, height=0.0, center_zero=False)]
    sx, sy = float(cfg.size[0]), float(cfg.size[1])
    pad = float(cfg.spawn_pad)
    spacing = float(getattr(cfg, "spacing", 1.7))
    cx, cy = 0.5 * sx, 0.5 * sy

    def _axis_positions(limit: float, center: float) -> list[float]:
        out = []
        x = pad
        while x <= limit - pad + 1e-6:
            if abs(x - center) >= pad:
                out.append(float(x))
            x += spacing
        return out

    # Retreat along x crosses these Y-long trapezoids (350 mm base, 70 mm high).
    for x in _axis_positions(sx, cx):
        meshes.append(
            make_trapezoid_speed_bump(
                bottom=cfg.bottom,
                top=cfg.top,
                height=cfg.height,
                length=max(1.2, sy - 0.4),
                center=(x, cy, 0.0),
                yaw=0.0,
            )
        )
    # After a yaw, retreat along y still meets the same hump profile.
    for y in _axis_positions(sy, cy)[::2]:
        meshes.append(
            make_trapezoid_speed_bump(
                bottom=cfg.bottom,
                top=cfg.top,
                height=cfg.height,
                length=max(1.2, sx - 0.4),
                center=(cx, y, 0.0),
                yaw=0.5 * np.pi,
            )
        )
    origin = np.asarray((cx, cy, 0.0))
    return meshes, origin


@configclass
class SpeedBumpStripsTerrainCfg(SubTerrainBaseCfg):
    """70 mm trapezoidal speed bump: 350 mm base, 100 mm crown."""

    function = speed_bump_strips_terrain
    bottom: float = 0.35
    top: float = 0.10
    height: float = 0.07
    num_strips: int = 4
    spawn_pad: float = 1.5
    spacing: float = 1.7
