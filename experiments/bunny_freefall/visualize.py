#!/home/bowen/miniconda3/envs/env_gipc/bin/python
"""Play saved PolyFEM surface frames in Polyscope; --screenshots exports key frames."""
import argparse
from pathlib import Path
import time
import meshio
import numpy as np
import polyscope as ps
import polyscope.imgui as ui

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--screenshots', action='store_true')
args = parser.parse_args()
data = np.load(ROOT / 'trajectory.npz')
positions, faces, ids = data['positions'], data['faces'], data['body_ids']
ps.init()
ps.set_up_dir('y_up')
ps.set_ground_plane_mode('none')
ps.set_window_size(1200, 900)
ps.set_background_color((0.94, 0.95, 0.97))
meshes = []
for bid, name, color in [(1, 'Ground', (0.55, 0.60, 0.65)), (2, 'Bunny', (0.87, 0.48, 0.23))]:
    face = faces[np.all(ids[faces] == bid, axis=1)]
    vertices, inverse = np.unique(face, return_inverse=True)
    surface = ps.register_surface_mesh(name, positions[0, vertices], inverse.reshape(-1, 3), color=color, smooth_shade=True)
    meshes.append((surface, vertices))
ps.set_automatically_compute_scene_extents(False)
ps.set_length_scale(2.)
ps.set_bounding_box(np.array([-1.,0.,-1.]), np.array([1.,1.5,1.]))
ps.look_at((2.1, 1.5, 2.6), (-0.1, 0.55, 0.0))
frame, playing, last, speed = 0, False, time.monotonic(), 1.
def update(i):
    for surface, vertices in meshes:
        surface.update_vertex_positions(positions[i, vertices])
def callback():
    global frame, playing, last, speed
    ui.TextUnformatted('PolyFEM: bunny free fall')
    ui.TextUnformatted('E = 10000 Pa | nu = 0.49 | dt = 0.01 s')
    _, playing = ui.Checkbox('Play', playing)
    changed, frame = ui.SliderInt('Frame', frame, 0, len(positions)-1)
    _, speed = ui.SliderFloat('Playback speed', speed, 0.1, 2.)
    if ui.Button('Restart'):
        frame, changed = 0, True
    now = time.monotonic()
    if playing and now-last >= 0.01/speed:
        frame = (frame+max(1,int((now-last)*speed/.01))) % len(positions)
        last, changed = now, True
    elif not playing:
        last = now
    if changed:
        update(frame)
    ui.TextUnformatted(f't = {frame*.01:.2f} s / 2.00 s')
ps.set_user_callback(callback)
if args.screenshots:
    for i in sorted(set((0, min(50, len(positions)-1), len(positions)-1))):
        frame = i
        update(i)
        ps.frame_tick()
        ps.screenshot(str(ROOT / f'frame_{i:03d}.png'))
else:
    ps.show()
