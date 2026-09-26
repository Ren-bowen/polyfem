"""Validate and compact all 201 surface states for Polyscope playback."""
from pathlib import Path
import json
import meshio
import numpy as np
root = Path(__file__).resolve().parent
positions = []
for step in range(201):
    m = meshio.read(root / 'result' / f'step_{step}_surf.vtu')
    p = m.points + m.point_data.get('displacement', m.point_data['solution'])
    assert np.isfinite(p).all(), step
    if step == 0:
        faces = m.cells_dict['triangle']
        ids = m.point_data['body_ids'].ravel().astype(int)
    positions.append(p)
x = np.array(positions)
np.savez_compressed(root/'trajectory.npz', positions=x, faces=faces, body_ids=ids, dt=.01)
bunny = x[:,ids==2]; ground = x[:,ids==1]
summary = {'steps':200,'saved_states':len(x),'dt':.01,'duration':2.,'initial_bunny_min_y':float(bunny[0,:,1].min()),'final_bunny_min_y':float(bunny[-1,:,1].min()),'minimum_bunny_y':float(bunny[:,:,1].min()),'ground_top_y':float(ground[0,:,1].max()),'max_ground_displacement':float(np.abs(ground-ground[0]).max()),'first_step_mean_y_displacement':float((bunny[1]-bunny[0])[:,1].mean())}
(root/'validation.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
