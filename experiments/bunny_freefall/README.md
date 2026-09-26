Current experiment: both bunnies raised by +0.5 m on Y; ground remains y=0.15. Initial clearance = 0.549254 m. Previous run saved in archive_height_0/.

# Single bunny free fall in PolyFEM

Run `./run.sh`, then `./view.sh`. The viewer provides play/pause, a frame slider,
restart, and playback speed. `./view.sh --screenshots` exports frames 0, 50, 200.

Reference: `/home/bowen/DiffIPC-data-original/json_scripts/fig1_bunnies/run.json`.
One bunny (the first) and the fixed flat-pool ground are retained. Initial
velocity is zero for release from rest. E=10000 Pa, nu=0.49, dt=0.01 s,
200 steps (2 s). Other reference parameters: NeoHookean, rho=1240,
scale=6, translation [0,0.5,0], BDF1, P1, quadrature order=5, no friction,
dhat=0.001, barrier stiffness=100000, RHS=[0,9.81,0]. PolyFEM's RHS sign
convention produces downward acceleration, verified from step 1.
The initial bunny-to-ground vertical gap is approximately 0.549254 m.

The final run uses `/home/bowen/polyfem-diffipc-original/build-make/PolyFEM_bin`
and the original nonlinear tolerance fields (grad_norm=1e-6).

Contact compatibility: `use_convergent_formulation=true`, matching the local
aligned fig1 configuration at
`/home/bowen/polyfem-original/agent_check/fig1_fd/case/run.json`.
This is an explicit difference from the raw published JSON, which omits that
flag (default false). Raw nonconvergent contact made negligible progress at
step 23 with gaps near 1e-11; those attempts are retained separately.
The final run retains fixed barrier stiffness 1e5, dhat 1e-3, and zero friction.

Outputs: `result/sim.pvd`, per-step volume and surface VTU files,
`run.log`, compact `trajectory.npz`, and `validation.json`.
Viewer geometry uses reference VTU points plus displacement, exactly once.
