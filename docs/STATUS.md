# What is validated, and what is not

Every number here was measured by a command in this repository, on
2026-11-06, on an Apple M-series Mac (Metal, macOS 14.5) with OpenMM 8.4.
Re-run them with the three lanes: **`npm run test:ci`** (12 s, no GPU and no
Python - the host suite, the OpenMM reference comparison, the gradients and the
triclinic oracles), **`npm run test:gpu`** (the fifteen kernel checks, and
`test:gpu:page` for the same fifteen in Chrome), and **`npm run test:slow`**
(9 minutes - the NPT unit cell, the thermostat against its setpoint across
timesteps, the drift's timestep scaling, and the live metrics against
MDAnalysis). `npm run test:page:all` drives the nine browser lanes.

🔴 **ONLY THE FIRST LANE RUNS IN CI, BECAUSE HOSTED RUNNERS HAVE NO GPU.** That
is under half the suite and `.github/workflows/ci.yml` says so rather than
implying a green tick means more than it does. It does catch the defect class
that has broken this repository seven times - a backtick in a WGSL template
literal - because `npm test` parse-checks the whole tree.

## Validated

| What | Against | Result |
|---|---|---|
| `prmtop` parse: 43 invariants (counts, charge/mass sums, force constants, index sums, box) | **ParmEd**, independent reader | 43/43 exact, worst 2.1e-14 |
| constraint and exception construction | **OpenMM** `createSystem`, 3 configurations | counts identical (1176 constraints, 1262 exceptions, 98 in vacuum) |
| Ewald `alpha` from cutoff and tolerance | OpenMM `getPMEParametersInContext` | 2.628261 vs 2.628261 |
| B-spline partition of unity / derivative sum | analytic | 3.3e-16 / 1.2e-16 |
| B-spline derivatives | central differences | 4.7e-10 |
| `erfc` | **SciPy**, 412 points | worst 1.5e-15 relative (~7 eps) |
| bonded energies (bond, angle, torsion) | OpenMM force groups | 1.4e-17 to 2.6e-15 |
| Lennard-Jones energy | OpenMM, charges-zeroed arm | 2.7e-11 |
| Coulomb energy (direct + reciprocal + self) | OpenMM, epsilon-zeroed arm | 1.2e-8 |
| total potential energy | OpenMM | 1.4e-8 |
| **per-atom forces** | OpenMM, full force dump | 5.2e-8 relative (worst component 2.7e-5 kJ/mol/nm against rms 516) |
| every analytic force | central difference of its own energy | all 8 terms below 1.1e-6 |
| **bonded WGSL kernel** | the host reference, on a real GPU | energy 1.8e-8 to 5.3e-7; force within 7–24% of the derived f32 bound |
| **cell list** | treated as a partition of the atoms | exact at 3 geometries: every atom in exactly one cell, every slice sorted, total correct |
| **neighbour list** | an O(N²) host enumeration, as a *set* | **0 missing, 0 spurious, 0 duplicated** at 2×2×2, 3×3×3 and 4×4×4 cells |
| **direct-space nonbonded kernel** | the host reference | LJ 5.1e-7 – 6.0e-7, Coulomb 1.4e-8 – 3.1e-7, worst force 6.5e-3 kJ/mol/nm against rms ~500 (1.3e-5) |
| **f32 `erfc` in the WGSL kernel** (`npm run test:gpu:erfc`) | Cody's reference in `shared/erf.js`, on the device at 65536 points | **1.4377e-6** worst relative over the αr ≤ 2.628 a 1.0 nm cutoff reaches, against a **derived** cap of 1.9e-6; identical to every digit in Dawn and in Chrome |
| **mixed-radix FFT index map** | a naive DFT, on the host, before any WGSL was written | 33 sizes (every 2/3/5-smooth value to 128), both directions, worst 9.5e-16 |
| **WGSL FFT** | the naive DFT, on random data, 4 grids including non-cubic | 2.9e-7 – 6.6e-7 over 9–12 stages |
| **PME spread grid** | the host's own spread, compared point by point | 3.0e-6 – 3.2e-6 relative; fixed-point headroom **0.15% of 128** charge units used |
| **PME energy and forces** | the host reference, 3 grids | energy 6.3e-8 – 2.1e-6, worst force 1.9e-3 kJ/mol/nm against rms 96 (2.0e-5) |
| **the whole GPU force path** | **OpenMM**, notebooks' configuration | bonded 1.7e-8, LJ 6.0e-7, Coulomb (direct + reciprocal + self) 1.3e-6, **total 1.4e-6**, **per-atom forces 8.8e-5** |
| counter-based PRNG uniform | the host's own, in f32 and f64 | **40000/40000 bit-identical** (the 2^-24 form it replaced: 20045/40000) |
| **one device step** | the host integrator, with the same noise *and* the same forces | positions 2.0e-6 nm, velocities 1.1e-3 nm/ps against a 4.8 nm/ps scale |
| **constraint geometry after 5000 device steps** | the constraint lengths themselves | worst 5.0e-6 of a bond length; 17 SHAKE iterations at most; iteration cap never reached |
| **energy conservation**, friction 0, **20 ps** (`npm run test:gpu:integrator`) | a least-squares fit to the total energy, over growing windows of one trace | **+0.484 kJ/mol per ns per atom at 20 ps**, fluctuation 6.36; the same trace gives **+0.370 at 10 ps** and **+2.63 at 4 ps**, so the figure is only meaningful beside its window — and it is **super-linear**, reaching +13.8 at 200 ps as the system heats 291 → 447 K |
| thermostat, friction 10/ps, 400 steps | the 298 K setpoint | mean 317.8 K |
| **DCD writer** | **byte-compared against `openmm.app.DCDFile`** | identical except the timestamp title and 2–7 ulp in the unit-cell cosines; MDAnalysis reads both files to the same 7.6e-7 Å |
| DCD read back by **cpptraj** | a 4 ps GPU run, 40 frames | opened as "Charmm DCD", 40 frames processed, box and `dt` correct |
| **constraints, measured from the written trajectory** | MDAnalysis, on the water O–H bond across 40 frames | mean **0.95720 Å**, spread 3.3e-6, worst deviation 7.0e-6 relative to the constrained length |
| AMBER restart writer | this port's own `inpcrd` reader | positions exact, velocities 1.0e-7 nm/ps (the format's precision) |
| **the page, driving a real GPU run** | headless Chrome, through the UI controls | GPU minimise, then 4000 NPT steps at 2.01 ms/step on the two-body fixture; nine frames; all five exports offered; render pass validated |
| **LJ tail correction** | **OpenMM**, isolated by toggling `useDispersionCorrection` | **-41.537158 kJ/mol, relative error 1.0e-8** |
| **barostat energy at a forced volume scale** | OpenMM, at the same molecule-COM-scaled coordinates | 1.4e-6 to 1.5e-6 relative at length scales 1.00, 1.02 and 0.97 |
| **NPT equilibration**, 48 ps at 1 bar / 298 K | TIP3P water's density | **0.5868 → 0.9924 ± 0.0116 g/cm³ at 298.7 K**, 50% move acceptance (TIP3P is ~0.98) |
| device minimiser | monotonicity, constraints, and whether dynamics runs afterwards | potential -11857 → -13450 kJ/mol in 23 iterations / 0.18 s, monotone, constraints 5.0e-6, 500 steps afterwards ran at 311.7 K |
| **the viewer actually draws** | a clipped screenshot of the canvas, decoded | 8.6% of pixels off-background, 2661 distinct colours, smooth shading gradients |
| neighbour overflow recovery | a forced 8× overflow (capacity 64 against 526 needed) | grew and rebuilt, **bit-identical** energies to the arm that never overflowed |
| SHAKE/RATTLE convergence | re-projection after a 0.004 nm perturbation | 5.0e-9 relative length error in 33 iterations |
| counter-based Gaussian noise | 60000 samples | mean 0.003, variance 1.001, reproducible per (atom, step) |
| **quaternion Kabsch superposition** | a known 30° rotation applied to a non-planar set | recovered to **2.2e-16 nm**, determinant +1 to 1e-12 (it cannot express a reflection) |
| **fit-on-receptor RMSD** | JIGGLE's own experiment: displace the binder 0.5 nm rigidly | reports **0.5000 nm** where whole-complex RMSD reports **0.2430 nm**; the binder's internal RMSD stays below 1e-9 |
| **`RunningRmsf`** | a two-pass mean/variance over the same aligned frames | below **1e-9 nm**; reports zeros on one frame rather than a meaningless number |
| **all seven live metrics** | **MDAnalysis 2.9.0**, on a DCD RainMD itself wrote, 31 frames | RMSD **2.2e-8 nm**, receptor 4.4e-8, binder 3.0e-8, **binder-on-receptor 4.0e-7**, Rg 4.0e-8, interface 2.5e-7, **RMSF 4.5e-8** — at the DCD's own float32 floor |
| **the PDB writer's columns** | the fixed-column spec, record by record | ` CA ` in columns 13–16 (an alpha carbon, not a calcium), TER per molecule, CRYST1 present, every coordinate parses |
| **the page's control wiring** | a static diff of `element("…")` against `index.html`'s ids | 33 controls, 0 missing — falsified by reintroducing the stale id, which names it |
| **pause actually holds the integrator** | headless Chrome: pause mid-run, wait 2.4 s, sample the step counter twice | held at step 500 with the run still in flight; falsified by deleting the gate (reports `ADVANCED PAST`) |
| **the six live panels draw** | a clipped screenshot of the panel grid, decoded | all six carry ink; the four-way RMSD split reaches 0.53 nm while the complex/receptor/binder traces stay under 0.1 nm |
| **the view dropdowns are wired** | headless Chrome: dispatch `change`, read the viewer's visible set | everything 2547 / backbone 28 / solute 68 on the two-body fixture; colour-by-body validates — falsified by emptying the listener list, which reports 68 for all three |
| **the device's max displacement** | a host loop over the same coordinates, one step | **2.2e-8 relative**; `atomicMax` over `bitcast<u32>` of a squared displacement is exact for non-negative floats, not a fixed-point approximation |
| **the skin detector fires** | a 0.004 nm skin with the interval pinned | caught, 4 violations, worst usage **95x** — falsified two ways: a kernel that reports zero fails arms 1 and 2, an origin that never re-seeds reports 36x usage over every step |
| **the skin controller converges** | the shipped defaults from a minimised structure, 500 steps | worst usage **1.14**, 2% of steps on a stale list, interval settles from 20 to 10; missed-pair force bound **0.71 kJ/mol/nm** against an rms of 516 |
| **the receptor/binder override** | the molecule list, flipped and by residue | an override swaps the two sets exactly, an empty one falls back rather than emptying the receptor, and every split partitions the solute |
| **secondary structure from CA geometry** | analytically ideal helix and strand coordinates | an ideal helix (r 0.23 nm, rise 0.15 nm, 100°/residue) measures **0.505 / 0.620 nm** at i→i+3 / i→i+4 and is assigned helix; an ideal 0.34 nm strand measures **1.023 / 1.360 nm** and is assigned sheet |
| **a declared chain break is refused** | one ideal helix with a break in the middle | the four residues before it lose every window that would cross it and come back coil — the first version of this test could not fail, because two helices joined end to end read as helix either way |
| **all three colour modes draw** | headless Chrome, each under its own error scope | element / receptor-binder / secondary structure all validate |
| **a real helix and strand are drawn as such** | two ff14SB fixtures built by tleap with `impose`, photographed in Chrome | 15/15 residues helix (red) and 15/15 sheet (yellow), ACE/NME caps neutral because they carry no alpha carbon; CA spacings 0.521/0.638 and 1.051/1.386 nm |
| **the structure is drawn on load** | the `--view` lane, which runs nothing | 162 atoms visible from a dropped topology alone — previously a black canvas until Minimise was pressed |
| **positional restraints** | **OpenMM**, the notebooks' own `CustomExternalForce`, at a displaced structure | host **exactly 0.00e+0**, device 6.7e-7 (f32 on a box-spanning reference); the restrained selection is 10 of 10 atoms, identical to pytraj's `!(:H*) & !(:WAT) & !(:Na+)...` |
| **a zero force constant is inert** | the same run built with no restraints at all | **bit-identical** potential — which is what separates "the restraint is right" from "the restraint does nothing" |
| **the restraint pulls towards its reference** | the angle between force and displacement | cosine **−1.0** — `k*d²` is even in `d`, so a sign error leaves every energy comparison exact while pushing the solute away |
| **the three-stage protocol runs** | headless Chrome, through the page's own buttons | minimise (1000 steps, 10 kJ/mol/nm) → equilibrate (NPT, 33 solute heavy atoms at 500 kJ/mol/nm²) → production; equilibration is locked until minimisation has run |
| **the restraint visibly holds the solute** | the live four-way RMSD, restrained against free | complex RMSD **0.034 nm restrained against 0.111 free** over the same 4000 steps, binder-on-receptor 0.085 against 0.207 |
| **Mol\* draws the structure** | the three structure fixtures, photographed in Chrome | secondary-structure colouring from Mol\*'s own assignment; a chain too short to trace falls back to ball-and-stick, which is why a capped tripeptide has no ribbon |
| **every representation and colour mode still works after a run** | headless Chrome, after 4000 NPT steps | cartoon / cartoon+sticks / atoms and structure / chain / element all build, with the atom count checked each time |
| **the LogMD payload matches their client's** | `logmd/logmd.py`'s `upload_worker_process`, read as source | the same seven keys, `frame_num` as a string, `user_id` pinned to `"public"` and `token` to `null` — a page must never send an account |
| **a failed share says which frame and why** | an injected `fetch`, so no test touches the network | 404 and a thrown `fetch` both reported with the frame index; the opaque-failure message names CORS *and* a dead endpoint, because a browser cannot tell them apart |
| **each stage's panels are its own** | three stages run through the page, reading the first panel's time column | strictly monotonic — a single step backwards means two stages superimposed, which is how it looked when it was broken: 21 rows with one reversal against production's own 16 |
| **the live animation actually moves** | the drawn coordinate read out of Mol*'s model during production | first atom x walks 10.77 → 9.86 → 8.74 → … over a 25,000-step run; the atom count alone cannot see this, and a stable count was what made it look fine |
| **the camera survives a frame update** | `keepCamera` on a frame reload | the snapshot is restored after the rebuild, so a reader's own rotation and zoom are not reset several times a second |
| **the viewer keeps up** | frames dropped while a reload is in flight, 1 ps reporting interval | **0 dropped** on the solvated fixture — a frame arriving mid-rebuild is dropped rather than queued, because a queue renders frames the simulation has already left behind |
| **the interface follows LocalFold's** | screenshots at 1440 px and 500 px, through the page's own controls | two columns with a tabbed result pane, white cards, capitalised field labels in bordered groups, one filled primary action; single column below 64rem with the structure first; no horizontal scroll at either width |
| **the ribbon's frame is a valid frame** | the three structure fixtures, guide by guide | side vectors unit length to 1e-5 and perpendicular to the backbone to **1e-5**; consecutive frames never oppose (min dot +0.62 helix, +0.83 strand) |
| **the flip correction is what stops a strand twisting** | deleting it | the minimum consecutive dot goes from +0.62 to **−0.75**, i.e. the frame reverses — the one ribbon property a still image of a helix cannot show |
| **a strand ends in an arrowhead, a helix does not** | the width profile along each chain | strand 0.110 nm → base **0.209** → tip **0.012**; helix constant at 0.120; a coil rod is 0.040, under half a helix's width so structure reads without colour |
| **no ribbon segment spans a chain break** | the two-body fixture, and one hidden mid-chain residue | two chains give two runs and two fewer segments; hiding a residue splits one run into two and drops two segments — falsified by emitting segments across run boundaries |
| **every representation draws during a live run** | headless Chrome, after 4000 NPT steps | atoms / cartoon ribbon / ribbon+atoms all validate and the ribbon modes build segments; the ribbon is rebuilt from the frame on screen at every report |
| **the ribbon WGSL compiles** | `getCompilationInfo()` on the shader module alone | 0 errors over 135 lines — `createRenderPipeline` only ever says "invalid due to a previous error", with no line number |
| **the browser lane survives being run back to back** | an 11-link `&&` chain and a 3-link one | both ports were fixed and `chrome.kill()` is asynchronous, so the 11th link failed on a port the dying browser still held — while all 11 passed alone. Ports now derive from `process.pid` |
| **the browser lane waits for the page, rather than sleeping** | three consecutive runs, and a forced module-scope throw | the old fixed 2500 ms delay failed about one run in three once the module graph grew, reporting `Cannot read properties of undefined (reading 'device')` — which is indistinguishable from the real `app.js` evaluation bug. It now polls for `window.rainmd` and names the cause |
| thermostat | 150 steps at 50/ps friction | 295.8 K against a 298 K setpoint (0.7%) |
| **the page, in real Chrome** | headless Chrome 157 over CDP, `npm run test:page` | loads, probes `apple / metal-3`, runs the bonded kernel: energy 1.75e-8, force 7.2% of the f32 bound — **identical to the Dawn lane** |
| **every kernel in both lanes** | the same fifteen check modules run through Dawn and through Chrome | 0 failures in both; the full force path agrees between implementations to 1e-8 on the total |
| determinism | two independent runs of the nonbonded check | **bit-identical** energies, including the arm that reaches them via a grow-and-rebuild |
| **the reported temperature holds the setpoint across the timestep range** | `npm run check:thermostat`, NVT friction 10/ps, 60 ps at each of 1 / 2 / 4 fs | offsets **+0.46 / −0.33 / +0.88 K**, growth over the range **0.42 K**; it was −0.02 / −2.08 / −6.36, and OpenMM's own is +0.42 / +1.63 / +4.45 |
| **a blown-up run is refused rather than completed** | `npm run test:gpu:blowup`, 40 fs on the solvated fixture | stops at step 20 with `NonFiniteStateError` naming the timestep; a 2 fs run of the same length is **not** refused and reports a finite potential and 382 K; identical in Dawn and Chrome |
| **coincident input atoms are refused before a step runs** | one water copied onto another, so SHAKE leaves them exactly superposed | refused at step 0 by the energy detector, separation after constraint projection **0.0 nm** — the arm asserts its own premise, because moving one oxygen instead of a whole molecule is silently undone by SHAKE |
| **the NVE drift no longer grows as the timestep shrinks** | `npm run check:timestep` — NVE at 1 / 0.5 / 0.25 fs, neighbour list rebuilt every step | drift ratio over a fourfold smaller step **2.03** against a bar of 10; it was **824**, and the end temperature now holds 299.5 / 297.4 / 299.1 K where 0.25 fs reached **398 K** |
| **one device step against the host integrator** | the same noise and the same forces | velocities **5.76e-5 nm/ps** against a 4.79 nm/ps scale — **20× closer** than the 1.15e-3 before the constrained-velocity fix |
| **the energy drift the docs quote is produced by the command** | `npm run test:gpu:integrator`, now 20 ps | window fits from one trace, so the figure carries its window |
| **the staleness measurement does not depend on how the run was submitted** | `npm run test:gpu:displacement`, arm 5 — the same 100 steps at three block sizes, controller pinned | worst usage **47.2308**, displacement **0.094462 nm** and bound **5.649** identical at blockSize 4, 20 and 100, with the same trajectory checksum; the shipped window gave **47.23 / 20.34 / 16.37** |
| **the barostat decides on the potential the run reports** | `npm run test:gpu:barostat`, restrained arm at the page's own equilibration settings | the two sums agree to **1.5e-4 kJ/mol** with a restraint of 8.44 present; with the term dropped the gap is `differenceAsFractionOfRestraint` **0.99998** — exactly the restraint energy |
| **the neighbour capacity tracks a compressing box** | `npm run test:gpu:capacity`, `ala-wat-big` driven through the barostat's own `scaleVolume` | 65 → 103 atoms/nm³ past bulk TIP3P with capacity growing **1499 → 2367** and nothing dropped; with the carry-forward restored the capacity stays at 1499 while the requirement climbs to 1395 |
| **an overflow is seen during a run, not only at startup** | the same check's second arm, capacity pinned to 64 | grown to 918 and the high-water mark reads 736; without the per-block check the mark stays **0**, which is how the defect looked |
| **the README's privacy claim matches the page** | `npm test`, a claim-consistency gate | the share button's presence forces the README to name LogMD and to qualify the claim; restoring "no install, no queue, no upload" fails it |
| **the trajectory's unit cell under NPT** | `npm run check:npt-box` — a real 4000-step NPT run read back by **MDAnalysis** | live volume walks 19.61 → 15.09 nm³ (23%) and every one of 16 frames matches to **8.3e-8**, against the DCD's float32 cell floor of ~2e-7; with the defect the error is 36% and every frame carries the same cell |
| **ff19SB is refused by name, not run without its CMAP** | `npm test`, the flag audit; measured against OpenMM on real ff19SB systems | −5.08 kJ/mol on a capped tripeptide, −56.67 on a 30-mer; all four ff19SB variants built by tleap are refused, and ff14SB / GLYCAM / protein+glycan / `solvateOct` / 52,200 atoms still parse |
| **4-point water is refused rather than constrained to a massless site** | the NUMEXTRA check; measured on the host | OPC and TIP4P-Ew refused; unrefused they gave **2653.6 K** at step 15 from a 298 K setpoint, with the extra points moving 0.00000 nm against a 412.9 K 3-point control |
| **an unrecognised `%FLAG` is refused and named** | the audit against a 44-flag classification | CHAMBER, 12-6-4, polarizable, solvent cap and LES all refused by name; harmless PDB metadata still ignored |
| **the PME reciprocal sum on a SHEARED cell** | an exact Ewald lattice sum written from the textbook formula, sharing no code with it | `ala-oct` (truncated octahedron, 1315 atoms) agrees to **2.4e-6** at a 48³ grid, the same convergence the rectangular fixture shows (1.9e-6) — and the transposed form, which was shipped, is grossly wrong there and **bit-identical** on a rectangular box |
| **the whole tree parses** | `node --check` over every first-party `.js`/`.mjs` under `shared/ cpu/ webgpu/ web/ tools/` | 49 files; the previous version named eleven by hand and omitted `webgpu/kernels/` entirely, where every WGSL template literal lives |
| **Stop keeps what the stage produced** | `npm run test:page:stop`, Stop pressed mid-equilibration with frames already written | trajectory kept (`equil:3`, 43 KiB), both downloads enabled, state carried to step 1040 with velocities, restart labelled `equil` and marked interrupted — falsified against the pre-fix page, which reports `trajectoryFiles []`, `trajectoryBytes 0`, both downloads disabled and `hostState` at step 0 |
| **a blown-up stage keeps its frames and hands on nothing** | the same lane's second arm, 40 fs after the stop | the opening frame is kept (`equil:1`), the carried checksum is **unchanged to the last digit**, and the earlier stage's restart is not replaced — the opposite assertion, so the fix cannot be "keep everything" |
| **the restart names the stage it came from** | the same lane | `<name>_equil.rst7`, and `restartStage` reports `equil` — a stopped production used to hand over the equilibration's state under `<name>.rst7` |
| **no file this program writes can contain `NaN`** | the DCD and restart writers, host tests | both refuse a non-finite frame by name; the DCD keeps the frames written before the refusal and `finish()` still produces a readable file |

Two tolerances in that table are **derived, not chosen**, which is the only kind
worth having:

- **PME grid floor, 8.4e-6 relative.** PME on a 24³ grid against an exact Ewald
  lattice sum on the same system (`--nonbonded Ewald`). No PME implementation on
  this grid can do better, so a tighter bar would be measuring the grid.
- **f32 bonded force bound, 0.16 kJ/mol/nm.** `k x span x eps` =
  `476976 x 2.82 x 5.96e-8`. The kernel measured 0.157.
- **f32 constraint-tolerance floor, 7.0e-6.** The SHAKE residual is
  `|d|²/L² − 1`, and `d` is a difference of two coordinates spanning the box, so
  its absolute f32 error is about `span × eps` whatever `d` is: `4 × 2.82 ×
  5.96e-8 / 0.09572 = 7.0e-6` for this system's shortest constraint. Asking for
  1e-6 — which the notebooks do, and which OpenMM reaches because it keeps
  positions in f64 — makes every one of 394 clusters run to its 150-iteration
  cap and report failure **on a correct result**. This is also why OpenMM's own
  default is 1e-5. The device clamps the request to the floor and says so in
  its diagnostics.
- **f32 `erfc` floor, 1.4377e-6 relative — measured on the device, by
  `npm run test:gpu:erfc`.** WGSL has no `erfc`, so the kernel evaluates
  `t·exp(−z² + P(t))` with a degree-9 `P` fitted by `npm run fit:erfc`. The
  floor is the *arithmetic*, not the fit: that exponential's argument is −6.91
  at a 1.0 nm cutoff, where one f32 ulp is 4.8e-7 of **absolute** argument
  error, and an exponential turns that straight into **relative** output error.
  The polynomial's own error is 1.0e-7 in exact arithmetic, seven times under
  the floor it sits on, so refitting it cannot help — and the same arithmetic
  emulated on the host with `Math.fround` gives only 7.4e-7, so **1.1e-6 of the
  1.44e-6 is the backend's `exp`**, which WGSL specifies merely to 3 ulp. The
  nonbonded test's Coulomb bar is four times whatever the running device
  measures, capped at a bound derived from that ulp spacing so it can never
  widen to admit a broken `erfc`.

  This number was previously recorded here as 1.191e-6, with no command behind
  it; the review found a third figure (1.478e-6) in a host emulation. The
  device settles it, and the earlier value **understated** what the GPU
  delivers by 21%.

## Not yet written

**The simulation is GPU-resident.** Forces and integration both run on the
device; nothing in a step reads back to the host. A block of steps is encoded
into one command buffer, so there is no per-step round trip.

Measured on this Mac (Apple Metal, 1190 atoms, PME 24³, 2 fs, HBonds):
**1.30 ms/step, or about 133 ns/day** on the shipped defaults — against 47–79
ms/step for the host reference, a factor of 40 to 60.

🔴 **THIS SAID 0.72–1.03 ms/step AND 215 ns/day, AND IT WAS STALE.** That was
measured in NVT, with a 0.2 nm skin and a 10-step rebuild, before the
displacement kernel, the restraint pass or the barostat existed — a
configuration the page no longer offers. A reader comparing their own ns/day
against it was comparing against something unreachable, and one did. Measured
now with `npm run bench:throughput`, in Chrome, 1190 atoms:

| configuration | ms/step | ns/day |
|---|---|---|
| **shipped default — NPT, derived skin, adaptive rebuild, 2 fs** | **1.299** | **133** |
| without the barostat (NVT) | 1.119 | 154 |
| NVT, skin 0.2, rebuild every 10 — the old documented setting | 1.018 | 170 |
| NPT with the equilibration restraint on | 1.268 | 136 |
| NVT, rebuild every 40 | 1.101 | 157 |
| NVT at 4 fs | 1.340 | **258** |

So the old number is not reachable even in its own configuration: the best this
machine now does there is 170 ns/day, because the displacement kernel runs every
step and the force path has grown. The barostat costs about 13%, which matches
the 0.3 ms/step the barostat note records.

**And throughput falls with system size**, which is what a reader comparing
numbers actually needs:

| system | atoms | ms/step | ns/day |
|---|---|---|---|
| alanine dipeptide, 8 Å padding (`ala-wat`) | 1190 | 1.299 | 133 |
| two peptides in water (`two-body`) | 2547 | 1.683 | 103 |
| alanine dipeptide, 12 Å padding (`ala-wat-big`) | 2854 | 2.346 | **74** |

A reader on an M3 Pro measured **53 ns/day** for an alanine dipeptide, which is
within 1.4x of the 2854-atom figure here — so it is the expected throughput for
a realistically solvated box on that hardware, not a fault. The levers, largest
first: a **4 fs timestep** (roughly doubles ns/day, and wants hydrogen mass
repartitioning to be safe — this port does not do that yet), **a smaller water
box**, and **NVT for production** if the density is already right.

🔴 **THIS LIST HAS BEEN WRONG BEFORE, WHICH COSTS MORE THAN A MISSING ONE.** It
named the barostat as "the only missing piece of physics" while the table above
recorded its validation, and named the viewer as absent after it was drawing. A
line here is deleted when it comes good, never widened.

What remains:

1. **LogMD's upload endpoint returns 404, so the share cannot be verified
   against a live service.** Measured 2026-11-07 from a server, where CORS does
   not apply: `upload-frame`, `upload-frame-bcif` and `list-project` all answer
   `404 modal-http: invalid function call`, and those are the URLs in the
   published logmd 0.1.46, not stale ones from a clone. The payload shape is
   gated against their source and both failure paths are tested with an
   injected fetch; what cannot be tested is a successful upload. The fallback —
   the same frames as a downloadable bundle plus the Python to post them — is
   the half that does not depend on anyone else's infrastructure.
2. **`web/viewer.js` and `shared/analysis/ribbon.js` are retained but the page
   no longer loads them.** The WGSL sphere-impostor and cartoon renderer is
   still gated — its tests and the `probe-ribbon-shader` lane still run — but
   3Dmol draws the page now. Keeping an unloaded renderer is exactly the trap
   LocalFold's own notes warn about, so this is a decision waiting to be taken
   rather than a state to settle into: either it finds a use or it goes.
3. **Performance work, which has barely begun.** One measurement so far: a stall
   break in SHAKE took a step from 2.40 ms to 1.03 ms, because a tolerance below
   the f32 floor made every cluster run its full iteration cap. Remaining
   candidates are unprofiled: the per-cell insertion sort, the
   single-workgroup prefix sum, a real-to-complex FFT (the charge grid is real,
   so the transform does twice the necessary work), reusing the spread's
   B-splines in the gather, and reducing the per-atom energy arrays on the
   device instead of reading them back.

The host reference implements all of these and is validated, so each kernel has
its oracle waiting before it is written. `shared/pme/parameters.js` already
restricts grids to sizes the planned FFT can transform.

## The stage handoff, which was broken

**Every stage restarted from the minimised structure.** The device run wrote its
final state into `finalState` — read only by the `.rst7` download button — and
left `hostState` untouched. So equilibration started from the minimised
structure correctly, and then production started from the minimised structure
*again*, discarding the entire equilibration. Found by a reader; no gate could
see it.

🔴 **AND NOTHING IN THE OUTPUT WOULD SHOW IT.** A production run that silently
restarts from the wrong structure still produces a valid trajectory, plausible
energies and a sensible RMSD — of the wrong starting point. Every force and
energy comparison in this document was correct throughout.

**Three things were being dropped, not one:**

| | before | after |
|---|---|---|
| positions | production re-read the minimised structure | each stage hands on what it finished with |
| velocities | re-drawn from Maxwell-Boltzmann each stage | carried, so production continues rather than restarting cold |
| **the box** | every reference read the *coordinate file's* box | carried, because NPT changes it |

The box was the worst of the three. An NPT equilibration exists to fix the
density, and `assembleSystem` rebuilt the system from `loaded.coordinates.box`
— whatever tleap's `solvateBox` originally guessed. Measured on the solvated
fixture, the volume now carries **20.52 → 14.39 → 11.96 nm³** across
minimise → equilibrate → production as the barostat compresses the deliberately
underdense starting box; before the fix it stayed at 20.52 for all three.

`readState()` did not even return the box, so there was nothing to carry.
`buildSystem` now takes `boxVectors` which overrides the file's `box`, and that
is the currency of the handoff because it is what the barostat produces.

**`npm run test:page:handoff`** records the structure checksum, the volume and
whether velocities exist at each stage boundary and asserts each stage moved
what it was handed. Falsified by removing the write-back, which reproduces the
original symptom exactly: checksum 240922.57967 and volume 20.5203 nm³ at all
three boundaries, with no velocities.

## What the notebooks do, and what this page now does

The Making-it-rain notebooks run three stages, and until now this page ran one.
Read out of `Amber.ipynb` rather than assumed:

| | notebook | this page |
|---|---|---|
| **1 minimise** | `minimizeEnergy(tolerance=10 kJ/mol/nm, maxIterations=N)`, N from a dropdown of 1000…100000 | the same, same dropdown, same tolerance |
| **2 equilibrate** | **NPT** with `MonteCarloBarostat`, solute heavy atoms restrained by `CustomExternalForce('k*periodicdistance(...)^2')` at 500 kJ/mol/nm², time in **ns** | the same |
| **3 production** | NPT, unrestrained, `Stride_Time × Number_of_strides` ns | the same |
| durations | ns, with trajectory and log intervals in **ps** | the same |
| timestep | a dropdown, 0.5–4 fs | a field, same default of 2 fs |

🔴 **THE EQUILIBRATION IS NPT, NOT NVT.** The cell adds a `MonteCarloBarostat`
before the integrator, so the box is free while the solute is held — the
restraint is what stops the solute moving, not a fixed volume. Implementing it
as NVT would hold the density at whatever tleap's `solvateBox` guessed, which is
the thing equilibration exists to fix: the test fixture starts at
0.587 g/cm³ and TIP3P wants 0.98.

🔴 **AND `k*d²` CARRIES NO HALF.** Every other harmonic term in this port uses
OpenMM's `½k x²`; the notebooks' restraint does not, so the slider's 500 means a
force of 1000 kJ/mol/nm at one nanometre. Reading it the other way would make
every restraint half as stiff as the number says and still produce a trajectory
that looked fine.

🔴 **THE REFERENCE DOES NOT FOLLOW THE BOX.** OpenMM captures `x0,y0,z0` when
the force is added and the barostat never rescales them, so a restraint pulls
towards where the atom was in the box as it was. That looks like an oversight
and is the behaviour the protocol depends on — references that rode the volume
changes would let the solute drift with them.

## How long a run actually takes, which the page now says out loud

Durations are in nanoseconds because that is how a protocol is written, and the
page shows what each stage costs **at a rate measured on the system in front of
you** (“Time this system” runs 300 steps and reports ms/step). That matters
because the honest answer is often discouraging: at the measured 2.24 ms/step
for the 2547-atom two-body fixture, 2 fs steps give about **77 ns/day**, so a
500 ns production is **roughly 6.5 days of a tab staying open**. A literature
protocol is not reachable here for a solvated protein, and a reader who types
500 should see that before pressing the button rather than after.

## Two defects a screenshot found

**The panels were drawing two runs as one.** Every stage builds a fresh
simulation whose step counter starts at zero, so its rows carry times 0..T
again — and `ensurePlot().clear()` had been dropped when the single Run button
became three stages. A panel still holding the previous stage's rows drew one
polyline right across the axis, back to the left, and right again, which renders
as **two traces per panel**: the equilibration transient and the production run
superimposed. The potential-energy panel showed a steep descent that had already
finished. Reported by a reader from a screenshot; nothing was watching for it.
The gate is mechanical — within one panel the time column must be monotonic —
and reproduces the symptom when the clear is removed.

**And the live animation was updating while looking broken.** The coordinates
were moving the whole time; what was wrong is that a frame update is a
clear-and-reload in Mol*, and it re-ran `zoomTo` every frame. So a reader who
had rotated or zoomed was snapped back to a default view several times a second,
with a blank between the clear and the load. "Not working" was the right
description of the experience and the wrong description of the cause — which is
why the gate reads the drawn coordinate out of Mol*'s own model rather than
trusting the atom count, which was stable and correct throughout.

## Three defects that were all one missing list

**The parser read the twenty-six `%FLAG` sections it names and dropped every
other one in silence.** There was no known-flag list to compare a file against,
no unknown-flag report, nothing. Found by a review pass; `docs/REVIEW.md` has
the flag-by-flag audit.

🔴 **SO THE NOTEBOOK'S OWN DEFAULT FORCE FIELD RAN AS A DIFFERENT ONE.**
`Amber.ipynb`'s dropdown is
`Force_field = "ff19SB" #@param ["ff19SB", "ff14SB"]`, and ff19SB's prmtop
carries `CMAP_COUNT`, `CMAP_RESOLUTION`, `CMAP_PARAMETER_01` and `CMAP_INDEX`.
None were read. Per term against OpenMM 8.4 on ACE-(ALA)₃-NME:

| term | OpenMM | RainMD |
|---|---|---|
| HarmonicBond | +0.088778 | +0.088778 |
| HarmonicAngle | +4.023395 | +4.023395 |
| PeriodicTorsion | +10.022601 | +10.022601 |
| Nonbonded | −53.246508 | −53.246507 |
| **CMAPTorsion** | **−5.082244** | **absent** |
| **TOTAL** | **−44.193978** | **−39.111732** |

and it grows with the chain: a 30-residue ff19SB helix carries 28 CMAP terms
worth **−56.671931 kJ/mol**. ff19SB *is* ff14SB with CMAP replacing the backbone
torsions, so dropping it is not a perturbation of that force field — it is an
unvalidated one. The file parsed with `warnings: []`.

🔴 **AND OPC WATER REACHED 2650 K.** `NUMEXTRA` was parsed into `pointers` and
never checked. `leaprc.water.opc` — ff19SB's recommended water, and
`Water_type = "TIP3P" #@param ["TIP3P", "OPC"]` in both notebooks — puts one
zero-mass `EPW` site per water, bonded to the oxygen. With no virtual-site
support, `rigidWater` constrained the O–EP bond: **933 of 3754** distance
constraints tied a movable oxygen to a frozen massless point. Host dynamics, 60
steps at 2 fs, 298 K setpoint:

```
step 15  T = 2653.6 K     step 30  T = 1669.1 K     step 60  T = 1872.1 K
max |dr|: water O 0.03187 nm, extra points 0.00000 nm, hydrogens 0.18199 nm
control (3-point ala-wat, identical code path): 412.9 K, water O 0.15278 nm
```

**What the fix is.** `shared/amber/flags.js` classifies all 44 flags the
fixtures carry, plus everything ParmEd's own source enumerates, into handled /
harmless / refused-by-name. Three further refusals are **pointers rather than
flags**, so the audit cannot see them: `NUMEXTRA`, `IFCAP` and `NCOPY`.

🔴 **AN UNRECOGNISED FLAG IS REFUSED, NOT WARNED ABOUT, AND THE ASYMMETRY IS THE
ARGUMENT.** A flag that changes the energy and is dropped produces a wrong
answer nothing downstream reveals; a harmless flag that causes a refusal costs a
user one message and this repository one line. A warning would not have been
enough either — the parser already pushes a note into `topology.warnings` when
the molecule partition disagrees with `ATOMS_PER_MOLECULE`, and the page shows
it as one line in a scrolling log.

🔴 **AND `IPOL` IS A SECTION IN MODERN FILES, IN NO POINTERS SLOT.** tleap writes
`%FLAG IPOL` with its own value, so `pointers.IPOL` was `undefined` and the
check meant to stop a polarizable topology would have been walked straight past.
Every fixture here writes `IPOL 0`, which is why it had to be read rather than
assumed.

**Gated without adding a single fixture.** A prmtop carries the parameters it
uses and AMBER's are not covered by this repository's licence, so committing an
ff19SB or TIP4P-Ew file to test a *refusal* would redistribute parameters for a
force field this engine declines to run. The tests append the offending flag to
`ala-vac.prmtop` in memory instead, which tests the audit exactly and adds
nothing. Falsified: disabling the audit fails all four arms. Verified against
real tleap output — all four ff19SB variants and TIP4P-Ew are refused, while
ff14SB, GLYCAM, a protein+glycan complex, `solvateOct` and a 52,200-atom box
still parse.

**What is still not covered.** A genuinely polarizable topology was never
tested, because `leaprc.water.opc3pol` writes `IPOL 0` in this AmberTools build;
the `IPOL` path is code-reading plus a synthetic arm, not a real file. And the
12-6-4 refusal fires on the flag's presence — the magnitude of what would have
been dropped is unmeasured, because no nonzero C₄ coefficients could be
produced here.

## The two open items, investigated

**The sheared-cell residual was OpenMM's, not this port's.** Two independent
oracles sharing no code with what they validate — a brute-force Lennard-Jones
sum over 125 lattice images, and a complete Ewald sum (real-space `erfc`,
exact k-space lattice sum, self, net charge, exclusion corrections, 1-4
exceptions):

| | independent exact | RainMD (48³) | OpenMM (48³) |
|---|---|---|---|
| ala-wat Coulomb (control) | −13652.474621 | **1.3e-7** | 6.2e-6 |
| **ala-oct Coulomb** | −15596.114539 | **5.3e-8** | **1.4e-4** |
| ala-wat LJ (control) | 1794.522906 | **3.8e-16** | 2.0e-10 |
| **ala-oct LJ** | 2673.565469 | **1.9e-15** | **3.07e-5** |

🔴 **THE RECTANGULAR CONTROL IS WHAT MAKES THAT READABLE.** There RainMD and
OpenMM agree with *each other* to 1.2e-8 at a 24³ grid while both sit 6e-6 from
the exact sum — the PME grid floor acting identically on two implementations of
one algorithm. Without that arm the octahedral numbers read as this port
drifting. At 48³ this port converges onto the exact answer and OpenMM does not.
Gated by `npm run check:triclinic`, which is how the octahedral case is held at
all: `check-against-openmm.mjs` cannot hold it, because on this box OpenMM is
the one that differs.

**And the cooling at large timestep is a reported-temperature question, now
proven rather than suspected.** This port runs `B A O A B` and reports the
kinetic temperature after the **closing** half kick; OpenMM's
`LangevinMiddleIntegrator` ends its step at the position update and reports
**one half kick earlier**. Same splitting, same trajectory; the estimator is
taken at different points, and a half kick is an O(dt) change in `v` and O(dt²)
in energy with the sign set by which side of it you stand on. Moving only the
sampling point, in f64, on the identical trajectory:

| | 1 fs | 2 fs | 4 fs | dependence |
|---|---|---|---|---|
| after the closing half kick (shipped) | 309.87 | 303.13 | 271.98 | **−37.89 K** |
| one half kick earlier (OpenMM's point) | 312.40 | 314.80 | 316.08 | **+3.68 K** |
| OpenMM itself, solvated | 299.37 | 299.31 | 302.38 | **+3.01 K** |

The dependence collapses tenfold and the sign flips to OpenMM's, matching it
within error — which accounts for the opposite signs, for both engines
converging as dt → 0, and for the effect surviving in f64. **The dynamics is not
at fault and the sampled ensemble was never in question.** Three hypotheses were
refuted on the way and are recorded in `docs/REVIEW.md` so they are not retried:
the constraint schedule, the RATTLE count, and an attempt to settle it from ⟨U⟩,
which at 60 ps per arm is too noisy by about 16×.

🔴 **FIXED BY SAMPLING INSIDE THE STEP, AND THE FIRST ATTEMPT AT IT WAS WRONG.**
The two velocities differ by `(halfStep/m)·F`, so the earlier estimator looks
recoverable from what is already read back — and on `ala-vac` that
reconstruction agrees to 0.2–0.8 K. It is wrong on a real system: the closing
kick is followed by RATTLE, so `v_post = RATTLE(v_pre + (halfStep/m)·F)`, and
subtracting the kick adds back the bond-directed velocity the projection
removed. `ala-vac` carries 12 constraints, 18% of its 3N; `ala-wat` carries
1176, **33%**. Implemented, the reconstruction reported **+86.85 K** at 4 fs
against a true +4.45. It was validated on the fixture where it happens to work —
the same mistake as checking a triclinic claim on a rectangular box.

What works is to take the sample *inside* the step: a kinetic pass encoded in
`encodeLangevinStep` after the second drift-and-project and before the closing
half kick, emitted only on the last step of a block since that is the one the
host reports. Offsets from a 298 K setpoint:

| dt | before | now | OpenMM |
|---|---|---|---|
| 1 fs | −0.02 | **+0.46 ± 0.57** | +0.42 |
| 2 fs | −2.08 | **−0.33 ± 0.54** | +1.63 |
| 4 fs | −6.36 | **+0.88 ± 0.57** | +0.88 → +4.45 |

Growth across the range is **0.42 K**, against 6.34 before and OpenMM's 4.03.
The reported temperature is now flat to within its own sampling error at every
timestep the page offers — better than the engine it was matched to.

🔴 **AND THE TOTAL ENERGY STILL USES THE POST-STEP SAMPLE, WHICH IS NOT A
DETAIL.** Pairing the early velocity with the end-of-step position is a
half-step mismatch: doing so inflated the measured NVE energy fluctuation from
9.31 to **15.51 kJ/mol**, a spurious oscillation that reads as drift. So
`energy()` keeps both samples — `kinetic` for the conserved total,
`kineticForTemperature` for the thermostat — and reports `kineticSampledAt` so
the number's provenance is never guessed. Separating them restored the
conservation figures exactly.

🔴 **A ZERO TEMPERATURE WAS REACHABLE, AND IS NOW GUARDED.** The host marks the
step it asked for an in-step sample and the encoder emits the pass; if those
disagree the buffer is read unwritten and the temperature comes out as 0 K. The
first falsification of this gate hit exactly that, reporting −298.00 K instead
of falling back. A zero kinetic energy beside a non-zero post-step one cannot be
physical, so `energy()` now takes the post-step value and says so.

**Superseded:** Two hypotheses tested this pass, both refuted:

- *The constraint schedule.* A patched host copy accumulating both half drifts
  and constraining the whole step **once**, OpenMM's shape, against the shipped
  double projection, in f64 so precision cannot confound it: the dt dependence
  is the same within error, **−37.89 K** against **−40.18 K** over 1 → 4 fs.
- *That OpenMM is flat.* It is not. Measured properly — CPU platform, minimised,
  20 ps equilibration, 60 ps sampled:

| dt | OpenMM | this port |
|---|---|---|
| 1 fs | 298.42 ± 0.47 (**+0.42**) | 297.98 ± 0.57 (**−0.02**) |
| 2 fs | 299.63 ± 0.48 (**+1.63**) | 295.92 ± 0.54 (**−2.08**) |
| 4 fs | 302.45 ± 0.51 (**+4.45**) | 291.64 ± 0.55 (**−6.36**) |

Both carry an O(dt²) on-step kinetic bias and **the signs are opposite**. Both
agree with the setpoint and with each other within error at 1 fs, so both are
correct integrators converging to the same limit and differing in their O(dt²)
term. That is a far narrower claim than "the ensemble is wrong", and it fits the
exact Lyapunov result: BAOAB's *configurational* temperature is 298 K at every
dt while its on-step *kinetic* temperature is biased.

**What a user should do meanwhile:** at 2 fs the reported temperature reads
about 2 K low and at 4 fs about 6 K low — under 1% and 2.5%. Treat the reported
temperature as having an O(dt²) offset, and do not tune a setpoint against it.
The remaining measurement, not done here, is a configurational temperature for
both engines: if those agree while the kinetic estimators diverge, this is
reporting and the sampled ensemble is sound.

## Superseded: how far these got on the first pass

**The direct-space Lennard-Jones sum on a sheared cell is RainMD's, not
OpenMM's.** An independent O(N²) sum with brute-force minimal images over 125
lattice shifts, sharing no code with `cpu/forces/nonbonded.js` and validated on
the rectangular fixture first, gives 2673.565469 on `ala-oct` — matching RainMD
to **1.5e-15** and differing from OpenMM by **3.07e-5**. On the rectangular
control all three agree. So the discrepancy that showed up when the PME
transpose was fixed is OpenMM's Reference platform departing from a true
minimum-image sum on a truncated octahedron, not a defect here. Consistent with
the direction (OpenMM's LJ is higher, so it includes fewer attractive pairs near
the cutoff) and with `c→b→a` changing **0 of 863,955** cutoff verdicts on this
box.

🔴 **THE COULOMB RESIDUAL IS NOT ATTRIBUTED, AND THE TIDY EXPLANATION DOES NOT
SURVIVE ARITHMETIC.** The obvious story is that the same boundary pairs explain
LJ and Coulomb together, but the two discrepancies are 0.082 and 2.115 kJ/mol —
a ratio of 25.8, where a screened Coulomb and an LJ term at r ≈ 1.0 nm for water
pairs differ by about 3.8. Building the full independent electrostatics oracle
(real + reciprocal + self + exclusion corrections + 1-4 exceptions) is what will
settle it; an attempt that compared against `coulombDirect` measured nothing,
because that field is not the screened real-space sum it was taken for.

**The cooling at large timestep survived the f32 fix, as its mechanism
predicted.** Re-measured on the device, NVT at friction 10/ps, 60 ps per step
from one shared equilibrated start:

| dt | before the f32 fix | after | OpenMM LangevinMiddle |
|---|---|---|---|
| 1 fs | — | 297.98 ± 0.57 | — |
| 2 fs | 296.76 | 295.92 ± 0.54 | 300.10 ± 0.23 |
| 4 fs | 291.37 | 291.64 ± 0.55 | 301.57 ± 0.23 |

The offsets moved by about one standard error. So the two halves of that finding
are independent defects of opposite sign: an f32 heating that grew as the step
shrank, now fixed, and an **O(dt²)** cooling that grows as the step grows, still
present. Zero within error at 1 fs, −2.08 at 2 fs, −6.36 at 4 fs; the 2→4 ratio
is 3.06 against 4 for exactly quadratic. OpenMM on the same system, with the
same splitting and — measured — the same uncorrected on-step kinetic estimator,
is flat to within 1.6 K, so the bias is this port's.

**The surviving hypothesis is a change, not a measurement.** This port applies
SHAKE after *each* half drift and RATTLE three times a step, where OpenMM
accumulates both half drifts into one displacement, constrains it **once** and
corrects velocities once. Projecting along a mid-step bond direction twice
rather than the step's once is a plausible O(dt²) source. Testing it means
restructuring the integrator, so it is not attempted here. One tempting test is
also **confounded**: running unconstrained does not isolate it, because without
constraints the water O–H stretch has a ~9 fs period and BAOAB's intrinsic
on-step kinetic bias is large at 2 fs for reasons unrelated to SHAKE.

**What a user should know meanwhile:** at the shipped 2 fs default the reported
temperature reads about 2 K below the setpoint, and at 4 fs about 6 K below —
under 1% and 2.5%, a sampling and reporting bias rather than an instability, but
visible to anyone comparing against a notebook run.

## The constrained velocity was recovered the ill-conditioned way

**`v = (positions - reference) / halfStep`, in both the host and the device.**
That is algebraically exactly `v + correction / halfStep` — `positions - reference`
*is* `halfStep·v + correction` — so it was never a wrong formula. It was a right
formula that obtains a ~1e-3 nm displacement by subtracting two coordinates
spanning the box, whose absolute f32 error is `span·eps` however small the answer
is. Divided by the half step, one ulp near 2.82 nm becomes a **2.4e-4 nm/ps**
velocity kick, on every constrained atom, every step, against a hydrogen's
thermal 1.6 nm/ps.

🔴 **SO A SMALLER TIMESTEP MADE IT WORSE, WHICH IS WHY IT LOOKED LIKE ANYTHING
BUT ARITHMETIC.** The error does not shrink with the step, so dividing by a
smaller half step amplifies it, and there are more steps per nanosecond to
accumulate. Measured three ways before the fix: NVE drift rose **29× per
halving** of dt with constraints on; it was **flat** with constraints off, where
this line never executes; and the f64 host *fell* over the same range, as a
correct integrator should.

**The fix is a conditioning change with no change of algorithm.** SHAKE now
accumulates the correction it applies, and the velocity update is
`v += correction / halfStep`. A ~1e-5 nm number in its own float carries
relative precision: measured per-step velocity error **5.89e-5 → 7.23e-9 nm/ps**
on this box, and the gain grows with the box because the old error scaled with
the span (98× at 0.5 nm, 8100× at 2.82, 120000× at 35).

| dt | drift before | drift after |
|---|---|---|
| 1 fs | 2.2 | 2.747 |
| 0.5 fs | **258.7** | **3.627** |
| 0.25 fs | **1811** | **5.583** |

and the end temperature holds near the setpoint at every step — 299.5 / 297.4 /
299.1 K — where 0.25 fs previously reached **398 K**. One device step against the
f64 host also came 20× closer, 1.15e-3 → 5.76e-5 nm/ps.

🔴 **AND THE f64 CHECK CAUGHT A FACTOR-OF-TWO BUG BEFORE IT REACHED THE KERNEL.**
The two forms must agree to roundoff in double precision, and the first version
did not: it differed by 1.5e-2 relative after a *single* step. The cause is that
iterating constraints touches a water oxygen twice — it is in O–H1 and O–H2 —
which was harmless when the velocity was *overwritten* with the same value both
times and is a doubled correction the moment the update becomes an addition. The
host now iterates unique constrained atoms; the device already did, because one
thread owns a cluster and the cluster's atom list has no repeats. After the fix
the two forms agree to **1.16e-13** relative.

**What this did not fix.** At 2 fs — the shipped default, where the f32 kick
mattered least — the 20 ps NVE drift moved from +0.484 to +1.222 against a
fluctuation of 9.3, which is inside the noise of that measurement rather than an
improvement or a regression; the trajectory also changed, so the two numbers are
not the same experiment. The **14b** half of the finding, the cooling at large
dt, is untouched and still unexplained: it is not this line, because the line is
algebraically correct and `check:timestep` now shows the constrained arm flat.

## The skin detector was measuring the wrong window

**It reported only the steps since the last rebuild inside the block it read.**
`#encodeForces` cleared the displacement accumulator at every rebuild, *inside*
the command buffer, while `#checkSkin` read it once per submitted block — and
`web/app.js` submits blocks of `rebuildInterval` steps (the user's requested
ceiling, 20) while the adaptive interval starts at 4. So a default block held
five rebuilds and the host saw the last four steps of twenty.

🔴 **AND THE RESOLUTION FELL EXACTLY WHEN IT MATTERED.** After a violation the
interval halves, so the fraction of each block the host could see shrank
further — the detector got blinder the moment it had something to report.

Controlled: the same 100 steps, the same rebuild interval, only the submit
boundary moved. The final position checksum is identical to 13 digits in all
three, so nothing about the physics differs.

| blockSize | worst usage | worst displacement | missed-pair bound |
|---|---|---|---|
| 4 | 47.23 | 0.09446 nm | 5.649 |
| 20 | **20.34** | 0.04068 nm | 1.346 |
| 100 | **16.37** | 0.03274 nm | 1.077 |

So every user-facing staleness number — `skinViolations`, `worstDisplacement`,
`worstSkinUsage`, and the `missedPairForceBound` the page prints as the
quantitative reason a run is still trustworthy — was understated by **2.3× to
5.2×** purely by a submit granularity that has nothing to do with the physics.

**What the fix is.** The displacement kernel keeps **two** slots and takes the
same per-step value into both. Slot 0 answers "has the list gone stale since it
was *built*?" and is cleared at every rebuild, as before. Slot 1 is cleared once
per **block** and so records the worst per-interval displacement anywhere in it.
Both are one `atomicMax` on a bit pattern in a diagnostic kernel, not the force
path.

🔴 **TAKING THE SAME VALUE INTO BOTH IS WHAT MAKES SLOT 1 THE RIGHT QUANTITY**
rather than a monotonically growing one. Each interval measures from its own
rebuild origin, so a maximum over those is the worst *per-interval* figure — not
the displacement since the start of the block, which would overstate.

**Gated by exact equality, not a tolerance.** The device reduces the same f32
values by `atomicMax` in every arm, and a maximum over a set is the same bits
however it is grouped — a tolerance would hide the defect the arm exists for.
Falsified: restoring the shipped window reproduces 47.23 / 20.34 / 16.37 while
the trajectory checksum stays identical.

**What is deliberately not compared, and why.** `skinViolations` counts *blocks*
that contained a violation, which the page's log says out loud ("over N
block(s)"), and `stepsWithStaleList` attributes the whole block when any
interval in it was stale — conservative, and now independent of submission.
With the adaptive controller **live**, block size still changes the run: a
different observation cadence makes the controller settle at a different
interval, so the trajectory itself differs. That is the determinism finding in
`docs/REVIEW.md`, not this one, and it is now the only remaining block-size
dependence. At the page's own shape, blockSize 20 and 10 agree exactly where
they previously read 0.991 and 1.161.

## The barostat was deciding on the wrong energy

**`#evaluatePotential` summed every term except the positional restraint.** It is
what `attemptVolumeChange` feeds to `metropolisWeight`, and `energy()` — which
only *reports* — did include the restraint. So through the whole shipped NPT
equilibration the reported potential energy was right and the accept/reject
decision was wrong. OpenMM's `MonteCarloBarostat` takes
`getState(State::Energy, ..., integrator.getIntegrationForceGroups())`, which
includes the notebooks' own `CustomExternalForce`.

🔴 **AND IT IS A BIAS ON THE VOLUME, NOT A SYMMETRIC PENALTY.** Centroid scaling
sends each displacement `d₀ → d₀ + (s−1)R`, so the restraint energy picks up
`2k(s−1)·Σd₀·R` — linear in `(s−1)`, therefore directional — as well as
`k(s−1)²·Σ|R|²`, which resists any move. Dropping both means the barostat
neither feels the restraint's resistance to changing the volume nor its
preferred direction. Measured at the page's own equilibration settings the
omitted term was **6.906 kJ/mol, about 2.8 kT**, against a whole Metropolis
weight of −2.510.

🔴 **`#encodeForces` ALREADY ENCODED THE RESTRAINT PASS**, so the trial *forces*
carried it and only the energy sum did not. That is why the trajectory looked
right and nothing in the output showed it.

**Why no gate saw it.** `check-barostat.mjs` ran entirely unrestrained, and
`check:openmm`'s restrained arm compares energies *at a configuration* and never
a decision. Two correct oracles with the defect in the gap between them. The new
arm runs the page's own settings, warms the solute off its reference so the term
has a value at all, and compares the trial potential against the reported one.
Falsified: with the term dropped the gap is **99.998% of the restraint energy**.

**Blast radius: bounded.** Equilibration is short and production releases the
restraint and re-equilibrates the density itself, so this corrupted the
equilibration density rather than the production ensemble.

## The neighbour capacity did not follow the density

**`#checkNeighbourCapacity` was called from exactly one place — `initialise()`.**
`run()` never asked again, and `#updateBox` carried the construction-time
capacity forward unchanged, so a barostat compressing the box raised the density
against a fixed capacity. `grownPlan` and `NeighbourOverflowError` both existed
and were **unreachable from the step loop**.

Measured on `ala-wat-big`, contracted the way a barostat goes: the requirement
crossed its 1499 capacity at **97.7 atoms/nm³** — bulk TIP3P density, which is
exactly where the shipped NPT equilibration is documented to land.

🔴 **AND THE DROPPED PAIRS WERE NOT AT THE CUTOFF**, so the skin detector's
`missedPairForceBound` did not bound them: the kernel truncates each over-full
list at whatever the cell stencil enumerated last, which is an arbitrary
distance. Against an f64 all-pairs host loop at identical coordinates, a
973-of-840 overflow cost **43.56 kJ/mol** of LJ, **42.12** of Coulomb and a worst
force component of **2803 kJ/mol/nm** — rms force error 9.8e-3 against 1.1e-6 for
the control.

🔴 **THE DIAGNOSTIC WAS A SNAPSHOT, NOT A MAXIMUM.** `required` is cleared inside
the command buffer at every rebuild, so the host read the *last* rebuild's
requirement — the overflow at 97.7 atoms/nm³ read "ok" at 1482 one step later
and had vanished from the log.

**What the fix is.** Prevention and detection:

- `#updateBox` **re-estimates** the capacity for the new box and takes the larger
  of that and the carried value, so the buffer grows with density and does not
  churn when the box expands again.
- `run()` checks the capacity at every block boundary — one `u32` at a boundary
  that already synchronises for validation, so it costs what `#checkSkin` costs —
  and keeps a high-water mark across the run.
- `neighbourSlack` and `maxNeighbours` are now **forwarded** from the caller.
  They were silently dropped, which also made a mid-run overflow impossible to
  provoke on purpose — a large part of why the detection path was missing.

**What the gate proves, exactly.** With the fix removed, capacity stays at 1499
while the requirement climbs 1071 → 1395 over 65 → 103 atoms/nm³, and the
high-water mark reads 0 because nothing asked. Both arms fail. The *spontaneous*
overflow is **not** reproduced by this arm: on its 40-step relaxation schedule
the headroom holds, where the review's 20-step schedule left the local density
peakier. Arm 2 forces the condition directly instead, pinning the capacity to 64
and asserting the run grows it to 918. Asking for a wider skin does not change
it either — 0.4 nm is already what `skinFor` derives for that box.

## The README claimed something the page could not honour

**"No install, no queue, no upload" and "Nothing leaves your machine … There is
no server to send a structure to"** — while `index.html:224` ships a **Share to
LogMD** button that posts frames to a third party. The page's own caveat was
honest about it; the README never mentioned the feature, and the README is the
document a reader meets first. It is also the project's central claim.

Corrected to say that nothing leaves the machine *unless you ask it to*, with the
one opt-in exception named. Three further false or stale statements in the same
document went with it: that **nothing in a step reads back to the host** (the
barostat does, once per volume move, 25 steps by default, at a measured
0.167 ms/step), that **the viewer reads the simulation's own position buffer**
(it is handed host-built PDB text — the Mol\* switch made that true), and the
0.992 g/cm³ density, now quoted as the converged **1.001 ± 0.013** against
OpenMM's 0.9997 ± 0.014 with the compressibility that the fluctuation implies.

🔴 **GATED, BECAUSE THIS IS THE RECURRING FAILURE MODE.** A test reads
`index.html` for the share button and, if it is there, requires the README to
name LogMD and to qualify the claim — and fails on the exact absolute phrasings
that were wrong. It also re-checks the half of the claim that is true: no
absolute-origin `src` or `href` anywhere, so opening the page makes no request.

## The PME reciprocal lattice, which was transposed

**Every non-orthorhombic box was wrong, and every validated number in this
document was unaffected.** `fractional` builds `s_d = Σᵢ rᵢ·R[i][d]`, so the
phase `k·s` gives a Cartesian wave vector `mᵢ = Σ_d k_d·R[i][d]` — **row** `i`
of the inverse box. Both `cpu/forces/pme.js` and `webgpu/kernels/pme.js` read
the **column**, which is the transpose.

🔴 **AND FOR A RECTANGULAR CELL THE TRANSPOSE OF A DIAGONAL MATRIX IS ITSELF.**
So the error was *exactly zero* on every fixture this repository had — all of
them built by `solvateBox` — and 91 kJ/mol on the first `solvateOct` box anyone
tried. Measured by patching only those three lines:

```
ala-wat  (rectangular)  shipped 933.776245   row-form 933.776245   diff  0.000000 kJ/mol
oct      (octahedral)   shipped 726.027026   row-form 635.144433   diff 90.882593 kJ/mol
```

🔴 **THREE SEPARATE INSTRUMENTS WERE POINTED AT THIS AND NONE COULD SEE IT.**

| instrument | why it was blind |
|---|---|
| OpenMM per-term energies | no triclinic fixture existed — `make_fixtures.sh` only ever called `solvateBox` |
| the self-gradient check | **structurally incapable.** The wave vector enters only the kernel *weights*, which do not depend on position, so the analytic force stays the exact gradient of the *wrong* energy. A central difference on the octahedral cell passes at relative error `0.00e+0` — verified, not assumed |
| the WGSL-against-host differential | both carried the same construction, so they agreed about a shared error. This is what `AGENTS.md`'s "a reference must share no code with the thing it validates" is for, in a form the `shared/ ← cpu/ ← webgpu/` layering does not catch |

**What gates it now.** An **exact Ewald k-space sum** in `test/physics.test.js`,
written from the textbook formula and sharing no code with the thing it
validates. It runs on both a rectangular and a truncated-octahedral fixture and
holds each to the two oracles' own floors (1e-5: the exact sum converges to
1.6e-6 between nmax 8 and 11, and a 48³ grid sits ~2e-6 from a 64³ one). Two
further arms make it impossible to pass with the bug reintroduced: the
transposed convention must be grossly wrong on the sheared cell **and
bit-identical on the rectangular one**, which is the whole explanation of why
the defect survived.

🔴 **THE ORACLE WAS WRONG FIRST, AND THE RECTANGULAR ARM IS WHY THAT WAS CAUGHT.**
The first version of the exact sum used a prefactor of `2π/V` where
`k = 2πm` requires `1/(2πV)` — out by exactly `4π² = 39.478`. It disagreed with
the *known-good* rectangular fixture by that factor, which is the only reason it
was not mistaken for a finding. An oracle that has not been checked on a case
whose answer is already known cannot be trusted on the case under test.

**`test/fixtures/ala-oct.*` is in `make_fixtures.sh`**, and 8.0 Å of padding is
not arbitrary: for an AMBER truncated octahedron `c_z = 0.8165·L`, so a 6.0 Å pad
gives a smallest half-width of 0.97 nm — under the 1.0 nm cutoff, which
`buildSystem` correctly refuses. 8.0 Å gives half-widths of 1.40 / 1.32 / 1.14 nm
and 1315 atoms.

### What this did NOT fix, measured

🔴 **THE DIRECT-SPACE TERMS STILL DISAGREE WITH OPENMM ON A SHEARED CELL.**
Against OpenMM at the same alpha and the same forced grid, `ala-oct` leaves
**LJ 0.082 kJ/mol (3.07e-5)** and **Coulomb 2.115 kJ/mol (1.36e-4)**. Both are
**identical at 24³ and 48³**, so neither is the grid floor and neither is the
reciprocal sum — which the exact lattice sum now puts at 2.4e-6. Hypotheses
eliminated by measurement, so nobody repeats them:

- *the LJ tail correction* — accounts for it exactly at the wrong volume
  (−60.62 kJ/mol) but the existing references are dumped with the correction
  **off**, and matching that convention leaves the 0.082;
- *a non-minimal minimum image* — the `c→b→a` reduction is non-minimal for
  14.5% of pairs on this box, but for **0 of 863,955** does it change the 1.0 nm
  cutoff verdict. All three perpendicular widths are 2.287 nm, so half of them
  is 1.1435 nm, comfortably above the cutoff: the box-width guard is what makes
  the simple reduction safe, and it is working;
- *the reduction order* — `a→b→c` disagrees with `c→b→a` about the cutoff on
  22,606 pairs, which is three orders of magnitude too many to explain
  0.082 kJ/mol.

So the octahedral case is deliberately **not** in `check-against-openmm.mjs`:
adding it with a tolerance wide enough to pass would be loosening a tolerance to
make a check pass. The reciprocal sum is gated; the direct-space residual is an
open finding in `docs/REVIEW.md` with its own evidence.

## Stop used to cost the whole stage

**Every write-back sat after the step loop.** `hostState`, `finalState`,
`writer.finish()`, `trajectories.push()` and both download buttons were all
below the `while`, with only `finally { simulation.release(); }` in between — so
`stopSignal.throwIfStopped()` threw straight past the lot. Found by a review
pass, not by a gate; the existing pause gate only ever checked that the step
counter was held.

Measured on the page: Stop at step 2,500 of an equilibration with **5 frames
already written**.

| | before | after |
|---|---|---|
| trajectory | `[]`, 0 bytes | `equil:3`, 43 KiB |
| Download DCD | disabled | enabled |
| Download restart | disabled | enabled |
| carried state | step 0, the post-minimise checksum | step 1040, with velocities |
| the log said | `  stopped` | what was kept, and that a further run continues from there |

🔴 **AND THE SAME `catch` SWALLOWED EVERY OTHER FAILURE.** A `DeferredValidation`
failure, a `NeighbourOverflowError`, a lost device and a non-finite state all
left by the same door, so **any** error at hour six cost the whole stage. A
single-stride production — the default — lost all of it; a multi-stride one kept
only the strides that had finished.

**What the fix is.** The commit moved into `finally`, which runs on the way out
whether the loop returned, threw, or was stopped. Three things that had to be
got right, each of which is now a separate assertion:

- 🔴 **WHAT IS SAFE TO KEEP DEPENDS ON THE STATE, NOT ON WHY THE LOOP ENDED.**
  Classifying the error would mean enumerating every way a run can stop and
  getting one wrong silently. The commit reads the state and asks whether it is
  a number: a clean stop hands its state on, a blown-up one must not, and a lost
  device cannot be read at all. The trajectory is kept in all three cases,
  because it is held in the page rather than on the GPU.
- 🔴 **A FAILURE TO SAVE MUST NOT REPLACE THE FAILURE THAT CAUSED IT.** An
  uncaught throw inside `finally` discards the in-flight exception, so a
  blow-up would be reported as whatever went wrong while saving it.
- 🔴 **THE RESTART NOW NAMES ITS STAGE.** Every stage wrote `<name>.rst7` and the
  button stayed enabled across stages, so stopping a production and pressing it
  handed over the **equilibration's** state under a filename that said nothing.
  That is the stage-handoff defect over again, reached through Stop. Files are
  now `<name>_equil.rst7` / `<name>_prod_1.rst7`, and the log says which stage
  and whether it was stopped early.

🔴 **ONE DEFECT IN THE FIX, FOUND BY ITS OWN GATE.** The first version returned
early when `done === 0`, on the reasoning that no steps meant nothing to save.
It is wrong: the opening frame is appended *before* the loop, so a stage that
goes non-finite inside its first block has one valid frame — the structure it
started from — and the early return discarded it and also skipped the message
explaining why nothing was carried forward. The blow-up arm caught it, reporting
`no step completed, so this stage produced nothing to save` where the stop arm
was green.

**Also corrected, because they were in the same code.** The skin line is no
longer printed for a stage whose state is not a number — the displacement
detector throws before it records a maximum, so a dead run was reporting
`worst displacement 0 pm — 0% of the budget used`, and the lane now fails if
that string appears at all. And the DCD download no longer logs
`wrote <name>.dcd` after writing `<name>_equil.dcd` and `<name>_prod_1.dcd`; the
one filename a reader could copy was the one that did not exist.

**What is still open from this.** A stopped stage is deliberately **not** marked
complete, so production stays locked until equilibration finishes — but the
state is carried, so pressing Equilibrate again continues from where it stopped
rather than restarting. Nothing is written to disk mid-run, so closing the tab
still loses everything held in the page; that is a separate entry in
`docs/REVIEW.md`.

## The blown-up run that completed

**Nothing in this port asked whether the state was still a number.** A grep over
`webgpu/`, `shared/`, `cpu/` and `web/` found no finiteness or magnitude test on
positions, forces, energy or temperature in any run path; the only
`Number.isFinite` calls were cosmetic formatters. Found by a review pass, not by
a gate — see `docs/REVIEW.md`.

🔴 **SO A DEAD RUN LOOKED LIKE A FINISHED ONE.** Measured on the page at a 40 fs
timestep, which `index.html` accepted without comment because the field had no
maximum: every atom non-finite after 50 steps, and then the run *completed* —
300 steps encoded at full speed, 13 frames written, both download buttons
enabled, the next stage unlocked, and a log line reading
`300 steps in 7.7 s — 25.79 ms/step`. The tiles read `–` and the CSV read `NaN`.

🔴 **AND NO TOOL DOWNSTREAM WOULD HAVE SAID SO EITHER.** The all-NaN DCD was read
by **cpptraj** with no warning at all (`Read 10 frames and processed 10 frames`)
and by **MDAnalysis** without one, so a user who took the file away had nothing
to tell them. The restart writer was worse than silent: its twelve-column guard
catches an atom that has flown away, but `Number.NaN.toFixed(7)` is the
three-character string `"NaN"`, which passes comfortably — so the writer emitted
a restart that **this port's own reader then refused**
(`non-numeric coordinate field "NaN" on line 3`).

**The information was already on the host, every block.** `#checkSkin` reads the
reduced squared displacement at every block boundary, and a non-finite position
makes that reduction non-finite — so the signal arrived every block and was
discarded. Worse than discarded: `Math.sqrt(Math.max(NaN, 0))` is `NaN`, which
left `worstDisplacement` at 0 while failing `usage <= 1`, so the one loud thing a
dead run printed was

```
skin: 243 pm, worst displacement 0 pm — 0% of the budget used
⚠ the neighbour list was stale for 300 of 300 steps … this costs
  exactness rather than stability — but the run is not exact.
```

two consecutive lines that contradict each other, ending in a reassurance that
is exactly backwards. The rebuild interval was also driven to 1, so a dead run
got 20× slower and read as a performance problem.

**What the fix is.** Three layers, in `shared/finite.js` and its two callers:

| where | cost | catches |
|---|---|---|
| `#checkSkin`, every block | **nothing** — the `u32` was already being read | positions, within one block |
| `energy()`, every report | one comparison | velocities a step before positions follow, and an input that is infinite on the first evaluation |
| `DcdWriter.appendFrame`, `writeRestart` | one scan per frame | the last line of defence: no file this program writes can contain `NaN` |

The free detector is sound rather than lucky because the displacement kernel
reduces `bitcast<u32>` of a squared displacement, and `NaN` (`0x7FC00000`) and
`+Inf` (`0x7F800000`) both sit above any finite squared displacement a
simulation can reach — a 1 nm displacement is `0x3F800000` — so a poisoned value
always wins the `atomicMax`.

🔴 **THE REFUSAL IS DELIBERATELY NOT IN `#evaluatePotential`.** The minimiser's
line search calls it on trial points, and a line search must be allowed to
evaluate a bad step and reject it; throwing there would turn a working back-off
into a failed minimisation. That path is already safe for a reason worth stating:
the accept test is `trial < potential`, and every comparison against `NaN` is
false, so a non-finite trial is always rejected.

**Falsified both ways.** With the detectors removed, `npm run test:gpu:blowup`
reports the original symptom exactly — `threw: false`, `stoppedAtStep: 200`, the
blown-up run running to completion — while the 2 fs arm still passes, so the
gate cannot pass by refusing everything. The two writer arms in `npm test` fail
with the guards removed and the restart round-trip then throws from the *reader*
instead of the writer, which is the original defect.

**What is still open from this.** The timestep field still has no maximum
(`index.html:73`, `min="0.5"`, where the notebooks' dropdown is 0.5–4 fs), so
the easiest route into this state is still one keystroke — now refused rather
than silent, but better prevented. And a refusal mid-run still costs the stage,
because every write-back in `web/app.js` sits after the loop; that is a separate
defect with its own entry in `docs/REVIEW.md`.

## What the viewer costs now

Mol\* replaced 3Dmol on a deliberate trade, and the numbers are here so the
decision can be reversed by someone who disagrees with it:

| | raw | gzipped |
|---|---|---|
| 3Dmol (removed) | 512 KiB | 149 KiB |
| **Mol\* 5.13.1 + css** | **5217 KiB** | **1486 KiB** |
| everything else that ships — parser, force field, PME, integrator, barostat, analysis, page | 461 KiB | 174 KiB |

So the viewer is now about **nine times the program it is a viewer for**, and a
first visit went from roughly 310 KiB to 1.66 MiB gzipped. It buys Mol\*'s
cartoon, its own secondary-structure assignment and its lighting. Both are MIT
and both were vendored rather than fetched from a CDN, so the page still works
with no network and still sends no request when it opens.

🔴 **AND A FRAME UPDATE IS NOW A RELOAD, WHICH 3Dmol DID BETTER.** 3Dmol exposes
`model.atoms` as a plain mutable array, so an animation frame was three writes
per atom. Mol\* holds coordinates in an immutable state tree, so the honest
implementation is clear-and-load. That is the one thing that got worse, and it
is why the live animation stays opt-in.

## The neighbour-list skin, which was wrong

**The default skin was 0.2 nm for every system, timestep and temperature, and
the solvated fixture needs 0.342 nm at a 20-step rebuild interval.** So the
neighbour list could be missing pairs that had come inside the cutoff, and
nothing reported it. The list holds pairs out to `cutoff + skin`, and a pair
outside it can only come within the cutoff if the two atoms close the gap by
more than `skin` — each contributing at most its own displacement — so the list
is exact exactly while `2·max|r − r_built| ≤ skin`. That question was never
asked: the skin was assumed to cover the motion.

**Why no existing gate could see it.** `docs/STATUS.md` previously recorded a
five-arm skin/interval sweep showing −1.3 to +3.7 kJ/mol/ns/atom with no
ordering, read as "fluctuation rather than staleness". Both were true: the drift
test cannot resolve a handful of missing pairs, because a pair that sneaks
inside the cutoff arrives *at* the cutoff, where the screened Coulomb force is
`erfc(α·rc)/rc` and the LJ force has fallen off as 1/r⁷. Measured, that is
**0.71 kJ/mol/nm** against an rms force of 516 — 0.14%. A geometric criterion
sees it; an energy-drift criterion cannot.

**What the fix is.** A device-side kernel reduces `max|r − r_built|` by
`atomicMax` over the raw bit pattern of the squared displacement (exact for
non-negative floats, so no fixed-point scale to choose), read at the block
boundary that already synchronises for validation. The rebuild interval then
becomes a *ceiling*: the run starts at 4 steps, grows by 1.25× only after three
consecutive blocks that used under 40% of the skin, and halves on any violation.

**And the interval cannot be predicted, which is why it is measured.** The worst
displacement scales as roughly `n^0.25` over the useful range — slower than
diffusive, because it is a maximum over a thousand atoms and the fastest one's
ballistic phase is short. Measured against the free-flight bound `v·n·dt` it is
3.17× at 2 steps, 4.16× at 5, 2.71× at 10 and 1.57× at 20. A linear model
overshoots by 3×, a diffusive one by 2×, and a bound using the worst ratio asks
for 0.87 nm of skin — more than half the test box.

**The default skin is now derived** from the timestep (linearly — a displacement
is a speed times a time), the temperature (as √T) and the lightest atom's mass
(as √(1/m)), anchored on the measurement at 2 fs / 298 K / hydrogen. It is also
**capped by the box**: `cutoff + skin` must stay under half the smallest width,
and on the test fixture that cap binds — a 2.54 nm box with a 1.0 nm cutoff
leaves only 0.243 nm for a skin that wants 0.342. The honest consequence is a
shorter interval, which the controller finds.

**And the old default was slow as well as wrong.** Measured with
`npm run bench:skin`, interleaved, two rounds, minimum per arm:

| skin / interval | ms/step | relative | neighbours | skin used |
|---|---|---|---|---|
| 0.2 nm / every 10 (the old default) | 1.379 | 1.000 | 648 | **131.5%** |
| 0.35 nm / every 20 | 1.266 | **0.918** | 854 | 97.6% |
| 0.4 nm / every 20 | 1.541 | 1.117 | 912 | 85.4% |
| 0.5 nm / every 20 | 2.130 | 1.545 | 1015 | 68.3% |
| 0.5 nm / every 40 | 1.657 | 1.202 | 1015 | 93.1% |

Rebuilding costs far more than carrying extra neighbours on this GPU, so the
old default sat in the worst quadrant: unsafe *and* 9% slower than a wider skin
with half the rebuilds. The page's rebuild default moved from 10 steps to 20.

## Known limitations

- **The host reference is ~47–79 ms/step for 1190 atoms**, against 0.8 ms on
  the device. It is the oracle, not a simulator.
- **Velocities drawn at a temperature must be projected onto the constraint
  manifold before the first step, and forgetting it looks exactly like
  integrator drift.** `setVelocitiesToTemperature` draws 3N independent
  components; this system has 2391 degrees of freedom, so at 298 K the draw
  carries 4422.7 kJ/mol where the manifold wants 2962.1. RATTLE discards the
  difference on step one — a **1510 kJ/mol** drop in the total energy between
  the first two reports, which read as a catastrophic drift until the trace
  showed the whole of it was in the first interval. `createState` now projects
  velocities as well as positions.
- **The FFT is complex-to-complex on a real input.** The charge grid is real, so
  a real-to-complex transform would halve both the memory and the work. Not
  done: it changes the index map, and the current one is verified against a
  naive DFT over 33 sizes, which is worth more right now than the factor of two.
- **The integrator is BAOAB (`LangevinMiddle`), not the notebooks'
  `LangevinIntegrator`.** Deliberate: BAOAB's configurational averages are
  correct to higher order in the timestep, and OpenMM now recommends it. The
  consequence is that a RainMD trajectory and a notebook trajectory from the
  same seed are *different trajectories sampling the same ensemble*. Energy and
  force comparisons against OpenMM must be single-point, which is what they are.
- **The minimiser is steepest descent, not L-BFGS.** It reaches a given gradient
  norm in more iterations than OpenMM's. A short run started from it is visibly
  hot — 358 K against a 298 K setpoint in one measured run — because the
  structure is still releasing ~675 kJ/mol of potential energy into a thermostat
  that has had less than half a relaxation time to remove it. That is physics,
  and it was confirmed by raising the friction (see the table), but it means
  "minimise then run briefly" is not a good demonstration.
- **The barostat is isotropic and synchronises with the host.** A Metropolis
  test is a decision about a number, so the trial energy must come back before
  the next step can be encoded — one round trip per `barostatInterval` steps
  (25 by default). It cost 1.10 ms/step against 0.86 in NVT on the test system.
- **A volume move that would stop the cutoff fitting the box is refused.**
  `buildSystem` checks that at construction, but the barostat changes the box
  afterwards, and once the box is under twice the cutoff the minimum-image
  convention breaks silently. On the deliberately underdense test fixture the
  box contracted to a 1×2×1 cell grid without ever triggering the refusal, but
  it was heading that way.
- **The device minimiser stalls before a tight force tolerance, and that is
  f32.** The line search compares two potential energies, and the potential is
  a sum of ~1190 f32 contributions totalling ~13000 kJ/mol — so it resolves to
  about 0.03 kJ/mol, below which "did the energy go down" is a coin flip. It
  reaches a *lower* energy than the f64 host minimiser does (−13450 against
  −13408) and then stops at a max force near 2100 kJ/mol/nm, which runs stably.
  A tighter minimisation needs the host path.
- **The neighbour capacity is a density estimate, and the estimate is low.** On
  the 1190-atom fixture the mean-density figure is 323 neighbours per atom and
  the densest atom actually has **526** — 1.63× the mean, where Poisson
  counting alone would predict about 1.17×. The difference is structural
  correlation: bonded neighbours, hydration shells, and a protein interior
  denser than the bulk around it. The default slack is 2.0, the kernel reports
  the true requirement through an `atomicMax`, and the caller grows and
  rebuilds. That recovery is tested by forcing an 8× overflow.
- **The prefix sum over cells is a single workgroup** walking the cells in
  chunks. It runs once per rebuild over a few hundred to a few thousand cells,
  so it is microseconds; it is the first thing to parallelise if a profile ever
  disagrees.
- **`HAngles` constraints are refused by name**, not silently approximated.
- **Legacy 10-12 hydrogen-bond terms are refused**, with a message saying to
  reparameterise. No force field the notebooks use emits them.
- **NetCDF restarts are refused by name**; ASCII `inpcrd`/`rst7` only.
- **Truncated-octahedron boxes** are implemented through the general triclinic
  path but have not been tested against OpenMM — the notebooks use
  `solvateBox`, so every validated number here is for a rectangular cell.
- **Node's WebGPU lane is pinned to `webgpu@0.3.0`.** 0.5.0 and 0.6.0 fail to
  load on macOS 14.5 (`Symbol not found: _OBJC_CLASS_$_MTLLogStateDescriptor` —
  they want a Metal from macOS 15). Both lanes are run: `npm run test:gpu`
  (Dawn) and `npm run test:page` (Chrome). The two implementations do not agree
  about everything — Chrome reports a 4096 MiB storage binding limit and
  `shader-f16` present, Dawn reports 9663 MiB and no `shader-f16` on a default
  device — which is exactly why both are run.
