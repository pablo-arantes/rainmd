# What is validated, and what is not

Every number here was measured by a command in this repository, on
2026-11-06, on an Apple M-series Mac (Metal, macOS 14.5) with OpenMM 8.4.
Re-run them with `npm run check:openmm`, `npm run check:gradients`,
`npm run test:gpu`, `npm run check:mdanalysis` and `npm test`.

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
| **direct-space nonbonded kernel** | the host reference | LJ 5.1e-7 – 6.0e-7, Coulomb 4.9e-8 – 3.1e-7, worst force 6.5e-3 kJ/mol/nm against rms ~500 (1.3e-5) |
| **mixed-radix FFT index map** | a naive DFT, on the host, before any WGSL was written | 33 sizes (every 2/3/5-smooth value to 128), both directions, worst 9.5e-16 |
| **WGSL FFT** | the naive DFT, on random data, 4 grids including non-cubic | 2.9e-7 – 6.6e-7 over 9–12 stages |
| **PME spread grid** | the host's own spread, compared point by point | 3.0e-6 – 3.2e-6 relative; fixed-point headroom **0.15% of 128** charge units used |
| **PME energy and forces** | the host reference, 3 grids | energy 6.3e-8 – 2.1e-6, worst force 1.9e-3 kJ/mol/nm against rms 96 (2.0e-5) |
| **the whole GPU force path** | **OpenMM**, notebooks' configuration | bonded 1.8e-8, LJ 6.0e-7, Coulomb (direct + reciprocal + self) 1.4e-6, **total 1.5e-6**, **per-atom forces 8.8e-5** |
| counter-based PRNG uniform | the host's own, in f32 and f64 | **40000/40000 bit-identical** (the 2^-24 form it replaced: 20045/40000) |
| **one device step** | the host integrator, with the same noise *and* the same forces | positions 2.0e-6 nm, velocities 1.1e-3 nm/ps against a 4.8 nm/ps scale |
| **constraint geometry after 5000 device steps** | the constraint lengths themselves | worst 5.0e-6 of a bond length; 17 SHAKE iterations at most; iteration cap never reached |
| **energy conservation**, friction 0, 20 ps | a least-squares fit to the total energy | **+0.30 kJ/mol per ns per atom**, fluctuation 3.45 kJ/mol, residual about the fit 2.73 |
| thermostat, friction 10/ps, 400 steps | the 298 K setpoint | mean 317.8 K |
| **DCD writer** | **byte-compared against `openmm.app.DCDFile`** | identical except the timestamp title and 2–7 ulp in the unit-cell cosines; MDAnalysis reads both files to the same 7.6e-7 Å |
| DCD read back by **cpptraj** | a 4 ps GPU run, 40 frames | opened as "Charmm DCD", 40 frames processed, box and `dt` correct |
| **constraints, measured from the written trajectory** | MDAnalysis, on the water O–H bond across 40 frames | mean **0.95720 Å**, spread 3.3e-6, worst deviation 7.0e-6 relative to the constrained length |
| AMBER restart writer | this port's own `inpcrd` reader | positions exact, velocities 1.0e-7 nm/ps (the format's precision) |
| **the page, driving a real GPU run** | headless Chrome, through the UI controls | GPU minimise, then 4000 NPT steps at 2.01 ms/step on the two-body fixture; nine frames; all five exports offered; render pass validated |
| **LJ tail correction** | **OpenMM**, isolated by toggling `useDispersionCorrection` | **-41.537158 kJ/mol, relative error 1.0e-8** |
| **barostat energy at a forced volume scale** | OpenMM, at the same molecule-COM-scaled coordinates | 1.5e-6 to 1.7e-6 relative at length scales 1.00, 1.02 and 0.97 |
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
| **every kernel in both lanes** | the same five check modules run through Dawn and through Chrome | 0 failures in both; the full force path agrees between implementations to 1e-8 on the total |
| determinism | two independent runs of the nonbonded check | **bit-identical** energies, including the arm that reaches them via a grow-and-rebuild |

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
- **f32 `erfc` floor, 1.19e-6 relative.** WGSL has no `erfc`, so the kernel uses
  the Numerical Recipes rational form. Its published accuracy is 1.2e-7
  *fractional* — in double precision; in f32 over the domain a 1.0 nm cutoff
  uses it is 1.191e-6, twenty times f32 epsilon. Refitting all ten coefficients
  by least squares against relative error over exactly that domain gives
  **1.191e-6** — the same number to four digits, which is the evidence that the
  limit is the arithmetic and not the fit: `exp(-x² + P(t))` has an argument
  near −7.8 at the cutoff, where one f32 ulp is 4.7e-7 of absolute argument
  error and an exponential turns that straight into relative output error. The
  nonbonded test's Coulomb bar is derived from this, not from f32 epsilon.

## Not yet written

**The simulation is GPU-resident.** Forces and integration both run on the
device; nothing in a step reads back to the host. A block of steps is encoded
into one command buffer, so there is no per-step round trip.

Measured on this Mac (Apple Metal, 1190 atoms, PME 24³, 2 fs, HBonds):
**0.72–1.03 ms/step, or about 215 ns/day** — against 47–79 ms/step for the host
reference, a factor of 50 to 100.

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
