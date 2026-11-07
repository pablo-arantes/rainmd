# What LocalFold does, and what this port borrows

This document is the reverse-engineering result: what was examined, what was
found, and which of it transfers to molecular dynamics. It is the reason the
rest of the tree is shaped the way it is.

## What was examined

`github.com/sokrypton/localfold` at `main`, cloned 2026-11-06 (tree SHA-256
`ee8f9970…`), read as source, plus a REA JavaScript-application analysis over
the runtime, kernel and page modules (`ev_8dd9356a…`). REA parsed 18 selected
sources with **0 parse failures and 0 bundler modules**, which is the machine
confirmation of the claim `AGENTS.md` makes in prose: there is no build step and
no bundle, only ES modules the browser loads as written.

Observed, not inferred: 735 files, 6 network-request relationships in the
selected set (weight fetching), 9 timers, no Electron surface. Coverage was
partial by construction — 717 files were left outside the analysis scope — so
nothing here should be read as a statement about the files that were not
selected.

## The eight things worth taking

**1. No build step.** Plain ES modules, relative specifiers with a `.js`
extension, served as static files. A clone plus `python3 -m http.server` is the
whole development loop. For a scientific tool this is not a style preference: it
means a reader can open any file in the browser's debugger and see exactly the
text that is running, and that a user who distrusts a result can read the code
that produced it without a toolchain. RainMD copies this exactly.

**2. The tree is split by backend, and a reference shares its kernel's name.**
LocalFold has `shared/` (device-agnostic), `cpu/` (reference implementations),
`webgpu/` (kernels and runtime), `web/` (the page), with layering
`shared <- cpu <- webgpu`. `cpu/af3/trunk/embedder.js` sits beside
`webgpu/af3/trunk/embedder.js`. RainMD uses the same split for the same reason:
the CPU file is not dead weight, it is the oracle.

**3. Every GPU kernel has a differential test against an independent
reference.** This is the first line of their `AGENTS.md`, and it is the single
most valuable thing in the repository. The corollary they state and that is easy
to miss: a reference that shares code with the thing under test cannot find a
bug in it. Their reciprocal-space check uses an independent oracle dumped from
JAX; RainMD's uses OpenMM, and its FFT reference is a naive per-axis DFT
precisely so that it shares nothing with the fast transform it validates.

**4. Limits and features are not inherited from the adapter.** A `GPUDevice`
gets the spec defaults unless it asks for more, and the spec default for
`maxStorageBufferBindingSize` is 128 MiB. Reproduced here: this Mac's adapter
advertises `shader-f16` and a device created with no arguments reports it as
**false**. `webgpu/runtime/device.js` requests every limit at exactly what the
adapter reports.

**5. A pipeline cache must be indexed by SOURCE, not only by key.** Every kernel
bakes constants into its WGSL and builds a key by hand; a key that omits one
hands the caller somebody else's kernel, which computes a plausible wrong answer.
LocalFold found two such collisions in shipped code. RainMD's cache compares the
text and reports the first differing line.

**6. Yield with a MessageChannel, never `setTimeout`.** A background tab clamps
timers to one second. LocalFold measured a fold going from two minutes to over
three hundred. MD is worse: a run yields hundreds of times a second, so a
backgrounded simulation does not slow down, it stops. And the yield must be a
real task — awaiting a resolved promise is a microtask and never repaints, so
the Stop button stops working.

**7. Defer validation error scopes.** `await device.popErrorScope()` inside the
step loop puts a host-device round trip between every pair of submits. Push
before encoding, pop without awaiting, read the promises at a boundary that
already synchronises.

**8. Pool buffers, and keep a switch to turn pooling off.** A pooled buffer
retains the previous step's bytes, so a kernel that reads a region it did not
write gives one answer on a fresh buffer and another on a recycled one — which
is deterministic and looks exactly like a race. The switch is how the two are
told apart.

## The mapping

| LocalFold | RainMD |
|---|---|
| model weights, fetched from Hugging Face and cached by the browser | **AMBER `prmtop` + `inpcrd`**, produced by the Making-it-rain notebooks and dropped on the page |
| featurise: sequence → batch tensors | `shared/topology/`: topology → flat GPU arrays, exclusions, constraint clusters, per-atom gather lists |
| trunk / pairformer / attention kernels | bonded, direct-space nonbonded, PME spread/FFT/convolve/gather |
| diffusion sampler loop | the integrator loop: Langevin, constraints, barostat |
| recycles streamed to the viewer | trajectory frames streamed to the viewer |
| CPU reference + JAX oracle dumps | `cpu/` reference + **OpenMM** oracle dumps |
| pLDDT / PAE confidence readouts | energy, temperature, RMSD, interaction energies |
| `Download All` as a zip in the AF3 server layout | DCD trajectory, restart state, logs |

The key structural insight is that **the prmtop is the weight bundle**. The
expensive, irreproducible-in-a-browser work — tleap, antechamber, AM1-BCC
charges — happens once, offline, exactly as training happens offline for
LocalFold. What crosses into the browser is the artifact. This is why RainMD
asks for a topology rather than reimplementing `tleap`: not because the latter is
impossible, but because it is the wrong half of the problem to put in a page.

## Where MD differs, and the design consequences

**A simulation runs the graph a million times, a fold runs it a few.** This
changes three things. Buffer pooling becomes a requirement rather than an
optimisation. Allocation must reach a steady state. And the per-step host work
must be near zero, because a 1 ms host overhead is 17 minutes over a million
steps.

**There is no `atomicAdd` on `f32` in WGSL, and MD wants to scatter forces.**
The obvious kernel is one thread per bond adding into three force components.
Fixed point would be needed, with a scale chosen against forces that span six
orders of magnitude between an equilibrated step and step zero of a
minimisation — and floating-point atomics are not associative, so two runs give
different answers and the differential test has no fixed value to compare
against. RainMD inverts the term lists instead (`shared/topology/gather.js`) so
that there is **one thread per atom and no atomics anywhere in the force path**.
The same step gives the same bits on every device. The cost is recomputing each
term once per atom it touches, which on a system whose work is almost entirely
nonbonded is not measurable.

**f32 has a measurable, derivable cost that must be stated rather than
discovered.** A bond force is `k(r - r0)`, and `r` is a difference of two
coordinates that span the whole box — so its f32 error is about `span * eps` and
the force error is `k` times that, whatever the force itself is. For the
solvated fixture: `476976 x 2.82 x 5.96e-8 = 0.16 kJ/mol/nm`, and the kernel
measured 0.157. The first version of the differential test asked for an
agreement f32 cannot give. It is also, incidentally, a quantitative argument for
why `constraints=HBonds` is standard: it removes exactly the stiffest springs
from the harmonic term, 1185 bonds down to 9 on this system.

**Positions must not be wrapped into the box during a run, but they must be
wrapped when a frame is written.** OpenMM's bonded forces are not periodic, so a
molecule has to stay whole in the coordinate array; wrapping the running state
would split one across the boundary and stretch a bond across the whole cell.

🔴 **THIS PARAGRAPH USED TO CLAIM BOTH HALVES AND ONLY THE SECOND WAS TRUE.**
The imaging on write was never implemented, so every trajectory this port
produced carried unwrapped coordinates — and because water diffuses freely, the
solvent drifts further from the solute the longer the run goes. Opened in VMD it
looks exactly like the waters are flying away from the protein, which is how a
reader found it. The physics was never affected: the nonbonded and PME paths
agree with OpenMM to 1.4e-6 and use the minimum image throughout. The *file* was
simply not what any other tool writes — OpenMM's own `DCDReporter` wraps by
default, so the notebooks' trajectories are wrapped and this port's were not.
Measured against it: 393 of 393 molecule *centroids* inside the cell against 387
of 393 first atoms, so the reference point is the centroid and single atoms are
allowed to stick out by up to a bond length. `shared/traj/imaging.js` now does
that for both the DCD and the PDB writers.

## The second target: JIGGLE's interface

The kernels came from LocalFold. The interface came from
`github.com/BioCoder007/JIGGLE`, cloned 2026-11-06 and read as source — a Qt
desktop application that runs OpenMM locally and reports while the simulation
runs. It is not a WebGPU program and shares no code with anything here; what it
has is the thing this page was missing, which is an answer to "what do I show
someone during a run".

**1. Analysis is incremental, or it does not exist.** `core/analysis.py` keeps
running sums rather than frames:
`RMSF_i = sqrt(<|r_i|^2> - |<r_i>|^2>)` per residue, so memory is
`O(residues)` however long the run goes. That identity is exact, not an
approximation. RainMD's `RunningRmsf` is the same algebra, and
`test/interface.test.js` holds it to a two-pass computation over the same
aligned frames (agreement below 1e-9).

**2. The four-way RMSD split is the whole idea.** `rmsd_fit_on(coords, ref,
fit_idx, measure_idx)` superposes on one atom group and measures another. With
`fit = receptor` and `measure = binder`, the receptor is held still and what is
reported is the binder's movement *relative to the binding site* — which is
invisible to a whole-complex RMSD, because that averages the binder's drift
across every receptor atom that did not move, and invisible to a centroid
distance, which cannot see a rotation or a slide that preserves the separation.
Measured here on the two-body fixture: displace the binder rigidly by 0.5 nm
and nothing else, and the fit-on-receptor RMSD reports **0.5000 nm** while the
whole-complex RMSD reports **0.2430 nm**. In a real 8 ps run the red trace
reaches 0.53 nm while the complex, receptor and binder traces all stay under
0.1 nm. That experiment is a standing test.

**3. Fit on the solute, never on the solvent.** JIGGLE's viewer and metrics both
exclude water by construction. Fitting on everything fits the water box, which
is the one thing that is not supposed to move. RainMD fits on *heavy* solute
atoms — hydrogens are light, fast and constrained, so they add noise without
information.

**4. A viewer that is told the atom count, and refuses a mismatch.**
`ui/viewer_template.html` carries a comment worth the whole file: 3Dmol's PDB
parser computes `stripH = !options.keepH` and silently discards every hydrogen,
so a hydrogenated coordinate array written into a heavy-atom model gives each
atom some other atom's position and scrambles the backbone. Their fix was
`{keepH: true}` plus an explicit count check that fails loudly. RainMD cannot
have that particular bug — the viewer is handed the same topology the
integrator uses, not a re-parsed PDB — but it keeps the lesson: `visibleIndices`
is one resolver used by both the viewer and the PDB export, so the file a reader
downloads is the structure they were looking at, by construction rather than by
coincidence.

**5. Style everything first, then layer.** Their comment records a real bug:
styling only the named receptor and binder chains left every other atom at
3Dmol's default wireframe, which read as "the receptor is broken". RainMD's
viewer assigns a colour to every atom and *then* applies visibility, for the
same reason.

**6. Secondary structure is a colour, not a measurement — so it need not be
DSSP.** JIGGLE gets a cartoon, with helices and strands, from 3Dmol. Doing that
properly means DSSP: backbone hydrogen-bond energies over N, H, C and O, then a
pattern pass over the bond map. What RainMD does instead is the classical
alpha-carbon distance criterion, because an ideal helix and an ideal strand put
their i→i+3 and i→i+4 alpha carbons a factor of two apart — 0.505 / 0.620 nm
against 1.023 / 1.360 — so two numbers per residue separate them. It disagrees
with DSSP at the frayed ends of elements, where DSSP itself turns on a single
hydrogen bond. That is the right trade *here* and only here: nothing in
`metrics.js` reads the assignment and no number in `docs/STATUS.md` depends on
it, so a residue drawn as coil costs a reader nothing. If an assignment ever
feeds an analysis, this is not the function to use, and `secondary.js` says so
in its header.

### What JIGGLE does that RainMD deliberately does not

JIGGLE asks the user for receptor and binder **chain letters**. A prmtop has no
chain IDs at all — AMBER throws them away — so there is nothing to ask for. The
split is taken from the molecule partition instead: the union-find over the bond
graph that `parsePrmtop` already computes, largest solute molecule as the
receptor.

That is strictly less expressive, and the limit is real: it cannot express a
two-chain receptor, where it would report one half of the receptor as the binder
and `binder on receptor` would measure the two halves moving relative to each
other — reading exactly like a pose coming apart. So the page offers the
**molecule list** in place of the chain list, each entry named by its residues
(`ACE..NME, 5 residues`) because "molecule 2" means nothing to anyone, plus a
residue-level override for a boundary that does not follow a molecule. The
derived split remains the default and `splitWasChosen` says which it was, so a
legend never claims more than it knows. The control is frozen while a run is in
flight: `LiveAnalysis` takes the two sets once, at the first frame, and
re-splitting mid-run would leave one column with two meanings.

JIGGLE also routes JS errors into the page title for Python to poll, because
QWebEngineView gives it no console. RainMD runs in a real browser with CDP
available, so the browser lane reads the exception directly.

## The third REA: LocalFold's structure viewer, and why it was not taken

LocalFold draws its predicted structures with **py2Dmol**, a fork of 3Dmol
vendored at `web/vendor/py2Dmol.full.min.js`, fed multi-MODEL PDB text so the
library animates the frames itself. The cartoons are genuinely good and the
wrapper around it, `web/viewer.js`, is only 363 lines — which is where the
impression of a light integration comes from.

🔴 **BUT THE LIBRARY IS 920 KB MINIFIED, 297 KB GZIPPED, AND RAINMD IN ITS
ENTIRETY IS 159 KB GZIPPED.** Parser, force field, PME, integrator, barostat,
analysis, viewer and page together are a little over half the size of the
viewer alone, so vendoring it would roughly triple what a visitor downloads in
order to add one representation. LocalFold's own `index.html` carries a comment
about the consequence: it is "765 KB of blocking classic script", and because
`app.js` is a module that runs only after that has parsed, the page is on
screen and interactive with its Fold button doing nothing at all for the
duration — "milliseconds on localhost, seconds over a network".

Two further reasons it does not transfer, both specific to running a simulation
rather than showing a prediction:

- **It wants the coordinates on the host, as text.** LocalFold predicts a
  structure once and hands over a PDB. RainMD's viewer binds the GPU buffer the
  integrator writes and redraws from a device-to-device copy, which is the
  reason drawing a frame costs nothing measurable. Rebuilding PDB text for
  thousands of atoms at every report would put string formatting on the
  reporting path and give up that property.
- **Its hidden behaviour has to be worked around.** The JIGGLE REA above records
  two real bugs from exactly this library: a PDB parser that silently discards
  every hydrogen unless `{keepH: true}` is passed, and a `molObj` cache that has
  to be nulled by hand for a coordinate update to show.

So the ribbon is native WGSL. What *was* taken is the geometry, which is common
to 3Dmol and PyMOL and is the part worth borrowing: a Catmull-Rom spline through
the alpha carbons, subdivided (eight samples a residue here, 3Dmol uses about
six); a per-residue frame whose direction is made continuous along the chain;
width and colour from the secondary structure; and an arrowhead on each strand.

**The split is host/device, and one step of it genuinely cannot be
parallelised.** The frame's "which way is up" has to be compared with the
previous residue's and flipped when they disagree — a scan, not a map, and
without it a strand corkscrews, because the backbone zigzags so the curvature
direction reverses at every residue. That scan is one iteration per residue,
hundreds of them, against the tens of thousands of vertices the sweep produces.
So the host builds guides and the vertex shader sweeps them procedurally from
`vertex_index` alone — no vertex buffer and no index buffer, the same trick the
sphere impostors already use, so a structure change rewrites a few hundred
guides rather than uploading geometry.

## The fourth REA: LogMD, and why it cannot replace this

`github.com/log-md/logmd`, cloned 2026-11-07, read as source. 416 KB, MIT,
`pip install logmd`. Its pitch is "Share MD While It Runs" and it does that
well — but it is a *client for a service*, not a library, and that decides it.

**Measured, from the repository:**

- **The viewer is not in it.** No `.js`, `.ts`, `.html`, `.svelte` or `.vue`
  anywhere. What ships is the uploader; the viewer is a hosted web app at
  `https://rcsb.ai/{run_id}` whose bundle is served from
  `logmd.b-cdn.net` and whose CDN refuses programmatic fetches.
- **Every frame is POSTed to a third party.** `upload_worker_process` sends
  `{file_contents, run_id, frame_num, token, user_id: token.email, data_dict}`
  to `https://alexander-mathiasen--logmd-upload-frame.modal.run`. The frame is
  a PDB string. The token is cached at `~/.logmd_token`.
- **It is Python.** Its two integration points are an ASE `dyn.attach` and an
  OpenMM `Reporter` (`describeNextReport` / `report`). RainMD has no Python at
  runtime; it is a page.
- **It does not compute analysis.** `data_dict` is whatever scalars the caller's
  own callback returns, which LogMD then plots. There is no RMSD, Rg or RMSF in
  it — `shared/analysis/metrics.js` would still have to exist.
- The hosted page also loads Google Analytics.

🔴 **SO "LIGHTER" IS TRUE AND MISLEADING.** The client is small *because* the
rendering, the storage and the frame history live on someone else's
infrastructure. That is not less complexity, it is relocated complexity — onto
an unversioned private endpoint (`*.modal.run`) this project would then depend
on.

🔴 **AND IT INVERTS THIS PROJECT'S CENTRAL CLAIM.** The README's first line is
"No install, no queue, no upload", and the reason this port exists at all is
that a reader should be able to run MD on their own machine without sending a
structure anywhere. Adopting LogMD as *the* viewer would mean uploading every
frame of someone's protein, with their email attached, to a third party — and
would stop the page working offline.

**What was taken instead.** The idea worth having is streaming frames as they
are produced, which is LogMD's actual contribution. That is now a control on the
page — *While running: animate every frame* — off by default because drawing
every report was measured at 17% of a run, and on when watching the water settle
during a short equilibration is the point. It renders locally, uploads nothing,
and works with no network.

**What remains open.** If the complaint behind the suggestion is the quality of
the drawing rather than the streaming, the honest answer is **Mol\*** — RCSB's
own viewer, MIT, and the best-looking molecular renderer there is. It can be
vendored and run locally exactly as 3Dmol is now, at roughly ten times the
bytes. That is a trade worth making deliberately, not by accident.

## What was deliberately not copied

LocalFold's per-device tuning-prior system (`webgpu/runtime/device-profile.js`,
1796 lines of measured knobs per GPU family) is the right answer to a problem
RainMD does not have yet: it presumes the kernels are correct and being
optimised. Correctness first, as their own `AGENTS.md` says. The same goes for
their tiered pipeline compilation, their quantised weight bundles (int3/int5 —
force-field parameters are kilobytes, not gigabytes) and their speculative
warm-up.
