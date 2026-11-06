# RainMD

Molecular dynamics that runs **in your browser**, on your own GPU, from the
AMBER topology you already have. No install, no queue, no upload.

The simulation is GPU-resident: forces *and* integration run on the device, and
nothing in a step reads back to the host. It reproduces OpenMM's energies to
1.5e-6 and its per-atom forces to 8.8e-5 on the notebooks' own configuration,
and runs this 1190-atom test system at **about 215 ns/day** on an Apple laptop
GPU — roughly 60x the host reference it is checked against.

A run on the page writes a **DCD trajectory and an AMBER restart**, so the
output goes straight into cpptraj, MDAnalysis, VMD or PyMOL alongside the
`.prmtop` you already have — the DCD writer is byte-compared against OpenMM's
own.

The whole workflow is on the GPU: **minimise, equilibrate, run NVT or NPT,
watch it, write it out.** The viewer reads the simulation's own position buffer,
so drawing a frame copies nothing to the host. In NPT at 1 bar it took the test
system's deliberately underdense tleap box from 0.587 to **0.992 g/cm³** — TIP3P
water is about 0.98.

**And you watch the analysis, not just the structure.** While the run is going
the page draws six live panels — RMSD, radius of gyration, potential energy,
per-residue RMSF, interface distance and temperature — and splits the RMSD
**four ways**: the whole complex, the receptor, the binder, and the binder
*fitted on the receptor*. That last one is the number that says whether a pose is
holding: a whole-complex RMSD averages the binder's drift over every receptor
atom that did not move, so a complex coming apart reads as a small number. The
structure view switches between solute, protein + ligand, ligand, backbone and
heavy atoms, solvent excluded by default, and colours by element, by body, or by
**secondary structure** — assigned live from alpha-carbon geometry, so you can
watch a helix fray. It draws a **real cartoon ribbon**, swept in WGSL from one
frame per residue: helices and strands as flat bands with thickness, strands
with arrowheads, loops as thin rods, atoms optional beside it. Because an AMBER topology carries no chain IDs, the
receptor/binder split is derived from the molecule partition and is **yours to
override**: the page offers the molecule list, named by its residues, in place
of the chain letters a PDB-based tool would ask for.
Pause holds the integrator between blocks and resumes from the identical state.
Everything exports: DCD, AMBER restart, a PDB of exactly the atoms on screen,
and the metrics and RMSF as CSV.

That design is [JIGGLE](https://github.com/BioCoder007/JIGGLE)'s, read as
source; `docs/ARCHITECTURE.md` says which five things were taken from it and the
one it asks for that an AMBER topology cannot answer.

[docs/STATUS.md](docs/STATUS.md) says exactly what is measured and what is not;
read it before trusting a number.

## The idea

[LocalFold](https://github.com/sokrypton/localfold) runs AlphaFold in a browser
tab on WebGPU. It cannot train a model in the page, so it does not try: the
weights are produced offline and the page runs inference on them.

Molecular dynamics has the same shape. A browser cannot run `tleap`,
`antechamber` or an AM1-BCC charge fit — but it does not need to. The
[Making-it-rain](https://github.com/pablo-arantes/making-it-rain) notebooks
already produce the artifact that matters:

```
Colab notebook (tleap / antechamber / GAFF2)  →  SYS.prmtop + SYS.crd
                                                      │  drag and drop
                                                      ▼
                              browser: parse → GPU buffers → simulate
```

**The prmtop is the weight bundle.** The expensive, irreproducible-in-a-page
work happens once, offline; what crosses into the browser is the result. That
means the same force field, the same charges and the same ligand parameters as
the Colab run — exactly, not approximately — and it means ligands work from day
one.

Why this is worth doing: WebGPU runs on Windows, macOS and Linux, on AMD, Intel,
NVIDIA and Apple silicon, with nothing to install. Anyone with a laptop and a
topology can run a simulation on hardware they already own.

## Running it

```bash
git clone <this repository> && cd rainmd
python3 tools/serve.py          # then open http://127.0.0.1:4173/
```

Static files, no build step — a plain file server works too, but `serve.py`
sends `no-store` so an edit is not hidden by Chrome's module cache.

Needs a browser with WebGPU: Chrome or Edge 113+, or Safari 18+.

## Checking it yourself

Every claim in [docs/STATUS.md](docs/STATUS.md) is a command.

```bash
npm run check:openmm      # per-term host energies and forces against OpenMM
npm run check:gradients   # every analytic force against a numerical gradient
npm test                  # the host suite
npm run test:gpu          # all nine kernel checks, through Dawn
npm run test:gpu:page     # the same nine, in a real headless Chrome
npm run check:mdanalysis  # the live metrics against MDAnalysis, on a DCD this wrote
node tools/run-reference-md.mjs 200    # a short simulation on the host oracle
npm run run:gpu -- --steps=2000       # a real GPU run, writing /tmp/rainmd-run.dcd
npm run test:page:simulate            # drive a GPU run through the UI in Chrome
npm run test:page:panels              # photograph the six live panels in Chrome
```

`npm test` includes the one gate that is pure text and still catches a whole
class of outage: `web/app.js` has top-level side effects, so a single
`element("…")` naming a control `index.html` no longer has throws *during module
evaluation* and silently abandons the rest of the file — handlers unregistered,
debug handle unassigned, and a page that still looks fine because the device
panel was filled by an earlier statement. The test diffs the two files.

The five GPU checks are layered so a failure localises itself:

| | |
|---|---|
| `test:gpu:bonded` | bonds, angles, torsions against the host reference |
| `test:gpu:fft` | the WGSL FFT against a naive DFT, on random data |
| `test:gpu:pme` | spread grid, reciprocal energy and forces |
| `test:gpu:nonbonded` | the cell list as a partition, the neighbour list as a *set*, then the forces |
| `test:gpu:forces` | **everything composed, against OpenMM** |
| `test:gpu:integrator` | one step against the host; then constraint geometry, energy conservation and the thermostat, on the device alone |
| `test:gpu:minimiser` | monotonicity, constraints, and whether dynamics runs from the result |
| `test:gpu:barostat` | the energy at a forced volume scale against OpenMM, then whether the sampler moves the box |

The oracle is OpenMM itself, built with the same arguments the notebooks pass to
`createSystem` — PME, a 1.0 nm cutoff, `HBonds`, rigid water. Regenerate it
with:

```bash
export PATH=/path/to/ambertools/bin:$PATH
tools/oracle/make_fixtures.sh
python3 tools/oracle/dump_openmm_reference.py \
    --prmtop test/fixtures/ala-wat.prmtop \
    --inpcrd test/fixtures/ala-wat.inpcrd \
    --out    test/fixtures/ala-wat-pme.json \
    --nonbonded PME --constraints HBonds --full-forces
```

That script also runs in Colab, so you can dump a reference for **your own**
system and check this engine against it before trusting it with a long run.

## How it is laid out

```
shared/     device-agnostic: the AMBER parser, topology, PME setup, units
cpu/        reference implementations — the oracle, not dead weight
webgpu/     runtime (device, allocator, pipeline cache) and WGSL kernels
web/        the page
tools/      oracles, differential checks, the dev server
docs/       ARCHITECTURE.md is the reverse-engineering result
```

`shared <- cpu <- webgpu`, and a kernel shares its reference's filename. This is
LocalFold's layout and it is deliberate: the CPU file next to each kernel is
what the kernel is tested against.

## Credit

The method, the pipeline and the reason this exists are
**[Making it rain: cloud-based molecular simulations for everyone](https://doi.org/10.1021/acs.jcim.1c00998)**
by **Pablo R. Arantes**, **Marcelo D. Polêto**, **Conrado Pedebos** and
**Rodrigo Ligabue-Braun**. The two notebooks in this repository's parent
directory are the specification this port follows.

The architecture is **[LocalFold](https://github.com/sokrypton/localfold)** by
Sergey Ovchinnikov and contributors, itself a fork of
**[AlphaFold2 WebGPU](https://github.com/martin-steinegger/alphafold2-webgpu)**
by Martin Steinegger — which is where running a scientific model in a browser on
WebGPU started. Nothing in `webgpu/runtime/` would be the shape it is without
reading theirs; `docs/ARCHITECTURE.md` says which eight things were taken and
why.

The live-analysis design is **[JIGGLE](https://github.com/BioCoder007/JIGGLE)**
by BioCoder007 — a Qt desktop application that runs OpenMM locally and reports
while the simulation runs. The incremental RMSF, the fit-on-receptor RMSD and
the rule that the viewer and the metrics both exclude solvent are all theirs.

The physics is **[OpenMM](https://openmm.org/)**, which is both what the
notebooks run and the oracle every force term here is checked against. The
analysis is additionally cross-checked against
**[MDAnalysis](https://www.mdanalysis.org/)**. The force
field is AMBER's, via **[AmberTools](https://ambermd.org/AmberTools.php)**.

> Eastman P, Swails J, Chodera JD, et al. OpenMM 7: Rapid development of high
> performance algorithms for molecular dynamics. *PLoS Comput Biol* 13(7):
> e1005659 (2017).

## License

MIT for the code; see `LICENSE`. Force-field parameters are not covered by it —
AMBER's remain under their own terms, and this repository redistributes none of
them beyond the small test systems under `test/fixtures/`, every one of which
`tools/oracle/make_fixtures.sh` regenerates from a local AmberTools.
