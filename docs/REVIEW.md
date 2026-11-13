# Pre-promotion review

A cold review pass at `ad095bc`, 2026-11-12, aimed deliberately at what the
existing suite does **not** cover. Six independent reviewers read the tree
without the context that built it, one per area; every finding below names a
file, a line, and either a command with its output or a number.

🔴 **NOTHING IN THIS DOCUMENT WAS FIXED.** It is a report. The suite was left
passing and no tracked file was changed.

## The existing suite passes, and was not re-litigated

Measured at the start of this pass, before anything else ran:

| command | result |
|---|---|
| `npm test` | **61 pass / 0 fail** |
| `npm run check:openmm` | **30/30** comparisons within tolerance |
| `npm run check:gradients` | all **8** terms are the gradient of their own energy |
| `npm run test:gpu` | exit 0 — 12 modules, Dawn |
| `npm run check:mdanalysis` | exit 0 |

So every number in the "Validated" table of `docs/STATUS.md` that has a command
still reproduces. **That is the problem this review is about.** The suite is
good and it is also, in six specific ways, pointed at the wrong configurations.

## What this review found, in one paragraph

For the Making-it-rain notebooks' **own default settings**, this port either
silently computes a different force field (ff19SB's CMAP term is dropped),
silently explodes (OPC/TIP4P-Ew water), or — on `solvateOct` — silently gets
electrostatics wrong by 91 kJ/mol. Separately, every NPT trajectory file it
writes carries the wrong unit cell, a blown-up run writes a full-length NaN
trajectory and says nothing, and pressing Stop discards the stage. None of
these is visible to any existing gate, and the reason is structural rather
than careless: the fixtures are all rectangular, all ff14SB, all 3-point
water, and all short.

---

# Blockers

Ordered by consequence. "Blast radius" is stated for each, because two of these
are narrower than they look and one is wider.

## 1. The PME reciprocal lattice is transposed — every non-orthorhombic box is wrong — ✅ FIXED (with a new finding behind it)

> **Fixed 2026-11-12.** Three lines in `cpu/forces/pme.js:174-176` and three in
> `webgpu/kernels/pme.js:244-246`, column → row. Every rectangular number is
> **identical to the digit** afterwards, so nothing in `docs/STATUS.md` moved.
> Gated by an **exact Ewald k-space sum** in `test/physics.test.js` sharing no
> code with what it validates: `ala-oct` agrees to 2.4e-6 at a 48³ grid, the
> same convergence the rectangular fixture shows. Two arms make the gate
> impossible to pass with the bug back — the transposed form must be grossly
> wrong on the sheared cell **and bit-identical on the rectangular one**.
> Falsified: reintroducing the host transpose fails the gate.
>
> `test/fixtures/ala-oct.*` (1315 atoms, truncated octahedron) and
> `ala-oct-pme.json` are added, and the tleap stanza is **in
> `make_fixtures.sh`** — a fixture that is not regenerable is finding 13's sin.
>
> 🔴 **AND FIXING IT EXPOSED A SECOND, SEPARATE TRICLINIC DEFECT — see
> finding 66.** The reciprocal sum is now right; the DIRECT-space terms still
> disagree with OpenMM on a sheared cell by LJ 0.082 kJ/mol and Coulomb 2.115,
> grid-independently. The octahedral case is therefore deliberately absent from
> `check-against-openmm.mjs` rather than present with a widened tolerance.
>
> Also fixed en route, both exposed by this work: the `node --check` gate named
> eleven files by hand while claiming to cover the tree and **omitted
> `webgpu/kernels/` entirely** — a backtick in a WGSL comment broke all thirteen
> GPU modules while `npm test` reported 65 passes; it now walks the tree (49
> files) and is falsified. And `dump_openmm_reference.py` silently ignored
> `--pme-grid` without `--pme-alpha`, because OpenMM reads `alpha == 0` as
> "choose alpha AND the grid automatically"; it now refuses that combination by
> name.

**`cpu/forces/pme.js:174-176`**, mirrored verbatim in **`webgpu/kernels/pme.js:244-246`**.

Line 95 builds fractional coordinates as `s_d = Σᵢ rᵢ·R[i][d]`, so the phase
`k·s` gives a Cartesian wave vector `mᵢ = Σ_d k_d·R[i][d]` — **row** `i` of the
inverse box, i.e. `recip[3i+0..2]`. The code uses `recip[0], recip[3], recip[6]`:
the column. `M` is diagonal for a rectangular cell, so `M = Mᵀ` and the error is
**exactly zero** on every box `solvateBox` produces.

Measured by patching only those three lines in a copy and changing nothing else:

```
ala-wat  (rectangular)  shipped 933.776245   row-form 933.776245   diff  0.000000 kJ/mol
oct      (octahedral)   shipped 726.027026   row-form 635.144433   diff 90.882593 kJ/mol
```

The row form lands on an independent exact Ewald k-space sum (635.06, converged
over nmax = 10/12/16/20); the shipped form sits 91 kJ/mol away from it at every
grid from 32×30×25 to 80³, so this is not a grid artefact. Against OpenMM on a
3360-atom `solvateOct` system: Coulomb −42437.88 against −42528.67, and forces
**1.8e-3 relative** against the 8.8e-5 `docs/STATUS.md` records for the
rectangular case.

**Why no gate caught it, which matters more than the bug.** `AGENTS.md` requires
two independent oracles per force kernel and both exist. Neither can see this:

- The OpenMM arm has **no triclinic fixture** — `make_fixtures.sh` only ever
  calls `solvateBox`.
- The self-gradient arm is **structurally blind**. The wave vector enters only
  the kernel *weights*, which do not depend on position, so the analytic force
  is the exact gradient of the wrong energy. Verified rather than assumed: a
  central-difference check of the reciprocal force on the octahedral cell passes
  at relative error `0.00e+0`. No gradient check at any box shape can ever
  see this.
- `webgpu/kernels/pme.js` carries the identical construction under the identical
  `COLUMNS` comment, so `test:gpu:pme` compares a kernel against a reference
  that **shares the error** — precisely what AGENTS.md's "a reference must share
  no code with the thing it validates" exists to prevent, in a form the
  `shared/ ← cpu/ ← webgpu/` layering does not catch.

`docs/STATUS.md` says octahedral boxes are "implemented through the general
triclinic path but have not been tested against OpenMM". They are implemented,
untested, and wrong.

**Blast radius: every `solvateOct` system, which is an option in both
notebooks. Zero effect on any rectangular box, so no published number moves.**

## 2. ff19SB's CMAP term is dropped with `warnings: []` — and ff19SB is the notebook's default — ✅ FIXED

> **Fixed 2026-11-12** by the flag audit in `shared/amber/flags.js`. Refused by name with a message naming ff14SB as the way out. Verified against all four real ff19SB systems tleap builds; gated in `npm test` by appending the flag to a committed fixture in memory, so no ff19SB parameters are redistributed to test a refusal. Falsified.

**`shared/amber/prmtop.js:110-311`** reads none of `CMAP_COUNT`,
`CMAP_RESOLUTION`, `CMAP_PARAMETER_NN`, `CMAP_INDEX`.

`Amber.ipynb`'s own dropdown is `Force_field = "ff19SB" #@param ["ff19SB", "ff14SB"]`.
Built with the notebook's literal defaults (`leaprc.protein.ff19SB` +
`leaprc.water.tip3p` + `solvatebox 10.0`): 2817 atoms, `CMAP_COUNT = 3 1`,
parses clean. Per-term against OpenMM 8.4, ACE-(ALA)₃-NME, NoCutoff:

| term | OpenMM | RainMD |
|---|---|---|
| HarmonicBond | +0.088778 | +0.088778 |
| HarmonicAngle | +4.023395 | +4.023395 |
| PeriodicTorsion | +10.022601 | +10.022601 |
| Nonbonded | −53.246508 | −53.246507 |
| **CMAPTorsion** | **−5.082244** | **absent** |
| **TOTAL** | **−44.193978** | **−39.111732** |

Scales with chain length: a 30-residue ff19SB helix carries 28 CMAP terms worth
**−56.671931 kJ/mol**, and the RainMD/OpenMM total difference is exactly that.

ff19SB *is* ff14SB with CMAP replacing the backbone torsions, so running it
without CMAP is not a perturbation — it is an unvalidated force field. Direct
violation of AGENTS.md's "Refuse by name rather than approximating".

**Blast radius: every ff19SB system, i.e. the default path a notebook user
arrives on.**

## 3. Extra points (OPC, TIP4P-Ew) are accepted silently; the run reaches 2650 K — ✅ FIXED

> **Fixed 2026-11-12.** `NUMEXTRA > 0` is refused by name, pointing at a 3-point water. `IFCAP` and `NCOPY` are refused alongside — all three are POINTERS values, which the `%FLAG` audit cannot see. Gated and falsified.

**`shared/amber/prmtop.js:28`** parses `NUMEXTRA` into `pointers` and never
checks it; **`shared/topology/system.js:112`** then constrains the O–EP bond
under `rigidWater`.

`Water_type = "TIP3P" #@param ["TIP3P", "OPC"]` is in both notebooks, and OPC is
ff19SB's recommended water. Measured:

```
ff19sb-wat (OPC) : natom 3774  NUMEXTRA 933  warnings: []  zeroMassAtoms 933
                   constraints 3754, of which 933 involve a zero-mass atom
host dynamics, 60 steps, 2 fs, 298 K setpoint:
  step 15  T = 2653.6 K     step 30  T = 1669.1 K     step 60  T = 1872.1 K
  max |dr|: water O 0.03187 nm, extra points 0.00000 nm, hydrogens 0.18199 nm
control (3-point ala-wat, identical code path): T = 412.9 K, water O 0.15278 nm
```

933 distance constraints tie a movable oxygen to a frozen massless site. The
extra points never move; the oxygens are dragged back. `tip4pew` is identical
(903 sites).

**Blast radius: every OPC or TIP4P-Ew system. Loud in the temperature panel
once running, silent at load — and the protocol minimises first, which masks it
at exactly the moment a user would look.**

## 4. There is no known-flag whitelist, so any unrecognised `%FLAG` is dropped in silence — ✅ FIXED

> **Fixed 2026-11-12.** `shared/amber/flags.js` classifies all 44 flags the fixtures carry plus everything ParmEd enumerates: handled, harmless, or refused by name. An unrecognised flag is **refused and named**, with the message saying where to classify it — the asymmetry is the argument, and `docs/STATUS.md` states it. CHAMBER, 12-6-4, polarizable, solvent cap and LES all have their own messages. Two things this exposed: `IPOL` is a *section* in modern files and in no POINTERS slot, so the existing check read `undefined`; and `RESIDUE_CHAINID` exists, which makes `ARCHITECTURE.md`'s "a prmtop has no chain IDs at all" false as an absolute (tleap does not write them, so the molecule-partition default still stands — see N53).

**`shared/amber/sections.js:87-147`** builds a `Map` of every flag;
**`prmtop.js:110-311`** reads the ~26 it names and never enumerates the rest.
No unknown-flag report exists. This is the mechanism behind findings 2 and 3,
and behind these, each confirmed to parse with `warnings: []`:

| flag | what is lost |
|---|---|
| `LENNARD_JONES_CCOEF` | AMBER 12-6-4 ion model (flag confirmed present and ignored; magnitude unmeasured) |
| `CTITLE`, `CHARMM_UREY_BRADLEY*`, `CHARMM_IMPROPER*`, `CHARMM_CMAP_*`, `LENNARD_JONES_14_ACOEF/_BCOEF` | CHAMBER: Urey-Bradley, CHARMM impropers, CMAP, and the separate 1-4 LJ table |
| `IPOL != 0`, `POLARIZABILITY` | a polarizable topology runs as fixed-charge (code reading) |
| `IFCAP != 0`, `CAP_INFO` | solvent cap ignored (code reading) |
| `NCOPY` | LES ignored (code reading) |

A *real* CHAMBER prmtop is refused, but by accident and uselessly:
`%FLAG FORCE_FIELD_TYPE` is `%FORMAT(i2,a78)` and `parseFortranFormat` throws
`unsupported AMBER %FORMAT descriptor: "i2,a78"` — a message that never says
"CHARMM". ParmEd routes on `'CTITLE' in parm.flag_list`; RainMD has no such test.

The full flag-by-flag audit is in the appendix.

## 5. No finiteness check anywhere on the integration path — ✅ FIXED

> **Fixed 2026-11-12.** `shared/finite.js` plus three call sites: `#checkSkin`
> (free — the `u32` was already being read every block), `energy()` (catches a
> velocity a step before positions follow, and an input infinite on the first
> evaluation), and both writers (no file this program writes can contain `NaN`).
> Gated by `npm run test:gpu:blowup` — now the 13th module in both GPU lanes —
> and three host arms in `npm test`. Falsified: with the detectors removed the
> gate reports `threw: false, stoppedAtStep: 200`, the original symptom, while
> the 2 fs arm still passes. `docs/STATUS.md` → "The blown-up run that
> completed". **Finding S22 (the restart writer's NaN) is fixed by the same
> change.** Still open: the timestep field has no maximum (N54), and a refusal
> mid-run still costs the stage (blocker 6).


**`webgpu/simulation.js:415`** (`run()`), **`:914`** (`energy()`),
**`shared/traj/dcd.js:95`** (`appendFrame`), **`web/app.js:1176-1224`**. A grep
over `webgpu/`, `shared/`, `cpu/`, `web/` finds no finiteness or magnitude test
on positions, forces, energy or temperature in any run path; the only
`Number.isFinite` guards are cosmetic formatters.

`index.html:73` is `<input type=number id=timestep min=0.5>` with **no max**, so
40 fs is a value the UI accepts. Measured on the page, headless Chrome:

```
tiles: step 300, potential "–", temperature "–", RMSD "–", frames 12
finished: true, equilibrateUnlockedProduction: true
downloadDcdDisabled: false, downloadStateDisabled: false
metricsCsvTail: "300,12,NaN,NaN,NaN,NaN,20.520340131422515,NaN,NaN,,,,"
log: "300 steps in 7.7 s — 25.79 ms/step" … "trajectory: 13 frames, 0.18 MiB"
```

Every atom is NaN after 50 steps. The engine keeps encoding blocks, writes every
frame, enables both downloads, and unlocks the next stage. The cliff is between
5 and 6 fs — 2, 3, 4 and 5 fs are all stable over 200 steps from an unminimised
start — so this is reachable by one keystroke, or by an `inpcrd` with two
coincident atoms (a common tleap/PDB artefact), which gives NaN on the *first*
force evaluation.

The all-NaN DCD is accepted **without a single warning** by both readers:

```
cpptraj:    'nan.dcd' is a CHARMM DCD file (coords) … Read 10 frames and processed 10 frames.
MDAnalysis: n_frames=10 read=10 allfinite_first=False allfinite_last=False
```

The information needed already comes back every block — `#checkSkin` does a
per-block readback at `webgpu/simulation.js:470`, and `diagnostics[1]` was
`true`. The **host** reference does refuse (`cpu/integrate/constraints.js:141`
returns `converged:false` with "the usual cause is a timestep too large for this
system"); the shipping engine only sets a flag nobody reads.

**Blast radius: any run that blows up, at any point, on any system. This is the
single cheapest finding to fix and the one that costs a user the most.**

## 6. Stop — and any mid-run exception — discards the entire stage — ✅ FIXED

> **Fixed 2026-11-12.** The commit moved into `finally`, which runs whether the
> loop returned, threw or was stopped. What is kept depends on **the state, not
> on why the loop ended**: the state is read and tested for finiteness, so a
> clean stop hands its state on, a blown-up one does not, and a lost device
> cannot be read — while the trajectory is kept in all three cases because it is
> held in the page. The restart now names its stage (`<name>_equil.rst7`), which
> closes **S26**'s sibling defect: a stopped production used to hand over the
> equilibration's state under a filename that said nothing. Gated by
> `npm run test:page:stop`, two arms — Stop must keep its output, and a blow-up
> must keep its frames and hand on nothing, so the fix cannot become "keep
> everything". Falsified against the pre-fix page: `trajectoryFiles []`,
> `trajectoryBytes 0`, both downloads disabled, `hostState` at step 0. One
> defect in the fix was caught by its own blow-up arm — see `docs/STATUS.md` →
> "Stop used to cost the whole stage". **Also fixed by the same change: S21**
> (the wrong diagnosis a blown-up run printed — the skin line is no longer
> emitted when the state is not a number, and the lane fails if
> `0% of the budget used` appears) and **N56** (the DCD download logged a
> filename it had not written). Still open: nothing is written to disk mid-run
> (S20), so closing the tab still loses the run.


**`web/app.js:1177`** throws from inside the loop, while `hostState`,
`finalState`, `writer.finish()`, `trajectories.push()` and both download buttons
all sit at **`:1271-1294`**, *after* it, with only `finally { simulation.release(); }`
in between.

Measured: equilibration at step 2,500 with 5 frames already written, Stop pressed.

```
framesAtStop "frames5", stepAtStop "step2,500", runningAtStop true
afterStop: trajectoryFiles []        trajectoryBytes 0
           downloadDcdDisabled true  downloadStateDisabled true
           hostStateStep 0           hostStateChecksum 4954.921021461487
                                     (identical to the post-minimise value)
log tail:  "  stopped"
```

The same `catch` at `:1010` swallows a `DeferredValidation` failure, a
`NeighbourOverflowError` and a device loss, so **any** error at hour six costs
the same. Only `download-metrics` stays enabled, which makes it look as though
something survived.

Worse, a *stopped* stage leaves `download-state` enabled with the **previous**
stage's state behind it, under the same filename: production stopped at step
3,000 reports `hostStateStep 2000` and offers only the equilibration DCD. That
is the "stage handoff, which was broken" defect reachable again through the Stop
path.

Mitigation that exists: a multi-stride production pushes each completed stride.
A single-stride production — the default, `index.html:121` — loses all of it.

## 7. The equilibration restraint is excluded from the barostat's Metropolis energy — ✅ FIXED

> **Fixed 2026-11-12.** The restraint energy is read and added in `#evaluatePotential`. Gated by a restrained arm in `check-barostat.mjs` at the page's own equilibration settings; the two sums agree to 1.5e-4 kJ/mol. Falsified — with the term dropped, `differenceAsFractionOfRestraint` is **0.99998**, i.e. the gap is exactly the restraint energy.

**`webgpu/simulation.js:606-630`** — `#evaluatePotential()` sums self +
dispersion + bonded + nonbonded + partials and never reads
`resources.restraintEnergies`. **`:972-979`** — `energy()` *does* add it. And
`#evaluatePotential` is exactly what feeds `metropolisWeight` at `:843`/`:879`.

OpenMM's `MonteCarloBarostat` evaluates the full context potential, so the
notebooks' `CustomExternalForce` participates in theirs. So the **reported**
potential energy is right and the **accept/reject decision** is wrong.

The algebra says why it is a bias rather than noise. Centroid scaling sends each
displacement `d₀ → d₀ + (s−1)R`:

```
k·Σ|d₀|²  +  2k(s−1)·Σd₀·R  +  k(s−1)²·Σ|R|²
             ^ linear, directional   ^ quadratic, resists any move
```

Omitting both means the barostat neither feels the restraint's resistance to
changing the volume nor its directional preference. Measured at kT = 2.478:

| length scale | omitted from Metropolis |
|---|---|
| 1.0033 (the shipped 1% initial move) | 0.789 kJ/mol = **0.32 kT** |
| 1.02 | 14.68 kJ/mol = 5.9 kT |
| 1.0914 (the 30%-of-volume cap) | 260.4 kJ/mol = 105 kT |

`tools/gpu/check-barostat.mjs` cannot see it because it runs **unrestrained**,
and `check:openmm`'s restrained arm is single-point — it compares energies at a
configuration, never an accept/reject. Two correct oracles, and the defect sits
in the gap between them.

🔴 **FOUND INDEPENDENTLY BY TWO REVIEWERS, AND THE SECOND MEASUREMENT IS LARGER.**
Driving the page's *own* equilibration configuration (10 solute heavy atoms at
500 kJ/mol/nm², 4000 steps of warm-up) rather than a synthetic displacement:

```
energyPotential          = -16763.98   (includes positionRestraint = +6.906)
barostatOldPotential     = -16770.89
differenceOldMinusEnergy = -6.9062     <- exactly -(restraint energy)
move: accepted=true weight=-2.510 lengthScale=0.99534
```

The omitted term is **6.9 kJ/mol ≈ 2.8 kT**, larger than the entire Metropolis
weight of −2.5 kJ/mol that decided the move. OpenMM takes
`getState(State::Energy, false, integrator.getIntegrationForceGroups())`, which
includes the `CustomExternalForce`. And because the references deliberately do
not follow the box, every volume move drags the restrained solute off its
references — which is exactly the term being dropped.

**Blast radius: bounded. Equilibration is short and production releases the
restraint and re-equilibrates the density itself, so this corrupts the
equilibration density rather than the production ensemble. It is still a blocker
because the page claims to implement the notebooks' protocol and does not.**

## 8. Under NPT, every DCD frame carries the stage's *starting* unit cell — ✅ FIXED

> **Fixed 2026-11-12.** Both `appendFrame` call sites in `web/app.js` now pass the live box (`energy.boxVectors`), and `tools/gpu/write-trajectory.mjs` passes `state.boxVectors` — its **restart** had the same defect and now carries the live box too. Gated by `npm run check:npt-box`: a real 4000-step NPT run whose volume moves 23%, read back by MDAnalysis, matching frame by frame to 8.3e-8 against the DCD's float32 cell floor of ~2e-7. Falsified — the shipped form gives 36% error with an identical cell on every frame. A text gate in `npm test` also refuses a regression of that one argument.

**`web/app.js:1175`** and **`:1219`** pass `system.boxVectors`.
`webgpu/simulation.js:134` copies the box into `this.boxVectors`, and
`#updateBox` at `:574` mutates **the simulation's own copy**. `system.boxVectors`
is never updated, so the writer never sees the live cell. Same call at
`tools/gpu/write-trajectory.mjs:58`.

Measured — 800 NPT steps written exactly as `app.js` does, live volume walking
20.4979 → 19.2124 nm³. MDAnalysis 2.9.0 reports, for all 8 frames:

```
27.1882 x 29.7181 x 25.3970 A = 20.5203 nm3   (the starting box, constant)
```

The documented equilibration takes the volume 20.52 → 14.39 nm³, so the file is
wrong by **43% in cell volume** for every frame. Three consequences: any
density, volume or box-dependent analysis in cpptraj/MDAnalysis/VMD is wrong;
`DcdWriter`'s own molecule imaging wraps centroids into a cell that is too
large, so molecules that left the real cell are not imaged back; and any
PBC-dependent analysis (RDF, minimum-image contacts, H-bonds across the
boundary) uses the wrong box.

Why no gate sees it: `check:mdanalysis` drives `tools/gpu/check-analysis.mjs:58`,
which constructs the simulation with **no `pressure`** — NVT, constant box.

**Blast radius: every NPT trajectory file the page has ever written, which is
every default run. The coordinates are correct; the cell record is not.**

## 9. A neighbour-list overflow during a run is never checked, and the capacity never grows with density — ✅ FIXED

> **Fixed 2026-11-12.** `#updateBox` re-estimates the capacity for the new box (taking the larger of carried and re-estimated); `run()` checks at every block boundary and keeps a high-water mark; `neighbourSlack`/`maxNeighbours` are forwarded from the caller, which they were not — closing the NICE-TO-HAVE noted under this finding and making a mid-run overflow provokable at all. Gated by `npm run test:gpu:capacity`, two arms, both falsified. Read the gate's own comment for what it does **not** show: the spontaneous overflow is not reproduced on its relaxation schedule, so arm 2 forces the condition directly.

**`webgpu/simulation.js:380`** — `#checkNeighbourCapacity()` is called from
exactly one site, inside `initialise()` (371-382). `run()` calls only
`#checkSkin`. **`:576`** — `#updateBox` re-plans with
`maxNeighbours: this.plan.maxNeighbours`, carried forward and never re-estimated.
**`:341`** — `required` is cleared at *every* rebuild, so it is a snapshot, not a
high-water mark. `grownPlan` and `NeighbourOverflowError` both exist and are
unreachable from `run()`.

Measured on ala-wat-big (2854 atoms), contracted in ≤5% volume steps with 20
relaxation steps at each stop — the path a barostat takes:

```
V=43.78 nm3  required 1081 / capacity 1499   ok
V=32.00      required 1442 / 1499            ok
V=29.20      required 1519 / 1499            OVERFLOW (20 dropped), run() threw nothing
V=28.06      required 1482 / 1499            "ok"   <- the overflow has already vanished
V=27.06      required 1439 / 1499            "ok"      from the diagnostics
```

V = 29.2 nm³ is 97.7 atoms/nm³ — **exactly TIP3P density, where the shipped NPT
equilibration is documented to land.**

Magnitude, device against `cpu/energy.js`'s all-pairs f64 loop at identical
coordinates, alpha and grid:

| | control (665/840) | overflowed (973/840, 133 dropped) |
|---|---|---|
| ΔLJ | +3.0e-4 kJ/mol | **−43.56 kJ/mol** |
| ΔCoulomb direct | −0.0166 kJ/mol | **−42.12 kJ/mol** |
| worst force component | 0.0059 kJ/mol/nm | **2803 kJ/mol/nm** |
| rms force error / rms force | 1.1e-6 | **9.8e-3** |

The dropped pairs are at *arbitrary* distances — whatever the cell stencil
enumerated last — not at the cutoff, so `missedPairForceBound` does not bound
them. The stage log can still read `neighbours: 1482 needed of 1499 capacity`
because `required` holds only the last rebuild.

## 10. The skin detector observes only the steps since the last rebuild *inside the block* — ✅ FIXED

> **Fixed 2026-11-12.** The displacement kernel keeps two slots: slot 0 cleared
> at each rebuild (stale-since-built), slot 1 cleared once per block (the worst
> per-interval displacement anywhere in it), both taking the same per-step
> value. The host reads slot 1. Gated by arm 5 of
> `npm run test:gpu:displacement` on **exact equality** — the same `atomicMax`
> over the same f32 values must give the same bits however it is grouped, so a
> tolerance would hide the defect. Falsified: the shipped window reproduces
> 47.23 / 20.34 / 16.37 with an identical trajectory checksum.
>
> `skinViolations` is deliberately still a count of *blocks* containing a
> violation (which the log states), and `stepsWithStaleList` still attributes
> the whole block — conservative, and now submission-independent. With the
> adaptive controller live, block size still changes the run itself, which is
> **S17**, now the only remaining block-size dependence.

**`webgpu/simulation.js:358`** clears the displacement buffer at every rebuild,
*inside* the command buffer; **`:446`/`:469`** read it once per block;
**`web/app.js:1186`** calls `run(take, { blockSize: rebuildInterval })` with the
user's requested ceiling (20), while the adaptive interval starts at 4. So a
default page block of 20 steps contains five rebuilds and the host inspects the
last four of the twenty — and after a violation the interval halves again, so
the fraction observed falls precisely when it matters.

Controlled: all three arms encode the *same* 100 steps with a rebuild every 4;
only the submit boundary moves. Final position checksum identical to 13 digits
in all three, so no physics changed:

```
blockSize   4 : 25 violations, worstSkinUsage 47.23, worstDisplacement 0.09446, bound 5.649
blockSize  20 :  5 violations, worstSkinUsage 20.34, worstDisplacement 0.04068, bound 1.346
blockSize 100 :  1 violation , worstSkinUsage 16.37, worstDisplacement 0.03275, bound 1.077
```

Every user-facing staleness number — `skinViolations`, `worstDisplacement`,
`worstSkinUsage`, and the `missedPairForceBound` that `web/app.js:1248-1259`
prints as the quantitative reason the run is still trustworthy — is understated
by 2.3× to 5.2× purely by submit granularity. In the page's own shape it reports
**0 violations** where a fully observed run reports 6, and the staleness is real:
a host f64 measurement over one pinned interval from a minimised structure gives
usage 1.428 at the page's default interval of 20.

**Blast radius: the physics error per missed pair is bounded (0.44-0.91 kJ/mol/nm
against an rms force of 516). This is a blocker because that detector is the
only defence and it under-reports itself, not because the error is large.**

## 11. "Nothing leaves your machine" is false, and the README omits the feature entirely — ✅ FIXED

> **Fixed 2026-11-12.** The claim now says "unless you ask it to" and names the opt-in LogMD share. **S28** (nothing reads back to the host), **S29** (the viewer reads the position buffer) and the density half of blocker **12** went with it, all in the same document. Gated by a claim-consistency test: the share button's presence in `index.html` forces the README to name LogMD and to qualify the claim, and the gate fails on the exact phrasings that were wrong.

**`README.md:6,8-10`** — "No install, no queue, no **upload**"; "Nothing leaves
your machine: … There is no server to send a structure to."

`index.html:224` ships a **Share to LogMD** button that POSTs frames to a
third-party endpoint. `grep -in "logmd\|share\|upload" README.md` returns
nothing but the word "upload" inside that denial.

`index.html:226-233` *is* honest about it ("This one leaves your machine…
Nothing else on this page makes a network request") and is the best text in the
project. The README is the document that is wrong, and it is the one a user
reads first, and it is the project's central claim.

Verified correct alongside it: there really is **no request on open** — only
relative `src`/`href`, the three `https://` are anchor links, and the only
`fetch()` calls are inside the *Load example* click handler.

## 12. Two headline numbers cannot be produced by any command — ✅ FIXED

> **Fixed 2026-11-12.** The conservative arm now runs **20 ps**, the length the docs quote, and reports window fits from one trace (4 ps +2.63, 10 ps +0.370, 20 ps **+0.484**) so the figure carries its window and its super-linearity. The old companions — a 3.45 fluctuation and a 2.73 residual — were computed by nothing and the residual is not an output; both are gone. `index.html` and STATUS now say 0.48. The density half was fixed with blocker 11.

- **Energy drift, `index.html:259` and `docs/STATUS.md:36`: 0.30 kJ/mol/ns/atom
  over 20 ps.** `check-integrator.mjs:131` runs 5000 steps = 10 ps.
  Its own comment at `:203-214` admits it: *"this arm measures +2.8 … The number
  to quote is the 20 ps one."* Reaching 0.30 requires editing a constant. The
  measured value today was **+1.12** with fluctuation 4.678, so neither 0.30 nor
  the code's own +2.8 reproduces, and the gate is `|drift| < 5`, which both pass.
  There is no residual-about-the-fit output at all. This is the number a user
  judges their own run's soundness by.
- **Density, `README.md:41` and `docs/STATUS.md:45`: 0.587 → 0.992 g/cm³, 48 ps,
  298.7 K, 50% acceptance.** `test:gpu:barostat` runs 3000 steps = 6 ps and
  reaches **0.80986 g/cm³** at 61.6% acceptance and 293.4 K. Nothing in the repo
  runs 48 ps of NPT. Separately, STATUS's own handoff volumes
  (`:231-233`, 20.52 → 14.39 → 11.96 nm³) imply equilibration ends at
  **0.837** and production at **1.007**, so two sections of the same document
  disagree about what equilibration achieves. The start density 0.5868 and the
  12.133 nm³ needed for 0.9924 both check out, so 0.992 is plausible — it is
  just not rerunnable.

  🔴 **AND A 300 ps RUN SAYS THE CLAIM IS TOO WEAK, NOT TOO STRONG.** A separate
  reviewer ran the same system to convergence: mean density **1.0014 ± 0.0129
  g/cm³** against OpenMM's 0.9997 ± 0.0141 on the same system — agreement to
  **0.17%**. So 0.992 is a 48 ps number that had not finished compressing, and
  the converged value is 1.000. The defect is that the figure has no command and
  no stated window, not that it flatters the engine.

## 13. Two committed AMBER systems are not regenerable, under the sentence that says they are — ✅ FIXED (one rebuilt, one honestly documented)

> **Fixed 2026-11-12.** `ala-wat-big`'s recipe was recovered by sweeping — `solvateBox TIP3PBOX 12.0 1.0`, closeness 1.0 and not ala-wat's 0.7 — and now rebuilds **byte-identically**. `two-body`'s could not be: the closest reconstruction reaches 2523 atoms and box x 4.1123 nm against 2547 and 4.1545, matching y and z exactly, so the original used geometry not recoverable from the output. It is recorded as a stated exception rather than overwritten with a near-miss, because it is the fixture the page offers visitors. README and LICENSE now say which fixtures rebuild byte-for-byte, which are equivalent-only (`addIonsRand` is random), and which is not built — and a gate fails if that list and the directory disagree.
>
> 🔴 **AND MY EARLIER FRAMING OF THIS AS A LICENCE PROBLEM WAS TOO STRONG.** AmberTools' parameters travel with every prmtop anyone publishes, and the README already says these test systems carry them. What was actually wrong is the reproducibility claim.

**`README.md:217`** — "every one of which `tools/oracle/make_fixtures.sh`
regenerates from a local AmberTools" — and **`LICENSE:29`** — "regenerates all
of them".

`make_fixtures.sh` builds five systems: `ala-vac`, `ala-wat`, `helix`, `strand`,
`mixed`. Seven prmtops are committed. `two-body.prmtop` (434 KB) and
`ala-wat-big.prmtop` (471 KB) have no recipe anywhere
(`grep -nE "two-body|big" tools/oracle/make_fixtures.sh` → no match). Both are
full ff14SB + TIP3P parameter sets, and AMBER's parameters are explicitly *not*
covered by the MIT grant.

`two-body` is the fixture the public page hands to visitors (`index.html:37`,
fetched at `web/app.js:197`). `ala-wat-big` is read by no code — but
`docs/STATUS.md:162` and the README's "2854 atoms and 74 ns/day" rest on it, so
deleting it would make a published claim unreproducible. **Both need a tleap
stanza, not a delete.**

## 14. A *smaller* timestep makes the engine heat uncontrollably, and the temperature error is non-monotonic in dt — ✅ BOTH FIXED

> **14b FIXED 2026-11-13.** The kinetic sample is taken inside the step, before
> the closing half kick, which is where OpenMM reports. Offsets from a 298 K
> setpoint went from −0.02 / −2.08 / −6.36 K at 1 / 2 / 4 fs to
> **+0.46 / −0.33 / +0.88**, flat to within the sampling error and better than
> OpenMM's own +0.42 / +1.63 / +4.45. Gated by `npm run check:thermostat`
> against a 2 K bar derived from the 0.55 K standard error; falsified by
> disabling the request, which reproduces the original offsets exactly.
>
> Three things that fell out of doing it:
> - **The cheap version is wrong.** Reconstructing the earlier velocity as
>   `v − (halfStep/m)·F` agrees to 0.2–0.8 K on a 12-constraint vacuum fixture
>   and is out by **+87 K** on the solvated one, because the closing kick is
>   followed by RATTLE and subtracting the kick does not invert the projection.
>   It was validated on the fixture where it happens to work.
> - **The total energy must keep the post-step sample.** Using the early one for
>   the conserved quantity is a half-step mismatch that inflated the NVE energy
>   fluctuation from 9.31 to 15.51 kJ/mol. `energy()` now carries both and
>   reports `kineticSampledAt`.
> - **A zero temperature was reachable** when the host's marker and the
>   encoder's pass disagreed, and is now guarded.

> **14b INVESTIGATED 2026-11-12 and CONFIRMED SEPARATE.** Re-measured on the
> device after the 14a fix — NVT, friction 10/ps, 60 ps at each step from one
> shared minimised and equilibrated start, 240 samples:
>
> | dt | before 14a | after 14a | OpenMM LangevinMiddle |
> |---|---|---|---|
> | 1 fs | — | 297.98 ± 0.57 (**−0.02**) | — |
> | 2 fs | 296.76 | 295.92 ± 0.54 (**−2.08**) | 300.10 ± 0.23 |
> | 4 fs | 291.37 | 291.64 ± 0.55 (**−6.36**) | 301.57 ± 0.23 |
>
> The offsets moved by about one standard error, so **14a was not its cause** —
> which is what the mechanism predicted, one being an f32 heating that grows as
> the step shrinks and the other a cooling that grows as the step grows, but it
> had to be measured rather than assumed.
>
> **Characterised:** the offset is zero within error at 1 fs, −2.08 at 2 fs and
> −6.36 at 4 fs, i.e. **O(dt²)** (the 2→4 fs ratio is 3.06 against 4 for exactly
> quadratic, slightly sub-quadratic). It is **RainMD-specific**: OpenMM on the
> same system, with the same splitting and — measured — the same uncorrected
> on-step kinetic estimator, is flat to within 1.6 K.
>
> 🔴 **THE CONSTRAINT-SCHEDULE HYPOTHESIS IS REFUTED.** Built as a patched host
> copy — both half drifts accumulated and the whole step's displacement
> constrained **once**, OpenMM's shape — and run against the shipped
> double-projection scheme in f64, same system, seed and thermostat, 120 ps per
> arm at 1 / 2 / 4 fs. The dt dependence is the same within error: **−37.89 K**
> (shipped) against **−40.18 K** (one projection) over 1 → 4 fs, with each
> timestep's pair differing by under 1.5 standard errors. The schedule is not the
> cause.
>
> 🔴 **AND OPENMM IS NOT FLAT, WHICH IS THE REAL FINDING.** The review's claim
> that it was came from arms this pass could not reproduce. Measured properly —
> OpenMM CPU platform, same system and settings, minimised, 20 ps equilibration
> then 60 ps sampled:
>
> | dt | OpenMM | RainMD |
> |---|---|---|
> | 1 fs | 298.42 ± 0.47 (**+0.42**) | 297.98 ± 0.57 (**−0.02**) |
> | 2 fs | 299.63 ± 0.48 (**+1.63**) | 295.92 ± 0.54 (**−2.08**) |
> | 4 fs | 302.45 ± 0.51 (**+4.45**) | 291.64 ± 0.55 (**−6.36**) |
>
> Both carry an O(dt²) on-step kinetic bias and **the signs are opposite**. Both
> agree with the setpoint and with each other within error at 1 fs, so both are
> correct integrators converging to the same limit and differing in their O(dt²)
> term — which is a much narrower statement than "this port samples the wrong
> ensemble".
>
> 🔴 **CAUSE FOUND AND PROVEN: IT IS WHERE IN THE CYCLE THE ESTIMATOR IS TAKEN,
> AND THE DYNAMICS IS NOT AT FAULT.** This port runs `B A O A B` and reports the
> kinetic temperature after the **closing** half kick. OpenMM's
> `LangevinMiddleIntegrator` ends its step at the position update and velocity
> correction, so it reports **one half kick earlier**. Both are the same
> splitting over a full step and generate the same trajectory; only the kinetic
> estimator's sampling point differs, and a half kick is an O(dt) change in `v`,
> O(dt²) in energy, with the sign set by which side of the kick you stand on.
>
> Measured by moving only the sampling point in a patched host copy — same
> system, seed, thermostat and trajectory, f64 so precision cannot confound it,
> 120 ps per arm:
>
> | | 1 fs | 2 fs | 4 fs | dependence (4 − 1) |
> |---|---|---|---|---|
> | after the closing half kick (shipped) | 309.87 | 303.13 | 271.98 | **−37.89 K** |
> | one half kick earlier (OpenMM's point) | 312.40 | 314.80 | 316.08 | **+3.68 K** |
> | OpenMM itself, solvated | 299.37 | 299.31 | 302.38 | **+3.01 K** |
>
> The dependence collapses by a factor of ten **and the sign flips to OpenMM's**,
> matching its +3.01 K within error. That accounts for every observation: the
> opposite signs, both engines converging as dt → 0, immunity to the SHAKE
> schedule and to the RATTLE count, and the effect surviving in f64.
>
> **Two hypotheses refuted along the way**, both recorded so they are not
> retried. *The constraint schedule*: a patched copy constraining the whole
> step's displacement once, OpenMM's shape, gives −40.18 K against the shipped
> −37.89. *The RATTLE count*: one velocity projection per step instead of three
> gives −41.73 K. Neither moves it.
>
> **And a configurational observable could not settle it at this precision.**
> ⟨U⟩ over 60 ps differs between the engines by 28 / 68 / 31 kJ/mol at 1 / 2 / 4
> fs against SEMs of ~8 — 2.6σ to 5.9σ, and non-monotonic. Notably at 4 fs the
> kinetic estimator says this port is 10.7 K *colder* while ⟨U⟩ says it is
> ~31 kJ/mol *hotter*, which is inconsistent with a uniformly colder ensemble;
> but resolving it properly needs about 16× the sampling. The sampling-point
> result above makes that unnecessary.
>
> 🔴 **THE OBVIOUS FIX WAS TRIED AND IS WRONG, AND THE WAY IT FAILED IS THE
> USEFUL PART.** The two velocities differ by `(halfStep/m)·F`, so the earlier
> estimator looks recoverable from quantities already read back. On `ala-vac`
> that reconstruction reproduces the true pre-kick temperature to 0.2–0.8 K:
>
> ```
>   dt/fs   post-kick (shipped)   reconstructed   true pre-kick
>       1        309.87              312.45          312.40
>       2        303.13              314.99          314.80
>       4        271.98              316.83          316.08
> ```
>
> Implemented on the device and the host, it made the reported temperature
> **far worse**: +5.88 / +21.44 / **+86.85** K at 1 / 2 / 4 fs, against OpenMM's
> +0.42 / +1.63 / +4.45 and the shipped −0.02 / −2.08 / −6.36. ⟨U⟩ was
> byte-identical, confirming only the estimator had moved.
>
> **Why: RATTLE.** The closing half kick is followed by a velocity projection, so
> `v_post = RATTLE(v_pre + (halfStep/m)·F)`. Subtracting the kick does not invert
> the projection — it adds back exactly the bond-directed velocity RATTLE had
> legitimately removed. `ala-vac` carries 12 constraints, 18% of its 3N;
> `ala-wat` carries 1176, **33%** of 3N. The reconstruction was validated on the
> fixture where it happens to work, which is the same error as validating a
> triclinic claim on a rectangular box.
>
> **The correct route, not taken here.** The estimator has to be computed
> *inside* the step, before the closing kick, rather than reconstructed after it:
> a kinetic pass encoded in `encodeLangevinStep` ahead of the final `kick()`,
> writing a buffer `energy()` reads, and emitted only on the last step of a block
> since that is the one the host reports. That is a change to the step encoding
> rather than to a reduction, so it wants doing deliberately.
>
> The change was reverted; the tree reports the post-kick estimator as before,
> and `docs/STATUS.md` tells a user it carries an O(dt²) offset.

> Either way the sampled ensemble was never in question, which is the thing a
> user actually needed to know and the reason this was worth chasing past three
> refuted hypotheses.

## 15. The documented energy drift is the initial slope of a super-linear process

**`docs/STATUS.md`** "energy conservation, friction 0, 20 ps" row; gate at
**`tools/gpu/check-integrator.mjs:216`** (`Math.abs(drift) < 5`).

One 200 ps NVE run (preheated 50 ps at 298 K, shipped defaults, 2 fs), refitted
over growing windows:

| window | drift (kJ/mol/ns/atom) | per DOF | T at end |
|---|---|---|---|
| 20 ps | **+0.459** | +0.228 | 287 K |
| 50 ps | +0.489 | +0.243 | 301 K |
| 100 ps | +2.500 | +1.244 | 292 K |
| 200 ps | **+13.844** | **+6.890** | **447 K** |

So the documented number *is* reproducible at the documented length (+0.46
against +0.30) and **does not extrapolate**: at 10× the length it is 46× larger
and the system has heated from 291 K to 447 K. The mechanism is positive
feedback — heat → larger displacements → skin exceeded (85 violations, 815 stale
steps, worst usage 1.42) → missed pairs → more heat. About **75%** of the 200 ps
drift is the stale list (+13.84 adaptive against +3.55 with `rebuild=1`); the
rest is finding 14. `|drift| < 5` passes only because the arm is 5000 steps.

`index.html:75` has `min="0"` on the friction field, so friction 0 is reachable
from the page. This supersedes the "unreproducible" verdict on the drift figure
in finding 12: the 20 ps number is real, the 10 ps gate is what cannot produce
it, and the number should not be quoted at all without the window.

## 66. The direct-space nonbonded sum disagrees with OpenMM on a sheared cell — ✅ RESOLVED: RainMD IS RIGHT ON BOTH TERMS

> **Investigated 2026-11-12.** An independent O(N²) Lennard-Jones sum with
> **brute-force minimal images over 125 lattice shifts**, sharing no code with
> `cpu/forces/nonbonded.js`, validated on the rectangular fixture first:
>
> | fixture | independent brute force | RainMD | OpenMM |
> |---|---|---|---|
> | ala-wat (control) | 1794.522906 | **6.3e-16** | 2.0e-10 |
> | **ala-oct** | 2673.565469 | **1.5e-15** | **3.07e-5** |
>
> All three agree on the control, so the oracle is sound; on the truncated
> octahedron the independent sum lands on **RainMD** to machine precision and
> OpenMM is the outlier. So the LJ half of this finding is not a RainMD defect:
> OpenMM's Reference platform differs from a true minimum-image sum on a sheared
> cell by 3.07e-5. Consistent with the direction — OpenMM's LJ is *higher*, so it
> includes fewer attractive pairs near the cutoff — and with the earlier
> measurement that RainMD's `c→b→a` reduction changes **0 of 863,955** cutoff
> verdicts on this box.
>
> **The Coulomb half resolved the same way.** A complete independent Ewald sum —
> real-space `erfc` over brute-force minimal images, the exact k-space lattice
> sum, the self and net-charge terms, the exclusion corrections and the 1-4
> exceptions, sharing no code with `cpu/forces/pme.js` or
> `cpu/forces/nonbonded.js`:
>
> | fixture | independent exact | RainMD (48³) | OpenMM (48³) |
> |---|---|---|---|
> | ala-wat (control) | −13652.474621 | **1.3e-7** | 6.2e-6 |
> | **ala-oct** | −15596.114539 | **5.3e-8** | **1.4e-4** |
>
> 🔴 **AND THE CONTROL IS WHAT MAKES THAT READABLE.** On the rectangular fixture
> RainMD and OpenMM agree with *each other* to 1.2e-8 at a 24³ grid while both
> sit 6e-6 from the exact sum — the PME grid floor acting identically on two
> implementations of the same algorithm. Without that arm the octahedral numbers
> could be read as this port drifting rather than OpenMM differing. At 48³ this
> port converges onto the exact answer and OpenMM does not: **2600× closer**.
>
> Gated by `npm run check:triclinic` (its own command — O(N²) over 125 lattice
> images, tens of seconds). So the octahedral case is now held to oracles that
> share no code with it, which is what `check-against-openmm.mjs` could not do.
>
> 🔴 **AND MY EARLIER RATIO ARGUMENT WAS WORTHLESS.** I noted the LJ and Coulomb
> discrepancies were 0.082 and 2.115 kJ/mol — ratio 25.8 against ~3.8 for a
> boundary-pair story — and concluded they were "probably not one cause". With
> both now attributed to OpenMM the ratio needed no explaining: electrostatics is
> long-ranged and simply more sensitive to the image convention. An argument from
> a ratio of two unexplained numbers was never going to be worth much.
>
> **Superseded, for the record:** my first attempt at this compared against
>  `cpu/energy.js`'s `coulombDirect`, which is not the screened real-space sum
> it was taken for — it reports +70075 where a screened sum over the same pairs
> is −14882. That comparison measured nothing and is replaced by the above.

**NEW, found while fixing finding 1.** `cpu/forces/nonbonded.js`,
`webgpu/kernels/nonbonded.js`. Against OpenMM at the same alpha and the same
forced PME grid, on `test/fixtures/ala-oct.*`:

| term | ours | OpenMM | relative |
|---|---|---|---|
| lennard-jones | 2673.565469 | 2673.647583 | **3.07e-5** |
| coulomb (all terms) | −15596.113719 | −15593.998600 | **1.36e-4** |
| per-atom forces | — | — | 2.32e-2 |

Both are **identical at a 24³ grid and a 48³ grid**, so neither is the PME grid
floor and neither is the reciprocal sum — which an exact lattice sum now puts at
2.4e-6 on this box. The rectangular fixture agrees to 2.72e-11 (LJ) and 1.18e-8
(Coulomb), so this is specific to a non-diagonal cell.

**Eliminated by measurement, so nobody repeats them:**

- *the LJ tail correction.* It accounts for the gap exactly if the wrong volume
  is used (|a||b||c| = 21.98 nm³ against the determinant 16.92, giving −60.62
  against −46.67 kJ/mol) — but the existing references are dumped with the
  correction **off**, and matching that convention leaves 0.082 kJ/mol.
- *a non-minimal minimum image.* The `c→b→a` reduction **is** non-minimal for
  14.5% of pairs on this box (worst excess 0.92 nm) and **0%** on a rectangular
  one — but for **0 of 863,955 pairs** does it change the 1.0 nm cutoff verdict.
  All three perpendicular widths are 2.287 nm, so half is 1.1435 nm, above the
  cutoff. The box-width guard is what makes the simple reduction safe and it is
  working as designed.
- *the reduction order.* `a→b→c` disagrees with `c→b→a` about the cutoff on
  22,606 pairs — three orders of magnitude too many to explain 0.082 kJ/mol.

**Severity: unresolved, and the honest answer is that it depends on which side
is right.** RainMD's pair set within the cutoff is the true minimal-image set
(measured above), so it is possible that OpenMM differs here and RainMD is
correct — but that is a claim about the oracle and it has not been established.
Until it is, a truncated-octahedron run should not be trusted to the tolerances
this project quotes for rectangular boxes. The next step is to compute the
direct-space LJ sum independently, pair by pair, and see which of the two it
lands on.

---

# Should-fix

## Numerics and kernels

| # | finding | evidence |
|---|---|---|
| S14 | **SHAKE's stall break makes non-convergence a non-event.** `webgpu/kernels/integrate.js:267` abandons any cluster improving < 1% per sweep and `:288` explicitly does **not** count it as a failure. `:289` clamps the residual diagnostic, so `worstStalledResidual` saturates at 4.0 — it read exactly `4` with `shakeHitIterationCap: false` on a run whose final state was entirely NaN. Forcing `maxConstraintIterations: 2` gives a worst length error of 7.85e-4, **100× the tolerance**, and the run carries on. `cpu/integrate/constraints.js:136-144` returns `converged:false` and no caller tests it. |
| S15 | **The skin is never re-derived after a box change.** `webgpu/simulation.js:576` carries `this.plan.skin`; `skinFor` is never called again. After contracting ala-wat to 11.96 nm³: skin still 0.24287, **cells 1×1×1**, `listCutoff` 1.2429 against half-width 1.0607 — the neighbour build has degenerated to all-pairs, and a skin of 0.12 is 14% faster at that box. Not a minimum-image break (the barostat's guard at `:866-876` correctly holds the cutoff inside half the width), but the controller's `usage` is measured against a budget the box cannot deliver. |
| S16 | **The Ewald exclusion correction loses all precision below r ≈ 1e-3 nm and is NaN at r = 0.** `webgpu/kernels/nonbonded.js:185-188`, `cpu/forces/nonbonded.js:145-148`: `1 - erfc(αr)` cancels against 1, then a difference of two quantities both tending to `2α/√π` is divided by `r²`. Exact to 2.9e-6 at the bonded length; relative error 3.3e-3 at r = 0.003, **1005 and the wrong sign** at r = 1e-5, NaN at 0. Error grows as 1/r², exactly as the cancellation predicts. Reachable from an unminimised start, too large a timestep, or duplicate input coordinates. |
| S17 | **"Same seed, same trajectory" holds only for an identical `run()` schedule.** `webgpu/simulation.js:514` sets `stepsSinceRebuild = Infinity` **outside** the `adaptiveRebuild` guard, so a detected violation forces an off-schedule rebuild at whatever step the block ended on. `run(200)` vs `run(50)+run(50)+run(100)` differ in 3570/3570 words (worst 0.0197 nm); pinning adaptive off *and* widening the skin so no violation fires restores bit-identity. Since `web/app.js:1186` uses `chunk = trajectoryInterval`, **the trajectory depends on the user's output interval**. Same ensemble, not wrong science — but a bug report is unreproducible unless the reporting interval is also quoted. |
| S18a | **Every stage and every production stride replays the same random-number realisation, and no seed control exists.** `web/app.js:375` hardcodes `seed: 20261106`; `webgpu/simulation.js:327-330` resets the device step counter to `options.step ?? 0` and `web/app.js:1126` calls `setState` with no `step`; `web/app.js:1074-1085` builds a fresh `GpuSimulation` per stride. The thermostat draws `rainmdGaussian(seed, atom, stepCounter[0])`, so **stride 2's step 0 draws the identical triple as stride 1's step 0** (confirmed `identical: true`). `grep -i seed index.html` finds no control. So "run it twice for an error bar" returns the same trajectory, a 5×2 ns strided production is one noise realisation five times over, and any error estimate built from the strides is meaningless. |
| S18b | **The SHAKE stall break has no stated origin and fires above the derived f32 floor.** `webgpu/kernels/integrate.js:267`/`:350` — `worst >= previousWorst * 0.99`. At 298 K the kernel's own `worstStalledResidual` is **2.44e-5 = 3.5× the 7.03e-6 floor**, and in hot NVE runs it reaches **1.31e-4 = 19× the floor**. `toleranceWasClamped: false`, so the break — not the floor — is what stops the solver, and `0.99` is a chosen constant against AGENTS.md's "a tolerance must have a stated origin". The host solver has no such break. Nothing asserts on `worstStalledResidual`. This is the likely unexplained remainder of finding 14, and the residual is **worse at smaller dt**, in the same direction. (Overlaps blocker 14 above — same line, different consequence.) |
| S18c | **The rebuild-interval ceiling is never lowered, so a long run keeps re-entering the stale-list state.** `webgpu/simulation.js:113` sets `requestedRebuildInterval` once; `:491` only grows back toward it; the violation branch halves `rebuildInterval` but never touches the ceiling. Over 125k-150k steps on `ala-wat` at the shipped derived skin: NVT 2 fs **370 violations / 1491 stale steps**, NVT 0.5 fs 127/508, NPT 2 fs 48/460, NVE 2 fs 85/815, worst usage 1.29-1.43 throughout. `STATUS.md`'s "the skin controller converges — worst usage 1.14, 2% of steps on a stale list" is a **500-step** measurement; over 125,000 steps the controller oscillates back to a ceiling it has already been caught violating. Not silent (`web/app.js:1251-1255` warns with the counts), but it never converges and the warning count grows without limit. |
| S18 | **Nothing pre-flights the allocation against `maxStorageBufferBindingSize`.** `webgpu/runtime/allocator.js:42` — `budgetBytes` is `options.budgetBytes ?? null` and **no caller ever sets it**, so the `GpuBudgetError` branch at `:68-77` is unreachable from the page and its message ("A smaller system, a shorter neighbour list, or a coarser PME grid would fit") never appears. Measured with the repo's own planner on a 52,200-atom box: neighbour buffer **190.0 MiB** against the WebGPU spec default of 128 MiB, crossover ≈ 35,000 atoms. On this Mac the adapter maximum hides it; on a device reporting the spec default, an ordinary solvated protein fails as a raw WebGPU validation error. Confirmed separately that an over-limit buffer does **not** throw — `allocate()` returns an invalid `GPUBuffer` and the error goes to `onuncapturederror`. |

## Failure handling

| # | finding | evidence |
|---|---|---|
| S19 | **Device loss is detected, recorded, and never read.** `webgpu/runtime/device.js:85-91` fills `report.lost` and `report.lastUncapturedError`; both have **zero** readers across the tree. Measured with `device.destroy()` under a running equilibration: `lostSeen { reason: "destroyed" }`, and the user sees `error: AbortError: Failed to execute 'mapAsync' … Buffer was unmapped before mapping was resolved.` The device panel still says "WebGPU available", and the run's data is gone via finding 6. |
| S20 | **Nothing is written to disk mid-run.** `shared/traj/dcd.js:85` holds every frame as a `Uint8Array` in `this.chunks`; the file exists only after `finish()`. There is no `beforeunload` handler anywhere. Measured frame size 14,360 B at 1190 atoms; a 500 ns production on the 2854-atom fixture at the default 10 ps interval is 50,000 frames × 34,328 B = **1.60 GiB of ArrayBuffers held in the tab** — for the run the docs describe as "roughly 6.5 days of a tab staying open". `shareFrames` is capped at 200; the trajectory is not capped at all. |
| S21 | ✅ **FIXED (with blockers 5 and 6).** **The one loud line a blown-up run prints is the wrong diagnosis.** `webgpu/simulation.js:470-481`: `Math.sqrt(Math.max(NaN, 0))` → `NaN`, so `worstDisplacement` stays 0 while `usage = NaN` fails `usage <= 1` and counts **every** block as a violation. Rendered as two consecutive lines reading `worst displacement 0 pm — 0% of the budget used` and `the neighbour list was stale for 300 of 300 steps`, followed by the reassurance "this costs exactness rather than stability" — which is exactly backwards. The interval is also driven to 1, so a dead run gets 20× slower and reads as a performance problem. |
| S22 | ✅ **FIXED (with blocker 5).** **`writeRestart` emits `NaN` fields its own reader then refuses.** `shared/traj/rst7.js:25-37` guards against a coordinate overflowing 12 columns ("the simulation blew up") but `Number.NaN.toFixed(7)` is `"NaN"`, 3 characters, which passes. Round-trip: `writeThrew: null`, then `Error: non-numeric coordinate field "NaN" on line 3`. The guard catches the flown-away atom and misses the NaN, so the page writes a restart it cannot read back and the user finds out in AMBER. |
| S23 | **Pause can lag by a whole trajectory-write interval while the log says it is already held.** The gate is awaited at `web/app.js:1182` but the chunk is `trajectoryInterval` (`:1155`), and `simulation.run()` has no pause check; the handler logs "paused — the integrator is not being advanced" immediately at `:873`. Measured at a 1 ps interval: paused at step 1,500, after a 3 s hold the step was 2,000. At the **default** 10 ps interval that is 5,000 steps (~6.5 s at 1190 atoms, ~12 s at 2854); at the 1000 ps option, roughly **11 minutes** of "paused" that is still integrating. The existing pause gate passes only because it uses a 1 ps interval and a 1.2 s wait. |

## Parser and input handling

| # | finding | evidence |
|---|---|---|
| S24 | **No section is length-checked against its declared count.** `sections.js:13-17` claims the reader "stops on the declared element count, not on the line geometry"; there is no declared element count anywhere in the reader, and no caller compares a section's length against `NATOM`/`NRES`/`NUMBND`. Re-declaring `ATOM_TYPE_INDEX` as `%FORMAT(10I7)` with 8-wide data gives **no error, no NaN, and an energy 110 kJ/mol from the right one** (+54.293 against −55.794). Dropping one `CHARGE` line gives `E = NaN` with no message naming `CHARGE`. |
| S25 | **A duplicate `%FLAG` silently overrides the first.** `sections.js:99` (`sections.set`). A second `%FLAG CHARGE` block appended to the fixture parses with 0 warnings and gives `E = +74.702` against −55.794. This is what a hand-concatenated prmtop does. |
| S26 | **A periodic restart that lost its box line falls back to the topology's stale box.** `inpcrd.js:82-105`, `system.js:53`. Dropping the final box line leaves `3N` numbers, a *valid* count, so the file reads as `box null` and `buildSystem` takes tleap's original guess — an NPT-equilibrated restart silently reverts to the pre-equilibration volume. Same shape for a `6N` file. |
| S27 | **A topology/molecule-partition disagreement is a log line, not a refusal.** `prmtop.js:276-282` pushes a string into `topology.warnings`, surfaced only as `note: …` in the scrolling text log at `web/app.js:293`. The barostat scales molecular centres of mass, so a partition disagreement is a physics problem. (It did not fire on any of eight systems built for this review.) |

## Claims, documentation and hygiene

| # | finding | evidence |
|---|---|---|
| S28 | ✅ **FIXED (with blocker 11).** **"nothing in a step reads back to the host" is contradicted 28 lines later in the same page.** `README.md:17-18`, `index.html:256-257`, `STATUS:127` against `index.html:284` ("synchronises with the browser once per move") and `STATUS:421-424` ("one round trip per `barostatInterval` steps"). Pressure defaults to 1, so NPT — and the round trip — is the shipped default. Measured cost **0.167 ms/step**, not the "about 0.3 ms" the page states. |
| S29 | ✅ **FIXED (with blocker 11).** **"The viewer reads the simulation's own position buffer, so drawing a frame copies nothing to the host" is false.** `README.md:39-40`. `app.js:1189` does `await simulation.readPositions()`, then `viewerPdb()` → `writePdb()` → `MolstarViewer.setFrame(pdb)`. `docs/ARCHITECTURE.md:259-264` argues *against* exactly this design as the reason py2Dmol was rejected. |
| S30 | **"the DCD writer is byte-compared against `openmm.app.DCDFile`" is backed by no command.** `README.md:29-30`, `STATUS:38`. `grep -rn DCDFile` hits only a source comment and STATUS itself; `test/physics.test.js:253` checks against hard-coded expectations, not against a file OpenMM wrote. The "2-7 ulp in the unit-cell cosines" and "7.6e-7 Å" figures have no rerunnable origin. This is the only claim about the *output format* with no gate. |
| S31 | **The bonded f32 bound's arithmetic does not evaluate to its stated answer.** `STATUS:102-103` and `ARCHITECTURE:121`: `476976 × 2.82 × 5.96e-8` = **0.0802**, not 0.16; 0.16 needs `1.19e-7`. Both numbers get called "f32 epsilon" (unit roundoff vs spacing near 1), and the *bound* is right — the kernel measured 0.157 — but the product as printed is half the stated answer, and the constraint-floor bullet three lines down uses `5.96e-8` and **does** evaluate correctly (7.02e-6, matching the 7.025e-6 the code computes). Two bullets, two conventions, one self-consistent. |
| S32 | **The PME grid floor 8.4e-6 is unreproducible and its fixture is orphaned.** `STATUS:99-101` presents it as the first "derived, not chosen" tolerance. `grep -rn ala-wat-ewald` across the tree returns **nothing** — the 355 KB fixture is read by no test, and `make_fixtures.sh` has no Ewald arm. The number survives as a comment in `tools/gpu/check-barostat.mjs:55`. |
| S33 | **The erfc floor 1.19e-6 is hard-coded, not derived by any command.** `tools/gpu/check-nonbonded.mjs:268` holds it literally. An independent `Math.fround` emulation of the shipped coefficients against `shared/erf.js` over αr ∈ (0, 2.8] gives worst relative **1.478e-6**. The emulation is not a GPU, so this is not proof the kernel is worse — but the claimed 1.191e-6 is not checkable today and the Coulomb bar rests on it. |
| S34 | **The WGSL `erfc` ships Numerical Recipes' `erfcc` coefficients verbatim, with the attribution written into the source.** `webgpu/kernels/wgsl-common.js:128-161`; `:129` says "These are Numerical Recipes' `erfcc` constants". NR licenses per-machine and explicitly not for redistribution; this repo is MIT and public. The same comment at `:132-134` records that a least-squares refit over the same domain gives **1.191e-6, the same number to four digits** — so a licence-clean substitute is already measured as numerically indistinguishable. (`shared/erf.js:12` uses Cody's netlib CALERF, which is freely redistributable — that one is fine.) Not a legal opinion; the point is that the fix costs nothing. |
| S35 | **`molstar-probe.html` and `tmp-helix.pdb` are live on the promoted site, and the probe page is broken there.** Both tracked, both fetched successfully from `pablo-arantes.github.io/rainmd/`, neither referenced by anything but each other. All three of the probe's URLs are absolute (`/web/vendor/molstar.js`, `/tmp-helix.pdb`), which is the exact mistake `web/app.js:178-182` warns about: a project Pages site serves from `/<repo>/`, so every fetch 404s and the page renders blank. |
| S36 | **The browser lane finds Chrome only at five fixed paths.** `tools/gpu-page.mjs:21-27` — three `/Applications/…` plus `/usr/bin/google-chrome` and `/usr/bin/chromium`, resolved by `statSync`, with no `$CHROME_PATH` override and no `PATH` lookup. Misses Debian's `chromium-browser`, snap, Flatpak, `google-chrome-stable`, and every Windows path. Breaks `test:gpu:page`, `test:page:simulate` and `test:page:panels` — 3 of the 9 commands under "Checking it yourself" — on a platform `README:14` explicitly supports. |
| S37 | **Four documented check-counts disagree with `package.json`.** `README:120` "all nine kernel checks", `README:136` "The five GPU checks" above a table of **eight** rows, `STATUS:93` "the same five check modules", `index.html:254` tells the user five. `test:gpu` and `test:gpu:page` each chain **12**. Also `index.html:253` says the button runs "the **bonded** differential test"; `app.js:666` runs `check-forces` and `check-integrator`. |
| S38 | **`docs/STATUS.md` item 2 names the wrong renderer, in the list that carries its own red warning.** `:190` — "**3Dmol draws the page now**". 3Dmol was deleted at `f57ce6d`; Mol\* draws it, as `:74` and `:312` of the same file say. `:171-174` flags this list in red as having "BEEN WRONG BEFORE". `index.html:306-310` likewise still comments "3Dmol defines a global" above `<script src="web/vendor/molstar.js">`, and `tools/gpu-page.mjs:768-797`'s `--probe-3dmol` block reads `window.$3Dmol`, which has not existed since that commit. |
| S39 | **`docs/ARCHITECTURE.md` has drifted furthest of the three documents.** `:246` "RAINMD IN ITS ENTIRETY IS 159 KB GZIPPED" was written at the first commit, before any vendored viewer; at HEAD it is **1.63-1.64 MiB**. `:250` "vendoring it would roughly triple what a visitor downloads" was overtaken by vendoring something 10× larger. `:259-264` argues against host-built PDB text, which is what the page now does. `:270-285` "So the ribbon is native WGSL" describes a renderer the page does not load. `:331-335` is written as a pending decision taken at `f57ce6d`. `:209` "no number in `docs/STATUS.md` depends on it" is contradicted by `STATUS:64-67`. |
| S40 | **The shipped-bytes row is stale and disagrees with the page.** `STATUS:319` says 461 KiB raw / 174 KiB gzipped; measured **521 KiB raw / 187 KiB** (161 KB as a single stream). `index.html:269` tells a user **159 KB** for the same quantity. `STATUS:321` "about nine times the program" measures **7.9×**. |
| S41 | **"0.8 ms on the device" and "1.10 against 0.86" survive in Known limitations.** `STATUS:393` and `:424` are the pre-displacement-kernel numbers that `STATUS:134` flags in red as stale and unreachable. Measured today: **1.259 NPT / 1.092 NVT**. A reader who skips to Known limitations gets the retracted figure. |
| S42 | **The two-body ms/step figure appears five times with five values**: `STATUS:283` (2.24 → 77 ns/day → "roughly 6.5 days"), `STATUS:161` (1.683 → 103 → 4.9 days), `STATUS:42` (2.01), `app.js:423` (2.69→2.09), `app.js:884`/`:948` (2.7, and "at the measured 2.7 ms/step a **1190-atom** box does about 64 ns/day", which attributes a two-body rate to a system measured at 1.259/137). |
| S43 | **"roughly 60x the host reference" is the top of the range quoted as typical.** `README:21`; `STATUS:132` "a factor of 40 to 60". Host measured at **59 ms/step** against 1.259 → **46.9×**; the claimed 47-79 host range spans 37-63×. |
| S44 | **The live animation is not "off by default".** `index.html:275-276` and `ARCHITECTURE:326-328` say it is; `index.html:186` selects `animate production only`, so every production frame is drawn. The "17% of a run" cost cited as the reason is recorded elsewhere as **22%** (`app.js:423`) and, by `app.js:1212`'s own admission, was measured at a 1 ps interval where the default is 10 ps. `app.js:410-421` still says "A LIVE ANIMATION IS NOT ONE OF THEM". |
| S45 | **The retained-but-unloaded renderer is 1092 lines and three modules, not two.** `STATUS:187` names `web/viewer.js` and `shared/analysis/ribbon.js`; a transitive import walk from `web/app.js` (49 modules reached) shows exactly three first-party modules outside the page graph — those two plus **`shared/analysis/secondary.js`**, superseded by Mol\*'s own assignment. `STRUCTURE_COLOURS` (`selection.js:275`) is a fourth member. It costs 8.5% of first-party source, 0 bytes served, **10 of 61 host tests**, 1 of 12 modules in each GPU lane, and **8 rows** of the STATUS validated table — one of which (`:67`) reports colours from a palette the page no longer uses. Two gates make false page-level claims about it: `test/interface.test.js:71` ("every module the page imports is on disk") iterates a list containing `web/viewer.js`, which the page does not import, and omitting `web/molstar-viewer.js`, which it does. |

---

# Nice-to-have

| # | finding |
|---|---|
| N46 | **The documented displacement scaling is wrong, which shrinks the controller's stated margin 8×.** `shared/topology/cells.js:56-63` and STATUS claim ≈ n^0.25; measured (host f64, pinned interval, warm) 0.01283 nm at 1 step → 0.34296 at 80 = **n^0.75**, and n^0.80 over the 4→20 range. So a 1.25× growth raises usage by 1.195, not 1.057 — margin from the 0.4 gate is 2.1×, not 17×. The conclusion survives; the derivation does not. |
| N47 | **`calmBlocks` counts the wrong thing.** `webgpu/simulation.js:483` increments for any block with `usage <= 1` while the comment and the gate say `usage < 0.4`. Two blocks at 99% of the skin then one at 39% triggers a growth. |
| N48 | **The PME grid is fixed across a barostat move with no diagnostic for an expansion.** `webgpu/simulation.js:566-571`, deliberate and matching OpenMM, safe under contraction. K = 24 stays at or above OpenMM's rule only up to L = 2.9952 nm — 10.2% in length, **33.7% in volume** — against the barostat's own 30%-of-volume per-move cap at `:902`. |
| N49 | **f32 energy resolution scales ~linearly with N, and the barostat depends on it.** Measured absolute resolution of the device total potential against the f64 host: **0.018** kJ/mol (1190 atoms), **0.037** (2547), **0.059** (2854), tracking \|coulombDirect\|. kT at 298 K is 2.478. Both the Metropolis test (`:879-885`) and the minimiser's line search (`:786`) decide on differences of that quantity; the minimiser half is documented, the barostat half is not. Extrapolation only. |
| N50 | **`shared/pme/parameters.js:88-98` uses \|a\|,\|b\|,\|c\| where OpenMM uses the diagonal widths.** On the octahedral box RainMD picks 32³ where OpenMM picks 31×29×25 — conservative, not wrong, ~1.4× wasted reciprocal work on every triclinic cell. |
| N51 | **`prmtop.js:55-57`** — `atomFromStride` is `Math.abs(v)/3` with no integrality or range check, so a non-multiple-of-3 index truncates and names a different atom. `parameter = raw[…] - 1` (`:335,357,404`) is never range-checked, so an out-of-range type index gives `undefined → NaN` for `k`/`r0`/`theta0`. |
| N52 | **`prmtop.js:197-219`** — the 10-12 scan counts type *pairs* over the full `NTYPES²` square, so one h-bond pair is reported as 2. Message is otherwise excellent. |
| N53 | **`ARCHITECTURE.md`'s "A prmtop has no chain IDs at all — AMBER throws them away"** is the stated basis for the molecule-partition receptor/binder split. The format does have `RESIDUE_CHAINID`/`RESIDUE_NUMBER`/`RESIDUE_ICODE` and ParmEd writes them; this tleap does not, so the design decision holds for notebook output but the absolute claim is false. |
| N54 | **`index.html:73`** — the timestep field has `min="0.5"` and no `max`, while the notebooks' dropdown is 0.5-4 fs. A maximum, or a warning above 4 fs, costs nothing and closes the easiest route into finding 5. |
| N55 | **Three exports are referenced by nothing, tests included**: `shared/units.js:81` `ljFromAmberCoefficients`, `webgpu/runtime/allocator.js:20` `bufferPoolingEnabled`, `webgpu/runtime/execution.js:88` `encodePass`. (A fourth, `web/viewer.js:329` `StructureViewer`, is covered by finding S45.) Separately 54 symbols are exported but used only in their own file — almost all per-kernel shader builders, which looks deliberate; listed so it is not mistaken for a finding. |
| N56 | ✅ **FIXED (with blocker 6).** **`web/app.js:1364`** logs `wrote ala-wat.dcd` after writing `ala-wat_equil.dcd` / `ala-wat_prod_1.dcd`; the named file does not exist. |
| N57 | **`STATUS:3-4` — "Every number here was measured … on 2026-11-06" — is false.** `STATUS:180` itself says 2026-11-07; the throughput tables depend on files dated Nov 9; rows 78-80 come from Nov 9 commits; rows 36-37, 45, 46, 56, 393, 424 pre-date Nov 6's configuration and no longer reproduce. Name a range, or date the rows. |
| N58 | **Stale counts and orphan rows.** `STATUS:56` "33 controls" → **46** measured, 0 missing (and the test asserts only `> 20`). `STATUS:91`'s thermostat row ("150 steps at 50/ps → 295.8 K") describes an arm `check-integrator.mjs:131` does not have. `STATUS:50` ("60000 samples, mean 0.003, variance 1.001") does not match that command's 3570/−0.0113/0.9916. `STATUS:37` 317.8 K → measured 319.40 K. `STATUS:96` "Two tolerances … are derived" sits above four bullets; AGENTS.md says "Two of the three". `README:66` "which five things were taken" from JIGGLE → ARCHITECTURE has six. |
| N59 | **`README:11-12` "the solvated alanine dipeptide runs a real NPT simulation on arrival"** — it needs a Load-example click, and Equilibrate is disabled until Minimise has run. |
| N60 | **`localfold.png` is a JPEG** (`file` → `JPEG image data, JFIF 1.01, 1280x1212`) served as `image/png`. Both it and `issue_rain.jpg` are unreferenced artefacts at the root of a published site; both have real documentary value next to the prose that explains them (`ARCHITECTURE.md` for the first, `STATUS.md:290` for the second) and neither contains personal data. Move to `docs/images/` and link, rather than delete. |
| N61 | **Mol\* 5.13.1 is verified but the pin exists only in prose.** `grep -oE '5\.13\.[0-9]' web/vendor/molstar.js` → `5.13.1` in `var uf="5.13.1"`, and the size matches `STATUS:318` to the kilobyte. But `package.json` has no `molstar` entry and there is no recorded upstream URL, tag or checksum, so re-vendoring is not reproducible from the repo. Both `molstar.js` and `molstar.css` also end with a `sourceMappingURL` whose `.map` is not vendored — a 404 pair whenever anyone opens devtools. |
| N62 | **Licence notices are complete, with two small gaps.** `web/vendor/molstar-LICENSE.txt` is the full unmodified MIT text with the correct holder, and the bundle itself carries full MIT texts for Immutable and React plus four further holders, so notice retention is met. But the root `LICENSE` says nothing about Mol\* at all (its only third-party section covers AMBER), and `molstar-LICENSE.txt` does not name which files it covers. `molstar.css` carries normalize.css's one-line MIT pointer rather than its full text — upstream's own output, so inherited rather than introduced. |
| N63a | **The DOF count subtracts 3 for a centre-of-mass motion removal that does not exist.** `shared/topology/system.js:203` is `3·massiveAtoms − constraints − 3`, and AGENTS.md states that formula as an invariant. A repo-wide grep for `CMMotion\|centre of mass\|center of mass\|momentum` finds nothing but an unrelated comment in `barostat.js`. OpenMM earns the −3 because `createSystem` adds a `CMMotionRemover` (confirmed present); RainMD never removes COM momentum, and under Langevin dynamics that is a genuine thermostatted degree of freedom. The **count matches OpenMM exactly** (2391 both) and both readouts use it consistently, so the only consequence is a reported temperature biased high by `2394/2391` = **+0.125%** (+0.37 K at 298 K) — negligible, but the −3 is unearned as written, and AGENTS.md should say why. |
| N63b | **"Different trajectories sampling the same ensemble" is false at finite timestep.** `cpu/integrate/langevin.js:13-18` and STATUS's Known-limitations bullet. By exact Lyapunov solution for a harmonic mode (no sampling, friction-independent): BAOAB's **configurational** temperature is exactly 298.00 K at every dt, while the notebooks' older `LangevinIntegrator` gives **330.63 K at 2 fs and 492.39 K at 4 fs** for a 20 fs-period mode. The two splittings sample *different* configurational distributions and the gap grows steeply with ω·dt. The project's choice of BAOAB is right and worth stating **more** strongly; "the same ensemble" is the part that is wrong. (Aggregate effect on the whole system is small because most DOF are soft — a `<PE>` difference was not resolvable above ~50 kJ/mol correlated-sampling error over 200 ps.) |
| N63c | **Two small departures from OpenMM's barostat.** (a) `webgpu/simulation.js:899-904` resets the acceptance window every 10 attempts *unconditionally*; OpenMM resets `numAttempted/numAccepted` only **inside** the two adaptation branches, so its window grows while the rate sits in [0.25, 0.75] — RainMD re-tunes on 10-sample noise. No impact detectable in the measured volume distribution. (b) `webgpu/simulation.js:425` fires the move at a *block* boundary when `stepsSinceBarostat >= barostatInterval`, and `blockSize = rebuildInterval`, so the measured rate was 5004 attempts over 150,000 steps = **one every 30.0 steps**, not the configured and reported 25. |
| N63d | **The device's Gaussian generator is never measured by the suite.** `check-integrator.mjs` arm 1 measures the **host** f64 generator; arm 2 feeds the device *supplied* noise, so `rainmdGaussian`'s f32 `log`/`cos`/`sin` path is checked against nothing. A 0.7% variance deficit there would read as a 2 K thermostat offset and nothing would catch it — which matters given finding 14. It was checked for this review and is **fine** (variance 1.00033 ± 0.00100 over 2.0e6 normals, reproduced with `Math.fround` at every operation); the gate is what is missing. |
| N63 | **Two `.gitignore` rules match nothing.** `*.pdb.out` has no producer anywhere; `/tmp/` is root-anchored and so matches a `rainmd/tmp/` directory, while every tool writes to the *system* `/tmp`. Neither would have caught `tmp-helix.pdb`. A root-anchored `/tmp-*` would. (The two `.DS_Store` files are already correctly ignored and were never committed — `git log --all -- '*.DS_Store'` is empty.) |

---

# Checked and found correct

Stated at length because it is as much of the result as the defects are, and
because several of these were the obvious places to expect a bug.

**Physics and numerics**

- **The mixed-precision architecture is sound.** Direct-space and bonded terms
  accumulate in an f32 register **per atom** (length set by density, not by N)
  and are summed across atoms in f64 on the host; the PME reciprocal is
  tree-reduced in 256-wide workgroups (error ~log n, not n) and finished in f64;
  self energy and the dispersion tail are f64-only. **No sum whose length grows
  with the system is done sequentially in f32.** Confirmed by per-term agreement
  against the f64 host with no growth from 1190 → 2854 atoms and 13824 → 46656
  grid points.
- **There is no virial anywhere** — the barostat is Monte Carlo, which removes
  the usual worst f32 sum in an MD engine.
- **PME grid selection matches OpenMM at every box size tested.** `alpha` agrees
  to all 16 digits (2.628260884878466); `gridSizeFor`'s pre-rounding integer
  reproduces OpenMM's choice at **all 15** cubic boxes from L = 2.05 to 10.0 nm;
  rounding to 2^a·3^b·5^c is always **up**; `nextTransformableSize` returns a
  value below its argument for **0 of 508** integers in 5..512.
- **Determinism holds for a fixed call schedule** — bit-identical over 200 steps
  in NVT *and* NPT, 0 of 3570 position words and 0 of 3570 velocity words, with
  identical box and identical barostat attempt/accept counts. The barostat's host
  round trip is **not** a source of non-determinism. The no-atomics-in-the-force-path
  rule does what AGENTS.md says it does.
- **The adaptive rebuild controller cannot outrun the displacement**, even with
  the ceiling removed: block-aligned from a minimised start with a requested
  ceiling of 400, the interval grew 4 → 33 over 488 steps with **0 violations**
  and worst usage flat at 0.734. Holds despite finding N46.
- **The displacement kernel itself is right** — no minimum image (correct, since
  positions are never wrapped), origin seeded only beside the rebuild,
  `atomicMax` over `bitcast<u32>` exact for non-negative floats. Finding 10 is
  about the window the host attributes it to, not the number.
- **The barostat's box guard works**: a move that would put the cutoff past half
  the smallest width is refused, measured at the fully contracted box.
- **Constraint tolerance and achieved geometry are honest.** Worst relative
  length error 5.05e-6 at 2 fs and 5.03e-6 at 3, 4 and 5 fs — flat, i.e. at the
  documented f32 floor rather than the requested 1e-5. OpenMM 8.4 Reference on
  the same fixture achieves 3.59e-7, 14× tighter, because it keeps positions in
  f64. The clamp and its report are correct and say so.
- **The 4 fs lever is genuinely available** (modulo the missing hydrogen-mass
  repartitioning): 2, 3, 4 and 5 fs all stable over 200 steps from an
  *unminimised* start with HBonds.
- **The exclusion correction is exact at physical distances** — 2.9e-6 relative
  at the constrained bond length, i.e. at the f32 `erfc` floor.

**Ensemble and sampling — the strongest positive result in the review**

- **This really is BAOAB, proved rather than asserted.** `cpu/integrate/langevin.js`
  is **bit-identical** to an independently hand-written BAOAB map (one step,
  fixed noise, relative difference **0** in both x and v) and differs from OBABO
  (8.6e-5 / 1.1e-2) and from OpenMM's original `LangevinIntegrator`
  (1.5e-2 / 8.8e-3). The device sequence is BAOAB with one force evaluation per
  step, and chunking a run into many `run()` calls does not perturb it — the two
  half-kicks at a block boundary use the same force.
- **And the notebooks really do use the older integrator**, read rather than
  assumed: `Amber.ipynb` cell18:136 / cell21:105 and
  `Protein_ligand.ipynb` cell17:153 / cell20:120 all build
  `LangevinIntegrator(temperature, friction, dt)` with friction 1.0,
  `setConstraintTolerance(1e-6)` and `MonteCarloBarostat` at frequency 25.
  RainMD's page defaults match. STATUS's claim here is correct.
- **The DOF count is identical to OpenMM's** — 2391 on the same prmtop (1190
  particles, 1176 constraints) — and both the host and device readouts use that
  same count. (See finding N63a for *why* it matches.)
- **The temperature distribution's shape is right.** Against the analytic
  χ²/Maxwell law for 2391 DOF: KS statistic **0.021** at 2 fs, **0.012-0.014**
  at friction 10, **0.017** at 4 fs; excess kurtosis 0.0041 against analytic
  0.0050; sd ratio 1.010 and 0.999. Only the *mean* is displaced (finding 14).
  Useful nuance: OpenMM's own LangevinMiddle sd sits 2-3% below the analytic χ²
  value, so χ² is not the exact reference for on-step BAOAB velocities.
- **The barostat scales molecular centroids, not atoms** — the question that
  matters most for rigid water. Over all 1176 constrained distances the worst
  change under a 1.05 scale is **1.24e-6** relative, over all harmonic bonds
  **0**, and the centroid separation between two molecules scales by 1.0499999.
  `webgpu/kernels/barostat.js:25-71` uses a **mass-weighted** centroid per
  molecule from the union-find bond-graph partition (not from
  `ATOMS_PER_MOLECULE`). OpenMM's reference kernel uses the **unweighted
  geometric** centre — a different but equally valid move, since both are linear
  per-molecule maps leaving internal coordinates fixed, so the Jacobian is
  `(V'/V)^N_mol` either way and both sample the same NPT ensemble. Molecule count
  for that Jacobian: RainMD 393, OpenMM `getMolecules()` 393.
- **The Metropolis criterion and the move-size adaptation match OpenMM line for
  line**: `volumeScale = 0.01·V`, `deltaVolume = volumeScale·2·(u−0.5)`,
  `lengthScale = cbrt(V'/V)`, `w = ΔU + P·ΔV − N_mol·kT·ln(V'/V)`, accept unless
  `w > 0 && u > exp(−w/kT)`, adapt every 10 attempts by /1.1 below 25% and ×1.1
  above 75%, capped at 0.3·V. Identical except finding N63c. Independently
  validated by running RainMD's own `metropolisWeight` plus accept rule as an
  ideal gas of 200 molecules, whose exact volume distribution is
  Gamma(N+1, kT/P): **mean error −0.003% / +0.07%, variance error +0.6% / +0.07%**,
  acceptance 50.2% over 400,000 attempts, two seeds.
- **The NPT volume *distribution* — the second moment, which is the sharp test —
  is right.** 300 ps at 1 bar / 298 K, last 75 ps, against OpenMM NPT on the
  same system:

  | | mean V (nm³) | sd V | mean ρ (g/cm³) | implied κ_T (/bar) |
  |---|---|---|---|---|
  | RainMD | 12.026 | 0.154 | **1.0014 ± 0.0129** | 4.80e-5 |
  | OpenMM LangevinMiddle | 12.047 | 0.169 | 0.9997 ± 0.0141 | 5.78e-5 |
  | OpenMM LangevinIntegrator (notebooks') | 12.039 | 0.147 | 1.0003 ± 0.0122 | 4.36e-5 |

  RainMD's volume fluctuation sits *between* the two OpenMM arms and implies an
  isothermal compressibility of 4.8e-5 /bar against water's experimental 4.5e-5
  and TIP3P's ~5.8e-5. Acceptance 50.2%, `volumeScale` converged to 1.55% of V,
  0 cutoff refusals.
- **Thermostat noise is white.** Lag-1/2/3 autocorrelations along `step` for four
  atoms are all within 1.5 SE of zero (|ρ| ≤ 0.0039 at 200,000 samples), as are
  the correlations across atoms at fixed step and between the three components
  of one draw. Mean 0.001-0.004, variance 0.9964-1.0007.
- **The device generator's f32 arithmetic is clean** — variance
  **1.00033 ± 0.00100** over 2.0e6 normals, reproduced with `Math.fround` at
  every operation. The f32 Box-Muller path is **not** a source of any temperature
  offset, which is what rules it out as an explanation for finding 14.
- **Velocity initialisation projects the 3N Maxwell draw onto the constraint
  manifold** in both position and velocity, which is what stops the first-step
  RATTLE discard reading as drift.
- **Barostat housekeeping matches OpenMM**: velocities are not rescaled, `alpha`
  and the PME grid are held fixed across a move while the reciprocal vectors,
  Ewald self-energy and volume-dependent dispersion correction are re-derived,
  an accepted move forces a neighbour rebuild, and a move that would put the
  cutoff over half the smallest box width is refused and counted.
- **The hard LJ cutoff is not a drift source here** — same as OpenMM's
  `NonbondedForce` default, and the OpenMM NVE control bounds its contribution at
  ≤ 0.014 kJ/mol/ns/atom on this system.

**Parser**

- **1-4 scaling is genuinely per-dihedral-type.** Confirmed on a *mixed* system
  (ff14SB peptide + GLYCAM_06j tetrasaccharide in one topology):
  `distinct SCEE: 1.2, 1` and `distinct SCNB: 2, 1`.
- **Exception construction is exact against OpenMM on all six systems built for
  this review** — ff19SB, GLYCAM, mixed, octahedral, TIP4P-Ew and a
  ParmEd-written 12-6-4 file — 0 only-in-one in every case, worst Δε 2.22e-16,
  including ring-closure 1-4 suppression.
- **Truncated-octahedron box vectors are exactly OpenMM's**, digit for digit.
- **LJ and direct-space Coulomb are exact on the octahedral cell** (LJ
  8481.754697 both), which is what localises finding 1 to the reciprocal sum and
  proves the triclinic minimum image and pair enumeration are right.
- **Molecule imaging on a sheared cell is exact** — max deviation **4.44e-16 nm**
  against OpenMM's `enforcePeriodicBox`, 0 of 3360 atoms differing by > 1e-6.
- **The DCD unit-cell record is correct for a non-orthorhombic cell** and
  byte-identical to OpenMM's; MDAnalysis reads both back to
  `[37.4969 37.4969 37.4969 109.47122 109.47122 109.47122]`.
- **Velocity parsing matches ParmEd bit for bit**; all four inpcrd variants
  (3N / 3N+6 / 6N / 6N+6) are discriminated correctly, and an impossible count
  throws a message naming all four expected totals.
- **Atom-count mismatch, swapped file types, NetCDF restarts, `IFPERT`, legacy
  10-12 terms and `HAngles` are all refused by name with actionable messages.**
  `%COMMENT` anywhere is skipped correctly; `%FLAG` order does not matter; a
  `%FLAG` with no `%FORMAT` is caught downstream.
- **52,200 atoms parses cleanly** with correct counts, no NaN, correct box.

**Failure handling**

- **Pause/resume is bit-identical, as the README claims** — confirmed twice, and
  strictly stronger than the existing step-counter gate. Engine: positions
  `0 of 3570` differing, velocities `0 of 3570`, box identical. Page: two full
  sessions (500-step minimise + 10,000-step NPT) with FNV-1a over the raw f32
  bit patterns gave `positionsHash 4cf78664` and `velocitiesHash 7bc857d2` in
  both. Only the *latency* is wrong (finding S23).
- **The DCD header is always consistent with the body**, because it is built at
  `finish()` from the real frame count — the "header says 5000, body holds 1200"
  file cannot be produced by this writer. Tested anyway by truncating mid-frame
  and by patching NSET to 5000 by hand: cpptraj and MDAnalysis both report the
  actual count with explicit warnings, and neither is misled.
- **The reported potential energy includes the restraint**, matching what the
  notebooks' `StateDataReporter` reports: `fullPotential − metropolisEnergy`
  equals the restraint term exactly.
- **The restraint reference is not rescaled by the barostat** and the
  displacement uses the current box's minimum image — which is what OpenMM does
  and what the protocol depends on.
- **The yield design is right for a background tab**: a `MessageChannel` task,
  with the `setTimeout` branch only for non-browser hosts, and no
  `requestAnimationFrame` or `setTimeout` anywhere in the step or report loop.
- **Stop itself lands promptly** — within one rebuild interval. The problem is
  what happens after, not the latency.
- **Neighbour overflow recovery works where it is reachable** (at initialise),
  and `release()` drops the pool, so steady-state allocation over a long run
  looks sound: peak 4.9 MiB after a blown-up run.

**Hygiene**

- **No debug logging in shipping code** — **zero** `console.*` across `web/`,
  `shared/`, `cpu/`, `webgpu/`. All 100+ live in `tools/`, where they are the
  deliberate report output. Unusually clean.
- **No TODO/FIXME/XXX/HACK anywhere in first-party code** — zero over all 125
  tracked files. The only 9 hits are inside the minified vendor bundle.
- **No secrets, ever** — no match for any common key, token or PEM pattern
  across all tracked files; zero email addresses; every outbound host is a
  public citation. `git log --all --diff-filter=D` deletes exactly three files
  across all 10 commits, all from the 3Dmol→Mol\* swap.
- **`shared/traj/logmd.js` is correct about credentials** — `user_id: "public"`
  and `token: null` hardcoded; no account path exists in the code.
- **No personal or machine-specific paths** — **zero** matches for `/Users/`,
  `/opt/`, `/usr/local/` over all tracked files. `make_fixtures.sh` needs only
  `tleap` on `PATH` and builds in `mktemp -d` with a cleanup trap; the README
  uses a placeholder. The regeneration instructions are portable as written.
- **The AGENTS.md browser-loadability invariant holds** for `shared/` and
  `cpu/`: every specifier relative and ending in `.js`, zero bare specifiers,
  zero `node:` imports, no build step.
- **`window.rainmd` is a feature, not a risk** — a facade of read-only getters,
  no setters, nothing a visitor could not already reach from devtools on a
  static page holding their own file. Load-bearing for the browser lane.

**And one coverage hole with no defect behind it.** Every GPU kernel does have
its own differential check, as AGENTS.md requires. The **runtime layer has
none** — `pipeline-cache.js`, `allocator.js`, `device.js`, `execution.js` and
`yield.js` are matched by nothing in `test/`. That matters most for the pipeline
cache, since `ARCHITECTURE.md` records that LocalFold shipped two key collisions
and the consequence here is "a plausible wrong force field, essentially
undebuggable". Probed on the host — it needs no GPU, only `device.features` and
`createComputePipeline` — and all eight arms behave correctly, including naming
the differing line exactly:

```
THREW   same key DIFFERENT source
        pipeline cache key collision for k2 - line 3 of 4:
        "let grid = 24u;" against "let grid = 32u;"
```

So this is a correct mechanism with nothing holding it there. A regression in
the file that exists to catch silent wrong answers would itself be silent, and
the fix is eight assertions and no GPU.

---

# The shape of the gap, if only one thing is taken from this

Every one of blockers 1, 2, 3, 7, 8, 9, 14 and 15 is invisible to the current
suite for the **same** reason: the fixtures are all rectangular, all ff14SB, all
3-point water, all unrestrained-or-NVT, all short, and **all at 2 fs**. The
oracles are right and the discipline is real; they are pointed at one corner of
the input space.

Five fixtures and one axis would close most of it:

| add | would have caught |
|---|---|
| `solvateOct` + OpenMM per-term | finding 1 |
| ff19SB + OpenMM per-term | finding 2 |
| OPC or TIP4P-Ew (refusal test) | finding 3 |
| a restrained NPT arm in `check-barostat` | blockers 7, 15 and finding N63c |
| **the same NVT/NVE arms at 0.5, 1 and 4 fs** | **blocker 14, finding S18b** |
| **any arm longer than 10 ps** | blocker 15, finding S18c |

And two assertions on output that already exists: a finiteness check per block
(finding 5) and the live box passed to `appendFrame` (finding 8).

🔴 **THE SHARPEST SINGLE LESSON IS ABOUT THE ORACLES THEMSELVES.** AGENTS.md
requires "two independent oracles: OpenMM for the energy, and a central
difference of its own energy for the force", and treats that pair as sufficient.
It is not, in two distinct ways this review measured:

- **The self-gradient oracle cannot see an error in a position-independent
  factor.** The PME wave vector (finding 1) enters only the kernel weights, so
  the analytic force is the exact gradient of the wrong energy. Verified: the
  central-difference check passes on an octahedral box at relative error
  `0.00e+0`. No gradient check at any box shape can ever catch that class.
- **Neither oracle is a dynamical oracle.** Both are single-point. Findings 14,
  15 and 18c are properties of a *trajectory*, and the only gates on trajectories
  are `|T − 298| < 25` over 400 steps and `|drift| < 5` over 5000 — bars loose
  enough to pass a 13 K offset and a run that reaches 447 K.

A third oracle class is missing and is cheap: **an independent energy reference
on a triclinic cell** (an exact Ewald lattice sum already exists in the repo's
own history as `ala-wat-ewald.json`, orphaned per finding S32), and **a
long-trajectory ensemble check** against OpenMM at more than one timestep.

---

# Review artefacts: what was promoted and what was not

The probes behind the measurements above were **not committed**. Five of them
became real gates and are; the rest were scratch and are deliberately left out
of the repository, because a probe that nothing runs rots into a claim nobody
checks — which is the failure mode half of this review is about.

**Promoted into gates, and committed:**

| gate | command | the finding it holds |
|---|---|---|
| `tools/gpu/check-blowup.mjs` | `npm run test:gpu:blowup` | blocker 5, the finiteness check |
| `tools/gpu/check-capacity.mjs` | `npm run test:gpu:capacity` | blocker 9, neighbour capacity under a barostat |
| `tools/gpu/check-timestep.mjs` | `npm run check:timestep` | 14a, the drift's timestep scaling |
| `tools/check-triclinic.mjs` | `npm run check:triclinic` | blockers 1 and 66, on a sheared cell |
| `tools/oracle/check_npt_box.py` | `npm run check:npt-box` | blocker 8, the NPT unit cell |

plus new arms inside existing checks: the restrained Metropolis arm in
`check-barostat.mjs` (blocker 7), the block-size invariance arm in
`check-displacement.mjs` (blocker 10), the 20 ps window fits in
`check-integrator.mjs` (blocker 12), and host arms in `test/physics.test.js`
and `test/amber-parse.test.js` (blockers 2, 3, 4, 11, 13 and the writers).

**Not committed**, and listed so nobody looks for them: the eighteen
`tools/gpu/review-*.mjs` probes from the six review passes, the host scripts
under `/tmp/rainmd-review/`, and the eight tleap systems built there (ff19SB in
four variants, OPC, TIP4P-Ew, GLYCAM, a protein+glycan complex, a ParmEd 12-6-4
file and a 52,200-atom box). The ff19SB and 4-point-water systems are what the
refusals in `shared/amber/flags.js` were verified against; rebuilding them takes
one `tleap` run each and the recipes are in that file's header.

---

# Suggested order

1. **Finding 5** — the finiteness check. Cheapest fix, largest user cost, and
   the information already comes back every block.
2. **Finding 14** (with **S18b**) — the timestep-dependent heating. This is the
   one that affects *every* run, and it is the only finding whose mechanism is
   not fully pinned down, so it needs investigation before it needs a patch.
   Start by narrowing the timestep field to the notebooks' 0.5-4 fs range and
   adding NVT arms at 0.5, 1 and 4 fs — the gate is a one-line change and it
   fails today.
3. **Findings 1, 2, 3** — the three silent-wrong-science cases on the notebooks'
   own defaults, each with its fixture from the table above. Finding 1 is three
   lines of code in two files plus one fixture.
4. **Finding 8** — pass the live box to `appendFrame`. One argument.
5. **Findings 6, 11** — the two that cost a user their work or mislead them
   about the central promise.
6. **Findings 7, 9, 10, 15** — the four that need a new arm in an existing check
   more than they need new code. Finding 15 in particular is satisfied by
   quoting the window beside the number.
7. **Finding 13** — before promoting, because the README and LICENSE both assert
   something untrue about redistributed AMBER parameters.
8. **Finding 4** — the flag whitelist, which closes the class rather than the
   three instances.
9. **Finding S18a** — a seed field. Small, and without it no error bar the page
   produces means anything.
10. The rest of should-fix, then the documentation sweep (S28-S45, N57-N59, N63b) as
    one pass, then nice-to-have.

One thing worth deciding early rather than late: **AGENTS.md's two-oracle
invariant should be amended**, per the lesson above. It is the document every
future change is measured against, and as written it certifies a pair of oracles
that provably cannot see two of this review's blockers.
