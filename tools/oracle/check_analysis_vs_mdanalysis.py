"""Hold RainMD's live analysis to MDAnalysis, on a trajectory RainMD wrote.

Run `tools/gpu/check-analysis.mjs` first; it writes the DCD, the CSVs and the
atom selections it used, then this recomputes every metric independently:

    node tools/gpu-node.mjs tools/gpu/check-analysis.mjs
    python3 -I tools/oracle/check_analysis_vs_mdanalysis.py \
        --prefix=/tmp/rainmd-analysis --prmtop=test/fixtures/two-body.prmtop

🔴 THE POINT IS THAT NOTHING IS SHARED BUT THE FILE. RainMD computes these
numbers live, frame by frame, from coordinates that never touch the disk.
MDAnalysis reads the DCD afterwards with its own reader, superposes with its own
QCP implementation, and accumulates RMSF in one pass over stored coordinates.
Agreement is therefore a statement about the algorithms, not about shared code -
and for RMSF specifically it is a statement about the running-sums identity,
since the two accumulate in completely different ways.

🔴 AND THE SELECTIONS COME FROM THE JSON, NOT FROM A SELECTION STRING. Asking
MDAnalysis for "protein and not name H*" would compare two different atom sets
and call the difference a disagreement; the indices RainMD actually fitted on
are written out so the comparison is of arithmetic alone.

🔴 AND UNITS DIFFER BY TEN. MDAnalysis works in angstrom, RainMD in nanometres.
Every MDAnalysis length here is divided by 10 at the point it is read, once.
"""
import argparse
import json
import sys

import numpy as np

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import align, rms
except ImportError:  # pragma: no cover - reported, not raised
    print("MDAnalysis is not installed in this interpreter; skipping", file=sys.stderr)
    raise SystemExit(0)

A_TO_NM = 0.1


def kabsch_rmsd(mobile, reference, fit_index, measure_index):
    """MDAnalysis' own superposition, fitted on one group and measured on another."""
    # `rotation_matrix` returns (R, rmsd) for the atoms it is given.
    mobile_fit = mobile[fit_index]
    reference_fit = reference[fit_index]
    mobile_centre = mobile_fit.mean(axis=0)
    reference_centre = reference_fit.mean(axis=0)
    rotation, _ = align.rotation_matrix(mobile_fit - mobile_centre,
                                        reference_fit - reference_centre)
    moved = (rotation @ (mobile[measure_index] - mobile_centre).T).T + reference_centre
    difference = moved - reference[measure_index]
    return float(np.sqrt((difference ** 2).sum(axis=1).mean()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="/tmp/rainmd-analysis")
    parser.add_argument("--prmtop", required=True)
    # The floor is the DCD format itself: coordinates are stored as float32, so
    # a round trip through the file costs about 1e-7 of a box span. Anything at
    # that level is the file, not the arithmetic.
    parser.add_argument("--tolerance", type=float, default=2e-6)
    arguments = parser.parse_args()

    with open(f"{arguments.prefix}.fit.json", encoding="utf-8") as handle:
        plan = json.load(handle)
    fit = np.asarray(plan["fit"], dtype=int)
    receptor = np.asarray(plan["receptor"], dtype=int)
    binder = np.asarray(plan["binder"], dtype=int)
    residue_atoms = np.asarray(plan["residueAtoms"], dtype=int)

    universe = mda.Universe(arguments.prmtop, f"{arguments.prefix}.dcd")
    frames = np.asarray([universe.atoms.positions.copy() * A_TO_NM
                         for _ in universe.trajectory])
    if len(frames) != plan["frames"]:
        print(f"the DCD holds {len(frames)} frames, the run reported {plan['frames']}")
        return 1
    reference = frames[0]
    masses = universe.atoms.masses

    live = np.genfromtxt(f"{arguments.prefix}.metrics.csv", delimiter=",", names=True)
    if len(live) != len(frames):
        print(f"{len(live)} metric rows against {len(frames)} frames")
        return 1

    worst = {}
    for index, frame in enumerate(frames):
        mine = {
            "rmsd_nm": kabsch_rmsd(frame, reference, fit, fit),
            "rmsd_receptor_nm": kabsch_rmsd(frame, reference, receptor, receptor),
            "rmsd_binder_nm": kabsch_rmsd(frame, reference, binder, binder),
            # The pose-stability number: fitted on the receptor, measured on the
            # binder. This is the one a whole-complex RMSD cannot see.
            "rmsd_binder_on_receptor_nm": kabsch_rmsd(frame, reference, receptor, binder),
        }
        # Radius of gyration over the fit atoms, mass weighted.
        weights = masses[fit]
        centre = (frame[fit] * weights[:, None]).sum(axis=0) / weights.sum()
        offsets = frame[fit] - centre
        mine["rg_nm"] = float(np.sqrt(
            (weights * (offsets ** 2).sum(axis=1)).sum() / weights.sum()))
        mine["interface_distance_nm"] = float(np.linalg.norm(
            frame[receptor].mean(axis=0) - frame[binder].mean(axis=0)))
        for key, value in mine.items():
            error = abs(value - float(live[key][index]))
            if error > worst.get(key, (0.0, -1))[0]:
                worst[key] = (error, index)

    # RMSF, in one pass over every stored frame, superposed the same way.
    aligned = np.empty((len(frames), len(residue_atoms), 3))
    for index, frame in enumerate(frames):
        centre = frame[fit].mean(axis=0)
        reference_centre = reference[fit].mean(axis=0)
        rotation, _ = align.rotation_matrix(frame[fit] - centre,
                                            reference[fit] - reference_centre)
        aligned[index] = (rotation @ (frame[residue_atoms] - centre).T).T + reference_centre
    mean = aligned.mean(axis=0)
    reference_rmsf = np.sqrt(((aligned - mean) ** 2).sum(axis=2).mean(axis=0))
    live_rmsf = np.genfromtxt(f"{arguments.prefix}.rmsf.csv", delimiter=",",
                              names=True, dtype=None, encoding="utf-8")["rmsf_nm"]
    worst["rmsf_nm"] = (float(np.abs(reference_rmsf - live_rmsf).max()),
                        int(np.abs(reference_rmsf - live_rmsf).argmax()))

    failures = 0
    width = max(len(key) for key in worst)
    print(f"{len(frames)} frames, {len(fit)} fit atoms, "
          f"{len(receptor)} receptor / {len(binder)} binder, "
          f"{len(residue_atoms)} residues")
    print(f"MDAnalysis {mda.__version__}, tolerance {arguments.tolerance:.1e} nm\n")
    for key in ["rmsd_nm", "rmsd_receptor_nm", "rmsd_binder_nm",
                "rmsd_binder_on_receptor_nm", "rg_nm", "interface_distance_nm",
                "rmsf_nm"]:
        error, where = worst[key]
        ok = error <= arguments.tolerance
        failures += 0 if ok else 1
        print(f"{'ok  ' if ok else 'FAIL'} {key:<{width}}  worst "
              f"{error:.3e} nm at index {where}")
    print()
    if failures:
        print(f"{failures} metric(s) disagree with MDAnalysis")
        return 1
    print("every live metric agrees with MDAnalysis within the DCD's own precision")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
