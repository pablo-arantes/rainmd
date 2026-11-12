"""Does the DCD this port writes carry the box the simulation actually had?

🔴 EVERY NPT TRAJECTORY IT EVER WROTE CARRIED THE STAGE'S STARTING CELL.
`web/app.js` passed `system.boxVectors` - fixed at `assembleSystem()` - while
the barostat mutates the simulation's own copy, so no volume move ever reached
the writer. Measured over 800 NPT steps whose live volume walked
20.4979 -> 19.2124 nm3, MDAnalysis read all 8 frames as 20.5203 nm3. The
documented equilibration takes 20.52 -> 14.39 nm3, so the file was out by 43%
in cell volume on every frame.

🔴 AND NO EXISTING GATE COULD SEE IT, FOR ONE REASON: `check:mdanalysis` drives
`tools/gpu/check-analysis.mjs`, which builds its simulation with no `pressure`.
Under NVT the box never moves, so ignoring the live box produces a correct file
by accident. A constant box cannot test a varying one.

The reader here is MDAnalysis, which shares no code with the writer. Run it
through `npm run check:npt-box`, which writes the trajectory first.

    python3 -I tools/oracle/check_npt_box.py --prefix=/tmp/rainmd-npt \\
        --prmtop=test/fixtures/ala-wat.prmtop
"""

import argparse
import json
import math
import sys

import MDAnalysis
import numpy


def cell_volume(dimensions):
    """nm^3 from MDAnalysis' [a, b, c, alpha, beta, gamma] in angstrom/degrees."""
    a, b, c, alpha, beta, gamma = (float(v) for v in dimensions)
    ca, cb, cg = (math.cos(math.radians(v)) for v in (alpha, beta, gamma))
    # The general triclinic determinant, so this is right for solvateOct too.
    factor = 1.0 - ca * ca - cb * cb - cg * cg + 2.0 * ca * cb * cg
    return a * b * c * math.sqrt(max(factor, 0.0)) / 1000.0


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--prmtop", required=True)
    args = parser.parse_args(argv)

    with open(f"{args.prefix}-volumes.json", encoding="utf8") as handle:
        recorded = json.load(handle)
    live = [float(v) for v in recorded["volumes"]]

    universe = MDAnalysis.Universe(args.prmtop, f"{args.prefix}.dcd")
    read = [cell_volume(ts.dimensions) for ts in universe.trajectory]

    print(f"frames: {len(read)} in the file, {len(live)} volumes recorded "
          f"(pressure {recorded['pressure']})")
    if len(read) != len(live):
        print(f"  FAILED: the file holds {len(read)} frames and the run "
              f"recorded {len(live)}")
        return 1

    failures = 0

    # 🔴 THE RUN MUST ACTUALLY HAVE MOVED THE BOX, OR THIS PROVES NOTHING. A
    # barostat that never accepted a move would make a writer that ignores the
    # live box look correct, which is the trap the NVT gate fell into.
    spread = (max(live) - min(live)) / max(live)
    print(f"  live volume {min(live):.4f} -> {max(live):.4f} nm3, "
          f"spread {spread * 100:.2f}%")
    if spread < 1e-3:
        print("  FAILED: the live volume barely moved, so a constant cell "
              "record would pass - run longer or raise the pressure")
        failures += 1

    # And the file's cell must track it, frame by frame.
    worst = 0.0
    worst_at = -1
    for index, (want, got) in enumerate(zip(live, read)):
        relative = abs(got - want) / abs(want)
        if relative > worst:
            worst = relative
            worst_at = index
    print(f"  worst per-frame cell-volume error {worst:.3e} at frame {worst_at}")
    for index, (want, got) in enumerate(zip(live, read)):
        print(f"    frame {index:3d}  simulation {want:9.4f} nm3   "
              f"file {got:9.4f} nm3   rel {abs(got - want) / abs(want):.2e}")

    # 🔴 THE BAR IS THE DCD'S OWN PRECISION, NOT A CHOSEN NUMBER. The unit cell
    # is written as six float32 values, so a length carries about 6e-8 relative
    # and a volume - three lengths multiplied - about 2e-7. 1e-6 is a few times
    # that. With the defect present the error is the full drift, 4e-2 upwards.
    bar = 1e-6
    if worst > bar:
        print(f"  FAILED: the file's cell does not match the simulation's "
              f"(worst {worst:.3e}, float32 cell floor about 2e-7)")
        failures += 1

    # A constant cell record is the precise signature of the original defect.
    if len(set(round(v, 6) for v in read)) == 1 and spread >= 1e-3:
        print("  FAILED: every frame carries the SAME cell while the "
              "simulation's volume changed - this is the original defect")
        failures += 1

    if failures == 0:
        print("\nthe trajectory's unit cell is the one the simulation had, "
              "frame by frame, as MDAnalysis reads it")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
