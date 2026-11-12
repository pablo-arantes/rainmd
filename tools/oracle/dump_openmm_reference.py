#!/usr/bin/env python3
"""Dump OpenMM's own energies and forces for an AMBER system, as JSON.

This is the oracle. Every force kernel in this repository is checked against
the numbers this script produces, because OpenMM is what the Making-it-rain
notebooks run and "agrees with the notebook" is the only correctness claim
worth making.

🔴 IT BUILDS THE SYSTEM THE NOTEBOOK BUILDS, NOT A CONVENIENT ONE. PME, a
1.0 nm cutoff, HBonds constraints, rigid water, ewaldErrorTolerance 5e-4 - the
same arguments Protein_ligand.ipynb and Amber.ipynb pass to createSystem. A
reference taken with different options is a reference for a different physics,
and the agreement it reports is worthless.

🔴 AND IT RECORDS THE PME PARAMETERS OPENMM CHOSE. alpha and the grid are
derived from the cutoff and the error tolerance by a rule with rounding in it,
and two codes that pick different grids disagree on the reciprocal energy by far
more than either is wrong. Comparing against this dump without reading
`pme.alpha` and `pme.grid` back out of it compares two different Ewald splits
and blames the kernel.

🔴 AND THE DECOMPOSITION IS BY ZEROING, NOT BY FORCE GROUP. OpenMM puts all of
LJ, direct-space Coulomb, the reciprocal sum, the exclusion corrections and the
self-energy inside one NonbondedForce, and no force group can separate them. So
the split is made by building the same system three times - whole, with every
charge zeroed, and with every epsilon zeroed - which gives LJ and Coulomb
exactly rather than approximately, at the cost of three contexts.

Usage:
    python dump_openmm_reference.py --prmtop S.prmtop --inpcrd S.crd --out ref.json
    python dump_openmm_reference.py ... --nonbonded NoCutoff   # vacuum systems
    python dump_openmm_reference.py ... --nonbonded Ewald      # exact lattice sum

The Ewald arm is how PME itself is validated: exact Ewald and PME on the same
system differ only by the grid approximation, so the PME kernel is held to the
Ewald dump at the tolerance the grid justifies.
"""
import argparse
import json
import sys

import openmm
import openmm.app as app
import openmm.unit as unit

METHODS = {
    "PME": app.PME,
    "Ewald": app.Ewald,
    "NoCutoff": app.NoCutoff,
    "CutoffPeriodic": app.CutoffPeriodic,
    "CutoffNonPeriodic": app.CutoffNonPeriodic,
}

CONSTRAINTS = {"None": None, "HBonds": app.HBonds, "AllBonds": app.AllBonds,
               "HAngles": app.HAngles}


def build(prmtop, method, cutoff_nm, constraints, rigid_water, ewald_tol):
    kwargs = dict(
        nonbondedMethod=METHODS[method],
        constraints=CONSTRAINTS[constraints],
        rigidWater=rigid_water,
    )
    if method != "NoCutoff":
        kwargs["nonbondedCutoff"] = cutoff_nm * unit.nanometer
        kwargs["ewaldErrorTolerance"] = ewald_tol
    return prmtop.createSystem(**kwargs)


def add_restraints(system, positions, force_constant, prmtop):
    """The notebooks' own restraint, built the notebooks' own way.

    🔴 COPIED FROM THE NOTEBOOK RATHER THAN REIMPLEMENTED, because the point of
    an oracle is to be independent of the thing it checks. `k*periodicdistance^2`
    with no half, per-particle k/x0/y0/z0, and the selection pytraj writes as
    `!(:H*) & !(:WAT) & !(:Na+) & !(:Cl-) & !(:Mg+) & !(:K+)` - which is done
    here through OpenMM's own topology so this script needs no pytraj.

    Returns the restrained atom indices so the caller can report them.
    """
    restrained = []
    solvent = {"WAT", "HOH", "H2O", "SOL", "TIP", "TP3", "SPC", "T3P",
               "NA", "NA+", "CL", "CL-", "K", "K+", "MG", "MG2", "CA2", "ZN",
               "LI", "LI+", "RB", "RB+", "CS", "CS+", "BR", "BR-", "IB", "CIO"}
    for atom in prmtop.topology.atoms():
        name = atom.residue.name.strip().upper()
        if name in solvent:
            continue
        if atom.element is not None and atom.element.symbol == "H":
            continue
        restrained.append(atom.index)
    if force_constant <= 0 or not restrained:
        return []
    restraint = openmm.CustomExternalForce(
        "k*periodicdistance(x, y, z, x0, y0, z0)^2;")
    restraint.addPerParticleParameter("k")
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")
    for index in restrained:
        here = positions[index].value_in_unit(unit.nanometer)
        restraint.addParticle(index, [force_constant, here[0], here[1], here[2]])
    system.addForce(restraint)
    return restrained


def nonbonded_of(system):
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            return force
    return None


def single_point(system, positions, box_vectors):
    """Energy and forces on the Reference platform, in double precision.

    🔴 THE REFERENCE PLATFORM, DELIBERATELY. CUDA and OpenCL accumulate forces
    in fixed point and reorder the sums, so two runs of the same system on them
    differ in the last few digits - which is fine for a simulation and useless
    for a tolerance. The Reference platform is the one that gives the same
    answer twice.
    """
    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(system, integrator, platform)
    if box_vectors is not None:
        context.setPeriodicBoxVectors(*box_vectors)
    context.setPositions(positions)
    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    forces = state.getForces(asNumpy=True).value_in_unit(
        unit.kilojoule_per_mole / unit.nanometer)
    groups = {}
    for index, force in enumerate(system.getForces()):
        force.setForceGroup(index)
    # Force groups must be set before the context exists, so rebuild once the
    # groups are assigned. Cheap at fixture size and unambiguous.
    del context, integrator
    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)
    context = openmm.Context(system, integrator, platform)
    if box_vectors is not None:
        context.setPeriodicBoxVectors(*box_vectors)
    context.setPositions(positions)
    for index, force in enumerate(system.getForces()):
        sub = context.getState(getEnergy=True, getForces=True, groups={index})
        groups[f"{index}:{type(force).__name__}"] = {
            "energy": sub.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
            "forceSum": float(abs(sub.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer)).sum()),
        }
    # 🔴 getPMEParametersInContext IS PME/LJPME ONLY. Asking it about an Ewald
    # context raises "This Context is not using PME or LJPME", which killed the
    # whole Ewald arm - the one the PME kernel is validated against. Ewald
    # derives alpha from the same error tolerance by the same formula, so the
    # tolerance is recorded instead and shared/pme/parameters.js reproduces it.
    pme = None
    force = nonbonded_of(system)
    if force is not None and force.getNonbondedMethod() == openmm.NonbondedForce.PME:
        alpha, nx, ny, nz = force.getPMEParametersInContext(context)
        pme = {
            "alpha": alpha.value_in_unit(unit.nanometer ** -1)
            if unit.is_quantity(alpha) else float(alpha),
            "grid": [int(nx), int(ny), int(nz)],
        }
    elif force is not None and force.getNonbondedMethod() == openmm.NonbondedForce.Ewald:
        pme = {"alpha": None, "grid": None,
               "ewaldErrorTolerance": force.getEwaldErrorTolerance()}
    return energy, forces, groups, pme


def zero_charges(system):
    force = nonbonded_of(system)
    for index in range(force.getNumParticles()):
        charge, sigma, epsilon = force.getParticleParameters(index)
        force.setParticleParameters(index, 0.0, sigma, epsilon)
    for index in range(force.getNumExceptions()):
        p1, p2, qq, sigma, epsilon = force.getExceptionParameters(index)
        force.setExceptionParameters(index, p1, p2, 0.0, sigma, epsilon)


def zero_lj(system):
    force = nonbonded_of(system)
    for index in range(force.getNumParticles()):
        charge, sigma, epsilon = force.getParticleParameters(index)
        force.setParticleParameters(index, charge, sigma, 0.0)
    for index in range(force.getNumExceptions()):
        p1, p2, qq, sigma, epsilon = force.getExceptionParameters(index)
        force.setExceptionParameters(index, p1, p2, qq, sigma, 0.0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prmtop", required=True)
    parser.add_argument("--inpcrd", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--nonbonded", default="PME", choices=sorted(METHODS))
    parser.add_argument("--cutoff", type=float, default=1.0, help="nm")
    parser.add_argument("--constraints", default="HBonds", choices=sorted(CONSTRAINTS))
    parser.add_argument("--no-rigid-water", action="store_true")
    parser.add_argument("--ewald-tolerance", type=float, default=0.0005)
    parser.add_argument("--volume-scale", type=float, default=None,
                        help="scale molecular centres of mass and the box by "
                             "this length factor before evaluating, which is "
                             "what a Monte Carlo barostat move does")
    parser.add_argument("--no-dispersion-correction", action="store_true",
                        help="switch off NonbondedForce's analytical LJ tail term")
    parser.add_argument("--pme-alpha", type=float, default=None,
                        help="force the Ewald splitting parameter, 1/nm")
    parser.add_argument("--pme-grid", default=None,
                        help="force the reciprocal grid, e.g. 24,32,24")
    parser.add_argument("--displace", type=float, default=None,
                        help="nm; shift every atom by a deterministic amount "
                             "AFTER capturing the restraint reference, so the "
                             "restraint has a non-zero displacement to score. "
                             "The displaced positions are written into the dump, "
                             "because a restraint evaluated at its own reference "
                             "is exactly zero and proves nothing.")
    parser.add_argument("--restraint-force", type=float, default=None,
                        help="k in kJ/mol/nm^2 for the notebooks' positional "
                             "restraint on solute heavy atoms; omitted means none")
    parser.add_argument("--full-forces", action="store_true",
                        help="write every per-atom force component, not just checksums")
    args = parser.parse_args(argv)

    prmtop = app.AmberPrmtopFile(args.prmtop)
    inpcrd = app.AmberInpcrdFile(args.inpcrd)
    rigid_water = not args.no_rigid_water
    box = inpcrd.boxVectors if args.nonbonded != "NoCutoff" else None
    positions = inpcrd.positions

    if args.volume_scale is not None:
        # 🔴 MOLECULAR CENTRES OF MASS, NOT ATOMS - the same move the barostat
        # makes. Scaling atoms individually would stretch every bond, and the
        # energy difference would be dominated by that strain rather than by
        # the change in intermolecular distances this is meant to measure.
        #
        # 🔴 AND THE MOLECULE PARTITION COMES FROM THE BOND GRAPH, by the same
        # union-find the port uses. ATOMS_PER_MOLECULE assumes a molecule's
        # atoms are contiguous, which is true of what tleap writes and is not a
        # property of the format - so deriving it both sides guarantees the two
        # scale the same groups.
        import numpy as np
        system_for_bonds = prmtop.topology
        n = system_for_bonds.getNumAtoms()
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for bond in system_for_bonds.bonds():
            a, b = find(bond[0].index), find(bond[1].index)
            if a != b:
                parent[b] = a
        groups = {}
        for atom in range(n):
            groups.setdefault(find(atom), []).append(atom)
        # 🔴 THE TOPOLOGY'S ELEMENT MASSES ARE NOT THE FORCE FIELD'S. AMBER
        # writes its own MASS section - 1.008 for hydrogen where the periodic
        # table says 1.00794 - and a centroid weighted by the wrong ones is
        # slightly displaced, which shows up as a disagreement in the scaled
        # energy that has nothing to do with the kernels.
        import parmed as pmd
        amu = np.array([atom.mass for atom in pmd.load_file(args.prmtop).atoms])
        xyz = np.array(positions.value_in_unit(unit.nanometer))
        scale = args.volume_scale
        for members in groups.values():
            m = amu[members]
            total = m.sum()
            centre = ((xyz[members] * m[:, None]).sum(axis=0) / total) if total > 0 \
                else xyz[members].mean(axis=0)
            xyz[members] += (scale - 1.0) * centre
        positions = xyz * unit.nanometer
        if box is not None:
            box = [[component * scale for component in row] for row in
                   np.array(box.value_in_unit(unit.nanometer))] * unit.nanometer
        print(f"  scaled molecular centroids and the box by {scale}")

    # 🔴 THE GRID MUST BE FORCED OR IT IS A FREE VARIABLE IN EVERY COMPARISON.
    # OpenMM's Reference platform picked [22, 24, 21] for this box, and 22 has a
    # factor of 11 - a size a mixed-radix 2/3/5/7 FFT cannot do at all. Left to
    # each code's own rule, this port and OpenMM would split the Ewald sum
    # differently and the difference would be read as a kernel defect. Forcing
    # alpha and the grid on BOTH sides makes the remaining difference the only
    # thing under test.
    def force_pme(system):
        # 🔴 THE LJ TAIL CORRECTION IS ON BY DEFAULT AND IS NOT A PAIR SUM.
        # NonbondedForce.useDispersionCorrection defaults to true for every
        # periodic method, so an OpenMM PME energy carries an analytical
        # estimate of the dispersion beyond the cutoff - a single volume-
        # dependent constant that no pair loop will ever reproduce. Left on, it
        # appears as a fixed offset in the LJ energy and reads as a broken
        # kernel. The differential tests switch it off, and a separate arm with
        # it on is what the tail term itself is checked against.
        force = nonbonded_of(system)
        if force is not None and args.no_dispersion_correction:
            force.setUseDispersionCorrection(False)
        if args.pme_alpha is None and args.pme_grid is None:
            return
        if force is None:
            return
        # 🔴 OPENMM TREATS alpha == 0 AS "CHOOSE EVERYTHING AUTOMATICALLY", SO A
        # GRID WITHOUT AN ALPHA IS SILENTLY IGNORED. Asking for --pme-grid
        # 24,24,24 alone produced a dump whose header still read
        # grid=[23, 22, 19] - the auto choice - and nothing said so. Refuse it
        # rather than hand back a reference computed on a different grid from
        # the one the caller asked for.
        if args.pme_grid is not None and args.pme_alpha is None:
            raise SystemExit(
                "--pme-grid needs --pme-alpha as well: OpenMM reads alpha == 0 "
                "as 'choose alpha AND the grid automatically', so the grid you "
                "asked for would be silently discarded. Pass the alpha the "
                "automatic choice gives (it is printed in the dump header).")
        alpha = args.pme_alpha if args.pme_alpha is not None else 0.0
        if args.pme_grid is None:
            grid = (0, 0, 0)
        else:
            parts = [int(value) for value in args.pme_grid.split(",")]
            if len(parts) != 3:
                raise SystemExit("--pme-grid wants three comma-separated integers")
            grid = tuple(parts)
        force.setPMEParameters(alpha, *grid)

    # 🔴 THE REFERENCE IS THE UNDISPLACED STRUCTURE, AND THE DISPLACEMENT COMES
    # AFTER. A restraint evaluated at its own reference is exactly zero, so the
    # first version of this dumped a restrained system whose restraint energy
    # was 0.000000 and whose total matched the unrestrained run to every digit -
    # a fixture that proved the restraint was wired to nothing.
    reference_positions = positions
    restrained = []
    if args.displace:
        import math
        shifted = []
        for index, point in enumerate(positions.value_in_unit(unit.nanometer)):
            shifted.append(openmm.Vec3(
                point[0] + args.displace * math.sin(index * 0.7),
                point[1] + args.displace * math.cos(index * 1.3),
                point[2] + args.displace * math.sin(index * 2.1),
            ) * unit.nanometer)
        positions = unit.Quantity(shifted)

    def run(mutate=None):
        nonlocal restrained
        system = build(prmtop, args.nonbonded, args.cutoff, args.constraints,
                       rigid_water, args.ewald_tolerance)
        force_pme(system)
        # Added inside run(), because run() builds a fresh system per arm and a
        # force added to the outer one would be evaluated by nothing.
        restrained = add_restraints(system, reference_positions,
                                    args.restraint_force or 0.0, prmtop)
        if mutate is not None:
            mutate(system)
        return single_point(system, positions, box)

    total_energy, total_forces, groups, pme = run()
    lj_energy, lj_forces, lj_groups, _ = run(zero_charges)
    coul_energy, coul_forces, coul_groups, _ = run(zero_lj)

    # 🔴 A ZEROED ARM STILL CARRIES THE BONDED ENERGY, AND READING ITS TOTAL AS
    # "the LJ energy" IS A 42 kJ/mol MISTAKE. Zeroing every charge removes
    # Coulomb and leaves bonds, angles and torsions exactly where they were, so
    # `ljOnly` as a system total is bonded + LJ. That read as a broken LJ kernel
    # on a port whose LJ was exact to 1e-16, and the tell was that the error
    # equalled the bonded sum to seven digits. The NonbondedForce force group
    # alone is the unambiguous quantity, so it is what gets written out.
    def nonbonded_group_energy(table):
        for key, value in table.items():
            if key.endswith(":NonbondedForce"):
                return value["energy"]
        return None

    lj_only = nonbonded_group_energy(lj_groups)
    coulomb_only = nonbonded_group_energy(coul_groups)

    # The constraint and exception tables, so the port can check it built the
    # same ones rather than inferring them from a matching total.
    system = build(prmtop, args.nonbonded, args.cutoff, args.constraints,
                   rigid_water, args.ewald_tolerance)
    force_pme(system)
    # 🔴 ADDED AFTER force_pme AND BEFORE ANY CONTEXT EXISTS. A force added to a
    # system that already has a context is silently ignored by that context, so
    # the restraint would contribute nothing and the comparison would report
    # this port's restraint as a pure error.
    constraints = []
    for index in range(system.getNumConstraints()):
        p1, p2, distance = system.getConstraintParameters(index)
        constraints.append([int(p1), int(p2),
                            distance.value_in_unit(unit.nanometer)])
    force = nonbonded_of(system)
    exceptions = []
    if force is not None:
        for index in range(force.getNumExceptions()):
            p1, p2, qq, sigma, epsilon = force.getExceptionParameters(index)
            exceptions.append([
                int(p1), int(p2),
                qq.value_in_unit(unit.elementary_charge ** 2),
                sigma.value_in_unit(unit.nanometer),
                epsilon.value_in_unit(unit.kilojoule_per_mole),
            ])

    payload = {
        "producer": {
            "openmm": openmm.version.version,
            "platform": "Reference",
            "script": "tools/oracle/dump_openmm_reference.py",
        },
        "inputs": {
            "prmtop": args.prmtop,
            "inpcrd": args.inpcrd,
            "nonbondedMethod": args.nonbonded,
            "cutoff": args.cutoff,
            "constraints": args.constraints,
            "rigidWater": rigid_water,
            "ewaldErrorTolerance": args.ewald_tolerance,
            "forcedPmeAlpha": args.pme_alpha,
            "forcedPmeGrid": args.pme_grid,
            "volumeScale": args.volume_scale,
            "dispersionCorrection": not args.no_dispersion_correction,
        },
        "atomCount": system.getNumParticles(),
        "pme": pme,
        "energy": {
            "total": total_energy,
            "ljOnly": lj_only,
            "coulombOnly": coulomb_only,
            "ljOnlySystemTotal": lj_energy,
            "coulombOnlySystemTotal": coul_energy,
            "byForceGroup": groups,
        },
        "forceChecksum": {
            "totalAbsSum": float(abs(total_forces).sum()),
            "ljAbsSum": float(abs(lj_forces).sum()),
            "coulombAbsSum": float(abs(coul_forces).sum()),
            "maxComponent": float(abs(total_forces).max()),
        },
        "constraintCount": len(constraints),
        "constraints": constraints,
        "exceptionCount": len(exceptions),
        "exceptions": exceptions,
    }
    # 🔴 THE RESTRAINED ATOM LIST, NOT JUST THE ENERGY. The selection is where
    # this can go wrong without the number looking wrong: restrain the whole
    # system instead of the solute and the energy is still a plausible
    # magnitude. Dumping the indices makes the two selections comparable.
    payload["restraint"] = {
        "force": args.restraint_force or 0.0,
        "atomCount": len(restrained),
        "atoms": [int(index) for index in restrained],
        "displace": args.displace or 0.0,
    }
    if args.displace:
        # The evaluated coordinates, so the port scores the same configuration
        # rather than reproducing the displacement and hoping it matches.
        payload["positions"] = [
            float(value) for point in positions.value_in_unit(unit.nanometer)
            for value in point
        ]
    if args.full_forces:
        payload["forces"] = {
            "total": total_forces.tolist(),
            "ljOnly": lj_forces.tolist(),
            "coulombOnly": coul_forces.tolist(),
        }

    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=1)
    print(f"{args.out}: {args.nonbonded} total={total_energy:.6f} kJ/mol "
          f"(LJ {lj_only:.6f}, Coulomb {coulomb_only:.6f}), "
          f"{len(constraints)} constraints, {len(exceptions)} exceptions"
          + (f", PME alpha={pme['alpha']:.6f} grid={pme['grid']}"
             if pme and pme.get("alpha") is not None else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
