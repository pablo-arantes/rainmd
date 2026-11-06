/**
 * A parsed topology plus run options -> the arrays a simulation actually runs.
 *
 * This is the step OpenMM calls `createSystem`, and it must make the SAME
 * decisions, because the oracle is an OpenMM system built with the notebook's
 * arguments. Three of those decisions change the energy:
 *
 *   1. A CONSTRAINED BOND IS REMOVED FROM THE HARMONIC TERM. OpenMM does not
 *      constrain a bond and also keep its spring - that would double-count the
 *      interaction and fight the constraint solver. A port that constrains and
 *      keeps gets a bonded energy too high by the sum of every constrained
 *      bond's residual, which at 1176 constraints is not subtle.
 *   2. EVERY EXCLUDED PAIR BECOMES AN EXCEPTION, not a deletion. Under PME the
 *      reciprocal sum is over all atoms and cannot skip a pair, so an excluded
 *      pair needs its reciprocal contribution subtracted back off in real
 *      space. "Excluded" means "handled separately", never "ignored".
 *   3. A 1-4 PAIR IS AN EXCEPTION WITH SCALED PARAMETERS, and the scale factors
 *      come from the torsion that produced it, not from a global constant.
 *      ff19SB and GAFF2 do not share them.
 */
import { ONE_4PI_EPS0 } from "../units.js";
import { boxVectorsFromLengthsAndAngles } from "../amber/inpcrd.js";

/** Residue names that `rigidWater` applies to. */
const WATER_RESIDUES = new Set(["WAT", "HOH", "H2O", "SOL", "TIP", "TP3", "SPC"]);

/**
 * @param {object} topology from parsePrmtop
 * @param {object} [options]
 * @param {"none"|"HBonds"|"AllBonds"} [options.constraints]
 * @param {boolean} [options.rigidWater]
 * @param {number} [options.cutoff] nm
 * @param {{lengths: number[], angles: number[]}} [options.box] from the inpcrd,
 *   which overrides the topology's own - the coordinate file is written later
 *   and a restart's box is the current one.
 */
export function buildSystem(topology, options = {}) {
  const constraintMode = options.constraints ?? "HBonds";
  if (!["none", "HBonds", "AllBonds"].includes(constraintMode)) {
    throw new Error(
      `unsupported constraints mode ${JSON.stringify(constraintMode)}. `
      + "This port implements none, HBonds and AllBonds; HAngles additionally "
      + "constrains angles and needs a cluster solver this engine does not have. "
      + "The notebooks use HBonds.",
    );
  }
  const rigidWater = options.rigidWater ?? true;
  const cutoff = options.cutoff ?? 1.0;

  const boxSource = options.box ?? topology.boxFromTopology;
  const boxVectors = boxSource === null || boxSource === undefined
    ? null : boxVectorsFromLengthsAndAngles(boxSource);
  if (topology.isPeriodic && boxVectors === null) {
    throw new Error(
      "topology declares a periodic box (IFBOX > 0) but no box vectors were "
      + "supplied by either the topology or the coordinate file",
    );
  }
  if (boxVectors !== null) {
    // 🔴 A CUTOFF OVER HALF THE SMALLEST BOX WIDTH BREAKS THE MINIMUM IMAGE
    // CONVENTION, and nothing in WebGPU will say so: each pair quietly
    // interacts with more than one image of its partner, the pressure comes out
    // wrong, and the structure drifts. tleap's `solvateBox 8.0` on a small
    // solute can genuinely produce a box too small for a 1.0 nm cutoff, so this
    // is a user-facing refusal and not an assertion.
    const widths = [boxVectors[0], boxVectors[4], boxVectors[8]];
    const smallest = Math.min(...widths.map(Math.abs));
    if (cutoff > 0.5 * smallest) {
      throw new Error(
        `a ${cutoff} nm cutoff does not fit in a box whose smallest width is `
        + `${smallest.toFixed(4)} nm: the cutoff must be at most half of it `
        + `(${(0.5 * smallest).toFixed(4)} nm). Re-solvate with a larger buffer, `
        + "or lower the cutoff.",
      );
    }
  }

  // --- which bonds are constraints ---------------------------------------
  const isWaterAtom = new Uint8Array(topology.atomCount);
  if (rigidWater) {
    for (let atom = 0; atom < topology.atomCount; atom += 1) {
      const residue = topology.residueOfAtom[atom];
      if (WATER_RESIDUES.has(topology.residueNames[residue])) isWaterAtom[atom] = 1;
    }
  }
  const constrained = new Uint8Array(topology.bonds.count);
  for (let index = 0; index < topology.bonds.count; index += 1) {
    const a = topology.bonds.atomA[index];
    const b = topology.bonds.atomB[index];
    const touchesHydrogen = topology.atomicNumbers[a] === 1 || topology.atomicNumbers[b] === 1;
    if (constraintMode === "AllBonds") constrained[index] = 1;
    else if (constraintMode === "HBonds" && touchesHydrogen) constrained[index] = 1;
    // 🔴 RIGID WATER IS INDEPENDENT OF THE CONSTRAINTS MODE. With
    // constraints=none and rigidWater=true, OpenMM still makes water rigid -
    // and AMBER's TIP3P is three bonds (O-H, O-H and an H-H whose length
    // encodes the angle), so constraining "bonds to hydrogen" happens to catch
    // all three. It is the H-H bond that holds the angle: miss it and the water
    // is a floppy two-constraint molecule whose angle oscillates at a period
    // the timestep cannot resolve.
    if (rigidWater && isWaterAtom[a] === 1 && isWaterAtom[b] === 1) constrained[index] = 1;
  }

  const harmonicBonds = { count: 0, atomA: [], atomB: [], k: [], r0: [] };
  const constraints = { count: 0, atomA: [], atomB: [], distance: [] };
  for (let index = 0; index < topology.bonds.count; index += 1) {
    if (constrained[index] === 1) {
      constraints.atomA.push(topology.bonds.atomA[index]);
      constraints.atomB.push(topology.bonds.atomB[index]);
      constraints.distance.push(topology.bonds.r0[index]);
      constraints.count += 1;
    } else {
      harmonicBonds.atomA.push(topology.bonds.atomA[index]);
      harmonicBonds.atomB.push(topology.bonds.atomB[index]);
      harmonicBonds.k.push(topology.bonds.k[index]);
      harmonicBonds.r0.push(topology.bonds.r0[index]);
      harmonicBonds.count += 1;
    }
  }

  // --- exceptions ---------------------------------------------------------
  // The 1-4 pairs, keyed for lookup while walking the exclusion list.
  /** @type {Map<number, number>} */
  const pair14Index = new Map();
  const pairs14 = topology.dihedrals.pairs14;
  for (let index = 0; index < pairs14.count; index += 1) {
    const low = Math.min(pairs14.atomA[index], pairs14.atomB[index]);
    const high = Math.max(pairs14.atomA[index], pairs14.atomB[index]);
    pair14Index.set(low * 0x4000000 + high, index);
  }

  const exceptions = {
    count: 0, atomA: [], atomB: [], chargeProduct: [], ljA: [], ljB: [], isScaled: [],
  };
  const typeCount = topology.typeCount;
  for (let atom = 0; atom < topology.atomCount; atom += 1) {
    const start = topology.exclusions.start[atom];
    const end = topology.exclusions.start[atom + 1];
    for (let cursor = start; cursor < end; cursor += 1) {
      const partner = topology.exclusions.partners[cursor];
      const low = Math.min(atom, partner);
      const high = Math.max(atom, partner);
      const found = pair14Index.get(low * 0x4000000 + high);
      const typePair = typeCount * topology.atomTypes[atom] + topology.atomTypes[partner];
      exceptions.atomA.push(low);
      exceptions.atomB.push(high);
      if (found === undefined) {
        exceptions.chargeProduct.push(0);
        exceptions.ljA.push(0);
        exceptions.ljB.push(0);
        exceptions.isScaled.push(0);
      } else {
        const scee = pairs14.scee[found];
        const scnb = pairs14.scnb[found];
        exceptions.chargeProduct.push(
          topology.charges[atom] * topology.charges[partner] / scee,
        );
        exceptions.ljA.push(topology.ljA[typePair] / scnb);
        exceptions.ljB.push(topology.ljB[typePair] / scnb);
        exceptions.isScaled.push(1);
      }
      exceptions.count += 1;
    }
  }

  // Molecule membership, as index ranges, for the barostat and for imaging.
  const moleculeStart = new Int32Array(topology.moleculeCount + 1);
  const counts = new Int32Array(topology.moleculeCount);
  for (let atom = 0; atom < topology.atomCount; atom += 1) {
    counts[topology.moleculeOfAtom[atom]] += 1;
  }
  for (let molecule = 0; molecule < topology.moleculeCount; molecule += 1) {
    moleculeStart[molecule + 1] = moleculeStart[molecule] + counts[molecule];
  }
  const moleculeAtoms = new Int32Array(topology.atomCount);
  const cursor = moleculeStart.slice(0, topology.moleculeCount);
  for (let atom = 0; atom < topology.atomCount; atom += 1) {
    const molecule = topology.moleculeOfAtom[atom];
    moleculeAtoms[cursor[molecule]] = atom;
    cursor[molecule] += 1;
  }

  // 🔴 A CONSTRAINED ATOM HAS FEWER DEGREES OF FREEDOM, AND THE TEMPERATURE
  // READOUT DEPENDS ON COUNTING THEM. 3N - constraints - 3 (for the removed
  // centre-of-mass motion) is what OpenMM reports a temperature against; using
  // 3N on this solvated fixture would read the temperature 1176/3570 = 33% low
  // and the thermostat would look broken while being correct.
  let massiveAtoms = 0;
  for (let atom = 0; atom < topology.atomCount; atom += 1) {
    if (topology.masses[atom] > 0) massiveAtoms += 1;
  }
  const degreesOfFreedom = 3 * massiveAtoms - constraints.count - 3;

  return {
    topology,
    options: { constraints: constraintMode, rigidWater, cutoff },
    boxVectors,
    isPeriodic: boxVectors !== null,
    coulombConstant: ONE_4PI_EPS0,
    harmonicBonds: {
      count: harmonicBonds.count,
      atomA: Int32Array.from(harmonicBonds.atomA),
      atomB: Int32Array.from(harmonicBonds.atomB),
      k: Float64Array.from(harmonicBonds.k),
      r0: Float64Array.from(harmonicBonds.r0),
    },
    constraints: {
      count: constraints.count,
      atomA: Int32Array.from(constraints.atomA),
      atomB: Int32Array.from(constraints.atomB),
      distance: Float64Array.from(constraints.distance),
    },
    exceptions: {
      count: exceptions.count,
      atomA: Int32Array.from(exceptions.atomA),
      atomB: Int32Array.from(exceptions.atomB),
      chargeProduct: Float64Array.from(exceptions.chargeProduct),
      ljA: Float64Array.from(exceptions.ljA),
      ljB: Float64Array.from(exceptions.ljB),
      isScaled: Uint8Array.from(exceptions.isScaled),
    },
    molecules: { count: topology.moleculeCount, start: moleculeStart, atoms: moleculeAtoms },
    degreesOfFreedom,
  };
}

/**
 * The excluded-pair set as a hash set the force loops can query in O(1).
 *
 * 🔴 A LINEAR SCAN OF THE EXCLUSION LIST PER PAIR IS WHAT MAKES A REFERENCE
 * IMPLEMENTATION TOO SLOW TO USE AS AN ORACLE. The direct-space loop asks "is
 * this pair excluded" once per pair within the cutoff, which is millions of
 * times for the solvated fixture; asking it against a 1262-entry array turns a
 * two-second reference into an hour and the oracle stops being run.
 */
export function buildExclusionSet(system) {
  const set = new Set();
  const { atomA, atomB, count } = system.exceptions;
  for (let index = 0; index < count; index += 1) {
    set.add(atomA[index] * 0x4000000 + atomB[index]);
  }
  return set;
}
