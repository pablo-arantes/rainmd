/**
 * AMBER parm7 -> a topology in this engine's units.
 *
 * This is the keystone of the whole port. It is the analogue of LocalFold's
 * weight bundle: the expensive, irreproducible-in-a-browser work (tleap,
 * antechamber, AM1-BCC) happens once, offline, in the Making-it-rain notebooks,
 * and what crosses into the browser is the artifact it produced. Everything
 * downstream - every kernel, every reference, every energy - reads what this
 * function returns, so a mistake here is a mistake everywhere and shows up as
 * physics rather than as a parse error.
 *
 * What it does NOT do: units are converted here, exactly once (see
 * shared/units.js), and nothing downstream may convert again. Indices are
 * converted to plain 0-based atom numbers here, and nothing downstream may
 * divide by three.
 */
import {
  AMBER_CHARGE_SCALE, ANGLE_K_SCALE, BOND_K_SCALE, ENERGY_SCALE,
  NM_PER_ANGSTROM, ljCoefficientsToInternal,
} from "../units.js";
import { readAmberSections } from "./sections.js";
import { auditFlags } from "./flags.js";

/** POINTERS, by name, in the order AMBER writes them. */
const POINTER_NAMES = [
  "NATOM", "NTYPES", "NBONH", "MBONA", "NTHETH", "MTHETA", "NPHIH", "MPHIA",
  "NHPARM", "NPARM", "NNB", "NRES", "NBONA", "NTHETA", "NPHIA", "NUMBND",
  "NUMANG", "NPTRA", "NATYP", "NPHB", "IFPERT", "NBPER", "NGPER", "NDPER",
  "MBPER", "MGPER", "MDPER", "IFBOX", "NMXRS", "IFCAP", "NUMEXTRA", "NCOPY",
];

/** AMBER's defaults for files written before the scale factors were sections. */
const DEFAULT_SCEE = 1.2;
const DEFAULT_SCNB = 2.0;

function requireSection(sections, flag) {
  const section = sections.get(flag);
  if (section === undefined) {
    throw new Error(`AMBER topology is missing required %FLAG ${flag}`);
  }
  return section.values;
}

function optionalSection(sections, flag) {
  return sections.get(flag)?.values;
}

/**
 * 🔴 AN ATOM INDEX IN A BOND/ANGLE/DIHEDRAL ARRAY IS `3 * atom`, AND IT IS
 * SIGNED. The factor of three is a fossil of AMBER's coordinate-array
 * addressing, and the sign carries meaning on dihedral atoms 3 and 4 - so the
 * magnitude must be taken before the division and the sign read separately.
 * Dividing a negative index by three and flooring gives an atom one off and a
 * dihedral that silently involves the wrong atom.
 */
function atomFromStride(value) {
  return Math.abs(value) / 3;
}

/**
 * Group atoms into molecules over the bond graph, with union-find.
 *
 * 🔴 THE BAROSTAT NEEDS THIS AND CANNOT GET IT FROM ATOMS_PER_MOLECULE ALONE.
 * A Monte Carlo barostat scales molecular centres of mass, not atoms: scaling
 * atoms individually stretches every bond in the system and the trial move is
 * rejected essentially always, so the box never equilibrates and the density
 * stays at whatever tleap's solvatebox guessed. ATOMS_PER_MOLECULE is written
 * only when IFBOX > 0 and assumes atoms of a molecule are contiguous, which is
 * true of what tleap writes and is not a property of the format - so the graph
 * is the source of truth and the section is a cross-check.
 */
function buildMolecules(atomCount, bonds) {
  const parent = new Int32Array(atomCount);
  for (let index = 0; index < atomCount; index += 1) parent[index] = index;
  const find = (node) => {
    let root = node;
    while (parent[root] !== root) root = parent[root];
    while (parent[node] !== root) {
      const next = parent[node];
      parent[node] = root;
      node = next;
    }
    return root;
  };
  for (let index = 0; index < bonds.count; index += 1) {
    const a = find(bonds.atomA[index]);
    const b = find(bonds.atomB[index]);
    if (a !== b) parent[b] = a;
  }
  /** @type {Map<number, number>} */
  const rootToMolecule = new Map();
  const moleculeOfAtom = new Int32Array(atomCount);
  for (let index = 0; index < atomCount; index += 1) {
    const root = find(index);
    let molecule = rootToMolecule.get(root);
    if (molecule === undefined) {
      molecule = rootToMolecule.size;
      rootToMolecule.set(root, molecule);
    }
    moleculeOfAtom[index] = molecule;
  }
  return { moleculeCount: rootToMolecule.size, moleculeOfAtom };
}

/**
 * Parse an AMBER parm7 topology.
 *
 * @param {string} text the whole `.prmtop` file
 * @returns {object} topology in nm / kJ/mol / ps / amu / e
 */
export function parsePrmtop(text) {
  const { version, sections } = readAmberSections(text);
  if (!sections.has("POINTERS")) {
    throw new Error(
      "not an AMBER parm7 topology: no %FLAG POINTERS. "
      + "A `.prmtop` written by tleap is what this expects - an OpenMM XML "
      + "system or a CHARMM psf is a different file.",
    );
  }
  const pointerValues = requireSection(sections, "POINTERS");
  /** @type {Record<string, number>} */
  const pointers = {};
  POINTER_NAMES.forEach((name, index) => {
    pointers[name] = pointerValues[index] ?? 0;
  });
  const atomCount = pointers.NATOM;
  if (!Number.isInteger(atomCount) || atomCount <= 0) {
    throw new Error(`AMBER topology declares NATOM = ${atomCount}`);
  }
  if (pointers.IFPERT !== 0) {
    throw new Error(
      "this topology carries perturbation (IFPERT != 0) parameters, which this "
      + "engine does not read. Re-save it without the perturbed state.",
    );
  }

  // 🔴 EVERY FLAG IN THE FILE IS AUDITED, BECAUSE UNTIL NOW NONE WERE. The
  // parser read the twenty-six flags it names and dropped the rest in silence -
  // so an ff19SB topology, which is the Making-it-rain notebook's OWN DEFAULT,
  // loaded with `warnings: []` and ran without its CMAP term. See
  // shared/amber/flags.js for the measurement and for why an unrecognised flag
  // is refused rather than warned about.
  const audit = auditFlags(sections.keys());
  if (audit.refusals.length > 0) {
    throw new Error(audit.refusals.join(" ALSO: "));
  }
  if (audit.unrecognised.length > 0) {
    throw new Error(
      `this topology carries ${audit.unrecognised.length} %FLAG section(s) this `
      + `engine does not know: ${audit.unrecognised.join(", ")}. It is refused `
      + "rather than ignored because a flag that changes the energy and is "
      + "dropped produces a wrong answer that nothing downstream will reveal - "
      + "which is exactly what happened with ff19SB's CMAP. If these carry no "
      + "parameters, add them to IGNORED_FLAGS in shared/amber/flags.js; if "
      + "they do, they need implementing or a refusal of their own.",
    );
  }

  // 🔴 AND THREE OF THE REFUSALS ARE POINTERS, NOT FLAGS, SO THE AUDIT ABOVE
  // CANNOT SEE THEM. NUMEXTRA is the one that matters: `leaprc.water.opc` and
  // `leaprc.water.tip4pew` - OPC is ff19SB's recommended water and BOTH
  // notebooks offer it as `Water_type` - put one zero-mass EP site per water,
  // bonded to the oxygen. With no virtual-site machinery this engine lets
  // `rigidWater` constrain the O-EP bond, so 933 of 3754 distance constraints
  // tie a movable oxygen to a frozen massless point. Measured on the host, 60
  // steps at 2 fs with a 298 K setpoint: 2653.6 K at step 15, while the extra
  // points moved 0.00000 nm and the oxygens 0.03187. The 3-point control on the
  // same code path reaches 412.9 K.
  if (pointers.NUMEXTRA > 0) {
    throw new Error(
      `this topology has ${pointers.NUMEXTRA} extra points (NUMEXTRA), so it is `
      + "a 4-point or 5-point water model such as OPC or TIP4P-Ew, or carries "
      + "lone pairs. This engine has no virtual-site support: it would treat "
      + "each massless site as an atom and constrain it to its oxygen, which "
      + "heats the system to thousands of kelvin within a few hundred steps. "
      + "Rebuild with a 3-point water - `source leaprc.water.tip3p`, which both "
      + "notebooks offer - or run it in OpenMM.",
    );
  }
  // IPOL lives in its own section in modern files and in no POINTERS slot, so
  // both places are read and whichever exists decides.
  const ipolSection = sections.get("IPOL");
  const ipol = ipolSection !== undefined && ipolSection.values.length > 0
    ? Number(ipolSection.values[0]) : (pointers.IPOL ?? 0);
  if (ipol !== 0) {
    throw new Error(
      `this topology declares IPOL = ${ipol}, so it was built for a `
      + "polarizable force field. This engine is fixed-charge and would run it "
      + "with the permanent charges alone - a different force field. Rebuild it "
      + "as fixed-charge.",
    );
  }
  if (pointers.IFCAP !== 0 && pointers.IFCAP !== undefined) {
    throw new Error(
      `this topology declares IFCAP = ${pointers.IFCAP}, a solvent cap, which `
      + "this engine does not implement - it would simulate the capped waters "
      + "as though they were a periodic box. Re-solvate with `solvateBox` or "
      + "`solvateOct`.",
    );
  }
  if (pointers.NCOPY > 1) {
    throw new Error(
      `this topology declares NCOPY = ${pointers.NCOPY}, so it uses Locally `
      + "Enhanced Sampling. LES replicates part of the system and scales its "
      + "interactions; this engine would simulate the copies as real, "
      + "interacting molecules. Rebuild without LES.",
    );
  }

  // --- per atom -----------------------------------------------------------
  const atomNames = requireSection(sections, "ATOM_NAME").slice(0, atomCount)
    .map((value) => String(value).trim());
  const rawCharge = requireSection(sections, "CHARGE");
  const charges = new Float64Array(atomCount);
  for (let index = 0; index < atomCount; index += 1) {
    charges[index] = rawCharge[index] / AMBER_CHARGE_SCALE;
  }
  const rawMass = requireSection(sections, "MASS");
  const masses = new Float64Array(atomCount);
  for (let index = 0; index < atomCount; index += 1) masses[index] = rawMass[index];

  const atomicNumbers = new Int32Array(atomCount);
  const rawAtomicNumber = optionalSection(sections, "ATOMIC_NUMBER");
  if (rawAtomicNumber !== undefined) {
    for (let index = 0; index < atomCount; index += 1) {
      atomicNumbers[index] = rawAtomicNumber[index];
    }
  } else {
    // 🔴 PRE-2010 TOPOLOGIES HAVE NO ATOMIC_NUMBER, and the constraint builder
    // needs to know which atoms are hydrogens. Mass is the only thing left to
    // ask, and the threshold has to admit deuterium and the 3-point masses
    // AMBER gives TIP3P hydrogens while refusing lithium at 6.94.
    for (let index = 0; index < atomCount; index += 1) {
      atomicNumbers[index] = masses[index] > 0 && masses[index] < 3.5 ? 1 : 0;
    }
  }
  const atomTypeIndex = requireSection(sections, "ATOM_TYPE_INDEX");
  const atomTypes = new Int32Array(atomCount);
  for (let index = 0; index < atomCount; index += 1) {
    atomTypes[index] = atomTypeIndex[index] - 1; // 0-based
  }
  const atomTypeNames = (optionalSection(sections, "AMBER_ATOM_TYPE") ?? [])
    .slice(0, atomCount).map((value) => String(value).trim());

  // --- residues -----------------------------------------------------------
  const residueNames = requireSection(sections, "RESIDUE_LABEL")
    .slice(0, pointers.NRES).map((value) => String(value).trim());
  const residuePointer = requireSection(sections, "RESIDUE_POINTER");
  const residueFirstAtom = new Int32Array(pointers.NRES);
  for (let index = 0; index < pointers.NRES; index += 1) {
    residueFirstAtom[index] = residuePointer[index] - 1; // 0-based
  }
  const residueOfAtom = new Int32Array(atomCount);
  for (let residue = 0; residue < pointers.NRES; residue += 1) {
    const start = residueFirstAtom[residue];
    const end = residue + 1 < pointers.NRES ? residueFirstAtom[residue + 1] : atomCount;
    for (let atom = start; atom < end; atom += 1) residueOfAtom[atom] = residue;
  }

  // --- Lennard-Jones ------------------------------------------------------
  const typeCount = pointers.NTYPES;
  const nonbondedParmIndex = requireSection(sections, "NONBONDED_PARM_INDEX");
  const rawA = requireSection(sections, "LENNARD_JONES_ACOEF");
  const rawB = requireSection(sections, "LENNARD_JONES_BCOEF");
  // A dense typeCount^2 table of (A, B) in internal units. Small - NTYPES is
  // tens, even for a protein with a ligand - and a direct index beats a
  // combining rule both in faithfulness to AMBER and in kernel cost.
  const ljA = new Float64Array(typeCount * typeCount);
  const ljB = new Float64Array(typeCount * typeCount);
  let hbondPairs = 0;
  for (let i = 0; i < typeCount; i += 1) {
    for (let j = 0; j < typeCount; j += 1) {
      const parmIndex = nonbondedParmIndex[typeCount * i + j];
      if (parmIndex > 0) {
        const { a, b } = ljCoefficientsToInternal(rawA[parmIndex - 1], rawB[parmIndex - 1]);
        ljA[typeCount * i + j] = a;
        ljB[typeCount * i + j] = b;
      } else if (parmIndex < 0) {
        // 🔴 A NEGATIVE ICO ENTRY IS A 10-12 HYDROGEN-BOND PAIR, a term AMBER
        // removed with ff94 and which no force field the notebooks use emits.
        // Counted and reported rather than silently treated as zero, because a
        // topology that has them is being read by the wrong engine.
        hbondPairs += 1;
      }
    }
  }
  if (hbondPairs > 0) {
    throw new Error(
      `topology uses the legacy 10-12 hydrogen-bond term on ${hbondPairs} type `
      + "pairs, which this engine does not implement. That term was dropped "
      + "after ff94; re-parameterise with ff14SB/ff19SB as the notebooks do.",
    );
  }

  // --- bonded terms -------------------------------------------------------
  const bondK = requireSection(sections, "BOND_FORCE_CONSTANT");
  const bondR0 = requireSection(sections, "BOND_EQUIL_VALUE");
  const bonds = readBonds(
    [...(optionalSection(sections, "BONDS_INC_HYDROGEN") ?? []),
      ...(optionalSection(sections, "BONDS_WITHOUT_HYDROGEN") ?? [])],
    bondK, bondR0,
    (optionalSection(sections, "BONDS_INC_HYDROGEN") ?? []).length / 3,
  );

  const angleK = requireSection(sections, "ANGLE_FORCE_CONSTANT");
  const angleTheta0 = requireSection(sections, "ANGLE_EQUIL_VALUE");
  const angles = readAngles(
    [...(optionalSection(sections, "ANGLES_INC_HYDROGEN") ?? []),
      ...(optionalSection(sections, "ANGLES_WITHOUT_HYDROGEN") ?? [])],
    angleK, angleTheta0,
  );

  const dihedralK = requireSection(sections, "DIHEDRAL_FORCE_CONSTANT");
  const dihedralPeriodicity = requireSection(sections, "DIHEDRAL_PERIODICITY");
  const dihedralPhase = requireSection(sections, "DIHEDRAL_PHASE");
  const scee = optionalSection(sections, "SCEE_SCALE_FACTOR");
  const scnb = optionalSection(sections, "SCNB_SCALE_FACTOR");
  const dihedrals = readDihedrals(
    [...(optionalSection(sections, "DIHEDRALS_INC_HYDROGEN") ?? []),
      ...(optionalSection(sections, "DIHEDRALS_WITHOUT_HYDROGEN") ?? [])],
    dihedralK, dihedralPeriodicity, dihedralPhase, scee, scnb,
  );

  // --- exclusions ---------------------------------------------------------
  const exclusions = readExclusions(
    atomCount,
    requireSection(sections, "NUMBER_EXCLUDED_ATOMS"),
    requireSection(sections, "EXCLUDED_ATOMS_LIST"),
  );

  // --- box ----------------------------------------------------------------
  let boxFromTopology = null;
  const boxDimensions = optionalSection(sections, "BOX_DIMENSIONS");
  if (pointers.IFBOX > 0 && boxDimensions !== undefined) {
    // BOX_DIMENSIONS is [OLDBETA, BOX(1), BOX(2), BOX(3)] - the angle FIRST.
    // Reading it as three lengths plus an angle gets a box 90 degrees wrong on
    // a truncated octahedron and silently right on a rectangular one.
    boxFromTopology = {
      lengths: [
        boxDimensions[1] * NM_PER_ANGSTROM,
        boxDimensions[2] * NM_PER_ANGSTROM,
        boxDimensions[3] * NM_PER_ANGSTROM,
      ],
      angles: [boxDimensions[0], boxDimensions[0], boxDimensions[0]],
    };
  }

  const molecules = buildMolecules(atomCount, bonds);
  const atomsPerMolecule = optionalSection(sections, "ATOMS_PER_MOLECULE");
  const warnings = [];
  if (atomsPerMolecule !== undefined && atomsPerMolecule.length !== molecules.moleculeCount) {
    warnings.push(
      `ATOMS_PER_MOLECULE declares ${atomsPerMolecule.length} molecules; the bond `
      + `graph finds ${molecules.moleculeCount}. Using the bond graph.`,
    );
  }

  return {
    source: "amber-parm7",
    version,
    pointers,
    atomCount,
    atomNames,
    atomTypeNames,
    atomicNumbers,
    charges,
    masses,
    atomTypes,
    typeCount,
    ljA,
    ljB,
    residueNames,
    residueFirstAtom,
    residueOfAtom,
    bonds,
    angles,
    dihedrals,
    exclusions,
    boxFromTopology,
    isPeriodic: pointers.IFBOX > 0,
    moleculeCount: molecules.moleculeCount,
    moleculeOfAtom: molecules.moleculeOfAtom,
    warnings,
  };
}

/**
 * @param {number[]} raw concatenated BONDS_INC_HYDROGEN then BONDS_WITHOUT
 * @param {number[]} forceConstants kcal/mol/A^2
 * @param {number[]} equilibria A
 * @param {number} withHydrogenCount how many of the entries came from the
 *   INC_HYDROGEN section - the constraint builder needs to know, and asking
 *   "is either atom a hydrogen" later is not the same question: AMBER's split
 *   is what `ntc`/HBonds means.
 */
function readBonds(raw, forceConstants, equilibria, withHydrogenCount) {
  const count = raw.length / 3;
  if (!Number.isInteger(count)) {
    throw new Error(`bond array length ${raw.length} is not a multiple of 3`);
  }
  const atomA = new Int32Array(count);
  const atomB = new Int32Array(count);
  const k = new Float64Array(count);
  const r0 = new Float64Array(count);
  const containsHydrogen = new Uint8Array(count);
  for (let index = 0; index < count; index += 1) {
    atomA[index] = atomFromStride(raw[index * 3]);
    atomB[index] = atomFromStride(raw[index * 3 + 1]);
    const parameter = raw[index * 3 + 2] - 1;
    k[index] = forceConstants[parameter] * BOND_K_SCALE;
    r0[index] = equilibria[parameter] * NM_PER_ANGSTROM;
    containsHydrogen[index] = index < withHydrogenCount ? 1 : 0;
  }
  return { count, atomA, atomB, k, r0, containsHydrogen };
}

function readAngles(raw, forceConstants, equilibria) {
  const count = raw.length / 4;
  if (!Number.isInteger(count)) {
    throw new Error(`angle array length ${raw.length} is not a multiple of 4`);
  }
  const atomA = new Int32Array(count);
  const atomB = new Int32Array(count);
  const atomC = new Int32Array(count);
  const k = new Float64Array(count);
  const theta0 = new Float64Array(count);
  for (let index = 0; index < count; index += 1) {
    atomA[index] = atomFromStride(raw[index * 4]);
    atomB[index] = atomFromStride(raw[index * 4 + 1]);
    atomC[index] = atomFromStride(raw[index * 4 + 2]);
    const parameter = raw[index * 4 + 3] - 1;
    k[index] = forceConstants[parameter] * ANGLE_K_SCALE;
    theta0[index] = equilibria[parameter]; // already radians
  }
  return { count, atomA, atomB, atomC, k, theta0 };
}

/**
 * Dihedrals, proper and improper, plus the 1-4 bookkeeping they carry.
 *
 * 🔴 THE SIGNS ON ATOMS 3 AND 4 ARE THE WHOLE INTERFACE TO THE 1-4 TERM. A
 * negative FOURTH atom marks an improper. A negative THIRD atom says "do not
 * compute the 1-4 interaction for this torsion", which AMBER writes both on
 * the second and later terms of a multi-term torsion (so the pair is not
 * counted once per term) and on a torsion closing a ring (where the 1-4 pair is
 * already a 1-2 or 1-3 pair, and counting it would double a bonded
 * interaction). Reading those signs as noise and taking abs() on all four gives
 * a system whose bonded energy is right and whose 1-4 energy is wrong by
 * however many duplicate terms the file happens to contain.
 */
function readDihedrals(raw, forceConstants, periodicities, phases, scee, scnb) {
  const count = raw.length / 5;
  if (!Number.isInteger(count)) {
    throw new Error(`dihedral array length ${raw.length} is not a multiple of 5`);
  }
  const atomA = new Int32Array(count);
  const atomB = new Int32Array(count);
  const atomC = new Int32Array(count);
  const atomD = new Int32Array(count);
  const k = new Float64Array(count);
  const periodicity = new Float64Array(count);
  const phase = new Float64Array(count);
  const isImproper = new Uint8Array(count);
  /** 1-4 pairs, deduplicated, with the scale factors of the torsion that made them. */
  const pair14A = [];
  const pair14B = [];
  const pair14Scee = [];
  const pair14Scnb = [];
  const seen14 = new Set();

  for (let index = 0; index < count; index += 1) {
    const rawC = raw[index * 5 + 2];
    const rawD = raw[index * 5 + 3];
    const a = atomFromStride(raw[index * 5]);
    const b = atomFromStride(raw[index * 5 + 1]);
    const c = atomFromStride(rawC);
    const d = atomFromStride(rawD);
    const parameter = raw[index * 5 + 4] - 1;
    atomA[index] = a;
    atomB[index] = b;
    atomC[index] = c;
    atomD[index] = d;
    k[index] = forceConstants[parameter] * ENERGY_SCALE;
    periodicity[index] = Math.abs(periodicities[parameter]);
    phase[index] = phases[parameter];
    isImproper[index] = rawD < 0 ? 1 : 0;

    const skip14 = rawC < 0 || rawD < 0;
    if (skip14) continue;
    const low = Math.min(a, d);
    const high = Math.max(a, d);
    const key = low * 0x4000000 + high;
    if (seen14.has(key)) continue;
    seen14.add(key);
    pair14A.push(low);
    pair14B.push(high);
    pair14Scee.push(scee === undefined ? DEFAULT_SCEE : scee[parameter]);
    pair14Scnb.push(scnb === undefined ? DEFAULT_SCNB : scnb[parameter]);
  }

  return {
    count, atomA, atomB, atomC, atomD, k, periodicity, phase, isImproper,
    pairs14: {
      count: pair14A.length,
      atomA: Int32Array.from(pair14A),
      atomB: Int32Array.from(pair14B),
      scee: Float64Array.from(pair14Scee),
      scnb: Float64Array.from(pair14Scnb),
    },
  };
}

/**
 * The excluded-pair set, as a sorted per-atom list.
 *
 * 🔴 A COUNT OF ONE WITH A VALUE OF ZERO MEANS NO EXCLUSIONS, NOT "EXCLUDE
 * ATOM 0". AMBER cannot write a zero-length entry, so an atom that excludes
 * nothing gets one placeholder. Taking it literally excludes the first atom of
 * the system from the nonbonded interaction of every terminal atom in it -
 * which is a handful of pairs in a protein and so moves the energy by a little
 * rather than by a lot, and is therefore the kind of mistake that survives.
 */
function readExclusions(atomCount, counts, list) {
  const start = new Int32Array(atomCount + 1);
  let cursor = 0;
  const partners = [];
  for (let atom = 0; atom < atomCount; atom += 1) {
    start[atom] = partners.length;
    const declared = counts[atom];
    for (let offset = 0; offset < declared; offset += 1) {
      const value = list[cursor + offset];
      if (value === 0) continue;
      partners.push(value - 1);
    }
    cursor += declared;
  }
  start[atomCount] = partners.length;
  return {
    pairCount: partners.length,
    start,
    partners: Int32Array.from(partners),
  };
}
