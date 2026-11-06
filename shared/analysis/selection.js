/**
 * Classify an AMBER topology's atoms, and build the selections a viewer and a
 * live analysis need.
 *
 * 🔴 AN AMBER TOPOLOGY HAS NO CHAIN IDs, WHICH IS THE ONE THING THAT DOES NOT
 * TRANSFER FROM A PDB-BASED TOOL. JIGGLE splits receptor from binder by
 * `atom.residue.chain.id`, because it starts from a PDB and OpenMM's Topology
 * carries chains. A `prmtop` carries residues and nothing above them - tleap
 * discards chain identity entirely. So the split here comes from the MOLECULE
 * PARTITION, which buildSystem already derives from the bond graph and which
 * the barostat already depends on: the largest solute molecule is the
 * receptor, everything else that is not solvent is the binder. For a protein
 * with a ligand, a protein-protein complex, or a monomer, that gives the right
 * answer without a chain ID existing.
 *
 * 🔴 AND THE SOLVENT IS EXCLUDED FROM BOTH THE VIEWER AND THE METRICS. This is
 * JIGGLE's point and it is worth restating: a live cartoon of twelve thousand
 * water molecules is neither useful nor fast, and a radius of gyration over a
 * water box measures the box. The solvent is fully simulated; it is simply not
 * what anyone is looking at.
 */

/** AMBER residue names, including the terminal and protonation variants. */
const AMINO_ACIDS = new Set([
  "ALA", "ARG", "ASN", "ASP", "ASH", "CYS", "CYM", "CYX", "GLN", "GLU", "GLH",
  "GLY", "HIS", "HID", "HIE", "HIP", "ILE", "LEU", "LYS", "LYN", "MET", "PHE",
  "PRO", "SER", "THR", "TRP", "TYR", "VAL", "HYP", "ACE", "NME", "NHE", "NMA",
  "SEP", "TPO", "PTR", "ASX", "GLX", "UNK",
]);
const NUCLEIC = new Set([
  "DA", "DT", "DG", "DC", "DU", "A", "U", "G", "C", "T", "RA", "RU", "RG", "RC",
]);
const WATER = new Set(["WAT", "HOH", "H2O", "SOL", "TIP", "TP3", "SPC", "T3P"]);
const IONS = new Set([
  "NA", "NA+", "CL", "CL-", "K", "K+", "MG", "MG2", "CA", "CA2", "ZN", "ZN2",
  "LI", "LI+", "RB", "RB+", "CS", "CS+", "F", "F-", "BR", "BR-", "I", "I-",
  "IB", "CIO", "MN", "FE", "FE2", "CU", "CU1",
]);

/**
 * 🔴 A TERMINAL RESIDUE IS PREFIXED OR SUFFIXED AND MUST BE STRIPPED. tleap
 * writes `NALA` for an N-terminal alanine, `CALA` for a C-terminal one, and
 * `DA5`/`DA3` for the ends of a nucleic chain. Matching the raw name leaves
 * every terminus unclassified, so it falls through to "ligand" - and the
 * receptor/binder split then treats both ends of the protein as separate
 * ligands.
 */
function canonicalResidue(name) {
  const upper = name.trim().toUpperCase();
  if (AMINO_ACIDS.has(upper) || NUCLEIC.has(upper) || WATER.has(upper) || IONS.has(upper)) {
    return upper;
  }
  if (upper.length === 4 && (upper.startsWith("N") || upper.startsWith("C"))) {
    const stripped = upper.slice(1);
    if (AMINO_ACIDS.has(stripped)) return stripped;
  }
  if (upper.length >= 2 && (upper.endsWith("5") || upper.endsWith("3"))) {
    const stripped = upper.slice(0, -1);
    if (NUCLEIC.has(stripped)) return stripped;
  }
  return upper;
}

export const KIND_PROTEIN = "protein";
export const KIND_NUCLEIC = "nucleic";
export const KIND_WATER = "water";
export const KIND_ION = "ion";
export const KIND_LIGAND = "ligand";

export function classifyResidue(name) {
  const canonical = canonicalResidue(name);
  if (AMINO_ACIDS.has(canonical)) return KIND_PROTEIN;
  if (NUCLEIC.has(canonical)) return KIND_NUCLEIC;
  if (WATER.has(canonical)) return KIND_WATER;
  if (IONS.has(canonical)) return KIND_ION;
  return KIND_LIGAND;
}

/**
 * Build every selection from a topology.
 *
 * @param {object} system from buildSystem
 * @returns {object} index arrays and per-atom labels
 */
/**
 * A label for one solute molecule, for a control that has to offer a choice.
 *
 * 🔴 RESIDUE NAMES, NOT A MOLECULE NUMBER, because "molecule 2" means nothing
 * to anyone. A short molecule is named by its residues outright (`GOL`), a
 * peptide by its length and its ends (`ACE..NME, 9 residues`) - which is as
 * close to a chain label as a prmtop can get.
 */
function describeMolecule(system, atoms) {
  const topology = system.topology;
  const residues = [...new Set(atoms.map((atom) => topology.residueOfAtom[atom]))];
  const names = residues.map((residue) => (topology.residueNames[residue] ?? "?").trim());
  if (names.length === 0) return "empty";
  if (names.length <= 3) return names.join("-");
  return `${names[0]}..${names[names.length - 1]}, ${names.length} residues`;
}

/**
 * @param {object} system from buildSystem
 * @param {object} [options]
 * @param {number[]} [options.binderMolecules] solute molecule ids that form the
 *   binder; everything else solute becomes the receptor. Overrides the
 *   largest-molecule rule.
 * @param {number[]} [options.binderResidues] residue indices that form the
 *   binder, for a split that does not follow molecule boundaries.
 */
export function buildSelections(system, options = {}) {
  const topology = system.topology;
  const atomCount = topology.atomCount;
  const kindOfResidue = topology.residueNames.map(classifyResidue);
  const kindOfAtom = new Array(atomCount);
  for (let atom = 0; atom < atomCount; atom += 1) {
    kindOfAtom[atom] = kindOfResidue[topology.residueOfAtom[atom]];
  }

  const groups = {
    all: [], solute: [], protein: [], nucleic: [], ligand: [], water: [], ion: [],
    backbone: [], alphaCarbon: [], heavy: [],
  };
  const BACKBONE = new Set(["N", "CA", "C", "O"]);
  for (let atom = 0; atom < atomCount; atom += 1) {
    const kind = kindOfAtom[atom];
    const name = (topology.atomNames[atom] ?? "").trim().toUpperCase();
    groups.all.push(atom);
    if (topology.atomicNumbers[atom] !== 1) groups.heavy.push(atom);
    if (kind !== KIND_WATER && kind !== KIND_ION) groups.solute.push(atom);
    if (kind === KIND_PROTEIN) {
      groups.protein.push(atom);
      if (BACKBONE.has(name)) groups.backbone.push(atom);
      if (name === "CA") groups.alphaCarbon.push(atom);
    }
    if (kind === KIND_NUCLEIC) groups.nucleic.push(atom);
    if (kind === KIND_LIGAND) groups.ligand.push(atom);
    if (kind === KIND_WATER) groups.water.push(atom);
    if (kind === KIND_ION) groups.ion.push(atom);
  }

  // Receptor / binder, from the molecule partition rather than from chains.
  const soluteByMolecule = new Map();
  for (const atom of groups.solute) {
    const molecule = topology.moleculeOfAtom[atom];
    const list = soluteByMolecule.get(molecule);
    if (list === undefined) soluteByMolecule.set(molecule, [atom]); else list.push(atom);
  }
  const molecules = [...soluteByMolecule.entries()]
    .sort((left, right) => right[1].length - left[1].length);

  /**
   * 🔴 THE DERIVED SPLIT IS A DEFAULT, NOT A FACT, AND IT CANNOT EXPRESS A
   * TWO-CHAIN RECEPTOR. "Largest molecule is the receptor" is right for a
   * protein with a ligand and for a monomer, and wrong the moment the receptor
   * is a dimer - which it silently was, reporting one of the two chains as the
   * binder and measuring the complex coming apart as a pose failure. JIGGLE
   * asks for chain letters; a prmtop has none, so what is offered instead is
   * the molecule list, named by its residues.
   */
  let receptor;
  let binder;
  if (Array.isArray(options.binderResidues) && options.binderResidues.length > 0) {
    const wanted = new Set(options.binderResidues);
    receptor = [];
    binder = [];
    for (const atom of groups.solute) {
      (wanted.has(topology.residueOfAtom[atom]) ? binder : receptor).push(atom);
    }
  } else if (Array.isArray(options.binderMolecules) && options.binderMolecules.length > 0) {
    const wanted = new Set(options.binderMolecules);
    receptor = [];
    binder = [];
    for (const atom of groups.solute) {
      (wanted.has(topology.moleculeOfAtom[atom]) ? binder : receptor).push(atom);
    }
  } else {
    receptor = molecules.length > 0 ? molecules[0][1] : [];
    binder = molecules.slice(1).flatMap(([, atoms]) => atoms);
  }
  // What a control needs to offer the choice: every solute molecule, largest
  // first, with the id the override takes and a name a reader recognises.
  const soluteMolecules = molecules.map(([molecule, atoms]) => ({
    molecule,
    atomCount: atoms.length,
    label: describeMolecule(system, atoms),
    isBinder: binder.length > 0 && binder.includes(atoms[0]),
  }));
  const splitWasChosen = (Array.isArray(options.binderMolecules)
    && options.binderMolecules.length > 0)
    || (Array.isArray(options.binderResidues) && options.binderResidues.length > 0);

  /**
   * 🔴 ONE REPRESENTATIVE ATOM PER RESIDUE, AND NOT ALWAYS A CA. JIGGLE's RMSF
   * is per-residue over alpha carbons, which is right for a protein and empty
   * for a ligand or a nucleic acid. Falling back to the phosphorus of a
   * nucleotide and then to the residue's first heavy atom means the RMSF plot
   * has a bar for every residue in the system rather than only the ones that
   * happen to be amino acids.
   */
  const residueAtoms = [];
  const residueLabels = [];
  for (let residue = 0; residue < topology.residueNames.length; residue += 1) {
    const kind = kindOfResidue[residue];
    if (kind === KIND_WATER || kind === KIND_ION) continue;
    const first = topology.residueFirstAtom[residue];
    const last = residue + 1 < topology.residueNames.length
      ? topology.residueFirstAtom[residue + 1] : atomCount;
    let chosen = -1;
    for (const wanted of ["CA", "P", "C1'", "C1*"]) {
      for (let atom = first; atom < last; atom += 1) {
        if ((topology.atomNames[atom] ?? "").trim().toUpperCase() === wanted) {
          chosen = atom;
          break;
        }
      }
      if (chosen >= 0) break;
    }
    if (chosen < 0) {
      for (let atom = first; atom < last; atom += 1) {
        if (topology.atomicNumbers[atom] !== 1) { chosen = atom; break; }
      }
    }
    if (chosen < 0) chosen = first;
    residueAtoms.push(chosen);
    residueLabels.push(`${topology.residueNames[residue]}${residue + 1}`);
  }

  const toArray = (list) => Int32Array.from(list);
  return {
    kindOfAtom,
    kindOfResidue,
    groups: Object.fromEntries(Object.entries(groups).map(([k, v]) => [k, toArray(v)])),
    receptor: toArray(receptor),
    binder: toArray(binder),
    soluteMoleculeCount: molecules.length,
    soluteMolecules,
    splitWasChosen,
    residueAtoms: toArray(residueAtoms),
    residueLabels,
    counts: Object.fromEntries(Object.entries(groups).map(([k, v]) => [k, v.length])),
  };
}

/** The presets the viewer offers, in the order they appear. */
/**
 * The colour of each body, in both of the forms the page needs.
 *
 * 🔴 ONE DECLARATION, TWO FORMS, AND THE TWO FORMS ARE DELIBERATELY DIFFERENT
 * COLOURS. `line` is drawn on a light panel and `rgb` lights a sphere on a dark
 * one, so the same hue has to be darker in the first and brighter in the
 * second to read at all. That is a real constraint, not an inconsistency - but
 * it was expressed as two unrelated literals in `web/app.js` and
 * `web/viewer.js`, which is how "the orange trace is the orange molecule"
 * stops being true without anything failing. A reader maps the plot onto the
 * structure by colour; if that mapping is wrong the plot is worse than absent.
 */
export const BODY_COLOURS = {
  receptor: { line: "#4b8fd6", rgb: [0.42, 0.72, 0.92] },
  binder: { line: "#b07d2a", rgb: [0.95, 0.62, 0.25] },
  ligand: { line: "#b07d2a", rgb: [0.95, 0.62, 0.25] },
  complex: { line: "#2a6b6b", rgb: [0.26, 0.56, 0.56] },
  water: { line: "#5a7080", rgb: [0.32, 0.44, 0.50] },
  ion: { line: "#8a5fb0", rgb: [0.62, 0.42, 0.85] },
  other: { line: "#8f9296", rgb: [0.60, 0.62, 0.66] },
};

/**
 * Secondary-structure colours, in the same two forms as BODY_COLOURS.
 *
 * 🔴 THE CONVENTIONAL ONES, because a reader already knows them: helices warm
 * red, strands yellow, coil grey. Inventing a palette here would make every
 * published figure of the same protein look like a different molecule.
 */
export const STRUCTURE_COLOURS = {
  helix: { line: "#c0504d", rgb: [0.85, 0.37, 0.34] },
  sheet: { line: "#b8a13a", rgb: [0.92, 0.80, 0.30] },
  coil: { line: "#8f9296", rgb: [0.62, 0.64, 0.68] },
};

export const VIEW_PRESETS = [
  { id: "solute", label: "Solute only", groups: ["solute"] },
  { id: "protein-ligand", label: "Protein + ligand", groups: ["protein", "ligand"] },
  { id: "protein", label: "Protein only", groups: ["protein"] },
  { id: "ligand", label: "Ligand only", groups: ["ligand"] },
  { id: "backbone", label: "Backbone", groups: ["backbone"] },
  { id: "heavy-solute", label: "Solute, heavy atoms", groups: ["solute"], heavyOnly: true },
  { id: "everything", label: "Everything, solvent included", groups: ["all"] },
];
