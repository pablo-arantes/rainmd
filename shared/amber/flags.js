/**
 * Every `%FLAG` this parser knows about, and what happens to the rest.
 *
 * 🔴 THERE WAS NO SUCH LIST, AND THAT IS WHY ff19SB RAN WITHOUT ITS CMAP TERM.
 * `sections.js` builds a Map of every flag in the file and `prmtop.js` reads
 * the twenty-six it names; everything else was dropped with no message at all.
 * Measured on the Making-it-rain notebook's OWN DEFAULT force field
 * (`Force_field = "ff19SB"`), ACE-(ALA)3-NME, NoCutoff, against OpenMM 8.4:
 *
 *   term              OpenMM        RainMD
 *   HarmonicBond      +0.088778     +0.088778
 *   HarmonicAngle     +4.023395     +4.023395
 *   PeriodicTorsion  +10.022601    +10.022601
 *   Nonbonded        -53.246508    -53.246507
 *   CMAPTorsion       -5.082244     absent
 *   TOTAL            -44.193978   -39.111732
 *
 * and it scales with chain length: a 30-residue ff19SB helix carries 28 CMAP
 * terms worth -56.671931 kJ/mol. ff19SB IS ff14SB with CMAP replacing the
 * backbone torsions, so running it without CMAP is not a perturbation of the
 * force field, it is a different one - and the file parsed with
 * `warnings: []`.
 *
 * 🔴 SO AN UNRECOGNISED FLAG IS REFUSED, NOT WARNED ABOUT. The asymmetry
 * decides it: a flag that changes the energy and is ignored produces wrong
 * science that nothing downstream will reveal, while a harmless flag that
 * causes a refusal costs a user one message and this file one line. AGENTS.md
 * is explicit - "refuse by name rather than approximating" and "a silent
 * approximation in a scientific tool is worse than a refusal".
 *
 * 🔴 AND A WARNING WOULD NOT HAVE BEEN ENOUGH. `prmtop.js` already pushes a
 * note into `topology.warnings` when the molecule partition disagrees with
 * `ATOMS_PER_MOLECULE`, and the page surfaces it as one line in a scrolling
 * text log - which is how much attention a reader gives it. The flags below
 * change the energy; they get a throw.
 */

/**
 * Flags this parser reads and turns into force-field terms. Present for
 * documentation and for the audit; the parser's own `requireSection` calls are
 * what actually enforce the ones it needs.
 */
export const HANDLED_FLAGS = new Set([
  "VERSION", "TITLE", "POINTERS",
  "ATOM_NAME", "CHARGE", "MASS", "ATOM_TYPE_INDEX", "AMBER_ATOM_TYPE",
  "ATOMIC_NUMBER",
  "RESIDUE_LABEL", "RESIDUE_POINTER",
  "NUMBER_EXCLUDED_ATOMS", "EXCLUDED_ATOMS_LIST",
  "NONBONDED_PARM_INDEX", "LENNARD_JONES_ACOEF", "LENNARD_JONES_BCOEF",
  "BOND_FORCE_CONSTANT", "BOND_EQUIL_VALUE",
  "BONDS_INC_HYDROGEN", "BONDS_WITHOUT_HYDROGEN",
  "ANGLE_FORCE_CONSTANT", "ANGLE_EQUIL_VALUE",
  "ANGLES_INC_HYDROGEN", "ANGLES_WITHOUT_HYDROGEN",
  "DIHEDRAL_FORCE_CONSTANT", "DIHEDRAL_PERIODICITY", "DIHEDRAL_PHASE",
  "DIHEDRALS_INC_HYDROGEN", "DIHEDRALS_WITHOUT_HYDROGEN",
  "SCEE_SCALE_FACTOR", "SCNB_SCALE_FACTOR",
  "BOX_DIMENSIONS", "ATOMS_PER_MOLECULE", "SOLVENT_POINTERS",
  // 🔴 IPOL IS A SECTION IN MODERN FILES, NOT A POINTER. tleap writes
  // `%FLAG IPOL` with its own value; the POINTERS array does not carry it, so
  // reading `pointers.IPOL` gives undefined and a polarizable topology would
  // have walked straight past the check meant to stop it. Every fixture in
  // this repository writes IPOL 0, which is why this had to be read rather
  // than assumed.
  "IPOL",
  // Read only to refuse the topologies that actually use them; the negative-ICO
  // scan in prmtop.js is what enforces that.
  "HBOND_ACOEF", "HBOND_BCOEF", "HBCUT",
]);

/**
 * Flags that carry no energy and are safe to ignore - AMBER itself ignores
 * most of them. Each one is listed deliberately rather than matched by a
 * pattern, because "it looked like metadata" is how CMAP would have been
 * missed a second time.
 */
export const IGNORED_FLAGS = new Set([
  // Unused by AMBER's own engines.
  "SOLTY", "TREE_CHAIN_CLASSIFICATION", "JOIN_ARRAY", "IROTAT",
  // Generalised Born only; this engine is explicit-solvent PME and refuses a
  // non-periodic implicit-solvent run on other grounds.
  "RADIUS_SET", "RADII", "SCREEN",
  // PDB provenance ParmEd writes. Note that RESIDUE_CHAINID means
  // docs/ARCHITECTURE.md's "a prmtop has no chain IDs at all" is false as an
  // absolute - tleap does not write them, so the molecule-partition split is
  // still the right default for notebook output.
  "RESIDUE_CHAINID", "RESIDUE_NUMBER", "RESIDUE_ICODE",
  "ATOM_NUMBER", "ATOM_BFACTOR", "ATOM_OCCUPANCY", "ATOM_ELEMENT",
  // Written by some converters, carries no parameters.
  "FORCE_FIELD_TYPE_COMMENT", "CHARMM_FORCE_FIELD_NAME",
]);

/**
 * Flags that change the energy and are not implemented. Each maps to a message
 * saying what to do instead, because "unsupported" on its own sends a reader
 * to the source.
 */
export const REFUSED_FLAGS = new Map([
  ["CMAP_COUNT", cmap], ["CMAP_RESOLUTION", cmap],
  ["CMAP_PARAMETER", cmap], ["CMAP_INDEX", cmap],
  ["LENNARD_JONES_CCOEF", () =>
    "this topology uses AMBER's 12-6-4 ion model (%FLAG "
    + "LENNARD_JONES_CCOEF), which adds an r^-4 term this engine does not "
    + "implement. Rebuild the ions with the ordinary 12-6 parameters - "
    + "frcmod.ionsjc_tip3p rather than frcmod.ions234lm_1264_tip3p - or run it "
    + "in AMBER."],
  ["CTITLE", chamber],
  ["CHARMM_UREY_BRADLEY_COUNT", chamber], ["CHARMM_UREY_BRADLEY", chamber],
  ["CHARMM_UREY_BRADLEY_FORCE_CONSTANT", chamber],
  ["CHARMM_UREY_BRADLEY_EQUIL_VALUE", chamber],
  ["CHARMM_NUM_IMPROPERS", chamber], ["CHARMM_IMPROPERS", chamber],
  ["CHARMM_NUM_IMPR_TYPES", chamber],
  ["CHARMM_IMPROPER_FORCE_CONSTANT", chamber], ["CHARMM_IMPROPER_PHASE", chamber],
  ["CHARMM_CMAP_COUNT", chamber], ["CHARMM_CMAP_RESOLUTION", chamber],
  ["CHARMM_CMAP_PARAMETER", chamber], ["CHARMM_CMAP_INDEX", chamber],
  ["LENNARD_JONES_14_ACOEF", chamber], ["LENNARD_JONES_14_BCOEF", chamber],
  ["POLARIZABILITY", () =>
    "this topology carries atomic polarizabilities (%FLAG POLARIZABILITY), so "
    + "it was built for a polarizable force field. This engine is fixed-charge "
    + "and would run it with the permanent charges alone, which is a different "
    + "force field. Rebuild it as fixed-charge."],
  ["CAP_INFO", cap], ["CAP_INFO2", cap],
  ["AMOEBA_FORCEFIELD", () =>
    "this is an AMOEBA topology (%FLAG AMOEBA_FORCEFIELD). AMOEBA is "
    + "polarizable and multipolar; nothing in this engine applies to it. Run "
    + "it in OpenMM or Tinker."],
]);

function cmap() {
  return "this topology carries CMAP backbone-correction terms (%FLAG CMAP_*), "
    + "which this engine does not implement. That makes it ff19SB or a CHARMM "
    + "conversion: ff19SB replaces ff14SB's backbone torsions with a CMAP, so "
    + "dropping it is not a small error - measured against OpenMM it is "
    + "-5.08 kJ/mol on a capped tripeptide and -56.67 on a 30-residue helix, "
    + "and it grows with the chain. Rebuild with `source "
    + "leaprc.protein.ff14SB`, which the notebooks also offer, or run ff19SB "
    + "in OpenMM.";
}

function chamber() {
  return "this topology was converted from CHARMM by `chamber` (it carries "
    + "%FLAG CTITLE / CHARMM_*). It needs Urey-Bradley terms, CHARMM "
    + "impropers, a CMAP and a separate 1-4 Lennard-Jones table, none of which "
    + "this engine implements - and its own 1-4 table means scaling the full "
    + "LJ table by SCNB, which is what this parser would do, is wrong. Run it "
    + "in OpenMM, which reads CHARMM files directly.";
}

function cap() {
  return "this topology uses a solvent cap (%FLAG CAP_INFO), which this engine "
    + "does not implement - it would simulate the capped waters as though they "
    + "were a periodic box. Re-solvate with `solvateBox` or `solvateOct`.";
}

/**
 * Audit the flags a file actually carries.
 *
 * 🔴 THE PREFIX MATCH IS FOR THE NUMBERED FAMILIES. CMAP and CHARMM CMAP write
 * one flag per map - `CMAP_PARAMETER_01`, `CMAP_PARAMETER_02` - so an exact
 * match would miss every one of them and catch only the count. Checked against
 * a real ff19SB file, which carries CMAP_COUNT, CMAP_RESOLUTION,
 * CMAP_PARAMETER_01 and CMAP_INDEX.
 *
 * @param {Iterable<string>} flags every `%FLAG` in the file
 * @returns {{refusals: string[], unrecognised: string[]}}
 */
export function auditFlags(flags) {
  const refusals = [];
  const unrecognised = [];
  const seen = new Set();
  for (const flag of flags) {
    const name = String(flag).trim();
    if (HANDLED_FLAGS.has(name) || IGNORED_FLAGS.has(name)) continue;
    let matched = null;
    for (const [prefix, message] of REFUSED_FLAGS) {
      if (name === prefix || name.startsWith(`${prefix}_`)) { matched = message; break; }
    }
    if (matched !== null) {
      const text = matched();
      if (!seen.has(text)) { seen.add(text); refusals.push(text); }
      continue;
    }
    unrecognised.push(name);
  }
  return { refusals, unrecognised };
}
