/**
 * A PDB writer, for a saved frame and for the topology a DCD needs.
 *
 * 🔴 PDB IS FIXED-COLUMN AND THE COLUMNS ARE NOT NEGOTIABLE. Every reader
 * slices by position, so a one-character overflow in the atom serial or the
 * residue number shifts the coordinates and the file parses into nonsense
 * rather than failing. The widths below are the v3.3 specification's.
 *
 * 🔴 AND AN ATOM NAME IS NOT LEFT-JUSTIFIED IN ITS FIELD. Columns 13-16 hold
 * the name with the ELEMENT right-justified in 13-14, so a two-letter element
 * starts at 13 and a one-letter element starts at 14. `CA` in columns 13-14 is
 * calcium; the alpha carbon is ` CA ` starting at 14. Readers use that to
 * assign elements when there is no element column, so getting it wrong turns
 * every alpha carbon into a metal ion.
 */

function atomNameField(name, atomicNumber) {
  const trimmed = (name ?? "").trim().slice(0, 4);
  if (trimmed.length >= 4) return trimmed;
  // A two-letter element (Cl, Na, Mg, Zn, Fe) starts in column 13; everything
  // else is indented by one.
  const twoLetter = atomicNumber === 11 || atomicNumber === 12 || atomicNumber === 17
    || atomicNumber === 20 || atomicNumber === 26 || atomicNumber === 30
    || atomicNumber === 35 || atomicNumber === 53;
  return twoLetter ? trimmed.padEnd(4) : ` ${trimmed}`.padEnd(4);
}

const ELEMENT_SYMBOLS = {
  1: "H", 6: "C", 7: "N", 8: "O", 11: "NA", 12: "MG", 15: "P", 16: "S",
  17: "CL", 19: "K", 20: "CA", 26: "FE", 30: "ZN", 35: "BR", 53: "I",
};

/**
 * @param {object} system from buildSystem
 * @param {ArrayLike<number>} positions 3N, nm
 * @param {object} [options] indices to write, header remarks, box
 */
export function writePdb(system, positions, options = {}) {
  const topology = system.topology;
  const indices = options.indices ?? null;
  const lines = [];
  for (const remark of options.remarks ?? []) {
    // REMARK text is free-form but must stay inside the 80-column record.
    lines.push(`REMARK ${String(remark).slice(0, 72)}`);
  }
  const box = options.box ?? null;
  if (box !== null) {
    const [a, b, c] = box.lengths.map((value) => value * 10);
    const [alpha, beta, gamma] = box.angles;
    lines.push(
      "CRYST1"
      + a.toFixed(3).padStart(9) + b.toFixed(3).padStart(9) + c.toFixed(3).padStart(9)
      + alpha.toFixed(2).padStart(7) + beta.toFixed(2).padStart(7)
      + gamma.toFixed(2).padStart(7) + " P 1           1",
    );
  }

  const list = indices === null
    ? { length: topology.atomCount, at: (index) => index }
    : { length: indices.length, at: (index) => indices[index] };
  let serial = 0;
  let previousMolecule = -1;
  for (let position = 0; position < list.length; position += 1) {
    const atom = list.at(position);
    const molecule = topology.moleculeOfAtom[atom];
    // 🔴 A TER RECORD BETWEEN MOLECULES, or a reader bonds the last atom of one
    // chain to the first of the next and draws a line across the box.
    if (previousMolecule >= 0 && molecule !== previousMolecule) {
      serial += 1;
      lines.push(`TER   ${String(serial % 100000).padStart(5)}`);
    }
    previousMolecule = molecule;
    serial += 1;
    const residue = topology.residueOfAtom[atom];
    const element = ELEMENT_SYMBOLS[topology.atomicNumbers[atom]] ?? "";
    lines.push(
      "ATOM  "
      // 🔴 WRAPPED AT 100000, because the field is five columns. A system past
      // that many atoms is normal in MD and the serial simply recycles; readers
      // use the record order, not the number.
      + String(serial % 100000).padStart(5)
      + " " + atomNameField(topology.atomNames[atom], topology.atomicNumbers[atom])
      + " " + (topology.residueNames[residue] ?? "UNK").slice(0, 3).padEnd(3)
      + " A" + String((residue + 1) % 10000).padStart(4) + "    "
      + (positions[atom * 3] * 10).toFixed(3).padStart(8)
      + (positions[atom * 3 + 1] * 10).toFixed(3).padStart(8)
      + (positions[atom * 3 + 2] * 10).toFixed(3).padStart(8)
      + "  1.00  0.00          "
      + element.padStart(2),
    );
  }
  lines.push("END");
  return `${lines.join("\n")}\n`;
}
