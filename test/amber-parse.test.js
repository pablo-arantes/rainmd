import assert from "node:assert/strict";
import { test } from "node:test";
import { readFileSync, readdirSync } from "node:fs";
import { parsePrmtop } from "../shared/amber/prmtop.js";
import { parseInpcrd, boxVectorsFromLengthsAndAngles } from "../shared/amber/inpcrd.js";
import { parseFortranFormat, sliceFields } from "../shared/amber/sections.js";
import { buildSystem } from "../shared/topology/system.js";
import { buildBondedGather } from "../shared/topology/gather.js";

const fixture = (name) => new URL(`fixtures/${name}`, import.meta.url);
const read = (name) => readFileSync(fixture(name), "utf8");

test("FORTRAN format descriptors", () => {
  assert.deepEqual(parseFortranFormat("10I8"), { perLine: 10, kind: "integer", width: 8 });
  assert.deepEqual(parseFortranFormat("5E16.8"), { perLine: 5, kind: "float", width: 16 });
  assert.deepEqual(parseFortranFormat("20a4"), { perLine: 20, kind: "string", width: 4 });
  assert.throws(() => parseFortranFormat("nonsense"));
});

test("fields are sliced by width, not split on whitespace", () => {
  // Two eight-wide integers with no space between them: the case that makes
  // whitespace splitting silently wrong.
  const fields = sliceFields("-1234567-7654321", { perLine: 10, kind: "integer", width: 8 });
  assert.equal(fields[0], "-1234567");
  assert.equal(fields[1], "-7654321");
});

test("a short final line yields only the fields it holds", () => {
  const fields = sliceFields("      1       2", { perLine: 10, kind: "integer", width: 8 });
  assert.equal(fields.length, 2);
});

test("vacuum topology parses to known values", () => {
  const topology = parsePrmtop(read("ala-vac.prmtop"));
  assert.equal(topology.atomCount, 22);
  assert.equal(topology.residueNames.length, 3);
  assert.equal(topology.moleculeCount, 1);
  assert.equal(topology.bonds.count, 21);
  assert.equal(topology.angles.count, 36);
  assert.equal(topology.dihedrals.count, 67);
  assert.equal(topology.dihedrals.pairs14.count, 41);
  assert.equal(topology.exclusions.pairCount, 98);
  // Neutral to within tleap's own rounding.
  const chargeSum = Array.from(topology.charges).reduce((a, b) => a + b, 0);
  assert.ok(Math.abs(chargeSum) < 1e-6, `charge sum ${chargeSum}`);
});

test("AMBER water is three constrained bonds and no angle", () => {
  const topology = parsePrmtop(read("ala-wat.prmtop"));
  const residue = topology.residueNames.indexOf("WAT");
  const first = topology.residueFirstAtom[residue];
  const last = topology.residueFirstAtom[residue + 1];
  let bonds = 0;
  for (let index = 0; index < topology.bonds.count; index += 1) {
    const a = topology.bonds.atomA[index];
    const b = topology.bonds.atomB[index];
    if (a >= first && a < last && b >= first && b < last) bonds += 1;
  }
  assert.equal(bonds, 3, "O-H, O-H and the H-H that encodes the angle");
  let angles = 0;
  for (let index = 0; index < topology.angles.count; index += 1) {
    if (topology.angles.atomB[index] >= first && topology.angles.atomB[index] < last) angles += 1;
  }
  assert.equal(angles, 0);
});

test("coordinate file atom count is checked against the topology", () => {
  const topology = parsePrmtop(read("ala-vac.prmtop"));
  assert.throws(
    () => parseInpcrd(read("ala-wat.inpcrd"), { expectedAtomCount: topology.atomCount }),
    /not from the same tleap run/,
  );
});

test("a NetCDF restart is refused by name", () => {
  assert.throws(() => parseInpcrd("CDF\u0001rest of a binary file"), /NetCDF/);
});

test("a rectangular box reduces to a diagonal matrix with exact zeros", () => {
  const vectors = boxVectorsFromLengthsAndAngles({
    lengths: [2.71882, 2.97181, 2.5397], angles: [90, 90, 90],
  });
  for (const index of [1, 2, 3, 5, 6, 7]) {
    assert.equal(vectors[index], 0, `off-diagonal ${index} must be exactly zero`);
  }
  assert.ok(Math.abs(vectors[0] - 2.71882) < 1e-12);
});

test("constraints and exceptions match OpenMM's counts", () => {
  const topology = parsePrmtop(read("ala-wat.prmtop"));
  const coordinates = parseInpcrd(read("ala-wat.inpcrd"),
    { expectedAtomCount: topology.atomCount });
  const reference = JSON.parse(read("ala-wat-pme.json"));
  const system = buildSystem(topology, {
    constraints: "HBonds", rigidWater: true, cutoff: 1.0, box: coordinates.box,
  });
  assert.equal(system.constraints.count, reference.constraintCount);
  assert.equal(system.exceptions.count, reference.exceptionCount);
  assert.equal(system.degreesOfFreedom, 3 * topology.atomCount - reference.constraintCount - 3);
});

test("a cutoff larger than half the box is refused", () => {
  const topology = parsePrmtop(read("ala-wat.prmtop"));
  const coordinates = parseInpcrd(read("ala-wat.inpcrd"),
    { expectedAtomCount: topology.atomCount });
  assert.throws(
    () => buildSystem(topology, { cutoff: 2.0, box: coordinates.box }),
    /does not fit in a box/,
  );
});

test("HAngles is refused rather than approximated", () => {
  const topology = parsePrmtop(read("ala-vac.prmtop"));
  assert.throws(() => buildSystem(topology, { constraints: "HAngles" }), /HAngles/);
});

test("the bonded gather reaches every term exactly once per role", () => {
  const topology = parsePrmtop(read("ala-vac.prmtop"));
  const system = buildSystem(topology, { constraints: "none", rigidWater: false, cutoff: 0 });
  const gather = buildBondedGather(system);
  const expected = system.harmonicBonds.count * 2
    + topology.angles.count * 3 + topology.dihedrals.count * 4;
  assert.equal(gather.list.length, expected);
  const seen = new Set();
  for (const word of gather.list) {
    const key = `${word >>> 30}:${(word >>> 27) & 3}:${word & 0x7ffffff}`;
    assert.ok(!seen.has(key), `duplicate gather entry ${key}`);
    seen.add(key);
  }
});

// ------------------------------------------------- the %FLAG audit (2, 3, 4) ----

/**
 * Append a section to a real topology's text, so these cases need no new
 * fixtures.
 *
 * 🔴 NO NEW FIXTURES, DELIBERATELY. A prmtop carries the parameters it uses, and
 * AMBER's are not covered by this repository's licence - README.md says none
 * are redistributed beyond `test/fixtures/`, every one of which
 * `make_fixtures.sh` builds. Committing an ff19SB or TIP4P-Ew file to test a
 * REFUSAL would add parameter sets for force fields this engine declines to
 * run. Appending the flag to a file that is already here tests the audit
 * exactly and adds nothing.
 */
const withFlag = (text, flag, format, values) =>
  `${text.trimEnd()}\n%FLAG ${flag}\n%FORMAT(${format})\n${values}\n`;

/** Rewrite one POINTERS slot in place, keeping the fixed 10I8 geometry. */
function withPointer(text, index, value) {
  const lines = text.split("\n");
  const at = lines.findIndex((line) => line.startsWith("%FLAG POINTERS"));
  assert.ok(at >= 0, "the fixture must have a POINTERS section");
  const row = at + 2 + Math.floor(index / 10);
  const column = (index % 10) * 8;
  const line = lines[row].padEnd(80);
  lines[row] = line.slice(0, column) + String(value).padStart(8) + line.slice(column + 8);
  return lines.join("\n");
}

test("an ff19SB topology is refused by name rather than run without its CMAP", () => {
  // 🔴 ff19SB IS THE MAKING-IT-RAIN NOTEBOOK'S OWN DEFAULT - its dropdown reads
  // `Force_field = "ff19SB" #@param ["ff19SB", "ff14SB"]` - and it parsed with
  // `warnings: []` and ran without its CMAP term. Measured against OpenMM 8.4:
  // -5.082244 kJ/mol on a capped tripeptide and -56.671931 on a 30-residue
  // helix, growing with the chain. ff19SB IS ff14SB with CMAP replacing the
  // backbone torsions, so that is a different force field, not a small error.
  const base = read("ala-vac.prmtop");
  assert.ok(parsePrmtop(base).atomCount === 22, "the unmodified fixture must parse");
  for (const flag of ["CMAP_COUNT", "CMAP_RESOLUTION", "CMAP_INDEX",
    // 🔴 THE NUMBERED FAMILY TOO. A real ff19SB file writes one section per
    // map - CMAP_PARAMETER_01, _02 - so an exact-match audit would catch only
    // the count and miss every parameter table.
    "CMAP_PARAMETER_01", "CMAP_PARAMETER_02"]) {
    assert.throws(
      () => parsePrmtop(withFlag(base, flag, "10I8", "       3       1")),
      (error) => /CMAP/.test(error.message) && /ff14SB/.test(error.message),
      `${flag} must be refused with a message naming the way out`,
    );
  }
});

test("4-point water is refused rather than constrained to a massless site", () => {
  // 🔴 OPC IS ff19SB's RECOMMENDED WATER AND BOTH NOTEBOOKS OFFER IT as
  // `Water_type = "TIP3P" #@param ["TIP3P", "OPC"]`. With no virtual-site
  // support, `rigidWater` constrained each water's O-EP bond: 933 of 3754
  // constraints tied a movable oxygen to a frozen massless point, and the host
  // reached 2653.6 K in 15 steps at a 298 K setpoint while the extra points
  // moved 0.00000 nm. The 3-point control on the same code path reached 412.9 K.
  const base = read("ala-vac.prmtop");
  // NUMEXTRA is pointer 31 (index 30); IFCAP is 30 and NCOPY 32, so a wrong
  // index would land on a neighbour and this test would pass for the wrong
  // reason - hence the three separate arms below.
  assert.throws(
    () => parsePrmtop(withPointer(base, 30, 3)),
    (error) => /NUMEXTRA/.test(error.message) && /3-point/.test(error.message),
    "extra points must be refused with a message naming a 3-point water",
  );
  assert.throws(() => parsePrmtop(withPointer(base, 29, 1)), /IFCAP/,
    "a solvent cap must be refused");
  assert.throws(() => parsePrmtop(withPointer(base, 31, 4)), /NCOPY/,
    "a Locally Enhanced Sampling topology must be refused");
  // And the fixture with those slots left alone must still parse, or the three
  // arms above prove nothing about the slots and only that parsing is fragile.
  assert.equal(parsePrmtop(withPointer(base, 30, 0)).atomCount, 22);
});

test("a CHAMBER conversion and the 12-6-4 ion model are refused by name", () => {
  const base = read("ala-vac.prmtop");
  // ParmEd routes on `'CTITLE' in parm.flag_list`; this parser had no such
  // test, and a file carrying CTITLE plus CHARMM_UREY_BRADLEY_COUNT parsed
  // clean - dropping Urey-Bradley terms, CHARMM impropers and a CMAP, and
  // scaling the FULL LJ table by SCNB when the file carries its own 1-4 table.
  for (const flag of ["CTITLE", "CHARMM_UREY_BRADLEY_COUNT",
    "CHARMM_CMAP_COUNT", "LENNARD_JONES_14_ACOEF"]) {
    assert.throws(() => parsePrmtop(withFlag(base, flag, "10I8", "       1")),
      /CHARMM/, `${flag} must be refused with a message naming CHARMM`);
  }
  // 12-6-4 is written by ParmEd's own documented add12_6_4 action.
  assert.throws(
    () => parsePrmtop(withFlag(base, "LENNARD_JONES_CCOEF", "5E16.8",
      "  0.00000000E+00")),
    /12-6-4/, "the 12-6-4 ion model must be refused by name");
  // IPOL is a SECTION in modern files and in no POINTERS slot, so reading
  // `pointers.IPOL` gave undefined and a polarizable file walked past the check.
  assert.throws(() => parsePrmtop(withFlag(base, "IPOL", "1I8", "       1")),
    /IPOL/, "a polarizable topology must be refused");
});

test("an unrecognised %FLAG is refused and named, not dropped in silence", () => {
  // 🔴 THIS IS THE CLASS, AND THE THREE TESTS ABOVE ARE ITS INSTANCES. The
  // parser read the twenty-six flags it names and dropped every other one with
  // no message at all - there was no known-flag list to compare against. An
  // unrecognised flag is refused rather than warned about because the asymmetry
  // decides it: a flag that changes the energy and is ignored is wrong science
  // nothing downstream reveals, while a harmless flag that causes a refusal
  // costs one message and one line in shared/amber/flags.js.
  const base = read("ala-vac.prmtop");
  assert.throws(
    () => parsePrmtop(withFlag(base, "SOMETHING_NEW", "10I8", "       1")),
    (error) => /SOMETHING_NEW/.test(error.message)
      && /flags\.js/.test(error.message),
    "an unknown flag must be named, and the message must say where to classify it",
  );
  // Metadata that really is harmless must NOT be refused, or the audit is
  // useless in practice - ParmEd writes these on any file that came from a PDB.
  for (const flag of ["RESIDUE_CHAINID", "ATOM_BFACTOR", "RADIUS_SET"]) {
    assert.equal(parsePrmtop(withFlag(base, flag, "20a4", "A   ")).atomCount, 22,
      `${flag} carries no parameters and must be ignored, not refused`);
  }
});

// ------------------------------------------- the fixtures must be buildable ----

test("every committed fixture is built by make_fixtures.sh, or listed as not", () => {
  // 🔴 README.md AND LICENSE BOTH SAID "regenerates all of them" WHILE TWO
  // SYSTEMS HAD NO RECIPE ANYWHERE. `two-body.prmtop` (434 KB) and
  // `ala-wat-big.prmtop` (471 KB) were committed with no tleap stanza, and
  // two-body is the fixture the public page hands to visitors. ala-wat-big's
  // recipe was recovered by sweeping and now rebuilds byte-identically;
  // two-body's could not be, and is recorded here as an exception so the claim
  // and the directory cannot drift apart again.
  //
  // 🔴 AND THE EARLIER FRAMING OF THIS AS A LICENCE PROBLEM WAS TOO STRONG.
  // AmberTools' parameters travel with every prmtop anyone publishes and the
  // README already says these test systems carry them. What was actually wrong
  // is the REPRODUCIBILITY claim: a reader could not rebuild two of the files
  // the documentation told them they could.
  const script = readFileSync(
    new URL("../tools/oracle/make_fixtures.sh", import.meta.url), "utf8");
  const built = new Set(
    [...script.matchAll(/^saveamberparm\s+\S+\s+(\S+)\.prmtop/gm)].map((m) => m[1]));

  /** Fixtures with no recipe, each with the reason stated. */
  const EXCEPTIONS = new Map([
    ["two-body", "recipe never recorded and not recoverable: the closest "
      + "reconstruction reaches 2523 atoms and box x 4.1123 nm against the "
      + "committed 2547 and 4.1545, matching y and z exactly - see the note in "
      + "make_fixtures.sh. It is the fixture index.html offers visitors, so it "
      + "is not overwritten with a near-miss."],
  ]);

  const committed = readdirSync(new URL("fixtures/", import.meta.url))
    .filter((name) => name.endsWith(".prmtop"))
    .map((name) => name.replace(/\.prmtop$/, ""));

  const unexplained = committed.filter((name) =>
    !built.has(name) && !EXCEPTIONS.has(name));
  assert.deepEqual(unexplained, [],
    `these fixtures are committed with no recipe and no stated reason: `
    + `${unexplained.join(", ")}. Either add a stanza to make_fixtures.sh or `
    + "add it to EXCEPTIONS here with the reason, and say so in README.md.");

  // The exceptions list must not rot either: an entry for a fixture that is now
  // built, or that no longer exists, is a claim nobody is checking.
  for (const name of EXCEPTIONS.keys()) {
    assert.ok(committed.includes(name),
      `EXCEPTIONS names ${name}, which is not in test/fixtures/ - delete the entry`);
    assert.ok(!built.has(name),
      `EXCEPTIONS names ${name}, but make_fixtures.sh now builds it - delete the entry`);
  }

  // And the README must carry the caveat rather than the absolute claim.
  const readme = readFileSync(new URL("../README.md", import.meta.url), "utf8")
    .replace(/\s+/g, " ");
  assert.ok(!/every one of which .{0,40}make_fixtures\.sh.{0,40}regenerates/.test(readme),
    "README.md still claims every fixture is regenerable");
  assert.ok(/two-body/.test(readme),
    "README.md does not name the fixture that cannot be rebuilt");
});
