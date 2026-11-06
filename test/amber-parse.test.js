import assert from "node:assert/strict";
import { test } from "node:test";
import { readFileSync } from "node:fs";
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
