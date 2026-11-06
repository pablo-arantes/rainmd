/**
 * Dump a parsed topology's invariants as JSON, for comparison against ParmEd.
 *
 * 🔴 THE POINT IS AN INDEPENDENT READER, NOT A ROUND TRIP. Checking this
 * parser by re-serialising what it read and comparing would compare the parser
 * with itself and agree perfectly however wrong it is. ParmEd reads the same
 * bytes with no shared code, so a sum that matches is evidence and a sum that
 * does not names the section that is wrong.
 *
 * Sums rather than whole arrays because a sum over a few hundred thousand
 * values is one number that changes if any of them is misplaced, and because
 * an index sum catches a shift that a value sum does not.
 *
 *   node tools/dump-topology.mjs ala-vac > /tmp/js-ala-vac.json
 */
import { readFileSync } from "node:fs";
import { parsePrmtop } from "../shared/amber/prmtop.js";
import { parseInpcrd, boxVectorsFromLengthsAndAngles } from "../shared/amber/inpcrd.js";

const name = process.argv[2];
if (name === undefined) {
  console.error("usage: node tools/dump-topology.mjs <fixture-name>");
  process.exit(2);
}
const base = new URL(`../test/fixtures/${name}`, import.meta.url);
const top = parsePrmtop(readFileSync(new URL(`${base}.prmtop`), "utf8"));
const crd = parseInpcrd(readFileSync(new URL(`${base}.inpcrd`), "utf8"),
  { expectedAtomCount: top.atomCount });

const sum = (array) => {
  let total = 0;
  for (const value of array) total += value;
  return total;
};
const absSum = (array) => {
  let total = 0;
  for (const value of array) total += Math.abs(value);
  return total;
};

process.stdout.write(JSON.stringify({
  atomCount: top.atomCount,
  residueCount: top.residueNames.length,
  typeCount: top.typeCount,
  moleculeCount: top.moleculeCount,
  chargeSum: sum(top.charges),
  chargeAbsSum: absSum(top.charges),
  massSum: sum(top.masses),
  bondCount: top.bonds.count,
  bondKSum: sum(top.bonds.k),
  bondR0Sum: sum(top.bonds.r0),
  bondAtomIndexSum: sum(top.bonds.atomA) + sum(top.bonds.atomB),
  angleCount: top.angles.count,
  angleKSum: sum(top.angles.k),
  angleTheta0Sum: sum(top.angles.theta0),
  angleAtomIndexSum: sum(top.angles.atomA) + sum(top.angles.atomB) + sum(top.angles.atomC),
  dihedralCount: top.dihedrals.count,
  dihedralKSum: sum(top.dihedrals.k),
  dihedralPhaseSum: sum(top.dihedrals.phase),
  dihedralPeriodicitySum: sum(top.dihedrals.periodicity),
  dihedralAtomIndexSum: sum(top.dihedrals.atomA) + sum(top.dihedrals.atomB)
    + sum(top.dihedrals.atomC) + sum(top.dihedrals.atomD),
  improperCount: sum(top.dihedrals.isImproper),
  pair14Count: top.dihedrals.pairs14.count,
  pair14IndexSum: sum(top.dihedrals.pairs14.atomA) + sum(top.dihedrals.pairs14.atomB),
  exclusionPairCount: top.exclusions.pairCount,
  exclusionPartnerSum: sum(top.exclusions.partners),
  ljASum: sum(top.ljA),
  ljBSum: sum(top.ljB),
  positionSum: sum(crd.positions),
  box: crd.box,
  boxVectors: crd.box === null ? null : Array.from(boxVectorsFromLengthsAndAngles(crd.box)),
  hasVelocities: crd.velocities !== null,
  warnings: top.warnings,
}, null, 2));
