/**
 * Electrostatics and Lennard-Jones on a SHEARED cell, against oracles that
 * share no code with them.
 *
 * 🔴 THIS EXISTS BECAUSE FIXING THE PME TRANSPOSE LEFT A RESIDUAL AGAINST
 * OPENMM, AND THE RESIDUAL TURNED OUT TO BE OPENMM'S. On `ala-oct` the
 * direct-space LJ differed by 3.07e-5 and the total Coulomb by 1.36e-4,
 * grid-independently, and the obvious reading was a second defect here. It is
 * not: an independent brute-force LJ sum and a complete independent Ewald sum
 * both land on this port and leave OpenMM's Reference platform as the outlier.
 *
 * 🔴 AND THE RECTANGULAR ARM IS WHAT MAKES THAT READABLE. There, RainMD and
 * OpenMM agree with EACH OTHER to 1.2e-8 on a 24^3 grid while both sit 6e-6 from
 * the exact sum - the PME grid floor, acting identically on two implementations
 * of the same algorithm. Without that control the octahedral numbers could be
 * read as this port drifting rather than as OpenMM differing.
 *
 * Run with `npm run check:triclinic`. It is O(N^2) over 125 lattice images and
 * takes tens of seconds, which is why it is its own command rather than part of
 * `npm test`.
 */
// A COMPLETE independent Ewald electrostatics sum, plus a brute-force LJ sum.
//
//   E = real(erfc, non-excluded pairs within cutoff, brute-force minimal image)
//     + reciprocal (exact lattice sum, row convention)
//     + self  (-ONE_4PI_EPS0 * alpha/sqrt(pi) * sum q^2)
//     + net-charge correction
//     - exclusion corrections (the reciprocal sum counted full 1/r for every
//       excluded pair; erf(ar)/r is what has to come back off)
//     + the 1-4 exception terms, at their own scaled charge products
//
// Shares no code with cpu/forces/nonbonded.js or cpu/forces/pme.js. Validated
// on the rectangular fixture first, where RainMD and OpenMM agree to 1.2e-8.
const R = new URL("..", import.meta.url).pathname.replace(/\/$/, "");
const { readFileSync } = await import("node:fs");
const { parsePrmtop } = await import(`${R}/shared/amber/prmtop.js`);
const { parseInpcrd } = await import(`${R}/shared/amber/inpcrd.js`);
const { buildSystem } = await import(`${R}/shared/topology/system.js`);
const { potentialEnergy } = await import(`${R}/cpu/energy.js`);
const { erf, erfc } = await import(`${R}/shared/erf.js`);
const { ONE_4PI_EPS0 } = await import(`${R}/shared/units.js`);
const { reciprocalBoxVectors } = await import(`${R}/cpu/forces/geometry.js`);

const minimalImage = (dx, dy, dz, b, reach) => {
  let best = Infinity;
  for (let ia = -reach; ia <= reach; ia += 1)
  for (let ib = -reach; ib <= reach; ib += 1)
  for (let ic = -reach; ic <= reach; ic += 1) {
    const x = dx + ia*b[0] + ib*b[3] + ic*b[6];
    const y = dy + ib*b[4] + ic*b[7];
    const z = dz + ic*b[8];
    const d2 = x*x + y*y + z*z;
    if (d2 < best) best = d2;
  }
  return Math.sqrt(best);
};

function exactReciprocal(system, positions, alpha, nmax) {
  const b = system.boxVectors, recip = reciprocalBoxVectors(b);
  const V = Math.abs(b[0]*b[4]*b[8]);
  const q = system.topology.charges, n = system.topology.atomCount;
  const factor = Math.PI*Math.PI/(alpha*alpha);
  let sum = 0;
  for (let na = -nmax; na <= nmax; na += 1)
  for (let nb = -nmax; nb <= nmax; nb += 1)
  for (let nc = -nmax; nc <= nmax; nc += 1) {
    if (na === 0 && nb === 0 && nc === 0) continue;
    const mx = na*recip[0] + nb*recip[1] + nc*recip[2];
    const my = na*recip[3] + nb*recip[4] + nc*recip[5];
    const mz = na*recip[6] + nb*recip[7] + nc*recip[8];
    const m2 = mx*mx + my*my + mz*mz;
    const weight = Math.exp(-factor*m2)/m2;
    if (weight < 1e-18) continue;
    let re = 0, im = 0;
    for (let j = 0; j < n; j += 1) {
      const phase = 2*Math.PI*(mx*positions[j*3] + my*positions[j*3+1] + mz*positions[j*3+2]);
      re += q[j]*Math.cos(phase); im += q[j]*Math.sin(phase);
    }
    sum += weight*(re*re + im*im);
  }
  return (1/(2*Math.PI*V))*ONE_4PI_EPS0*sum;
}

function independentCoulomb(system, positions, cutoff, alpha) {
  const t = system.topology, n = t.atomCount, b = system.boxVectors, q = t.charges;
  const exc = system.exceptions;
  const excluded = new Set();
  for (let k = 0; k < exc.count; k += 1) {
    const a = exc.atomA[k], c = exc.atomB[k];
    excluded.add(a < c ? a*n + c : c*n + a);
  }
  // --- real space -------------------------------------------------------
  let real = 0;
  const rc = cutoff;
  for (let i = 0; i < n; i += 1) {
    for (let j = i + 1; j < n; j += 1) {
      if (excluded.has(i*n + j)) continue;
      const r = minimalImage(positions[i*3]-positions[j*3],
        positions[i*3+1]-positions[j*3+1], positions[i*3+2]-positions[j*3+2], b, 2);
      if (r > rc) continue;
      real += ONE_4PI_EPS0*q[i]*q[j]*erfc(alpha*r)/r;
    }
  }
  // --- reciprocal, self, net charge -------------------------------------
  const reciprocal = exactReciprocal(system, positions, alpha, 8);
  let q2 = 0, qnet = 0;
  for (let i = 0; i < n; i += 1) { q2 += q[i]*q[i]; qnet += q[i]; }
  const self = -ONE_4PI_EPS0*(alpha/Math.sqrt(Math.PI))*q2;
  const V = Math.abs(b[0]*b[4]*b[8]);
  const netCharge = -ONE_4PI_EPS0*(Math.PI/(2*alpha*alpha*V))*qnet*qnet;
  // --- exclusion corrections and the 1-4 exceptions ---------------------
  let correction = 0, exception = 0;
  for (let k = 0; k < exc.count; k += 1) {
    const i = exc.atomA[k], j = exc.atomB[k];
    const r = minimalImage(positions[i*3]-positions[j*3],
      positions[i*3+1]-positions[j*3+1], positions[i*3+2]-positions[j*3+2], b, 1);
    correction -= ONE_4PI_EPS0*q[i]*q[j]*erf(alpha*r)/r;
    exception += ONE_4PI_EPS0*exc.chargeProduct[k]/r;
  }
  return { real, reciprocal, self, netCharge, correction, exception,
    total: real + reciprocal + self + netCharge + correction + exception };
}

function independentLJ(system, positions, cutoff) {
  const t = system.topology, n = t.atomCount, b = system.boxVectors;
  const exc = system.exceptions;
  const excluded = new Set();
  for (let k = 0; k < exc.count; k += 1) {
    const a = exc.atomA[k], c = exc.atomB[k];
    excluded.add(a < c ? a*n + c : c*n + a);
  }
  const ljA = t.ljA, ljB = t.ljB, types = t.atomTypes, nt = t.typeCount;
  let direct = 0;
  for (let i = 0; i < n; i += 1) {
    for (let j = i + 1; j < n; j += 1) {
      if (excluded.has(i*n + j)) continue;
      const r = minimalImage(positions[i*3]-positions[j*3],
        positions[i*3+1]-positions[j*3+1], positions[i*3+2]-positions[j*3+2], b, 2);
      if (r > cutoff) continue;
      const index = types[i]*nt + types[j];
      const inv6 = 1/(r*r*r*r*r*r);
      direct += ljA[index]*inv6*inv6 - ljB[index]*inv6;
    }
  }
  let exception = 0;
  for (let k = 0; k < exc.count; k += 1) {
    const i = exc.atomA[k], j = exc.atomB[k];
    const r = minimalImage(positions[i*3]-positions[j*3],
      positions[i*3+1]-positions[j*3+1], positions[i*3+2]-positions[j*3+2], b, 1);
    const inv6 = 1/(r*r*r*r*r*r);
    exception += exc.ljA[k]*inv6*inv6 - exc.ljB[k]*inv6;
  }
  return direct + exception;
}

const alpha = 2.6282608939112584;
const OPENMM = { "ala-wat": -13652.389550, "ala-oct": -15593.949390 };
const OPENMM_LJ = { "ala-wat": 1794.522906, "ala-oct": 2673.647583 };
// 🔴 THE BAR IS THE ORACLE'S OWN CONVERGENCE, NOT A CHOSEN NUMBER. The exact
// k-space sum converges to about 1.6e-6 between nmax 8 and 11, and this port
// lands 5.3e-8 (octahedral) and 1.3e-7 (rectangular) from it at a 48^3 grid, so
// 1e-5 is well above the floor and well below OpenMM's 1.4e-4 departure.
const BAR = 1e-5;
let failures = 0;
for (const name of ["ala-wat", "ala-oct"]) {
  const t = parsePrmtop(readFileSync(`${R}/test/fixtures/${name}.prmtop`, "utf8"));
  const c = parseInpcrd(readFileSync(`${R}/test/fixtures/${name}.inpcrd`, "utf8"),
    { expectedAtomCount: t.atomCount });
  const s = buildSystem(t, { nonbonded: "PME", cutoff: 1.0,
    constraints: "HBonds", rigidWater: true, box: c.box });
  const mine = independentCoulomb(s, c.positions, 1.0, alpha);
  const ours = potentialEnergy(s, c.positions,
    { method: "PME", alpha, grid: [48,48,48] }).energy.coulomb;
  const rel = (a, bb) => Math.abs(a - bb)/Math.abs(bb);
  console.log(`${name}:`);
  console.log(`   independent total ${mine.total.toFixed(6)}`);
  console.log(`     real ${mine.real.toFixed(3)}  recip ${mine.reciprocal.toFixed(3)}  self ${mine.self.toFixed(3)}`
    + `  net ${mine.netCharge.toExponential(2)}  exclCorr ${mine.correction.toFixed(3)}  excep ${mine.exception.toFixed(3)}`);
  console.log(`   RainMD coulomb    ${ours.toFixed(6)}   rel to mine ${rel(ours, mine.total).toExponential(2)}`);
  console.log(`   OpenMM coulombOnly ${OPENMM[name].toFixed(6)}  rel to mine ${rel(OPENMM[name], mine.total).toExponential(2)}`);
  const lj = independentLJ(s, c.positions, 1.0);
  const oursLJ = potentialEnergy(s, c.positions,
    { method: "PME", alpha, grid: [48,48,48] }).energy.lennardJones;
  console.log(`   independent LJ    ${lj.toFixed(6)}`);
  console.log(`   RainMD LJ         ${oursLJ.toFixed(6)}   rel to mine ${rel(oursLJ, lj).toExponential(2)}`);
  console.log(`   OpenMM ljOnly     ${OPENMM_LJ[name].toFixed(6)}   rel to mine ${rel(OPENMM_LJ[name], lj).toExponential(2)}`);
  for (const [what, got, want] of [["coulomb", ours, mine.total], ["lennard-jones", oursLJ, lj]]) {
    const r = rel(got, want);
    if (r > BAR) {
      console.log(`   FAILED: ${name} ${what} is ${r.toExponential(2)} from the `
        + `independent sum, over the ${BAR} floor`);
      failures += 1;
    }
  }
}
console.log(failures === 0
  ? "\nboth terms match oracles that share no code with them, on a sheared cell "
    + "as well as a rectangular one"
  : `\n${failures} comparison(s) failed`);
if (failures > 0) process.exitCode = 1;
