import { buildConstraintClusters } from "../../shared/topology/clusters.js";

export { buildConstraintClusters };
/**
 * Distance constraints: SETTLE for rigid water, SHAKE/RATTLE for the rest.
 *
 * 🔴 WITHOUT THESE THERE IS NO 2 fs TIMESTEP. An O-H bond vibrates with a
 * period near 10 fs, so a 2 fs step samples it four times a period and the
 * integrator pumps energy into it until the water comes apart. The notebooks
 * ask for `constraints=HBonds` and a 2 fs step together, and the two are one
 * decision: removing the constraint while keeping the step is not a slower
 * simulation, it is an exploding one.
 *
 * The cluster partition itself lives in shared/topology/clusters.js, because
 * the device solver needs exactly the same grouping and the layering here is
 * `shared <- cpu <- webgpu`. It is re-exported so callers of this module do
 * not have to know that.
 *
 * 🔴 AMBER'S TIP3P IS THREE CONSTRAINTS, NOT TWO PLUS AN ANGLE. The topology
 * carries O-H, O-H and an H-H "bond" whose length (0.15136 nm) encodes the
 * 104.52 degree angle, and there is no angle term at all. So a water is a
 * three-constraint rigid cluster and SETTLE applies directly - see
 * findWaterClusters, which identifies it by that exact shape rather than by the
 * residue name, because a residue called WAT with four constraints is a
 * four-site model and SETTLE's closed form does not apply to it.
 */

/**
 * Which clusters are three-constraint rigid waters, in SETTLE's canonical order.
 *
 * Returns, for each such cluster, the central (heavy) atom and its two
 * partners, plus the three distances. A cluster that is three constraints over
 * three atoms where one atom appears in all three is a triangle; that is the
 * shape SETTLE solves.
 */
export function findWaterClusters(system, clusters) {
  const { atomA, atomB, distance } = system.constraints;
  const water = [];
  const other = [];
  for (const cluster of clusters) {
    if (cluster.length !== 3) { other.push(cluster); continue; }
    const degree = new Map();
    for (const index of cluster) {
      for (const atom of [atomA[index], atomB[index]]) {
        degree.set(atom, (degree.get(atom) ?? 0) + 1);
      }
    }
    if (degree.size !== 3) { other.push(cluster); continue; }
    // In a triangle every atom has degree 2, so the apex cannot be found from
    // the graph and is chosen by MASS: SETTLE's closed form wants the heavy
    // atom there.
    let centre = -1;
    let heaviest = -Infinity;
    for (const atom of degree.keys()) {
      if (system.topology.masses[atom] > heaviest) {
        heaviest = system.topology.masses[atom];
        centre = atom;
      }
    }
    const partners = [...degree.keys()].filter((atom) => atom !== centre);
    if (partners.length !== 2) { other.push(cluster); continue; }
    const lengthOf = (p, q) => {
      for (const index of cluster) {
        if ((atomA[index] === p && atomB[index] === q)
          || (atomA[index] === q && atomB[index] === p)) return distance[index];
      }
      return null;
    };
    const d1 = lengthOf(centre, partners[0]);
    const d2 = lengthOf(centre, partners[1]);
    const d3 = lengthOf(partners[0], partners[1]);
    if (d1 === null || d2 === null || d3 === null) { other.push(cluster); continue; }
    water.push({ centre, partners, distances: [d1, d2, d3], constraints: cluster });
  }
  return { water, other };
}

/**
 * Iterative constraint projection on positions (SHAKE).
 *
 * 🔴 THE TOLERANCE IS ON THE RELATIVE LENGTH ERROR AND MUST BE TIGHT. OpenMM's
 * default constraint tolerance is 1e-5 of the constrained distance, and the
 * notebooks set 1e-6. Loosening it does not make the simulation a little less
 * accurate: an unconverged constraint leaves a residual force that the
 * thermostat reads as heat, so the system runs warm by an amount nothing in the
 * output explains.
 *
 * 🔴 AND IT CAN FAIL TO CONVERGE, WHICH MUST BE REPORTED RATHER THAN IGNORED.
 * A cluster whose atoms have been pushed through each other - after a bad
 * minimisation, or a timestep far too large - has no solution, and iterating
 * forever or silently stopping both produce a trajectory that is not physics.
 */
export function applyPositionConstraints(system, positions, reference, options = {}) {
  const tolerance = options.tolerance ?? 1e-6;
  const maxIterations = options.maxIterations ?? 150;
  const { count, atomA, atomB, distance } = system.constraints;
  if (count === 0) return { iterations: 0, converged: true };
  const masses = system.topology.masses;
  const inverseMass = (atom) => (masses[atom] > 0 ? 1 / masses[atom] : 0);

  for (let iteration = 0; iteration < maxIterations; iteration += 1) {
    let worst = 0;
    for (let index = 0; index < count; index += 1) {
      const a = atomA[index];
      const b = atomB[index];
      const target = distance[index];
      const dx = positions[a * 3] - positions[b * 3];
      const dy = positions[a * 3 + 1] - positions[b * 3 + 1];
      const dz = positions[a * 3 + 2] - positions[b * 3 + 2];
      const current = dx * dx + dy * dy + dz * dz;
      const error = current - target * target;
      const relative = Math.abs(error) / (target * target);
      if (relative > worst) worst = relative;
      if (relative <= tolerance) continue;
      // The reference geometry is the pre-move one: SHAKE projects along the
      // OLD bond direction, which is what makes the correction a rigid-body
      // motion to first order rather than a slide along the new direction.
      const rx = reference[a * 3] - reference[b * 3];
      const ry = reference[a * 3 + 1] - reference[b * 3 + 1];
      const rz = reference[a * 3 + 2] - reference[b * 3 + 2];
      const projection = dx * rx + dy * ry + dz * rz;
      const inverseA = inverseMass(a);
      const inverseB = inverseMass(b);
      const denominator = 2 * (inverseA + inverseB) * projection;
      if (Math.abs(denominator) < 1e-20) continue;
      const lambda = -error / denominator;
      positions[a * 3] += lambda * inverseA * rx;
      positions[a * 3 + 1] += lambda * inverseA * ry;
      positions[a * 3 + 2] += lambda * inverseA * rz;
      positions[b * 3] -= lambda * inverseB * rx;
      positions[b * 3 + 1] -= lambda * inverseB * ry;
      positions[b * 3 + 2] -= lambda * inverseB * rz;
    }
    if (worst <= tolerance) return { iterations: iteration + 1, converged: true, worst };
  }
  return {
    iterations: maxIterations,
    converged: false,
    message:
      `constraints did not converge in ${maxIterations} iterations. The usual `
      + "cause is a timestep too large for this system, or starting from a "
      + "structure that was never minimised.",
  };
}

/**
 * Project velocities onto the constraint manifold (RATTLE).
 *
 * 🔴 CONSTRAINING POSITIONS WITHOUT CONSTRAINING VELOCITIES IS A COMMON AND
 * EXPENSIVE HALF-MEASURE. The positions stay rigid and the velocities keep a
 * component along each constrained bond, which contributes to the kinetic
 * energy and therefore to the reported temperature - so the thermostat cools
 * the real degrees of freedom to compensate and the simulation is quietly cold.
 * This is also why degreesOfFreedom subtracts the constraint count.
 */
export function applyVelocityConstraints(system, positions, velocities, options = {}) {
  const tolerance = options.tolerance ?? 1e-6;
  const maxIterations = options.maxIterations ?? 150;
  const { count, atomA, atomB, distance } = system.constraints;
  if (count === 0) return { iterations: 0, converged: true };
  const masses = system.topology.masses;
  const inverseMass = (atom) => (masses[atom] > 0 ? 1 / masses[atom] : 0);

  for (let iteration = 0; iteration < maxIterations; iteration += 1) {
    let worst = 0;
    for (let index = 0; index < count; index += 1) {
      const a = atomA[index];
      const b = atomB[index];
      const rx = positions[a * 3] - positions[b * 3];
      const ry = positions[a * 3 + 1] - positions[b * 3 + 1];
      const rz = positions[a * 3 + 2] - positions[b * 3 + 2];
      const vx = velocities[a * 3] - velocities[b * 3];
      const vy = velocities[a * 3 + 1] - velocities[b * 3 + 1];
      const vz = velocities[a * 3 + 2] - velocities[b * 3 + 2];
      const projection = rx * vx + ry * vy + rz * vz;
      const target = distance[index];
      const relative = Math.abs(projection) / (target * target);
      if (relative > worst) worst = relative;
      if (relative <= tolerance) continue;
      const inverseA = inverseMass(a);
      const inverseB = inverseMass(b);
      const denominator = (inverseA + inverseB) * (rx * rx + ry * ry + rz * rz);
      if (denominator < 1e-20) continue;
      const lambda = -projection / denominator;
      velocities[a * 3] += lambda * inverseA * rx;
      velocities[a * 3 + 1] += lambda * inverseA * ry;
      velocities[a * 3 + 2] += lambda * inverseA * rz;
      velocities[b * 3] -= lambda * inverseB * rx;
      velocities[b * 3 + 1] -= lambda * inverseB * ry;
      velocities[b * 3 + 2] -= lambda * inverseB * rz;
    }
    if (worst <= tolerance) return { iterations: iteration + 1, converged: true, worst };
  }
  return { iterations: maxIterations, converged: false };
}
