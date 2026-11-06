/**
 * Constraint clusters, flattened for the device.
 *
 * 🔴 CONSTRAINTS THAT SHARE AN ATOM MUST BE SOLVED TOGETHER, AND CLUSTERS THAT
 * SHARE NOTHING CAN BE SOLVED IN ANY ORDER. That is the whole reason SHAKE is a
 * GPU-shaped problem at all: one thread per cluster, each iterating over its
 * own handful of constraints, touching no atom any other thread touches. No
 * atomics, no barriers, and the same answer every run.
 *
 * 🔴 THE INVERSE MASSES ARE STORED PER CONSTRAINT, NOT PER ATOM, AND THAT IS A
 * BINDING-COUNT DECISION RATHER THAN A MEMORY ONE. WGSL guarantees only eight
 * storage buffers per stage, and the SHAKE kernel needs positions, a reference,
 * velocities, the cluster CSR, the constraint atoms, the constraint parameters
 * and a convergence flag. Carrying the two inverse masses alongside each
 * constraint's distance removes the per-atom mass array and brings it inside
 * the limit. Duplicating a float per constraint costs kilobytes.
 */

/** Group constraints into connected components over shared atoms. */
export function buildConstraintClusters(system) {
  const { count, atomA, atomB } = system.constraints;
  const atomCount = system.topology.atomCount;
  const parent = new Int32Array(atomCount);
  for (let atom = 0; atom < atomCount; atom += 1) parent[atom] = atom;
  const find = (node) => {
    let root = node;
    while (parent[root] !== root) root = parent[root];
    while (parent[node] !== root) { const next = parent[node]; parent[node] = root; node = next; }
    return root;
  };
  for (let index = 0; index < count; index += 1) {
    const a = find(atomA[index]);
    const b = find(atomB[index]);
    if (a !== b) parent[b] = a;
  }
  /** @type {Map<number, number[]>} */
  const byRoot = new Map();
  for (let index = 0; index < count; index += 1) {
    const root = find(atomA[index]);
    const list = byRoot.get(root);
    if (list === undefined) byRoot.set(root, [index]); else list.push(index);
  }
  const clusters = [...byRoot.values()];
  let widest = 0;
  for (const cluster of clusters) widest = Math.max(widest, cluster.length);
  return { clusters, widest };
}

/**
 * The flattened form the kernels read.
 *
 * One buffer holds both CSR tables with baked offsets, because four separate
 * bindings for `constraintStart`, `constraintList`, `atomStart` and `atomList`
 * would not fit beside everything else the solver needs.
 *
 * @returns {object} clusterData (u32), constraintAtoms (u32), constraintParams (f32)
 */
export function buildClusterTables(system) {
  const { clusters } = buildConstraintClusters(system);
  const clusterCount = clusters.length;
  const masses = system.topology.masses;
  const inverseMass = (atom) => (masses[atom] > 0 ? 1 / masses[atom] : 0);

  const constraintStart = new Uint32Array(clusterCount + 1);
  const atomStart = new Uint32Array(clusterCount + 1);
  const constraintList = [];
  const atomList = [];
  let widestCluster = 0;
  for (let cluster = 0; cluster < clusterCount; cluster += 1) {
    constraintStart[cluster] = constraintList.length;
    atomStart[cluster] = atomList.length;
    const members = clusters[cluster];
    widestCluster = Math.max(widestCluster, members.length);
    const atoms = new Set();
    for (const index of members) {
      constraintList.push(index);
      atoms.add(system.constraints.atomA[index]);
      atoms.add(system.constraints.atomB[index]);
    }
    // Sorted, so the velocity rewrite at the end of SHAKE visits a cluster's
    // atoms in a fixed order on every device.
    for (const atom of [...atoms].sort((left, right) => left - right)) atomList.push(atom);
  }
  constraintStart[clusterCount] = constraintList.length;
  atomStart[clusterCount] = atomList.length;

  const layout = {
    clusterCount,
    constraintStartBase: 0,
    constraintListBase: clusterCount + 1,
    atomStartBase: clusterCount + 1 + constraintList.length,
    atomListBase: clusterCount + 1 + constraintList.length + clusterCount + 1,
    widestCluster,
  };
  const clusterData = new Uint32Array(layout.atomListBase + atomList.length);
  clusterData.set(constraintStart, layout.constraintStartBase);
  clusterData.set(Uint32Array.from(constraintList), layout.constraintListBase);
  clusterData.set(atomStart, layout.atomStartBase);
  clusterData.set(Uint32Array.from(atomList), layout.atomListBase);

  const constraintAtoms = new Uint32Array(system.constraints.count * 2);
  const constraintParams = new Float32Array(system.constraints.count * 4);
  for (let index = 0; index < system.constraints.count; index += 1) {
    const a = system.constraints.atomA[index];
    const b = system.constraints.atomB[index];
    constraintAtoms[index * 2] = a;
    constraintAtoms[index * 2 + 1] = b;
    constraintParams[index * 4] = system.constraints.distance[index];
    constraintParams[index * 4 + 1] = inverseMass(a);
    constraintParams[index * 4 + 2] = inverseMass(b);
    constraintParams[index * 4 + 3] = 0;
  }
  return { layout, clusterData, constraintAtoms, constraintParams };
}

/**
 * The tightest constraint tolerance f32 coordinates can actually reach.
 *
 * 🔴 1e-6 IS BELOW THE f32 FLOOR AND ASKING FOR IT MAKES THE SOLVER SPIN TO ITS
 * ITERATION CAP AND REPORT FAILURE ON A CORRECT RESULT. The residual is
 * `|d|^2/L^2 - 1`, and `d` is a difference of two coordinates that span the
 * whole box - so its absolute f32 error is about `span * eps` whatever `d`
 * itself is, and the relative error in the residual is twice that divided by
 * the constraint length. Measured on the solvated fixture: a 2.82 nm span and a
 * 0.09572 nm O-H constraint give 3.5e-6, and the solver reached 2-3e-6 and then
 * stopped improving - against a requested 1e-6, so all 394 clusters ran 150
 * iterations and both failure flags were set.
 *
 * This is also why OpenMM's default is 1e-5 rather than something tighter: it
 * is what single-precision coordinates support. A mixed-precision code keeps
 * positions in f64 and can ask for more.
 *
 * @param {object} system from buildSystem
 * @param {ArrayLike<number>} positions used only for the coordinate span
 */
export function constraintToleranceFloor(system, positions) {
  if (system.constraints.count === 0) return 0;
  const epsilon = 2 ** -24;
  let span = 0;
  for (const value of positions) span = Math.max(span, Math.abs(value));
  let shortest = Infinity;
  for (let index = 0; index < system.constraints.count; index += 1) {
    shortest = Math.min(shortest, system.constraints.distance[index]);
  }
  // Two coordinates differenced, squared (hence the factor of two), with a
  // factor of two of headroom so the solver converges rather than grazing it.
  return (4 * span * epsilon) / shortest;
}
