/**
 * The two kernels a Monte Carlo barostat needs: molecular centroids, and
 * scaling about them.
 *
 * 🔴 A BAROSTAT SCALES MOLECULAR CENTRES OF MASS, NEVER ATOMS. Scaling atoms
 * individually stretches every bond and every constrained distance in the
 * system, so the trial energy is dominated by that strain, every move is
 * rejected, and the box never equilibrates - it simply stays at whatever
 * density `solvateBox` guessed. This is the single thing to get right in a
 * barostat, and it is why buildSystem derives a molecule partition from the
 * bond graph rather than trusting ATOMS_PER_MOLECULE.
 *
 * 🔴 TWO PASSES, NOT ONE. One thread per molecule doing both the centroid and
 * the scaling would be correct and would serialise a 5000-atom protein onto a
 * single lane while 20000 water threads idle. The centroid pass still has that
 * imbalance, but it is three loads and an add per atom; the scaling pass, which
 * is the one that touches every coordinate, is one thread per atom.
 */
import { pipelineCacheFor } from "../runtime/pipeline-cache.js";
import { dispatchGrid } from "../runtime/execution.js";

export const BAROSTAT_WORKGROUP = 64;

/** Pass 1: the mass-weighted centroid of every molecule. */
export function centroidShader(moleculeCount, gridWidth) {
  return `
const MOLECULE_COUNT: u32 = ${moleculeCount}u;
const GRID_WIDTH: u32 = ${gridWidth}u;
@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read> masses: array<f32>;
@group(0) @binding(2) var<storage, read> moleculeStart: array<u32>;
@group(0) @binding(3) var<storage, read> moleculeAtoms: array<u32>;
@group(0) @binding(4) var<storage, read_write> centroids: array<f32>;

@compute @workgroup_size(${BAROSTAT_WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let molecule = id.x + id.y * GRID_WIDTH * ${BAROSTAT_WORKGROUP}u;
  if (molecule >= MOLECULE_COUNT) { return; }
  let begin = moleculeStart[molecule];
  let end = moleculeStart[molecule + 1u];
  var weighted = vec3<f32>(0.0, 0.0, 0.0);
  var total = 0.0;
  for (var slot = begin; slot < end; slot = slot + 1u) {
    let atom = moleculeAtoms[slot];
    let mass = masses[atom];
    weighted = weighted + mass * vec3<f32>(
      positions[atom * 3u], positions[atom * 3u + 1u], positions[atom * 3u + 2u]);
    total = total + mass;
  }
  // 🔴 A MOLECULE OF ONLY MASSLESS PARTICLES HAS NO CENTRE OF MASS. AMBER
  // topologies carry zero-mass extra points, and a lone one would be a
  // molecule whose total mass is zero - dividing there puts a NaN into the
  // centroid and the scaling pass spreads it to every atom of that molecule.
  // Falling back to the geometric centre keeps such a particle moving with the
  // box, which is what scaling it at all is for.
  if (total > 0.0) {
    weighted = weighted / total;
  } else {
    let count = f32(end - begin);
    var sum = vec3<f32>(0.0, 0.0, 0.0);
    for (var slot = begin; slot < end; slot = slot + 1u) {
      let atom = moleculeAtoms[slot];
      sum = sum + vec3<f32>(
        positions[atom * 3u], positions[atom * 3u + 1u], positions[atom * 3u + 2u]);
    }
    weighted = sum / max(count, 1.0);
  }
  centroids[molecule * 3u] = weighted.x;
  centroids[molecule * 3u + 1u] = weighted.y;
  centroids[molecule * 3u + 2u] = weighted.z;
}
`;
}

/**
 * Pass 2: move every atom by its molecule's centroid displacement.
 *
 * 🔴 WRITTEN AS `position + centroid * (scale - 1)`, NOT
 * `centroid * scale + (position - centroid)`. The two are equal in exact
 * arithmetic; the second subtracts and re-adds the centroid, so in f32 it loses
 * bits of the intramolecular geometry on every accepted move - which
 * accumulates into a slow drift of every bond length over a long NPT run,
 * exactly where it is hardest to notice.
 */
export function scaleShader(atomCount, gridWidth) {
  return `
const ATOM_COUNT: u32 = ${atomCount}u;
const GRID_WIDTH: u32 = ${gridWidth}u;
struct Scale { factor: f32, reserved0: f32, reserved1: f32, reserved2: f32 };
@group(0) @binding(0) var<uniform> scaling: Scale;
@group(0) @binding(1) var<storage, read_write> positions: array<f32>;
@group(0) @binding(2) var<storage, read> centroids: array<f32>;
@group(0) @binding(3) var<storage, read> moleculeOfAtom: array<u32>;

@compute @workgroup_size(${BAROSTAT_WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let atom = id.x + id.y * GRID_WIDTH * ${BAROSTAT_WORKGROUP}u;
  if (atom >= ATOM_COUNT) { return; }
  let molecule = moleculeOfAtom[atom];
  let shift = (scaling.factor - 1.0) * vec3<f32>(
    centroids[molecule * 3u], centroids[molecule * 3u + 1u], centroids[molecule * 3u + 2u]);
  positions[atom * 3u] = positions[atom * 3u] + shift.x;
  positions[atom * 3u + 1u] = positions[atom * 3u + 1u] + shift.y;
  positions[atom * 3u + 2u] = positions[atom * 3u + 2u] + shift.z;
}
`;
}

/** Encode both passes for one trial scaling. */
export function encodeVolumeScale(device, encoder, resources, atomCount, moleculeCount) {
  const maxPerDimension = device.limits.maxComputeWorkgroupsPerDimension;
  const cache = pipelineCacheFor(device);
  const dispatchFor = (items) => {
    const workgroups = Math.ceil(items / BAROSTAT_WORKGROUP);
    const [x, y, z] = dispatchGrid(workgroups, maxPerDimension);
    return { x, y, z, gridWidth: x === workgroups ? workgroups : maxPerDimension };
  };
  const molecules = dispatchFor(moleculeCount);
  const atoms = dispatchFor(atomCount);
  const run = (key, code, bindings, dispatch) => {
    const pipeline = cache.get(key, code);
    const pass = encoder.beginComputePass({ label: key });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: bindings.map((allocation, binding) => ({
        binding, resource: { buffer: allocation.buffer },
      })),
    }));
    pass.dispatchWorkgroups(dispatch.x, dispatch.y, dispatch.z);
    pass.end();
  };
  run(`barostat:centroid:${moleculeCount}:${molecules.gridWidth}`,
    centroidShader(moleculeCount, molecules.gridWidth),
    [resources.positions, resources.masses, resources.moleculeStart,
      resources.moleculeAtoms, resources.centroids], molecules);
  run(`barostat:scale:${atomCount}:${atoms.gridWidth}`,
    scaleShader(atomCount, atoms.gridWidth),
    [resources.scaling, resources.positions, resources.centroids,
      resources.moleculeOfAtom], atoms);
}

/**
 * The Metropolis test for a volume move.
 *
 * 🔴 THE `N ln(V'/V)` TERM IS NOT OPTIONAL AND IT IS NOT ENTHALPY. It is the
 * volume entropy of the N molecules - the Jacobian of the coordinate scaling -
 * and leaving it out biases every run towards smaller volumes, giving a density
 * that is wrong by a few percent in a way that looks like a force-field
 * problem. It counts MOLECULES, not atoms, because molecules are what the move
 * scales.
 *
 * @param {object} move {deltaEnergy, deltaVolume, volumeRatio} kJ/mol and nm^3
 * @param {number} pressure in kJ/(mol nm^3)
 * @param {number} kT kJ/mol
 * @param {number} moleculeCount
 */
export function metropolisWeight(move, pressure, kT, moleculeCount) {
  return move.deltaEnergy
    + pressure * move.deltaVolume
    - moleculeCount * kT * Math.log(move.volumeRatio);
}
