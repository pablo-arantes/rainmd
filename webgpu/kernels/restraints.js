/**
 * Positional restraints on the device, for the equilibration stage.
 *
 * 🔴 ONE THREAD PER RESTRAINED ATOM, AND NO ATOMICS. A restrained atom appears
 * exactly once in the list, so its force is a plain read-modify-write that no
 * other invocation touches - which keeps the force path deterministic, the
 * property `docs/ARCHITECTURE.md` records as "determinism by construction".
 * An `atomicAdd` here would be correct and would also reorder the sum.
 *
 * 🔴 THE CONVENTION IS `k * d^2`, NOT `1/2 k d^2` - see
 * shared/topology/restraints.js. The notebooks write
 * `CustomExternalForce('k*periodicdistance(...)^2')`, so the force is `-2k d`.
 * Reading it as a half would make every restraint half as stiff as the slider
 * says, which is the kind of error that still produces a plausible trajectory.
 *
 * 🔴 AND THE ENERGY GOES IN ITS OWN BUFFER, INDEXED BY SLOT RATHER THAN BY
 * ATOM. Folding it into the bonded per-atom array would be one fewer buffer and
 * would break the per-term comparison this repository is built on: the bonded
 * total is checked against OpenMM's HarmonicBondForce group, which does not
 * contain a restraint. The buffer is as long as the restrained list, not as the
 * system, so it costs nothing on a solvated box where the solute is 2% of it.
 */
import { pipelineCacheFor } from "../runtime/pipeline-cache.js";
import { dispatchGrid } from "../runtime/execution.js";
import { PARAMS_STRUCT, GEOMETRY_FUNCTIONS } from "./wgsl-common.js";

export const RESTRAINT_WORKGROUP = 64;

export function restraintShader(count, gridWidth) {
  return `${PARAMS_STRUCT}${GEOMETRY_FUNCTIONS}
const RESTRAINT_COUNT: u32 = ${count}u;
const GRID_WIDTH: u32 = ${gridWidth}u;

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> atoms: array<u32>;
@group(0) @binding(3) var<storage, read> reference: array<f32>;
@group(0) @binding(4) var<storage, read_write> forces: array<f32>;
@group(0) @binding(5) var<storage, read_write> energies: array<f32>;
@group(0) @binding(6) var<uniform> settings: vec4<f32>;

@compute @workgroup_size(${RESTRAINT_WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let slot = id.x + id.y * GRID_WIDTH * ${RESTRAINT_WORKGROUP}u;
  if (slot >= RESTRAINT_COUNT) { return; }
  let atom = atoms[slot];
  let k = settings.x;
  let here = vec3<f32>(
    positions[atom * 3u], positions[atom * 3u + 1u], positions[atom * 3u + 2u]);
  let origin = vec3<f32>(
    reference[slot * 3u], reference[slot * 3u + 1u], reference[slot * 3u + 2u]);
  // The same minimum image the nonbonded path uses, for the same reason: an
  // accepted volume move can take an atom across a face relative to a
  // reference that does not move with the box.
  let delta = minimumImage(here - origin);
  energies[slot] = k * dot(delta, delta);
  let scale = 2.0 * k;
  forces[atom * 3u] = forces[atom * 3u] - scale * delta.x;
  forces[atom * 3u + 1u] = forces[atom * 3u + 1u] - scale * delta.y;
  forces[atom * 3u + 2u] = forces[atom * 3u + 2u] - scale * delta.z;
}
`;
}

/** Encode the restraint pass. A zero-length list encodes nothing. */
export function encodeRestraints(device, encoder, resources, count) {
  if (count === 0) return;
  const maxPerDimension = device.limits.maxComputeWorkgroupsPerDimension;
  const workgroups = Math.ceil(count / RESTRAINT_WORKGROUP);
  const [x, y, z] = dispatchGrid(workgroups, maxPerDimension);
  const gridWidth = x === workgroups ? workgroups : maxPerDimension;
  const key = `restraints:${count}:${gridWidth}`;
  const pipeline = pipelineCacheFor(device).get(key, restraintShader(count, gridWidth));
  const pass = encoder.beginComputePass({ label: key });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      resources.params, resources.positions, resources.restraintAtoms,
      resources.restraintReference, resources.forces, resources.restraintEnergies,
      resources.restraintSettings,
    ].map((allocation, binding) => ({
      binding, resource: { buffer: allocation.buffer },
    })),
  }));
  pass.dispatchWorkgroups(x, y, z);
  pass.end();
}
