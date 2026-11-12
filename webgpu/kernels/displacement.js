/**
 * How far the worst atom has moved since the neighbour list was built.
 *
 * 🔴 THIS IS THE ONE THING THAT MAKES A STALE NEIGHBOUR LIST LOUD. The list
 * holds every pair within `cutoff + skin`, and the nonbonded kernel then
 * ignores anything beyond `cutoff`. The skin is the whole safety margin: a pair
 * that was outside the list can only come inside the cutoff if the two atoms
 * between them close the gap by more than `skin`, and each can contribute at
 * most its own displacement. So the list is valid exactly while
 *
 *     2 * max|r - r_built| <= skin
 *
 * and once that fails some pair is interacting and not being counted. Nothing
 * reports it: the forces are simply wrong, smoothly and plausibly, and the
 * energy drifts in a way that reads like an integrator problem. The fixed
 * rebuild schedule this replaces was a guess that the inequality holds.
 *
 * 🔴 ATOMICMAX ON THE RAW BIT PATTERN, WHICH IS EXACT AND NOT A QUANTISATION.
 * WebGPU has no float atomics, and the usual workaround is a fixed-point
 * encoding that costs precision and needs a scale nobody can choose well. It is
 * not needed here: for non-negative IEEE-754 floats the u32 bit pattern is
 * monotonic in the value, so `atomicMax` over `bitcast<u32>` of a squared
 * displacement orders them correctly and returns the exact float. A squared
 * displacement cannot be negative, so the sign-bit case that breaks this trick
 * cannot arise.
 *
 * 🔴 SQUARED, AND THE HOST TAKES THE ROOT. One `sqrt` on the host instead of
 * one per atom on the device, and the comparison against `skin` is done on the
 * square there too, so the kernel is three subtractions and a dot product.
 *
 * 🔴 AND NO MINIMUM-IMAGE CONVENTION HERE, DELIBERATELY. Positions are never
 * wrapped into the box during a run - `docs/ARCHITECTURE.md` records why: the
 * bonded forces are not periodic, so a molecule has to stay whole in the
 * coordinate array. A plain difference is therefore the true displacement. If
 * wrapping were ever added this kernel would silently report a box-width jump
 * as motion and rebuild on every step.
 */
import { pipelineCacheFor } from "../runtime/pipeline-cache.js";
import { dispatchGrid } from "../runtime/execution.js";

export const DISPLACEMENT_WORKGROUP = 64;

export function displacementShader(atomCount, gridWidth) {
  return `
const ATOM_COUNT: u32 = ${atomCount}u;
const GRID_WIDTH: u32 = ${gridWidth}u;
@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read> origin: array<f32>;
@group(0) @binding(2) var<storage, read_write> worst: array<atomic<u32>>;
@compute @workgroup_size(${DISPLACEMENT_WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let atom = id.x + id.y * GRID_WIDTH * ${DISPLACEMENT_WORKGROUP}u;
  if (atom >= ATOM_COUNT) { return; }
  let delta = vec3<f32>(
    positions[atom * 3u] - origin[atom * 3u],
    positions[atom * 3u + 1u] - origin[atom * 3u + 1u],
    positions[atom * 3u + 2u] - origin[atom * 3u + 2u]);
  let squared = bitcast<u32>(dot(delta, delta));
  // 🔴 TWO SLOTS, AND THE SECOND IS THE WHOLE FIX FOR THE MEASUREMENT WINDOW.
  // Slot 0 answers "has the list gone stale since it was BUILT?", so the caller
  // clears it at every rebuild. That made the host's reading cover only the
  // steps since the last rebuild INSIDE the block it read - and the page
  // submits a block of rebuildInterval steps while the adaptive interval starts
  // at 4, so a 20-step block contained five rebuilds and the host saw the last
  // four steps of twenty. Slot 1 is cleared once per BLOCK instead, so it is
  // the worst per-interval displacement anywhere in the block.
  //
  // Both slots take the same per-step value, which is what makes slot 1 the
  // right quantity rather than a monotonically growing one: each interval
  // measures from its own origin, and slot 1 keeps the maximum over those.
  //
  // 🔴 AND NO BACKTICK MAY APPEAR IN THIS COMMENT - it is inside a JS template
  // literal, so one ends the WGSL string and the file stops parsing at a word
  // in a sentence. This is the seventh time; the tree-wide node --check gate in
  // npm test caught it before anything ran.
  atomicMax(&worst[0], squared);
  atomicMax(&worst[1], squared);
}
`;
}

/**
 * Encode one displacement pass.
 *
 * `worst[0]` accumulates since the last rebuild, so the caller clears it when
 * it copies `positions` into `origin`. `worst[1]` accumulates over the whole
 * submitted block and is cleared once per block, which is what the host reads -
 * see the note in the shader.
 */
export function encodeDisplacement(device, encoder, resources, atomCount) {
  const maxPerDimension = device.limits.maxComputeWorkgroupsPerDimension;
  const workgroups = Math.ceil(atomCount / DISPLACEMENT_WORKGROUP);
  const [x, y, z] = dispatchGrid(workgroups, maxPerDimension);
  const gridWidth = x === workgroups ? workgroups : maxPerDimension;
  const key = `displacement:${atomCount}:${gridWidth}`;
  const pipeline = pipelineCacheFor(device).get(key,
    displacementShader(atomCount, gridWidth));
  const pass = encoder.beginComputePass({ label: key });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [resources.positions, resources.rebuildOrigin, resources.displacement]
      .map((allocation, binding) => ({ binding, resource: { buffer: allocation.buffer } })),
  }));
  pass.dispatchWorkgroups(x, y, z);
  pass.end();
}
