/**
 * Build a Verlet neighbour list from the cell list.
 *
 * 🔴 EXCLUSIONS ARE APPLIED HERE, NOT IN THE FORCE KERNEL. A solvated system
 * has ~300 neighbours per atom and ~15 exclusions; testing every neighbour
 * against the exclusion set every step is a binary search per neighbour per
 * step, which costs more than the interaction it decides about. Doing it at
 * build time pays it once per rebuild - every twenty steps or so - and the
 * force kernel then walks a list it can trust completely.
 *
 * 🔴 AND THE EXCLUSION LIST MUST BE THE SYMMETRIC ONE. AMBER stores only the
 * partners with j > i. A gather kernel asks "is j excluded from i" for both
 * orders, so a one-sided list drops each excluded pair from exactly one of its
 * two atoms - the interaction is absent from one force and present in the
 * other, which is a non-conservative force field. See buildExclusionGather.
 *
 * 🔴 AND OVERFLOW IS DETECTED RATHER THAN TRUSTED. The capacity comes from a
 * mean-density estimate, and density is not uniform: a protein interior is
 * denser than bulk water. A truncated list does not fail, it silently drops
 * the pairs that did not fit - lowering the energy smoothly, which reads as a
 * slightly different force field rather than as a bug. So the kernel counts
 * every neighbour it finds, stores only what fits, and records the true maximum
 * through an atomicMax. The host then refuses the list and says what capacity
 * it needed, which is far more useful than a boolean.
 */
import { pipelineCacheFor } from "../runtime/pipeline-cache.js";
import { dispatchGrid } from "../runtime/execution.js";
import { PARAMS_STRUCT, GEOMETRY_FUNCTIONS, CELL_INDEX_FUNCTION } from "./wgsl-common.js";

export const NEIGHBOUR_WORKGROUP = 64;

/**
 * 🔴 THE PER-AXIS OFFSETS ARE A CONTIGUOUS RANGE, WHICH IS WHY THIS NEEDS NO
 * ARRAY. `axisOffsets` returns {-1,0,1}, {0,1} or {0} depending on the cell
 * count - all contiguous - so each axis is a base and a length, and the stencil
 * is three nested loops with baked bounds. A `const` array of vec3 indexed by a
 * loop variable would work too, but WGSL's rules about dynamically indexing a
 * const array are a corner this does not need to visit.
 */
function axisRange(cellCount) {
  if (cellCount >= 3) return { base: -1, count: 3 };
  if (cellCount === 2) return { base: 0, count: 2 };
  return { base: 0, count: 1 };
}

export function neighbourShader(plan, atomCount, gridWidth) {
  const [rx, ry, rz] = plan.counts.map(axisRange);
  return `
const ATOM_COUNT: u32 = ${atomCount}u;
const CELLS_X: u32 = ${plan.counts[0]}u;
const CELLS_Y: u32 = ${plan.counts[1]}u;
const CELLS_Z: u32 = ${plan.counts[2]}u;
const CELL_COUNT: u32 = ${plan.cellCount}u;
const MAX_NEIGHBOURS: u32 = ${plan.maxNeighbours}u;
const OFFSET_X_BASE: i32 = ${rx.base};
const OFFSET_X_COUNT: i32 = ${rx.count};
const OFFSET_Y_BASE: i32 = ${ry.base};
const OFFSET_Y_COUNT: i32 = ${ry.count};
const OFFSET_Z_BASE: i32 = ${rz.base};
const OFFSET_Z_COUNT: i32 = ${rz.count};
const GRID_WIDTH: u32 = ${gridWidth}u;
${PARAMS_STRUCT}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> cellStart: array<u32>;
@group(0) @binding(3) var<storage, read> cellAtoms: array<u32>;
@group(0) @binding(4) var<storage, read> exclusionStart: array<u32>;
@group(0) @binding(5) var<storage, read> exclusionPartners: array<u32>;
@group(0) @binding(6) var<storage, read_write> neighbours: array<u32>;
@group(0) @binding(7) var<storage, read_write> neighbourCount: array<u32>;
@group(0) @binding(8) var<storage, read_write> required: array<atomic<u32>>;
${GEOMETRY_FUNCTIONS}
${CELL_INDEX_FUNCTION}

fn cellCoordinate(p: vec3<f32>) -> vec3<u32> {
  let f = fractional(p);
  let wrapped = f - floor(f);
  let counts = vec3<f32>(f32(CELLS_X), f32(CELLS_Y), f32(CELLS_Z));
  let scaled = wrapped * counts;
  return vec3<u32>(
    min(u32(scaled.x), CELLS_X - 1u),
    min(u32(scaled.y), CELLS_Y - 1u),
    min(u32(scaled.z), CELLS_Z - 1u));
}

fn wrapCell(value: i32, count: i32) -> u32 {
  var v = value % count;
  if (v < 0) { v = v + count; }
  return u32(v);
}

/** The exclusion slice is sorted, so this is a binary search. */
fn isExcluded(atom: u32, partner: u32) -> bool {
  var low = exclusionStart[atom];
  var high = exclusionStart[atom + 1u];
  loop {
    if (low >= high) { return false; }
    let middle = low + (high - low) / 2u;
    let value = exclusionPartners[middle];
    if (value == partner) { return true; }
    if (value < partner) { low = middle + 1u; } else { high = middle; }
  }
}

@compute @workgroup_size(${NEIGHBOUR_WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let atom = id.x + id.y * GRID_WIDTH * ${NEIGHBOUR_WORKGROUP}u;
  if (atom >= ATOM_COUNT) { return; }
  let centre = atomPosition(atom);
  let coordinate = cellCoordinate(centre);
  let listCutoffSquared = params.scalars[PARAMS_LIST_CUTOFF] * params.scalars[PARAMS_LIST_CUTOFF];
  let base = atom * MAX_NEIGHBOURS;

  var stored = 0u;
  var found = 0u;
  for (var ox = 0; ox < OFFSET_X_COUNT; ox = ox + 1) {
    let cx = wrapCell(i32(coordinate.x) + OFFSET_X_BASE + ox, i32(CELLS_X));
    for (var oy = 0; oy < OFFSET_Y_COUNT; oy = oy + 1) {
      let cy = wrapCell(i32(coordinate.y) + OFFSET_Y_BASE + oy, i32(CELLS_Y));
      for (var oz = 0; oz < OFFSET_Z_COUNT; oz = oz + 1) {
        let cz = wrapCell(i32(coordinate.z) + OFFSET_Z_BASE + oz, i32(CELLS_Z));
        let cell = (cx * CELLS_Y + cy) * CELLS_Z + cz;
        let begin = cellStart[cell];
        let end = cellStart[cell + 1u];
        for (var slot = begin; slot < end; slot = slot + 1u) {
          let other = cellAtoms[slot];
          if (other == atom) { continue; }
          let delta = minimumImage(centre - atomPosition(other));
          if (dot(delta, delta) > listCutoffSquared) { continue; }
          if (isExcluded(atom, other)) { continue; }
          found = found + 1u;
          if (stored < MAX_NEIGHBOURS) {
            neighbours[base + stored] = other;
            stored = stored + 1u;
          }
        }
      }
    }
  }
  neighbourCount[atom] = stored;
  // Always recorded, not only on overflow: the host prints it as the observed
  // high-water mark, which is how the capacity gets tuned from measurement
  // rather than from the density estimate.
  atomicMax(&required[0], found);
}
`;
}

export function encodeNeighbours(device, encoder, resources, atomCount, plan) {
  const maxPerDimension = device.limits.maxComputeWorkgroupsPerDimension;
  const workgroups = Math.ceil(atomCount / NEIGHBOUR_WORKGROUP);
  const [x, y, z] = dispatchGrid(workgroups, maxPerDimension);
  const gridWidth = x === workgroups ? workgroups : maxPerDimension;
  const code = neighbourShader(plan, atomCount, gridWidth);
  const key = `neighbours:${atomCount}:${plan.counts.join("x")}:${plan.maxNeighbours}:${gridWidth}`;
  const pipeline = pipelineCacheFor(device).get(key, code);
  const pass = encoder.beginComputePass({ label: key });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      resources.params, resources.positions, resources.cellStart, resources.cellAtoms,
      resources.exclusionStart, resources.exclusionPartners,
      resources.neighbours, resources.neighbourCount, resources.required,
    ].map((allocation, binding) => ({ binding, resource: { buffer: allocation.buffer } })),
  }));
  pass.dispatchWorkgroups(x, y, z);
  pass.end();
}

/** The error the host raises when the list did not fit. */
export class NeighbourOverflowError extends Error {
  constructor(required, capacity) {
    super(
      `the neighbour list needed ${required} slots per atom and the capacity is `
      + `${capacity}. Pairs were dropped, so the forces from this build are not `
      + "usable. Rebuild with a larger neighbourSlack, or a smaller skin.",
    );
    this.name = "NeighbourOverflowError";
    this.required = required;
    this.capacity = capacity;
  }
}
