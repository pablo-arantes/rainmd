/**
 * Bin the atoms into cells: histogram, scan, scatter, sort.
 *
 * 🔴 THE SORT IS WHY THIS IS FOUR PASSES AND NOT THREE, AND IT IS THERE FOR
 * DETERMINISM RATHER THAN FOR SPEED. The scatter assigns each atom a slot with
 * an `atomicAdd`, so the ORDER of atoms within a cell depends on which threads
 * happen to arrive first - it differs between runs on the same machine. The
 * counts are exact either way (integer addition), but the order decides the
 * order of the f32 force summation in the gather, so without the sort the same
 * step gives slightly different forces every time. That costs two things worth
 * more than the pass: a reproducible trajectory, which is a scientific feature,
 * and a differential test with a fixed value to compare against.
 *
 * 🔴 AND AN INSERTION SORT IS THE RIGHT ONE HERE, WHICH IS NOT OBVIOUS. It is
 * O(occupancy^2) on a single thread per cell - but the occupancy depends only
 * on the density and the cell size, never on the system size, while the number
 * of cells grows with the box. So the SERIAL depth is constant as the
 * simulation scales and the parallel width grows, which is the opposite of the
 * usual objection to insertion sort. At 149 atoms per cell it is ~11k
 * comparisons on each of 8 threads for the test fixture; a radix sort is what
 * to reach for only if a profile says so.
 */
import { pipelineCacheFor } from "../runtime/pipeline-cache.js";
import { dispatchGrid } from "../runtime/execution.js";
import { PARAMS_STRUCT, GEOMETRY_FUNCTIONS, CELL_INDEX_FUNCTION } from "./wgsl-common.js";

export const ASSIGN_WORKGROUP = 64;
export const SCAN_WORKGROUP = 256;
export const SORT_WORKGROUP = 64;

function preamble(plan, atomCount, gridWidth) {
  return `
const ATOM_COUNT: u32 = ${atomCount}u;
const CELLS_X: u32 = ${plan.counts[0]}u;
const CELLS_Y: u32 = ${plan.counts[1]}u;
const CELLS_Z: u32 = ${plan.counts[2]}u;
const CELL_COUNT: u32 = ${plan.cellCount}u;
const GRID_WIDTH: u32 = ${gridWidth}u;
${PARAMS_STRUCT}`;
}

/** Pass 1: every atom finds its cell and bumps that cell's counter. */
export function assignShader(plan, atomCount, gridWidth) {
  return `${preamble(plan, atomCount, gridWidth)}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read_write> cellOfAtom: array<u32>;
@group(0) @binding(3) var<storage, read_write> cellCount: array<atomic<u32>>;
${GEOMETRY_FUNCTIONS}
${CELL_INDEX_FUNCTION}
@compute @workgroup_size(${ASSIGN_WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let atom = id.x + id.y * GRID_WIDTH * ${ASSIGN_WORKGROUP}u;
  if (atom >= ATOM_COUNT) { return; }
  let cell = cellOfPosition(atomPosition(atom));
  cellOfAtom[atom] = cell;
  atomicAdd(&cellCount[cell], 1u);
}
`;
}

/**
 * Pass 2: exclusive prefix sum over the cell counts, in one workgroup.
 *
 * 🔴 ONE WORKGROUP, WALKING THE CELLS IN CHUNKS, WITH A RUNNING OFFSET. A
 * multi-block scan would parallelise it, but it runs once per neighbour-list
 * rebuild - every twenty steps or so - over a few hundred to a few thousand
 * cells, so the whole pass is microseconds and a second kernel to scan the
 * block sums is machinery with nothing to buy. The chunk count is baked in so
 * the loop bound is a constant: a `workgroupBarrier` inside a loop whose trip
 * count WGSL cannot prove uniform is a uniformity error, not a slow kernel.
 *
 * It also zeroes each count on the way past, because the scatter reuses that
 * array as its per-cell cursor and would otherwise need a pass of its own.
 */
export function scanShader(plan, atomCount, gridWidth) {
  const chunks = Math.ceil(plan.cellCount / SCAN_WORKGROUP);
  return `${preamble(plan, atomCount, gridWidth)}
const SCAN_CHUNKS: u32 = ${chunks}u;
@group(0) @binding(0) var<storage, read_write> cellCount: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> cellStart: array<u32>;

var<workgroup> scratch: array<u32, ${SCAN_WORKGROUP}>;
var<workgroup> runningOffset: u32;

@compute @workgroup_size(${SCAN_WORKGROUP})
fn main(@builtin(local_invocation_id) local: vec3<u32>) {
  let t = local.x;
  if (t == 0u) { runningOffset = 0u; }
  workgroupBarrier();
  for (var chunk = 0u; chunk < SCAN_CHUNKS; chunk = chunk + 1u) {
    let index = chunk * ${SCAN_WORKGROUP}u + t;
    var value = 0u;
    if (index < CELL_COUNT) { value = atomicLoad(&cellCount[index]); }
    scratch[t] = value;
    workgroupBarrier();
    // Hillis-Steele inclusive scan. The barrier between the read and the write
    // is what makes it correct: without it a thread can overwrite a value its
    // neighbour has not read yet.
    for (var stride = 1u; stride < ${SCAN_WORKGROUP}u; stride = stride << 1u) {
      var addend = 0u;
      if (t >= stride) { addend = scratch[t - stride]; }
      workgroupBarrier();
      scratch[t] = scratch[t] + addend;
      workgroupBarrier();
    }
    if (index < CELL_COUNT) {
      cellStart[index] = runningOffset + scratch[t] - value;
      atomicStore(&cellCount[index], 0u);
    }
    workgroupBarrier();
    if (t == ${SCAN_WORKGROUP - 1}u) { runningOffset = runningOffset + scratch[t]; }
    workgroupBarrier();
  }
  // cellStart has CELL_COUNT + 1 entries; the last is the total, which is what
  // lets the sort and the gather read a cell's end as cellStart[cell + 1].
  if (t == 0u) { cellStart[CELL_COUNT] = runningOffset; }
}
`;
}

/** Pass 3: each atom claims a slot in its cell's slice. */
export function scatterShader(plan, atomCount, gridWidth) {
  return `${preamble(plan, atomCount, gridWidth)}
@group(0) @binding(0) var<storage, read> cellOfAtom: array<u32>;
@group(0) @binding(1) var<storage, read> cellStart: array<u32>;
@group(0) @binding(2) var<storage, read_write> cellCursor: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> cellAtoms: array<u32>;
@compute @workgroup_size(${ASSIGN_WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let atom = id.x + id.y * GRID_WIDTH * ${ASSIGN_WORKGROUP}u;
  if (atom >= ATOM_COUNT) { return; }
  let cell = cellOfAtom[atom];
  let slot = cellStart[cell] + atomicAdd(&cellCursor[cell], 1u);
  cellAtoms[slot] = atom;
}
`;
}

/** Pass 4: sort each cell's slice by atom index, so the order is fixed. */
export function sortShader(plan, atomCount, gridWidth) {
  return `${preamble(plan, atomCount, gridWidth)}
@group(0) @binding(0) var<storage, read> cellStart: array<u32>;
@group(0) @binding(1) var<storage, read_write> cellAtoms: array<u32>;
@compute @workgroup_size(${SORT_WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let cell = id.x + id.y * GRID_WIDTH * ${SORT_WORKGROUP}u;
  if (cell >= CELL_COUNT) { return; }
  let begin = cellStart[cell];
  let end = cellStart[cell + 1u];
  if (end <= begin + 1u) { return; }
  for (var i = begin + 1u; i < end; i = i + 1u) {
    let key = cellAtoms[i];
    var j = i;
    loop {
      if (j == begin) { break; }
      if (cellAtoms[j - 1u] <= key) { break; }
      cellAtoms[j] = cellAtoms[j - 1u];
      j = j - 1u;
    }
    cellAtoms[j] = key;
  }
}
`;
}

/**
 * Encode all four passes.
 *
 * 🔴 EACH PASS IS ITS OWN COMPUTE PASS BECAUSE EACH DEPENDS ON THE LAST. WebGPU
 * guarantees ordering between compute passes in a command encoder but gives no
 * barrier WITHIN one, so merging the histogram and the scan into a single pass
 * would let the scan read counts that are still being written - a race whose
 * symptom is a neighbour list that is almost right.
 */
export function encodeCellList(device, encoder, resources, atomCount, plan) {
  const maxPerDimension = device.limits.maxComputeWorkgroupsPerDimension;
  const cache = pipelineCacheFor(device);
  const key = `${atomCount}:${plan.counts.join("x")}`;

  const atomWorkgroups = Math.ceil(atomCount / ASSIGN_WORKGROUP);
  const [ax, ay, az] = dispatchGrid(atomWorkgroups, maxPerDimension);
  const atomGridWidth = ax === atomWorkgroups ? atomWorkgroups : maxPerDimension;

  const cellWorkgroups = Math.ceil(plan.cellCount / SORT_WORKGROUP);
  const [cx, cy, cz] = dispatchGrid(cellWorkgroups, maxPerDimension);
  const cellGridWidth = cx === cellWorkgroups ? cellWorkgroups : maxPerDimension;

  const run = (label, code, bindings, workgroups) => {
    const pipeline = cache.get(label, code);
    const pass = encoder.beginComputePass({ label });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: bindings.map((allocation, binding) => ({
        binding, resource: { buffer: allocation.buffer },
      })),
    }));
    pass.dispatchWorkgroups(...workgroups);
    pass.end();
  };

  // The histogram accumulates, so the counters must start at zero every
  // rebuild. clearBuffer is free next to a dispatch and removes a kernel.
  encoder.clearBuffer(resources.cellCount.buffer);

  run(`cells:assign:${key}:${atomGridWidth}`,
    assignShader(plan, atomCount, atomGridWidth),
    [resources.params, resources.positions, resources.cellOfAtom, resources.cellCount],
    [ax, ay, az]);

  run(`cells:scan:${key}`,
    scanShader(plan, atomCount, atomGridWidth),
    [resources.cellCount, resources.cellStart],
    [1, 1, 1]);

  run(`cells:scatter:${key}:${atomGridWidth}`,
    scatterShader(plan, atomCount, atomGridWidth),
    [resources.cellOfAtom, resources.cellStart, resources.cellCount, resources.cellAtoms],
    [ax, ay, az]);

  run(`cells:sort:${key}:${cellGridWidth}`,
    sortShader(plan, atomCount, cellGridWidth),
    [resources.cellStart, resources.cellAtoms],
    [cx, cy, cz]);
}
