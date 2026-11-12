/**
 * PME on the device: spread, convolve, gather.
 *
 * A transcription of cpu/forces/pme.js, which is validated against OpenMM to
 * 1.2e-8 on the Coulomb energy and against its own numerical gradient to
 * 4.7e-8. Four conventions must match it exactly, and each one has a comment
 * because each one is invisible if wrong:
 *
 *   - the forward transform carries exp(+2*pi*i*m*k/K)
 *   - the B-spline moduli DIVIDE the convolution
 *   - the grid index for weight k is (floor(u) + k - order + 1) mod K
 *   - the inverse transform is UNNORMALISED
 */
import { pipelineCacheFor } from "../runtime/pipeline-cache.js";
import { dispatchGrid } from "../runtime/execution.js";
import { PARAMS_STRUCT, GEOMETRY_FUNCTIONS } from "./wgsl-common.js";
import { encodeFft3d } from "./pme-fft.js";

export const PME_WORKGROUP = 64;
export const REDUCE_WORKGROUP = 256;

/**
 * The fixed-point scale for the charge grid.
 *
 * 🔴 THIS IS THE ONE PLACE IN THE ENGINE THAT USES ATOMICS, AND IT IS STILL
 * DETERMINISTIC. Spreading is a genuine scatter - each atom writes to 125 grid
 * points and each grid point receives from a dozen atoms - so the gather trick
 * that removes atomics everywhere else would cost ten times the work here.
 * Integer addition IS associative and commutative, so an `atomicAdd` on i32
 * gives the same total whatever order the threads arrive in. That is the whole
 * reason for fixed point: not range, but reproducibility.
 *
 * 🔴 THE HEADROOM IS +-128 CHARGE UNITS AND THAT IS AMPLE, BUT IT IS NOT
 * INFINITE. 2^24 leaves 2^31/2^24 = 128, and a grid point accumulates
 * `sum of q * weight` over the atoms whose spline support covers it - about a
 * dozen atoms with |q| <= 1 and weights summing to 1 per atom. A value above a
 * few units would be extraordinary. The quantisation step is 6e-8, which on a
 * grid value of order 0.5 is 1.2e-7 relative: f32 epsilon, by construction.
 * The differential test reads the grid back and asserts the maximum is far
 * below the limit, because an i32 `atomicAdd` wraps silently on overflow.
 */
export const FIXED_SCALE = 16777216.0;
export const FIXED_LIMIT = 128.0;

/**
 * Order-5 B-spline weights and derivatives, as WGSL.
 *
 * 🔴 THE DERIVATIVE IS TAKEN AT ORDER 4 AND THEN DIFFERENCED, BEFORE THE LAST
 * RECURSION STEP. `d/dw M_n(w) = M_{n-1}(w) - M_{n-1}(w-1)`, so the derivative
 * must be read off the order-(n-1) values - which is why the final recursion
 * step appears after it below, and not before. Differentiating the finished
 * order-5 weights instead gives something close enough to look right on an
 * energy and wrong enough that the reciprocal forces are not the gradient of
 * the reciprocal energy. That shows up as drift in a simulation that conserves
 * nothing, long after the kernel was called correct.
 */
export const BSPLINE_FUNCTION = `
const ORDER: u32 = 5u;

struct Spline {
  weights: array<f32, 5>,
  derivatives: array<f32, 5>,
};

fn spline(offset: f32) -> Spline {
  var w = array<f32, 5>(0.0, 0.0, 0.0, 0.0, 0.0);
  w[0] = 1.0 - offset;
  w[1] = offset;
  // Build up to order - 1.
  for (var k = 3u; k < ORDER; k = k + 1u) {
    let div = 1.0 / f32(k - 1u);
    w[k - 1u] = div * offset * w[k - 2u];
    for (var j = 1u; j < k - 1u; j = j + 1u) {
      w[k - j - 1u] = div * ((offset + f32(j)) * w[k - j - 2u]
        + (f32(k - j) - offset) * w[k - j - 1u]);
    }
    w[0] = div * (1.0 - offset) * w[0];
  }
  var d = array<f32, 5>(0.0, 0.0, 0.0, 0.0, 0.0);
  d[0] = -w[0];
  for (var j = 1u; j < ORDER; j = j + 1u) {
    d[j] = w[j - 1u] - w[j];
  }
  // The final recursion step, after the derivative has been read.
  let div = 1.0 / f32(ORDER - 1u);
  w[ORDER - 1u] = div * offset * w[ORDER - 2u];
  for (var j = 1u; j < ORDER - 1u; j = j + 1u) {
    w[ORDER - j - 1u] = div * ((offset + f32(j)) * w[ORDER - j - 2u]
      + (f32(ORDER - j) - offset) * w[ORDER - j - 1u]);
  }
  w[0] = div * (1.0 - offset) * w[0];
  var result: Spline;
  result.weights = w;
  result.derivatives = d;
  return result;
}
`;

function gridPreamble(grid, atomCount, gridWidth) {
  return `
const ATOM_COUNT: u32 = ${atomCount}u;
const GRID_X: u32 = ${grid[0]}u;
const GRID_Y: u32 = ${grid[1]}u;
const GRID_Z: u32 = ${grid[2]}u;
const GRID_POINTS: u32 = ${grid[0] * grid[1] * grid[2]}u;
const FIXED_SCALE: f32 = ${FIXED_SCALE};
const GRID_WIDTH: u32 = ${gridWidth}u;
${PARAMS_STRUCT}`;
}

/**
 * The scaled grid coordinate of an atom, and the wrapped grid index of a
 * spline point.
 *
 * 🔴 `wrapped` CAN EVALUATE TO EXACTLY 1.0 IN f32 even though `f - floor(f)` is
 * in [0, 1) in exact arithmetic, which puts `base` one past the last cell. It
 * is harmless here and only because the index is reduced modulo the grid size
 * below - `base = GRID_X` with offset 0 is the same point as `base = 0` shifted
 * by a whole period. Remove the modulo and this becomes an out-of-bounds write
 * that Metal clamps and Vulkan discards.
 */
const GRID_INDEX_FUNCTIONS = `
fn scaledGrid(p: vec3<f32>) -> vec3<f32> {
  let f = fractional(p);
  let wrapped = f - floor(f);
  return wrapped * vec3<f32>(f32(GRID_X), f32(GRID_Y), f32(GRID_Z));
}

fn wrapIndex(value: i32, count: i32) -> u32 {
  var v = value % count;
  if (v < 0) { v = v + count; }
  return u32(v);
}
`;

/** Pass 1: charges onto the grid, in fixed point. */
export function spreadShader(grid, atomCount, gridWidth) {
  return `${gridPreamble(grid, atomCount, gridWidth)}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> charges: array<f32>;
@group(0) @binding(3) var<storage, read_write> fixedGrid: array<atomic<i32>>;
${GEOMETRY_FUNCTIONS}
${GRID_INDEX_FUNCTIONS}
${BSPLINE_FUNCTION}

@compute @workgroup_size(${PME_WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let atom = id.x + id.y * GRID_WIDTH * ${PME_WORKGROUP}u;
  if (atom >= ATOM_COUNT) { return; }
  let charge = charges[atom];
  if (charge == 0.0) { return; }

  let scaled = scaledGrid(atomPosition(atom));
  let base = floor(scaled);
  let splineX = spline(scaled.x - base.x);
  let splineY = spline(scaled.y - base.y);
  let splineZ = spline(scaled.z - base.z);
  let baseX = i32(base.x);
  let baseY = i32(base.y);
  let baseZ = i32(base.z);

  for (var kx = 0u; kx < ORDER; kx = kx + 1u) {
    let ix = wrapIndex(baseX + i32(kx) - i32(ORDER) + 1, i32(GRID_X));
    let wx = charge * splineX.weights[kx];
    for (var ky = 0u; ky < ORDER; ky = ky + 1u) {
      let iy = wrapIndex(baseY + i32(ky) - i32(ORDER) + 1, i32(GRID_Y));
      let wxy = wx * splineY.weights[ky];
      for (var kz = 0u; kz < ORDER; kz = kz + 1u) {
        let iz = wrapIndex(baseZ + i32(kz) - i32(ORDER) + 1, i32(GRID_Z));
        let value = wxy * splineZ.weights[kz];
        let point = (ix * GRID_Y + iy) * GRID_Z + iz;
        // The grid is complex and interleaved; charge is real, so only the
        // even slot is touched and the odd one stays at the cleared zero.
        atomicAdd(&fixedGrid[2u * point], i32(round(value * FIXED_SCALE)));
      }
    }
  }
}
`;
}

/** Pass 2: fixed point to f32 complex, ready for the transform. */
export function convertShader(grid, atomCount, gridWidth) {
  return `${gridPreamble(grid, atomCount, gridWidth)}
@group(0) @binding(0) var<storage, read> fixedGrid: array<i32>;
@group(0) @binding(1) var<storage, read_write> floatGrid: array<f32>;
@compute @workgroup_size(${PME_WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let point = id.x + id.y * GRID_WIDTH * ${PME_WORKGROUP}u;
  if (point >= GRID_POINTS) { return; }
  floatGrid[2u * point] = f32(fixedGrid[2u * point]) / FIXED_SCALE;
  floatGrid[2u * point + 1u] = f32(fixedGrid[2u * point + 1u]) / FIXED_SCALE;
}
`;
}

/**
 * Pass 3: multiply by the Ewald kernel, and read the energy off the same product.
 *
 * 🔴 THE FREQUENCY INDEX IS SIGNED AND WRAPS AT HALF THE GRID. Index
 * GRID_X - 1 is frequency -1, not GRID_X - 1. Read unsigned, m^2 becomes
 * enormous over the whole upper half of the grid, the Gaussian kills those
 * terms, and the reciprocal energy comes out roughly a quarter of its true
 * value - which looks like a wrong prefactor and is not one.
 *
 * 🔴 AND THE MODULI DIVIDE. Essmann's b(m) is the RECIPROCAL of the DFT of the
 * B-spline, so |b|^2 = 1/moduli. Multiplying instead is a smooth error that
 * grows towards the Nyquist frequency - it does not look like a bug anywhere.
 */
export function convolveShader(grid, atomCount, gridWidth) {
  const [nx, ny, nz] = grid;
  return `${gridPreamble(grid, atomCount, gridWidth)}
const MODULI_Y_BASE: u32 = ${nx}u;
const MODULI_Z_BASE: u32 = ${nx + ny}u;
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> floatGrid: array<f32>;
@group(0) @binding(2) var<storage, read> moduli: array<f32>;
@group(0) @binding(3) var<storage, read_write> energy: array<f32>;

@compute @workgroup_size(${PME_WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let point = id.x + id.y * GRID_WIDTH * ${PME_WORKGROUP}u;
  if (point >= GRID_POINTS) { return; }
  let mz = point % GRID_Z;
  let my = (point / GRID_Z) % GRID_Y;
  let mx = point / (GRID_Z * GRID_Y);

  let kx = select(i32(mx) - i32(GRID_X), i32(mx), mx <= GRID_X / 2u);
  let ky = select(i32(my) - i32(GRID_Y), i32(my), my <= GRID_Y / 2u);
  let kz = select(i32(mz) - i32(GRID_Z), i32(mz), mz <= GRID_Z / 2u);

  if (kx == 0 && ky == 0 && kz == 0) {
    floatGrid[2u * point] = 0.0;
    floatGrid[2u * point + 1u] = 0.0;
    energy[point] = 0.0;
    return;
  }

  // 🔴 THE ROWS OF THE INVERSE BOX, NOT ITS COLUMNS. See the long note in
  // cpu/forces/pme.js: the fractional helper in wgsl-common.js builds s_d from
  // recip[3i + d], so the Cartesian wave vector is row i, and taking the column
  // computes the transpose. Identical for a diagonal (rectangular) inverse box,
  // 91 kJ/mol wrong for a truncated octahedron. This kernel and its host
  // reference carried the same error, so the differential test between them
  // agreed about it; the independent oracle is the exact lattice sum in
  // test/physics.test.js.
  //
  // 🔴 AND NO BACKTICK MAY APPEAR IN THIS COMMENT. It lives inside a JS
  // template literal, so one backtick ends the WGSL string and the file becomes
  // a syntax error - which is how the first draft of this comment broke every
  // GPU lane at once. docs/STATUS.md records this happening five times.
  let frequency = vec3<f32>(f32(kx), f32(ky), f32(kz));
  let m = vec3<f32>(
    dot(frequency, params.recipA.xyz),
    dot(frequency, params.recipB.xyz),
    dot(frequency, params.recipC.xyz));
  let mSquared = dot(m, m);

  let alpha = params.scalars[PARAMS_ALPHA];
  let factor = 9.869604401089358 / (alpha * alpha);
  let modulus = moduli[mx] * moduli[MODULI_Y_BASE + my] * moduli[MODULI_Z_BASE + mz];
  let kernel = exp(-factor * mSquared) / (mSquared * modulus);
  let volume = params.boxA.x * params.boxB.y * params.boxC.z;
  let prefactor = params.scalars[PARAMS_COULOMB] / (6.283185307179586 * volume);

  let re = floatGrid[2u * point];
  let im = floatGrid[2u * point + 1u];
  energy[point] = prefactor * kernel * (re * re + im * im);
  // 🔴 THE FACTOR OF TWO BELONGS HERE, NOT IN THE GATHER. d|F|^2/dF carries it,
  // so the grid written out is dE/dQ(k) exactly and the gather must NOT halve
  // anything. Putting a 0.5 in the gather as well - the instinct from the 1/2
  // in the energy prefactor - makes every reciprocal force half its true value.
  floatGrid[2u * point] = re * kernel * 2.0 * prefactor;
  floatGrid[2u * point + 1u] = im * kernel * 2.0 * prefactor;
}
`;
}

/** Pass 4: differentiate the splines against the convolved grid. */
export function gatherShader(grid, atomCount, gridWidth) {
  return `${gridPreamble(grid, atomCount, gridWidth)}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> charges: array<f32>;
@group(0) @binding(3) var<storage, read> floatGrid: array<f32>;
@group(0) @binding(4) var<storage, read_write> forces: array<f32>;
${GEOMETRY_FUNCTIONS}
${GRID_INDEX_FUNCTIONS}
${BSPLINE_FUNCTION}

@compute @workgroup_size(${PME_WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let atom = id.x + id.y * GRID_WIDTH * ${PME_WORKGROUP}u;
  if (atom >= ATOM_COUNT) { return; }
  let charge = charges[atom];
  if (charge == 0.0) { return; }

  let scaled = scaledGrid(atomPosition(atom));
  let base = floor(scaled);
  let splineX = spline(scaled.x - base.x);
  let splineY = spline(scaled.y - base.y);
  let splineZ = spline(scaled.z - base.z);
  let baseX = i32(base.x);
  let baseY = i32(base.y);
  let baseZ = i32(base.z);

  var gradient = vec3<f32>(0.0, 0.0, 0.0);
  for (var kx = 0u; kx < ORDER; kx = kx + 1u) {
    let ix = wrapIndex(baseX + i32(kx) - i32(ORDER) + 1, i32(GRID_X));
    for (var ky = 0u; ky < ORDER; ky = ky + 1u) {
      let iy = wrapIndex(baseY + i32(ky) - i32(ORDER) + 1, i32(GRID_Y));
      for (var kz = 0u; kz < ORDER; kz = kz + 1u) {
        let iz = wrapIndex(baseZ + i32(kz) - i32(ORDER) + 1, i32(GRID_Z));
        let point = (ix * GRID_Y + iy) * GRID_Z + iz;
        let value = floatGrid[2u * point];
        gradient = gradient + value * vec3<f32>(
          splineX.derivatives[kx] * splineY.weights[ky] * splineZ.weights[kz],
          splineX.weights[kx] * splineY.derivatives[ky] * splineZ.weights[kz],
          splineX.weights[kx] * splineY.weights[ky] * splineZ.derivatives[kz]);
      }
    }
  }

  // Chain rule into cartesian: du_d/dr_x = grid_d * recip[x][d].
  let scaledGradient = gradient * charge
    * vec3<f32>(f32(GRID_X), f32(GRID_Y), f32(GRID_Z));
  let force = -(scaledGradient.x * vec3<f32>(params.recipA.x, params.recipB.x, params.recipC.x)
    + scaledGradient.y * vec3<f32>(params.recipA.y, params.recipB.y, params.recipC.y)
    + scaledGradient.z * vec3<f32>(params.recipA.z, params.recipB.z, params.recipC.z));
  forces[atom * 3u] = forces[atom * 3u] + force.x;
  forces[atom * 3u + 1u] = forces[atom * 3u + 1u] + force.y;
  forces[atom * 3u + 2u] = forces[atom * 3u + 2u] + force.z;
}
`;
}

/**
 * Sum an array into one partial per workgroup.
 *
 * 🔴 THE TREE IS DETERMINISTIC AND A SEQUENTIAL LOOP WOULD NOT BE FASTER OR
 * MORE ACCURATE. A fixed tree reduction gives the same answer every run, and
 * pairwise summation over 256 values has error growing as log(n) rather than n
 * - so it is strictly better than the host loop it replaces, as well as
 * avoiding a readback of the whole grid every step.
 */
export function reduceShader(count, gridWidth) {
  return `
const COUNT: u32 = ${count}u;
const GRID_WIDTH: u32 = ${gridWidth}u;
@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read_write> partials: array<f32>;
var<workgroup> scratch: array<f32, ${REDUCE_WORKGROUP}>;

@compute @workgroup_size(${REDUCE_WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3<u32>,
        @builtin(local_invocation_id) local: vec3<u32>,
        @builtin(workgroup_id) group: vec3<u32>) {
  let index = id.x + id.y * GRID_WIDTH * ${REDUCE_WORKGROUP}u;
  var value = 0.0;
  if (index < COUNT) { value = source[index]; }
  scratch[local.x] = value;
  workgroupBarrier();
  for (var stride = ${REDUCE_WORKGROUP / 2}u; stride > 0u; stride = stride >> 1u) {
    if (local.x < stride) { scratch[local.x] = scratch[local.x] + scratch[local.x + stride]; }
    workgroupBarrier();
  }
  if (local.x == 0u) {
    partials[group.x + group.y * GRID_WIDTH] = scratch[0];
  }
}
`;
}

/**
 * Encode the whole reciprocal-space calculation.
 *
 * 🔴 THE TRANSFORM'S PARITY DECIDES WHICH BUFFER THE CONVOLUTION READS, and
 * that is tracked rather than assumed. A 24^3 grid runs twelve stages and ends
 * where it started; a 27x25x16 grid runs nine and does not. `encodeFft3d`
 * returns the buffer holding its result for exactly this reason - hard-coding
 * `gridA` is correct for half of all grid sizes and silently reads stale data
 * for the other half.
 *
 * The forces are ACCUMULATED, so the caller clears them (or runs the
 * direct-space pass first, which writes them).
 *
 * @returns {{energyPartials: object, partialCount: number}} the caller sums the
 *   partials; there are a few hundred of them, against a grid of tens of
 *   thousands of points.
 */
export function encodePme(device, encoder, resources, atomCount, grid) {
  const maxPerDimension = device.limits.maxComputeWorkgroupsPerDimension;
  const cache = pipelineCacheFor(device);
  const gridPoints = grid[0] * grid[1] * grid[2];
  const key = `${atomCount}:${grid.join("x")}`;

  const dispatchFor = (items, workgroupSize) => {
    const workgroups = Math.ceil(items / workgroupSize);
    const [x, y, z] = dispatchGrid(workgroups, maxPerDimension);
    return { x, y, z, gridWidth: x === workgroups ? workgroups : maxPerDimension };
  };
  const run = (label, code, bindings, dispatch) => {
    const pipeline = cache.get(label, code);
    const pass = encoder.beginComputePass({ label });
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

  const atomDispatch = dispatchFor(atomCount, PME_WORKGROUP);
  const pointDispatch = dispatchFor(gridPoints, PME_WORKGROUP);

  // The histogram of charge accumulates, so the fixed grid starts at zero.
  encoder.clearBuffer(resources.fixedGrid.buffer);

  run(`pme:spread:${key}:${atomDispatch.gridWidth}`,
    spreadShader(grid, atomCount, atomDispatch.gridWidth),
    [resources.params, resources.positions, resources.charges, resources.fixedGrid],
    atomDispatch);

  run(`pme:convert:${key}:${pointDispatch.gridWidth}`,
    convertShader(grid, atomCount, pointDispatch.gridWidth),
    [resources.fixedGrid, resources.gridA],
    pointDispatch);

  const forward = encodeFft3d(device, encoder, grid,
    { a: resources.gridA, b: resources.gridB }, 1);

  run(`pme:convolve:${key}:${pointDispatch.gridWidth}`,
    convolveShader(grid, atomCount, pointDispatch.gridWidth),
    [resources.params, forward.current, resources.moduli, resources.gridEnergy],
    pointDispatch);

  // 🔴 THE INVERSE TRANSFORM IS UNNORMALISED, which is the convention the host
  // reference uses: the factor of the grid volume that a forward-then-inverse
  // pair carries is absorbed into the convolution's prefactor. Dividing by
  // nx*ny*nz here to make it a "proper" inverse makes every reciprocal force
  // too small by exactly that factor.
  const inverse = encodeFft3d(device, encoder, grid,
    { a: forward.current, b: forward.other }, -1);

  run(`pme:gather:${key}:${atomDispatch.gridWidth}`,
    gatherShader(grid, atomCount, atomDispatch.gridWidth),
    [resources.params, resources.positions, resources.charges,
      inverse.current, resources.forces],
    atomDispatch);

  const reduceDispatch = dispatchFor(gridPoints, REDUCE_WORKGROUP);
  const partialCount = Math.ceil(gridPoints / REDUCE_WORKGROUP);
  run(`pme:reduce:${gridPoints}:${reduceDispatch.gridWidth}`,
    reduceShader(gridPoints, reduceDispatch.gridWidth),
    [resources.gridEnergy, resources.energyPartials],
    reduceDispatch);

  return { energyPartials: resources.energyPartials, partialCount };
}
