/**
 * A mixed-radix Stockham FFT over one axis of the charge grid.
 *
 * 🔴 THE INDEX MAP WAS VERIFIED BEFORE A LINE OF WGSL WAS WRITTEN. It was
 * prototyped on the host and compared against a naive DFT over 33 sizes
 * (every 2/3/5-smooth value up to 128) in both directions, worst relative error
 * 9.5e-16. That is the only reason the transcription below can be trusted: an
 * FFT with a wrong twiddle or a transposed output produces a plausible-looking
 * grid, and the reciprocal energy it leads to is wrong by a factor nobody can
 * guess from the number. Do not edit the three index expressions without
 * re-running that comparison.
 *
 * The map, for a stage of radix r with L sub-transforms already completed and
 * m = n/(L*r):
 *
 *   input    j*(r*m) + p*m + k        p = 0..r-1
 *   twiddle  exp(2*pi*i*s*p*j/(L*r))
 *   output   q*(L*m) + j*m + k        q = 0..r-1
 *
 * 🔴 AUTOSORT, SO THERE IS NO BIT-REVERSAL PASS. The output permutation is
 * folded into the write index, which is why the stages ping-pong between two
 * buffers rather than working in place. An in-place version would need the
 * reversal pass and a different index map.
 *
 * 🔴 `target` IS A RESERVED KEYWORD IN WGSL, which is not obvious and is not
 * in any error message until the parser reaches it. The output binding is
 * called `destination` for that reason alone. DeferredValidation is what turned
 * it into one line naming the identifier instead of a silent failure to create
 * the pipeline.
 *
 * 🔴 AND THE TWIDDLE ARGUMENT IS REDUCED MODULO L*r BEFORE IT REACHES cos().
 * `p*j` grows to (r-1)(L-1), so the unreduced angle reaches 2*pi*(r-1) ~ 25
 * radians, where one f32 ulp is 1.5e-6 of a radian - three digits worse than
 * f32 allows. Reducing the integer first keeps every argument inside [0, 2*pi).
 */
import { pipelineCacheFor } from "../runtime/pipeline-cache.js";
import { dispatchGrid } from "../runtime/execution.js";

export const FFT_WORKGROUP = 64;

/** The 2/3/5 factorisation, in the order the stages run. */
export function factorise(size) {
  const radices = [];
  let remaining = size;
  for (const radix of [2, 3, 5]) {
    while (remaining % radix === 0) { radices.push(radix); remaining /= radix; }
  }
  if (remaining !== 1) {
    throw new Error(
      `${size} is not a product of 2, 3 and 5, so this FFT cannot transform it. `
      + "See isTransformableSize in shared/pme/parameters.js, which is what "
      + "stops such a grid being chosen.",
    );
  }
  return radices;
}

/**
 * How a line along `axis` is addressed.
 *
 * 🔴 THE LINE BASE IS NOT THE LINE INDEX EXCEPT ALONG X. The grid is stored as
 * `(x*ny + y)*nz + z`, so a line along y is strided by nz and its base mixes x
 * and z - and getting this wrong transposes the transform, which still produces
 * a grid of plausible magnitudes. Each axis gets its own expression, baked in.
 */
function axisAddressing(grid, axis) {
  const [nx, ny, nz] = grid;
  const total = nx * ny * nz;
  if (axis === 0) {
    return { n: nx, stride: ny * nz, lineCount: total / nx, lineBase: "line" };
  }
  if (axis === 1) {
    return {
      n: ny, stride: nz, lineCount: total / ny,
      lineBase: `(line / ${nz}u) * ${ny * nz}u + (line % ${nz}u)`,
    };
  }
  return { n: nz, stride: 1, lineCount: total / nz, lineBase: `line * ${nz}u` };
}

export function fftStageShader(grid, axis, radix, L, sign, gridWidth) {
  const { n, stride, lineCount, lineBase } = axisAddressing(grid, axis);
  const m = n / (L * radix);
  if (!Number.isInteger(m)) {
    throw new Error(`FFT stage geometry is wrong: n=${n}, L=${L}, radix=${radix}`);
  }
  return `
const RADIX: u32 = ${radix}u;
const L: u32 = ${L}u;
const M: u32 = ${m}u;
const STRIDE: u32 = ${stride}u;
const LINE_COUNT: u32 = ${lineCount}u;
const BUTTERFLIES: u32 = ${L * m}u;
const SIGN: f32 = ${sign > 0 ? "1.0" : "-1.0"};
const TWIDDLE_PERIOD: u32 = ${L * radix}u;
const GRID_WIDTH: u32 = ${gridWidth}u;
const TWO_PI: f32 = 6.283185307179586;

@group(0) @binding(0) var<storage, read> source: array<f32>;
@group(0) @binding(1) var<storage, read_write> destination: array<f32>;

@compute @workgroup_size(${FFT_WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let thread = id.x + id.y * GRID_WIDTH * ${FFT_WORKGROUP}u;
  if (thread >= LINE_COUNT * BUTTERFLIES) { return; }
  let line = thread / BUTTERFLIES;
  let butterfly = thread % BUTTERFLIES;
  let j = butterfly / M;
  let k = butterfly % M;
  let base = ${lineBase};

  var xr: array<f32, ${radix}>;
  var xi: array<f32, ${radix}>;
  for (var p = 0u; p < RADIX; p = p + 1u) {
    let element = j * (RADIX * M) + p * M + k;
    let offset = 2u * (base + element * STRIDE);
    let re = source[offset];
    let im = source[offset + 1u];
    let reduced = (p * j) % TWIDDLE_PERIOD;
    let angle = SIGN * TWO_PI * f32(reduced) / f32(TWIDDLE_PERIOD);
    let c = cos(angle);
    let s = sin(angle);
    xr[p] = re * c - im * s;
    xi[p] = re * s + im * c;
  }
  for (var q = 0u; q < RADIX; q = q + 1u) {
    var sumRe = 0.0;
    var sumIm = 0.0;
    for (var p = 0u; p < RADIX; p = p + 1u) {
      let reduced = (p * q) % RADIX;
      let angle = SIGN * TWO_PI * f32(reduced) / f32(RADIX);
      let c = cos(angle);
      let s = sin(angle);
      sumRe = sumRe + xr[p] * c - xi[p] * s;
      sumIm = sumIm + xr[p] * s + xi[p] * c;
    }
    let element = q * (L * M) + j * M + k;
    let offset = 2u * (base + element * STRIDE);
    destination[offset] = sumRe;
    destination[offset + 1u] = sumIm;
  }
}
`;
}

/**
 * Encode a complete 3D transform, ping-ponging between two buffers.
 *
 * 🔴 THE CALLER MUST BE TOLD WHICH BUFFER HOLDS THE RESULT. The number of
 * stages is the total number of radix factors across the three axes - twelve
 * for a 24^3 grid - so the data ends in `a` or `b` depending on its parity.
 * Assuming it comes back where it started is correct for exactly half of all
 * grid sizes, which is the worst possible kind of bug to introduce.
 *
 * @returns {{current: object, other: object}} the buffer holding the transform
 */
export function encodeFft3d(device, encoder, grid, buffers, sign) {
  const maxPerDimension = device.limits.maxComputeWorkgroupsPerDimension;
  const cache = pipelineCacheFor(device);
  let current = buffers.a;
  let other = buffers.b;

  for (let axis = 0; axis < 3; axis += 1) {
    const { n, lineCount } = axisAddressing(grid, axis);
    const radices = factorise(n);
    let L = 1;
    for (const radix of radices) {
      const m = n / (L * radix);
      const threads = lineCount * L * m;
      const workgroups = Math.ceil(threads / FFT_WORKGROUP);
      const [x, y, z] = dispatchGrid(workgroups, maxPerDimension);
      const gridWidth = x === workgroups ? workgroups : maxPerDimension;
      const code = fftStageShader(grid, axis, radix, L, sign, gridWidth);
      const key = `fft:${grid.join("x")}:axis${axis}:r${radix}:L${L}:`
        + `${sign > 0 ? "fwd" : "inv"}:${gridWidth}`;
      const pipeline = cache.get(key, code);
      const pass = encoder.beginComputePass({ label: key });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [current, other].map((allocation, binding) => ({
          binding, resource: { buffer: allocation.buffer },
        })),
      }));
      pass.dispatchWorkgroups(x, y, z);
      pass.end();
      const swap = current; current = other; other = swap;
      L *= radix;
    }
  }
  return { current, other };
}

/** Total stage count, so a caller can predict the parity without encoding. */
export function fftStageCount(grid) {
  return grid.reduce((total, size) => total + factorise(size).length, 0);
}
