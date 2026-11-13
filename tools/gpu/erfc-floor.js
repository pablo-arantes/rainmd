/**
 * Measure the shipped WGSL `erfcApprox` against a near-exact reference, on the
 * device.
 *
 * 🔴 THIS NUMBER WAS A LITERAL IN THE SOURCE AND NOTHING PRODUCED IT. The
 * Coulomb tolerance of the nonbonded differential test is four times the
 * relative error of the f32 erfc, so that error IS the bar - and it was written
 * as `const erfcFloor = 1.19e-6` with a comment saying it had been measured,
 * by nothing a reader could run. Three different figures for it existed in the
 * tree at once (1.191e-6 in docs/STATUS.md, 1.478e-6 from a host emulation in
 * docs/REVIEW.md) and they could not all be right.
 *
 * 🔴 AND A HOST EMULATION CANNOT SETTLE IT, WHICH IS WHY THIS RUNS ON THE GPU.
 * `Math.fround` around each operation reproduces WGSL's rounding for +, -, *
 * and /, but `exp` is a different function on every backend: WGSL requires only
 * 3 ulp for `exp`, and `Math.exp` is a correctly-rounded f64 exponential that
 * is then rounded once. The emulation is therefore a LOWER bound on the device,
 * and the gap between them is reported rather than assumed.
 *
 * The bound the result is held to is computed, not chosen: see
 * `argumentUlpBound`.
 */
import { erfc } from "../../shared/erf.js";
import { ERFC_FUNCTION } from "../../webgpu/kernels/wgsl-common.js";
import { readBuffer, encodePass } from "../../webgpu/runtime/execution.js";

const WORKGROUP = 64;

/**
 * The relative error one f32 rounding of the exponential's argument costs.
 *
 * 🔴 THE WHOLE ACCURACY STORY IS IN THIS FUNCTION. The kernel evaluates
 * `t * exp(-z*z + P(t))`, and at the cutoff the argument of that exponential is
 * about -z^2 - so the argument is a number of magnitude z^2 held in f32, whose
 * spacing there is `2^(floor(log2(z^2)) - 23)`. An exponential turns an
 * ABSOLUTE error in its argument straight into a RELATIVE error in its result,
 * so that spacing is a floor on the relative accuracy of erfc that no
 * polynomial can cross: at a 1.0 nm cutoff z^2 is 7.4, one f32 ulp there is
 * 4.8e-7, and the measured error is a small multiple of it.
 *
 * A better fit cannot help. tools/fit-erfc.py reports the polynomial's own
 * error at 1.0e-7 in exact arithmetic - seven times smaller than this - and the
 * f32 result is unmoved by refitting it.
 */
export function argumentUlpBound(zMax, roundings = 4) {
  const argument = Math.max(zMax * zMax, 1);
  return roundings * 2 ** (Math.floor(Math.log2(argument)) - 23);
}

/** The shader's own arithmetic, in f64 with every step rounded to f32. */
export function emulateErfc(z) {
  const round = Math.fround;
  const t = round(1 / round(1 + round(0.5 * z)));
  const coefficients = [...ERFC_FUNCTION.matchAll(/p = (-?[\d.]+) \+ t \* p;/g)]
    .map((match) => round(Number(match[1])));
  const lead = round(Number(/var p = (-?[\d.]+);/.exec(ERFC_FUNCTION)[1]));
  let p = lead;
  for (const coefficient of coefficients) p = round(coefficient + round(t * p));
  return round(t * round(Math.exp(round(-round(z * z) + p))));
}

/**
 * Evaluate `erfcApprox` on the device at `samples` points of (0, zMax], and
 * compare with Cody's reference in f64.
 *
 * `perturb` textually alters one shipped coefficient, for the arm that has to
 * show this measurement can fail; it asserts the substitution landed, so the
 * string under test is always the one the kernels import.
 */
export async function measureErfcFloor(device, allocator, options = {}) {
  const { zMax = 2.8, samples = 1 << 16, perturb = null, negative = false } = options;
  let source = ERFC_FUNCTION;
  if (perturb !== null) {
    source = ERFC_FUNCTION.replace(perturb.from, perturb.to);
    if (source === ERFC_FUNCTION) {
      throw new Error(`perturbation "${perturb.from}" is not in the shipped erfc`);
    }
  }
  const points = new Float32Array(samples);
  for (let index = 0; index < samples; index += 1) {
    points[index] = (zMax * (index + 1)) / samples * (negative ? -1 : 1);
  }
  const inputs = allocator.upload("erfc.inputs", points, GPUBufferUsage.STORAGE);
  const outputs = allocator.allocate("erfc.outputs", samples * 4,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
  try {
    const code = `${source}
const COUNT: u32 = ${samples}u;
@group(0) @binding(0) var<storage, read> inputs: array<f32>;
@group(0) @binding(1) var<storage, read_write> outputs: array<f32>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= COUNT) { return; }
  outputs[id.x] = erfcApprox(inputs[id.x]);
}
`;
    const encoder = device.createCommandEncoder({ label: "erfc.probe" });
    encodePass(device, encoder, `erfc:${samples}:${perturb === null ? "shipped" : "perturbed"}:${negative}`,
      code, [inputs, outputs], Math.ceil(samples / WORKGROUP));
    device.queue.submit([encoder.finish()]);
    const values = await readBuffer(device, allocator, outputs, samples * 4, Float32Array);

    let worst = 0;
    let worstAt = 0;
    let worstEmulated = 0;
    let deviceVsEmulation = 0;
    for (let index = 0; index < samples; index += 1) {
      const z = points[index];
      const reference = erfc(z);
      const relative = Math.abs(values[index] - reference) / Math.abs(reference);
      if (relative > worst) { worst = relative; worstAt = z; }
      if (!negative) {
        const emulated = emulateErfc(z);
        worstEmulated = Math.max(worstEmulated,
          Math.abs(emulated - reference) / Math.abs(reference));
        deviceVsEmulation = Math.max(deviceVsEmulation,
          Math.abs(values[index] - emulated) / Math.abs(reference));
      }
    }
    return {
      zMax, samples, worst, worstAt, worstEmulated, deviceVsEmulation,
      bound: argumentUlpBound(zMax),
    };
  } finally {
    outputs.release();
    inputs.release();
  }
}
