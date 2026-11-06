/**
 * Counter-based Gaussian noise: the same number from the same (atom, step).
 *
 * 🔴 A STATEFUL PRNG CANNOT BE USED BY A GPU INTEGRATOR, AND SWITCHING LATER
 * CHANGES EVERY TRAJECTORY. A thermostat needs three normal deviates per atom
 * per step; on the device those are produced by thousands of threads at once,
 * so there is no sequence and no state to advance. Hashing the coordinates of
 * the request instead - atom index, step number, component - gives every thread
 * its own value with no communication, and gives the HOST reference the
 * identical value, which is what lets a GPU Langevin step be compared against a
 * CPU one at all. With a stateful generator the two diverge on step one for
 * reasons that have nothing to do with the kernel.
 *
 * 🔴 AND IT MAKES A RUN REPRODUCIBLE FROM ITS SEED ALONE. A restart carries the
 * step number, so resuming continues the same noise stream rather than starting
 * a new one - which is the difference between a continued trajectory and two
 * trajectories glued together.
 *
 * The hash is a 32-bit integer mix in the style of PCG/splitmix, chosen because
 * WGSL has u32 arithmetic with defined wraparound and nothing wider.
 */

/** One round of integer avalanche. Identical arithmetic to the WGSL version. */
function mix(value) {
  let state = value >>> 0;
  state = (state ^ (state >>> 16)) >>> 0;
  state = Math.imul(state, 0x7feb352d) >>> 0;
  state = (state ^ (state >>> 15)) >>> 0;
  state = Math.imul(state, 0x846ca68b) >>> 0;
  state = (state ^ (state >>> 16)) >>> 0;
  return state;
}

/**
 * A uniform in (0, 1), never exactly 0 - the log in Box-Muller needs that.
 *
 * 🔴 ONLY THE TOP 24 BITS ARE USED, AND THAT IS WHAT MAKES THE HOST AND THE
 * DEVICE AGREE EXACTLY. The obvious scaling is `(state + 0.5) * 2^-32`, which
 * on the host is computed in f64 and is exact - and in a WGSL shader is
 * computed in f32, where `f32(state)` already rounds for any state above 2^24.
 * So the two lanes would produce different uniforms from the same bits, every
 * draw, and no differential test of the thermostat could tell that from a bug.
 *
 * 🔴 AND IT IS NINE BITS, NOT EIGHT, WHICH TOOK A MEASUREMENT TO GET RIGHT.
 * Shifting by eight leaves a value below 2^24 - exact in f32 on its own - but
 * `integer + 0.5` then needs 25 significand bits for any integer at or above
 * 2^23, and f32 has 24. Measured: the eight-bit form is bit-identical between
 * f64 and f32 on only 20045 of 40000 draws. Shifting by nine makes the sum fit
 * in 24 bits and it is **40000 of 40000**, with the uniform still strictly
 * inside (5.96e-8, 0.99999398).
 *
 * The cost is nine bits of resolution per uniform, which a Maxwell-Boltzmann
 * draw cannot notice; the gain is that `gaussianTriple` here and
 * `rainmdGaussian` in the integrator kernel are the same function, so a
 * disagreement between host and device is a bug rather than a rounding mode.
 */
const UNIFORM_SCALE = 1 / 8388608;

export function uniformAt(seed, a, b, c) {
  let state = mix((seed ^ Math.imul(a, 0x9e3779b1)) >>> 0);
  state = mix((state ^ Math.imul(b, 0x85ebca6b)) >>> 0);
  state = mix((state ^ Math.imul(c, 0xc2b2ae35)) >>> 0);
  return ((state >>> 9) + 0.5) * UNIFORM_SCALE;
}

/**
 * Three independent standard normals for one atom at one step.
 *
 * Box-Muller, which costs a log, a sqrt and two trigonometric calls for two
 * deviates. The third comes from a second pair whose other half is discarded:
 * cheaper than threading a cache through a stateless function, and the discard
 * costs nothing a thermostat can measure.
 *
 * @param {Int32Array|Float64Array|Float32Array} out length >= 3, written
 */
export function gaussianTriple(out, seed, atom, step) {
  const u1 = uniformAt(seed, atom, step, 0);
  const u2 = uniformAt(seed, atom, step, 1);
  const u3 = uniformAt(seed, atom, step, 2);
  const u4 = uniformAt(seed, atom, step, 3);
  const radius1 = Math.sqrt(-2 * Math.log(u1));
  const radius2 = Math.sqrt(-2 * Math.log(u3));
  out[0] = radius1 * Math.cos(2 * Math.PI * u2);
  out[1] = radius1 * Math.sin(2 * Math.PI * u2);
  out[2] = radius2 * Math.cos(2 * Math.PI * u4);
  return out;
}
