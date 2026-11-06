/**
 * The Ewald split, the reciprocal grid, and the B-spline moduli behind them.
 *
 * Everything here is host-side setup computed once per system. It is in
 * `shared/` because the CPU reference and the WebGPU kernels must use the
 * SAME alpha, the SAME grid and the SAME moduli or their disagreement is about
 * the split rather than about the arithmetic - which is the single easiest way
 * to spend a day blaming a kernel.
 */

/**
 * B-spline interpolation order.
 *
 * 🔴 FIVE, BECAUSE OPENMM USES FIVE. `PME_ORDER = 5` in OpenMM's
 * ReferencePME.cpp and in its CUDA platform, and the oracle this port is
 * checked against is OpenMM. Order 4 is the more common textbook choice and
 * would be a perfectly good PME - just not the same one, differing by far more
 * than the grid tolerance and in a way that looks exactly like a spreading bug.
 *
 * 🔴 AND ORDER 5 IS PRECISELY THE CASE WITH A ZERO MODULUS, which is the
 * opposite of what this comment claimed for most of an afternoon. The order-5
 * B-spline has FOUR nonzero values at integer points - 1/24, 11/24, 11/24,
 * 1/24 - and at the Nyquist frequency the DFT alternates their signs, so they
 * cancel exactly. The modulus there came out 2.1e-33 rather than 0 through
 * floating point, so a `<= 0` guard let it through, the convolution divided by
 * it, and the reciprocal energy was 2.9e57 kJ/mol. See patchZeroModuli: AMBER's
 * own `dftmod` has carried this fix-up since 1995 and so does OpenMM, which is
 * why matching OpenMM requires it rather than merely benefiting from it.
 */
export const PME_ORDER = 5;

/**
 * The Ewald splitting parameter from the cutoff and the requested accuracy.
 *
 * `alpha = sqrt(-log(2*tol)) / cutoff`, which is OpenMM's rule. Verified
 * against the oracle: cutoff 1.0 nm with tolerance 5e-4 gives 2.628261, which
 * is what `getPMEParametersInContext` reports for the notebook's settings.
 *
 * @param {number} cutoff nm
 * @param {number} tolerance OpenMM's ewaldErrorTolerance
 * @returns {number} alpha in 1/nm
 */
export function alphaFromTolerance(cutoff, tolerance) {
  if (!(cutoff > 0)) throw new RangeError(`cutoff must be positive, got ${cutoff}`);
  if (!(tolerance > 0 && tolerance < 0.5)) {
    throw new RangeError(`ewald tolerance must be in (0, 0.5), got ${tolerance}`);
  }
  return Math.sqrt(-Math.log(2 * tolerance)) / cutoff;
}

/**
 * Sizes this port's FFT can transform: 2^a * 3^b * 5^c.
 *
 * 🔴 OPENMM'S REFERENCE PLATFORM PICKED 22 FOR ONE OF THESE BOXES, AND 22 IS
 * 2 x 11. A mixed-radix FFT with radices 2, 3 and 5 cannot do a factor of 11
 * at all, and neither can one with 2, 3, 5 and 7 - OpenMM's Reference platform
 * does not restrict its grid because it does not use an FFT of that kind. So
 * the grid is chosen HERE, from this set, and forced on the oracle too
 * (`--pme-grid` on dump_openmm_reference.py). A port that accepted whatever
 * OpenMM chose would be asked for a transform it cannot perform, at run time,
 * on a user's box and not on a fixture.
 */
export function isTransformableSize(value) {
  let remaining = value;
  for (const radix of [2, 3, 5]) {
    while (remaining % radix === 0) remaining /= radix;
  }
  return remaining === 1;
}

/** The smallest transformable size at or above `minimum`. */
export function nextTransformableSize(minimum) {
  let candidate = Math.max(2, Math.ceil(minimum));
  while (!isTransformableSize(candidate)) candidate += 1;
  return candidate;
}

/**
 * The reciprocal grid for a box, by OpenMM's spacing rule, rounded up to a
 * size this port can transform.
 *
 * OpenMM's rule is `ceil(2*alpha*L / (3*tol^0.2))`, which reproduces its own
 * [22, 24, 21] for the solvated fixture before the rounding below is applied.
 *
 * @param {Float64Array} boxVectors nine components, row-major
 * @param {number} alpha 1/nm
 * @param {number} tolerance
 * @returns {number[]} three grid sizes
 */
export function gridSizeFor(boxVectors, alpha, tolerance) {
  const lengths = [
    Math.hypot(boxVectors[0], boxVectors[1], boxVectors[2]),
    Math.hypot(boxVectors[3], boxVectors[4], boxVectors[5]),
    Math.hypot(boxVectors[6], boxVectors[7], boxVectors[8]),
  ];
  const denominator = 3 * Math.pow(tolerance, 0.2);
  return lengths.map(
    (length) => nextTransformableSize(Math.ceil(2 * alpha * length / denominator)),
  );
}

/**
 * Cardinal B-spline weights, and their derivatives, at a fractional offset.
 *
 * The de Boor recursion, in the form PME implementations use: `weights[k]` is
 * the weight of the grid point `k` away from the atom's floor, for
 * `k = 0 .. order-1`.
 *
 * 🔴 THE DERIVATIVE IS NOT THE DERIVATIVE OF THE FINAL WEIGHTS. It is taken at
 * order-1 and then differenced, because d/dw M_n(w) = M_{n-1}(w) - M_{n-1}(w-1).
 * Differentiating the order-n weights numerically, or analytically after the
 * last recursion step, gives something close enough to look right on an energy
 * and wrong enough to make the reciprocal forces not the gradient of the
 * reciprocal energy - which shows up as energy drift in a simulation that
 * conserves nothing, long after the kernel was declared correct.
 *
 * @param {number} offset fractional position in [0, 1)
 * @param {number} order
 * @returns {{weights: Float64Array, derivatives: Float64Array}}
 */
export function bsplineWeights(offset, order = PME_ORDER) {
  const weights = new Float64Array(order);
  weights[0] = 1 - offset;
  weights[1] = offset;
  // Build up to order-1, keeping the (order-1) values for the derivative.
  for (let k = 3; k < order; k += 1) {
    const div = 1 / (k - 1);
    weights[k - 1] = div * offset * weights[k - 2];
    for (let j = 1; j < k - 1; j += 1) {
      weights[k - j - 1] = div
        * ((offset + j) * weights[k - j - 2] + (k - j - offset) * weights[k - j - 1]);
    }
    weights[0] = div * (1 - offset) * weights[0];
  }
  const derivatives = new Float64Array(order);
  derivatives[0] = -weights[0];
  for (let j = 1; j < order; j += 1) {
    derivatives[j] = weights[j - 1] - weights[j];
  }
  // The final recursion step, after the derivative has been taken.
  const div = 1 / (order - 1);
  weights[order - 1] = div * offset * weights[order - 2];
  for (let j = 1; j < order - 1; j += 1) {
    weights[order - j - 1] = div
      * ((offset + j) * weights[order - j - 2] + (order - j - offset) * weights[order - j - 1]);
  }
  weights[0] = div * (1 - offset) * weights[0];
  return { weights, derivatives };
}

/**
 * The squared B-spline moduli |b(m)|^2 for one grid dimension.
 *
 * These divide the reciprocal-space convolution, correcting for the fact that
 * the charge was interpolated onto a grid rather than placed exactly. Computed
 * as the explicit DFT of the B-spline's integer-point values, which is the
 * definition - O(order * gridSize) once per system, so there is no reason to
 * use a recursive shortcut whose correctness is harder to see.
 *
 * @param {number} gridSize
 * @param {number} order
 * @returns {Float64Array} gridSize moduli
 */
export function bsplineModuli(gridSize, order = PME_ORDER) {
  // The B-spline's values at integer points: the weights at offset 0.
  const atZero = bsplineWeights(0, order).weights;
  const moduli = new Float64Array(gridSize);
  for (let m = 0; m < gridSize; m += 1) {
    let real = 0;
    let imaginary = 0;
    for (let k = 0; k < order; k += 1) {
      const angle = (2 * Math.PI * m * k) / gridSize;
      real += atZero[k] * Math.cos(angle);
      imaginary += atZero[k] * Math.sin(angle);
    }
    moduli[m] = real * real + imaginary * imaginary;
  }
  return patchZeroModuli(moduli, order);
}

/**
 * Replace a vanishing modulus with the mean of its neighbours.
 *
 * 🔴 THIS IS NOT A FUDGE, IT IS THE REFERENCE BEHAVIOUR. AMBER's `dftmod`
 * (pmesh_kspace.F90) and OpenMM's `pme_calculate_bsplines_moduli` both test
 * each modulus against a small epsilon and substitute the average of its two
 * cyclic neighbours. The Nyquist component of the charge grid carries no
 * physical information - it is the frequency the grid cannot represent - so the
 * value put there affects nothing except the division it prevents. Matching
 * OpenMM to 1e-10 requires using ITS substitution, not merely a finite one.
 *
 * 🔴 AND THE THRESHOLD CANNOT BE `=== 0`. The cancellation is exact in
 * arithmetic and 2.1e-33 in floating point, so an equality test passes the
 * value straight through to the divide. 1e-7 is AMBER's own threshold, and the
 * moduli are O(1) quantities, so it is nowhere near a legitimate value.
 */
function patchZeroModuli(moduli, order) {
  const patched = Float64Array.from(moduli);
  const size = moduli.length;
  for (let m = 0; m < size; m += 1) {
    if (moduli[m] < 1e-7) {
      patched[m] = 0.5 * (moduli[(m - 1 + size) % size] + moduli[(m + 1) % size]);
    }
  }
  for (let m = 0; m < size; m += 1) {
    if (!(patched[m] > 0)) {
      throw new Error(
        `B-spline modulus ${m} of ${size} is still ${patched[m]} after the `
        + `neighbour substitution at order ${order}. Two adjacent zeros means `
        + "the grid is too small for this interpolation order.",
      );
    }
  }
  return patched;
}

/**
 * Everything the reciprocal sum needs, derived once.
 *
 * @param {object} options
 * @param {Float64Array} options.boxVectors nine components
 * @param {number} options.cutoff nm
 * @param {number} [options.tolerance]
 * @param {number[]} [options.grid] force the grid instead of deriving it
 * @param {number} [options.alpha] force alpha instead of deriving it
 */
export function pmeParameters(options) {
  const tolerance = options.tolerance ?? 0.0005;
  const alpha = options.alpha ?? alphaFromTolerance(options.cutoff, tolerance);
  const grid = options.grid ?? gridSizeFor(options.boxVectors, alpha, tolerance);
  for (const size of grid) {
    if (!isTransformableSize(size)) {
      throw new Error(
        `PME grid size ${size} is not a product of 2, 3 and 5, which is what `
        + "this port's FFT can transform. See isTransformableSize.",
      );
    }
    if (size < PME_ORDER) {
      throw new Error(
        `PME grid size ${size} is smaller than the interpolation order `
        + `${PME_ORDER}; the box is too small for this cutoff.`,
      );
    }
  }
  return {
    alpha,
    tolerance,
    order: PME_ORDER,
    grid,
    moduli: grid.map((size) => bsplineModuli(size, PME_ORDER)),
  };
}
