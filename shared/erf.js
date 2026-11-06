/**
 * erf and erfc to near machine precision, because JavaScript has neither.
 *
 * 🔴 THE CHEAP ERFC APPROXIMATIONS ARE NOT GOOD ENOUGH FOR EWALD. The
 * Numerical Recipes one-liner and Abramowitz-Stegun 7.1.26 are accurate to
 * about 1e-7 relative, and the direct-space Coulomb sum of a solvated system is
 * a cancellation of numbers in the tens of thousands down to a result in the
 * thousands - so a 1e-7 error per pair accumulates into the first decimal place
 * of the total energy. Worse, it does so SMOOTHLY, which reads as a small
 * systematic offset in the force field rather than as a bug.
 *
 * This is W. J. Cody's rational Chebyshev approximation (the netlib CALERF),
 * which is what C libraries ship, and it is checked against SciPy in
 * test/erf.test.js.
 */

const A = [
  3.16112374387056560e+00, 1.13864154151050156e+02,
  3.77485237685302021e+02, 3.20937758913846947e+03,
  1.85777706184603153e-01,
];
const B = [
  2.36012909523441209e+01, 2.44024637934444173e+02,
  1.28261652607737228e+03, 2.84423683343917062e+03,
];
const C = [
  5.64188496988670089e-01, 8.88314979438837594e+00,
  6.61191906371416295e+01, 2.98635138197400131e+02,
  8.81952221241769090e+02, 1.71204761263407058e+03,
  2.05107837782607147e+03, 1.23033935479799725e+03,
  2.15311535474403846e-08,
];
const D = [
  1.57449261107098347e+01, 1.17693950891312499e+02,
  5.37181101862009858e+02, 1.62138957456669019e+03,
  3.29079923573345963e+03, 4.36261909014324716e+03,
  3.43936767414372164e+03, 1.23033935480374942e+03,
];
const P = [
  3.05326634961232344e-01, 3.60344899949804439e-01,
  1.25781726111229246e-01, 1.60837851487422766e-02,
  6.58749161529837803e-04, 1.63153871373020978e-02,
];
const Q = [
  2.56852019228982242e+00, 1.87295284992346047e+00,
  5.27905102951428412e-01, 6.05183413124413191e-02,
  2.33520497626869185e-03,
];

/** 1/sqrt(pi). */
const SQRT_PI_INVERSE = 5.6418958354775628695e-1;

/**
 * 🔴 THE `exp(-z*z) * exp(-del)` SPLIT IS NOT A TYPO AND MUST NOT BE FUSED.
 * Cody truncates y to a sixteenth, so `z*z` is exactly representable and its
 * exponential is computed without the rounding that `exp(-y*y)` would carry
 * into the leading digits. Writing it as one exponential loses two to three
 * digits in the tail, which is exactly where Ewald reads it.
 */
function scaledExponential(y) {
  const z = Math.trunc(y * 16) / 16;
  const del = (y - z) * (y + z);
  return Math.exp(-z * z) * Math.exp(-del);
}

export function erfc(x) {
  const y = Math.abs(x);
  let result;
  if (y <= 0.46875) {
    const z = y > 1e-20 ? y * y : 0;
    let numerator = A[4] * z;
    let denominator = z;
    for (let index = 0; index < 3; index += 1) {
      numerator = (numerator + A[index]) * z;
      denominator = (denominator + B[index]) * z;
    }
    result = 1 - (x * (numerator + A[3])) / (denominator + B[3]);
    return result;
  }
  if (y <= 4) {
    let numerator = C[8] * y;
    let denominator = y;
    for (let index = 0; index < 7; index += 1) {
      numerator = (numerator + C[index]) * y;
      denominator = (denominator + D[index]) * y;
    }
    result = ((numerator + C[7]) / (denominator + D[7])) * scaledExponential(y);
  } else if (y >= 26.543) {
    // Below the smallest normal double once multiplied by exp(-y*y).
    result = 0;
  } else {
    const z = 1 / (y * y);
    let numerator = P[5] * z;
    let denominator = z;
    for (let index = 0; index < 4; index += 1) {
      numerator = (numerator + P[index]) * z;
      denominator = (denominator + Q[index]) * z;
    }
    result = (z * (numerator + P[4])) / (denominator + Q[4]);
    result = ((SQRT_PI_INVERSE - result) / y) * scaledExponential(y);
  }
  return x < 0 ? 2 - result : result;
}

export function erf(x) {
  return 1 - erfc(x);
}
