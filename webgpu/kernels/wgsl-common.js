/**
 * WGSL fragments every nonbonded kernel shares.
 *
 * Shared as TEXT rather than as a module, because WGSL has no imports. The
 * consequence worth remembering: a kernel's cache key must cover anything these
 * snippets embed, or two kernels built from different parameters collide. See
 * PipelineCache, which compares the whole source and names the differing line.
 */

/**
 * The simulation parameters, as a uniform.
 *
 * 🔴 EVERYTHING HERE IS A vec4 EVEN WHERE THREE COMPONENTS WOULD DO. WGSL's
 * uniform address space aligns a vec3 to 16 bytes but gives it a size of 12, so
 * a struct of vec3s has padding the host must reproduce exactly - and getting
 * it wrong does not fail validation, it reads the next field's bytes as
 * coordinates. Writing vec4 throughout makes the host layout trivially
 * correct: seven vec4s, 112 bytes, no padding to reason about.
 *
 * 🔴 AND THE BOX IS A UNIFORM RATHER THAN A BAKED CONSTANT so that a barostat
 * can change it without recompiling every kernel. Baking it would work today -
 * there is no barostat yet - and would be a trap the moment one arrives.
 */
export const PARAMS_STRUCT = `
struct Params {
  boxA: vec4<f32>,
  boxB: vec4<f32>,
  boxC: vec4<f32>,
  recipA: vec4<f32>,
  recipB: vec4<f32>,
  recipC: vec4<f32>,
  scalars: vec4<f32>,
};
const PARAMS_CUTOFF: u32 = 0u;
const PARAMS_LIST_CUTOFF: u32 = 1u;
const PARAMS_ALPHA: u32 = 2u;
const PARAMS_COULOMB: u32 = 3u;
`;

/** Floats per Params struct, for the host-side buffer. */
export const PARAMS_FLOATS = 28;

/**
 * Pack the uniform exactly as the struct above expects.
 *
 * @param {Float64Array} boxVectors nine components, reduced
 * @param {Float64Array} reciprocal nine components, rows of the inverse
 */
export function packParams(boxVectors, reciprocal, scalars) {
  const data = new Float32Array(PARAMS_FLOATS);
  for (let row = 0; row < 3; row += 1) {
    data[row * 4] = boxVectors[row * 3];
    data[row * 4 + 1] = boxVectors[row * 3 + 1];
    data[row * 4 + 2] = boxVectors[row * 3 + 2];
    data[12 + row * 4] = reciprocal[row * 3];
    data[12 + row * 4 + 1] = reciprocal[row * 3 + 1];
    data[12 + row * 4 + 2] = reciprocal[row * 3 + 2];
  }
  data[24] = scalars.cutoff;
  data[25] = scalars.listCutoff;
  data[26] = scalars.alpha;
  data[27] = scalars.coulomb;
  return data;
}

/**
 * Minimum image, fractional coordinates, and the cell index.
 *
 * 🔴 `floor(x + 0.5)` AND NOT `round(x)`. WGSL's `round` breaks ties to the
 * EVEN whole number; JavaScript's `Math.round` breaks them towards positive
 * infinity. They differ at exactly d/L = +-0.5 - an atom precisely on the box
 * face - and there the two implementations would choose different periodic
 * images for that pair. `floor(x + 0.5)` reproduces `Math.round` on both signs,
 * so the kernel and the reference it is tested against cannot disagree even in
 * that measure-zero case.
 *
 * 🔴 AND THE THREE SUBTRACTIONS ARE ORDERED c, b, a. That order is only
 * correct for a REDUCED box, which boxVectorsFromLengthsAndAngles guarantees.
 * Any other order is right for a rectangular cell and wrong for a sheared one.
 */
export const GEOMETRY_FUNCTIONS = `
fn minimumImage(delta: vec3<f32>) -> vec3<f32> {
  var d = delta;
  d = d - floor(d.z / params.boxC.z + 0.5) * params.boxC.xyz;
  d = d - floor(d.y / params.boxB.y + 0.5) * params.boxB.xyz;
  d = d - floor(d.x / params.boxA.x + 0.5) * params.boxA.xyz;
  return d;
}

fn fractional(p: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(
    dot(p, vec3<f32>(params.recipA.x, params.recipB.x, params.recipC.x)),
    dot(p, vec3<f32>(params.recipA.y, params.recipB.y, params.recipC.y)),
    dot(p, vec3<f32>(params.recipA.z, params.recipB.z, params.recipC.z)));
}

fn atomPosition(atom: u32) -> vec3<f32> {
  return vec3<f32>(positions[atom * 3u], positions[atom * 3u + 1u], positions[atom * 3u + 2u]);
}
`;

/**
 * The cell index of a position. Needs CELLS_X/Y/Z as constants.
 *
 * 🔴 THE CLAMP IS NOT DEFENSIVE, IT IS NECESSARY. `wrapped` is computed as
 * `f - floor(f)`, which is in [0, 1) in exact arithmetic - but in f32 a value
 * of f just below an integer rounds so that `f - floor(f)` evaluates to exactly
 * 1.0. Multiplying by the cell count then gives CELLS_X, one past the end, and
 * the atom is written into another cell's slice or past the array. Atoms sit on
 * cell boundaries constantly, so this is a matter of steps, not of luck.
 */
export const CELL_INDEX_FUNCTION = `
fn cellOfPosition(p: vec3<f32>) -> u32 {
  let f = fractional(p);
  let wrapped = f - floor(f);
  let counts = vec3<f32>(f32(CELLS_X), f32(CELLS_Y), f32(CELLS_Z));
  let scaled = wrapped * counts;
  let cx = min(u32(scaled.x), CELLS_X - 1u);
  let cy = min(u32(scaled.y), CELLS_Y - 1u);
  let cz = min(u32(scaled.z), CELLS_Z - 1u);
  return (cx * CELLS_Y + cy) * CELLS_Z + cz;
}
`;

/**
 * erfc, and the Gaussian that its derivative needs.
 *
 * 🔴 THE FRACTIONAL ERROR IS 1.19e-6 AND THAT IS THE f32 FLOOR, NOT THE
 * COEFFICIENTS. These are Numerical Recipes' `erfcc` constants, whose published
 * claim is a fractional error below 1.2e-7 - in double precision. Measured
 * here in f32 against the Cody reference in shared/erf.js, over the domain a
 * 1.0 nm cutoff actually uses (alpha*r up to 2.8), the error is **1.191e-6**,
 * twenty times f32 epsilon. Refitting all ten coefficients by least squares
 * against relative error over exactly that domain gives **1.191e-6** - the same
 * number to four digits. So the limit is the arithmetic: `exp(-x*x + P(t))` has
 * an argument near -7.8 at the cutoff, and one f32 ulp there is 4.7e-7 of
 * absolute argument error, which an exponential turns straight into relative
 * output error. A better polynomial cannot help, and this is why the nonbonded
 * differential test's Coulomb tolerance is derived from 1.19e-6 rather than
 * from f32 epsilon.
 *
 * 🔴 AND AN ABSOLUTE-ERROR APPROXIMATION WOULD BE USELESS HERE. Abramowitz &
 * Stegun 7.1.26 is accurate to 1.5e-7 ABSOLUTE, and erfc at the cutoff is
 * 2.0e-4 - so it carries 7.5e-4 relative error on exactly the long-range terms
 * the Ewald split exists to get right. Fractional accuracy is the requirement.
 */
export const ERFC_FUNCTION = `
fn erfcApprox(x: f32) -> f32 {
  let z = abs(x);
  let t = 1.0 / (1.0 + 0.5 * z);
  var p = 0.17087277;
  p = -0.82215223 + t * p;
  p = 1.48851587 + t * p;
  p = -1.13520398 + t * p;
  p = 0.27886807 + t * p;
  p = -0.18628806 + t * p;
  p = 0.09678418 + t * p;
  p = 0.37409196 + t * p;
  p = 1.00002368 + t * p;
  p = -1.26551223 + t * p;
  let ans = t * exp(-z * z + p);
  return select(2.0 - ans, ans, x >= 0.0);
}

const INVERSE_SQRT_PI: f32 = 0.5641895835477563;
`;
