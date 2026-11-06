/**
 * Vector helpers and the minimum-image convention.
 *
 * Plain functions over a flat Float64Array of positions, because that is what
 * both the reference and the GPU upload path hold and because a Vec3 class here
 * would allocate a few million objects per energy evaluation.
 */

/**
 * The shortest vector between two points under periodicity.
 *
 * 🔴 THE ORDER OF THE THREE SUBTRACTIONS IS NOT ARBITRARY, and it is only
 * correct for a REDUCED box. Reducing along c first, then b, then a, is what
 * makes a sheared cell's nearest image actually nearest; doing it in the other
 * order, or independently per axis, is correct for a rectangular box and wrong
 * for a truncated octahedron. See boxVectorsFromLengthsAndAngles, which is
 * what guarantees the reduction this relies on.
 *
 * 🔴 AND `round` MUST BE ROUND-HALF-AWAY-FROM-ZERO, NOT FLOOR(x+0.5). They
 * differ at exactly -0.5, where a pair sits precisely on the box edge. One
 * atom landing on the wrong side of that is a single pair interacting with the
 * wrong image, which moves the energy by a little and is untraceable.
 *
 * @param {Float64Array} out length 3, written in place
 * @param {Float64Array|null} box nine components, row-major, reduced
 */
export function minimumImage(out, box) {
  if (box === null) return out;
  const scale2 = Math.round(out[2] / box[8]);
  out[0] -= scale2 * box[6];
  out[1] -= scale2 * box[7];
  out[2] -= scale2 * box[8];
  const scale1 = Math.round(out[1] / box[4]);
  out[0] -= scale1 * box[3];
  out[1] -= scale1 * box[4];
  const scale0 = Math.round(out[0] / box[0]);
  out[0] -= scale0 * box[0];
  return out;
}

/** The reciprocal box vectors (rows of the inverse), for PME and for scaling. */
export function reciprocalBoxVectors(box) {
  const determinant = box[0] * box[4] * box[8];
  if (!(Math.abs(determinant) > 0)) {
    throw new Error("box vectors are singular; volume is zero");
  }
  // Reduced upper-triangular form: a = (ax,0,0), b = (bx,by,0), c = (cx,cy,cz)
  // so the inverse is cheap and exact in closed form.
  const ax = box[0];
  const bx = box[3];
  const by = box[4];
  const cx = box[6];
  const cy = box[7];
  const cz = box[8];
  return Float64Array.from([
    1 / ax, 0, 0,
    -bx / (ax * by), 1 / by, 0,
    (bx * cy - by * cx) / (ax * by * cz), -cy / (by * cz), 1 / cz,
  ]);
}

export function boxVolume(box) {
  return box === null ? 0 : box[0] * box[4] * box[8];
}

export function cross(ax, ay, az, bx, by, bz, out) {
  out[0] = ay * bz - az * by;
  out[1] = az * bx - ax * bz;
  out[2] = ax * by - ay * bx;
  return out;
}
