/**
 * Optimal superposition, by the quaternion method.
 *
 * 🔴 QUATERNIONS RATHER THAN AN SVD, AND NOT FOR SPEED. The textbook Kabsch
 * takes the SVD of the 3x3 covariance and then has to repair the result: if
 * `det(V U^T)` is negative the solution is a REFLECTION, which superposes a
 * structure onto its mirror image and reports a small RMSD for a molecule that
 * has been turned inside out. Every implementation carries a sign fix-up for
 * it, and the fix-up is easy to get wrong. The largest-eigenvector quaternion
 * (Horn 1987; Coutsias, Seok & Dill 2004) cannot express a reflection at all -
 * every unit quaternion is a proper rotation by construction - so the failure
 * mode does not exist.
 *
 * It also avoids needing an SVD in JavaScript, which has none built in.
 *
 * 🔴 THE KEY MATRIX'S SIGN CONVENTION WAS DETERMINED BY EXPERIMENT, NOT BY
 * RECOLLECTION. Several published forms differ by a transpose or a sign, and all
 * of them look plausible; `test/analysis.test.js` recovers a known rotation from
 * synthetic data, which is what pins it.
 */

/** Jacobi eigendecomposition of a symmetric 4x4. Returns the top eigenvector. */
function largestEigenvector(matrix) {
  // Work on a copy; `vectors` accumulates the rotations so its columns are the
  // eigenvectors.
  const a = matrix.map((row) => row.slice());
  const vectors = [
    [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
  ];
  for (let sweep = 0; sweep < 64; sweep += 1) {
    let offDiagonal = 0;
    for (let p = 0; p < 4; p += 1) {
      for (let q = p + 1; q < 4; q += 1) offDiagonal += a[p][q] * a[p][q];
    }
    if (offDiagonal < 1e-30) break;
    for (let p = 0; p < 4; p += 1) {
      for (let q = p + 1; q < 4; q += 1) {
        if (Math.abs(a[p][q]) < 1e-300) continue;
        const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
        const sign = theta >= 0 ? 1 : -1;
        const t = sign / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < 4; k += 1) {
          const akp = a[k][p];
          const akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < 4; k += 1) {
          const apk = a[p][k];
          const aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < 4; k += 1) {
          const vkp = vectors[k][p];
          const vkq = vectors[k][q];
          vectors[k][p] = c * vkp - s * vkq;
          vectors[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  let best = 0;
  for (let index = 1; index < 4; index += 1) if (a[index][index] > a[best][best]) best = index;
  return [vectors[0][best], vectors[1][best], vectors[2][best], vectors[3][best]];
}

/** The centroid of a selection, as three numbers. */
export function centroid(positions, indices, weights = null) {
  let x = 0;
  let y = 0;
  let z = 0;
  let total = 0;
  for (const atom of indices) {
    const weight = weights === null ? 1 : weights[atom];
    x += positions[atom * 3] * weight;
    y += positions[atom * 3 + 1] * weight;
    z += positions[atom * 3 + 2] * weight;
    total += weight;
  }
  if (total === 0) return [0, 0, 0];
  return [x / total, y / total, z / total];
}

/**
 * The rotation that best superposes `mobile` onto `reference` over `indices`.
 *
 * Returns the rotation plus both centroids, so the transform can be applied to
 * a DIFFERENT set of atoms than it was derived from - which is the whole basis
 * of a "fit on the receptor, measure the ligand" RMSD.
 *
 * @returns {{rotation: number[], mobileCentroid: number[], referenceCentroid: number[]}}
 */
export function kabschTransform(mobile, reference, indices) {
  const mobileCentroid = centroid(mobile, indices);
  const referenceCentroid = centroid(reference, indices);
  // H[a][b] = sum (mobile - mc)_a (reference - rc)_b
  const h = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (const atom of indices) {
    const m = [
      mobile[atom * 3] - mobileCentroid[0],
      mobile[atom * 3 + 1] - mobileCentroid[1],
      mobile[atom * 3 + 2] - mobileCentroid[2],
    ];
    const r = [
      reference[atom * 3] - referenceCentroid[0],
      reference[atom * 3 + 1] - referenceCentroid[1],
      reference[atom * 3 + 2] - referenceCentroid[2],
    ];
    for (let a = 0; a < 3; a += 1) {
      for (let b = 0; b < 3; b += 1) h[a][b] += m[a] * r[b];
    }
  }
  const key = [
    [h[0][0] + h[1][1] + h[2][2], h[1][2] - h[2][1], h[2][0] - h[0][2], h[0][1] - h[1][0]],
    [h[1][2] - h[2][1], h[0][0] - h[1][1] - h[2][2], h[0][1] + h[1][0], h[2][0] + h[0][2]],
    [h[2][0] - h[0][2], h[0][1] + h[1][0], -h[0][0] + h[1][1] - h[2][2], h[1][2] + h[2][1]],
    [h[0][1] - h[1][0], h[2][0] + h[0][2], h[1][2] + h[2][1], -h[0][0] - h[1][1] + h[2][2]],
  ];
  const [w, x, y, z] = largestEigenvector(key);
  const norm = Math.hypot(w, x, y, z) || 1;
  const qw = w / norm;
  const qx = x / norm;
  const qy = y / norm;
  const qz = z / norm;
  const rotation = [
    1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy),
    2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx),
    2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy),
  ];
  return { rotation, mobileCentroid, referenceCentroid };
}

/** Apply a transform to one atom, into `out`. */
export function applyTransform(transform, positions, atom, out) {
  const { rotation, mobileCentroid, referenceCentroid } = transform;
  const x = positions[atom * 3] - mobileCentroid[0];
  const y = positions[atom * 3 + 1] - mobileCentroid[1];
  const z = positions[atom * 3 + 2] - mobileCentroid[2];
  out[0] = rotation[0] * x + rotation[1] * y + rotation[2] * z + referenceCentroid[0];
  out[1] = rotation[3] * x + rotation[4] * y + rotation[5] * z + referenceCentroid[1];
  out[2] = rotation[6] * x + rotation[7] * y + rotation[8] * z + referenceCentroid[2];
  return out;
}
