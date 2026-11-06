/**
 * The PME reciprocal sum, as a reference implementation.
 *
 * 🔴 IT USES A NAIVE PER-AXIS DFT, O(N^2) PER LINE, ON PURPOSE. This is the
 * oracle for the WebGPU Stockham FFT, and a reference that shared the FFT would
 * be unable to find a bug in it - a butterfly with a wrong twiddle agrees
 * perfectly with itself. At a 24^3 grid the naive transform is about a million
 * complex operations and runs in milliseconds, so there is no reason to risk
 * the fast path here.
 *
 * The conventions, all of which must match the GPU kernels exactly:
 *   - forward transform carries exp(+2*pi*i*m*k/K)
 *   - the B-spline moduli DIVIDE the convolution (|b(m)|^2 = 1/moduli[m])
 *   - the grid index for weight k is (floor(u) + k - order + 1) mod K
 */
import { ONE_4PI_EPS0 } from "../../shared/units.js";
import { bsplineWeights } from "../../shared/pme/parameters.js";
import { reciprocalBoxVectors, boxVolume } from "./geometry.js";

/**
 * Transform a complex array along one axis, in place.
 *
 * @param {Float64Array} data interleaved real/imaginary, length 2*nx*ny*nz
 * @param {number[]} grid [nx, ny, nz]
 * @param {number} axis 0, 1 or 2
 * @param {number} sign +1 forward, -1 inverse (unnormalised)
 */
export function dftAlongAxis(data, grid, axis, sign) {
  const [nx, ny, nz] = grid;
  const n = grid[axis];
  const strides = [ny * nz, nz, 1];
  const stride = strides[axis];
  const line = new Float64Array(2 * n);
  // Precompute the twiddles for this length: n^2 exp() calls become n.
  const cosTable = new Float64Array(n);
  const sinTable = new Float64Array(n);
  for (let index = 0; index < n; index += 1) {
    const angle = (sign * 2 * Math.PI * index) / n;
    cosTable[index] = Math.cos(angle);
    sinTable[index] = Math.sin(angle);
  }
  const outer = [nx, ny, nz];
  for (let x = 0; x < nx; x += 1) {
    for (let y = 0; y < ny; y += 1) {
      for (let z = 0; z < nz; z += 1) {
        const coordinate = [x, y, z];
        if (coordinate[axis] !== 0) continue;
        const base = x * strides[0] + y * strides[1] + z * strides[2];
        for (let k = 0; k < n; k += 1) {
          line[2 * k] = data[2 * (base + k * stride)];
          line[2 * k + 1] = data[2 * (base + k * stride) + 1];
        }
        for (let m = 0; m < n; m += 1) {
          let real = 0;
          let imaginary = 0;
          for (let k = 0; k < n; k += 1) {
            const twiddle = (m * k) % n;
            const c = cosTable[twiddle];
            const s = sinTable[twiddle];
            real += line[2 * k] * c - line[2 * k + 1] * s;
            imaginary += line[2 * k] * s + line[2 * k + 1] * c;
          }
          data[2 * (base + m * stride)] = real;
          data[2 * (base + m * stride) + 1] = imaginary;
        }
      }
    }
  }
  void outer;
}

/**
 * Spread the charges onto the grid, and return the per-atom spline state the
 * force gather needs so the splines are not computed twice.
 */
function spreadCharge(system, positions, parameters) {
  const topology = system.topology;
  const grid = parameters.grid;
  const order = parameters.order;
  const [nx, ny, nz] = grid;
  const recip = reciprocalBoxVectors(system.boxVectors);
  const charge = new Float64Array(2 * nx * ny * nz);
  const atomCount = topology.atomCount;
  const base = new Int32Array(atomCount * 3);
  const weights = new Float64Array(atomCount * 3 * order);
  const derivatives = new Float64Array(atomCount * 3 * order);

  for (let atom = 0; atom < atomCount; atom += 1) {
    const q = topology.charges[atom];
    const rx = positions[atom * 3];
    const ry = positions[atom * 3 + 1];
    const rz = positions[atom * 3 + 2];
    for (let d = 0; d < 3; d += 1) {
      // frac_d = r . (Bo^-1)[:, d]; recip is row-major Bo^-1.
      const fraction = rx * recip[d] + ry * recip[3 + d] + rz * recip[6 + d];
      // 🔴 WRAPPED WITH floor, NOT WITH %. A negative coordinate - which every
      // box centred on the origin has - gives a negative remainder from `%`,
      // and a negative grid index then reads the wrong end of the array or,
      // worse, indexes within bounds after a later `+ n` and spreads the atom's
      // charge onto a point on the opposite face of the box.
      const wrapped = fraction - Math.floor(fraction);
      const scaled = wrapped * grid[d];
      const integral = Math.floor(scaled);
      const offset = scaled - integral;
      base[atom * 3 + d] = integral;
      const spline = bsplineWeights(offset, order);
      for (let k = 0; k < order; k += 1) {
        weights[(atom * 3 + d) * order + k] = spline.weights[k];
        derivatives[(atom * 3 + d) * order + k] = spline.derivatives[k];
      }
    }
    for (let kx = 0; kx < order; kx += 1) {
      let ix = (base[atom * 3] + kx - order + 1) % nx;
      if (ix < 0) ix += nx;
      const wx = weights[(atom * 3) * order + kx];
      for (let ky = 0; ky < order; ky += 1) {
        let iy = (base[atom * 3 + 1] + ky - order + 1) % ny;
        if (iy < 0) iy += ny;
        const wxy = wx * weights[(atom * 3 + 1) * order + ky];
        for (let kz = 0; kz < order; kz += 1) {
          let iz = (base[atom * 3 + 2] + kz - order + 1) % nz;
          if (iz < 0) iz += nz;
          const weight = wxy * weights[(atom * 3 + 2) * order + kz];
          charge[2 * ((ix * ny + iy) * nz + iz)] += q * weight;
        }
      }
    }
  }
  return { charge, base, weights, derivatives, recip };
}

/**
 * The reciprocal-space energy and forces.
 *
 * @returns {number} the reciprocal energy in kJ/mol, forces accumulated
 */
export function reciprocalEnergy(system, positions, forces, parameters) {
  const topology = system.topology;
  const grid = parameters.grid;
  const order = parameters.order;
  const [nx, ny, nz] = grid;
  const alpha = parameters.alpha;
  const volume = boxVolume(system.boxVectors);
  const { charge, base, weights, derivatives, recip } = spreadCharge(system, positions, parameters);

  dftAlongAxis(charge, grid, 0, 1);
  dftAlongAxis(charge, grid, 1, 1);
  dftAlongAxis(charge, grid, 2, 1);

  // Convolution with the Ewald kernel, keeping the convolved grid for the
  // force gather. The energy is read off the same multiply.
  const factor = (Math.PI * Math.PI) / (alpha * alpha);
  const prefactor = ONE_4PI_EPS0 / (2 * Math.PI * volume);
  let energy = 0;
  for (let mx = 0; mx < nx; mx += 1) {
    // 🔴 THE FREQUENCY INDEX IS SIGNED AND WRAPS AT HALF THE GRID. Index
    // nx-1 is frequency -1, not nx-1. Reading it unsigned makes m^2 enormous
    // for the whole upper half of the grid, the Gaussian kills those terms, and
    // the reciprocal energy comes out roughly a quarter of its true value -
    // which looks like a prefactor mistake and is not one.
    const kx = mx <= nx / 2 ? mx : mx - nx;
    for (let my = 0; my < ny; my += 1) {
      const ky = my <= ny / 2 ? my : my - ny;
      for (let mz = 0; mz < nz; mz += 1) {
        const kz = mz <= nz / 2 ? mz : mz - nz;
        const index = (mx * ny + my) * nz + mz;
        if (kx === 0 && ky === 0 && kz === 0) {
          charge[2 * index] = 0;
          charge[2 * index + 1] = 0;
          continue;
        }
        // m = kx*a* + ky*b* + kz*c*, where the reciprocal vectors are the
        // COLUMNS of Bo^-1 (recip rows are indexed by cartesian component).
        const mxCart = kx * recip[0] + ky * recip[3] + kz * recip[6];
        const myCart = kx * recip[1] + ky * recip[4] + kz * recip[7];
        const mzCart = kx * recip[2] + ky * recip[5] + kz * recip[8];
        const mSquared = mxCart * mxCart + myCart * myCart + mzCart * mzCart;
        const modulus = parameters.moduli[0][mx]
          * parameters.moduli[1][my] * parameters.moduli[2][mz];
        const kernel = Math.exp(-factor * mSquared) / (mSquared * modulus);
        const real = charge[2 * index];
        const imaginary = charge[2 * index + 1];
        energy += prefactor * kernel * (real * real + imaginary * imaginary);
        charge[2 * index] = real * kernel * 2 * prefactor;
        charge[2 * index + 1] = imaginary * kernel * 2 * prefactor;
      }
    }
  }

  // 🔴 THE INVERSE TRANSFORM IS UNNORMALISED AND MUST STAY THAT WAY. The
  // forward DFT of the spread charge and this inverse together carry a factor
  // of the grid size; it is absorbed into the convolution's prefactor above
  // (via the 1/(2*pi*V) and the moduli) rather than applied here. Dividing by
  // nx*ny*nz as a "proper" inverse makes the reciprocal forces too small by
  // exactly the grid volume.
  dftAlongAxis(charge, grid, 0, -1);
  dftAlongAxis(charge, grid, 1, -1);
  dftAlongAxis(charge, grid, 2, -1);

  // Gather: dE/dr_i = q_i * sum_grid (dW/dr_i) * convolved(grid)
  for (let atom = 0; atom < topology.atomCount; atom += 1) {
    const q = topology.charges[atom];
    if (q === 0) continue;
    let gradientU = 0;
    let gradientV = 0;
    let gradientW = 0;
    for (let kx = 0; kx < order; kx += 1) {
      let ix = (base[atom * 3] + kx - order + 1) % nx;
      if (ix < 0) ix += nx;
      const wx = weights[(atom * 3) * order + kx];
      const dx = derivatives[(atom * 3) * order + kx];
      for (let ky = 0; ky < order; ky += 1) {
        let iy = (base[atom * 3 + 1] + ky - order + 1) % ny;
        if (iy < 0) iy += ny;
        const wy = weights[(atom * 3 + 1) * order + ky];
        const dy = derivatives[(atom * 3 + 1) * order + ky];
        for (let kz = 0; kz < order; kz += 1) {
          let iz = (base[atom * 3 + 2] + kz - order + 1) % nz;
          if (iz < 0) iz += nz;
          const wz = weights[(atom * 3 + 2) * order + kz];
          const dz = derivatives[(atom * 3 + 2) * order + kz];
          const value = charge[2 * ((ix * ny + iy) * nz + iz)];
          gradientU += dx * wy * wz * value;
          gradientV += wx * dy * wz * value;
          gradientW += wx * wy * dz * value;
        }
      }
    }
    // Chain rule into cartesian: du_d/dr_x = grid_d * recip[x][d].
    //
    // 🔴 NO FACTOR OF A HALF. The convolution above already folded the 2 from
    // d|F|^2/dF into the grid it wrote (`* 2 * prefactor`), so the grid this
    // reads is dE/dQ(k) exactly. Halving it again - the instinct from the 1/2
    // in the energy's own prefactor - makes every reciprocal force exactly half
    // its true value, which is a force field that is subtly too weak on long
    // range electrostatics and conserves nothing.
    const gu = gradientU * grid[0] * q;
    const gv = gradientV * grid[1] * q;
    const gw = gradientW * grid[2] * q;
    forces[atom * 3] -= gu * recip[0] + gv * recip[1] + gw * recip[2];
    forces[atom * 3 + 1] -= gu * recip[3] + gv * recip[4] + gw * recip[5];
    forces[atom * 3 + 2] -= gu * recip[6] + gv * recip[7] + gw * recip[8];
  }
  return energy;
}

/**
 * The self-energy and the net-charge correction, both constants.
 *
 * 🔴 THE SELF TERM IS LARGE AND NEGATIVE AND HAS NO FORCE. For the solvated
 * fixture it is thousands of kJ/mol - bigger than the whole Coulomb energy -
 * so leaving it out does not perturb a result, it replaces it. It contributes
 * no force at all, which is why a port can be missing it and still run a
 * perfectly stable simulation with a badly wrong energy.
 */
export function ewaldSelfEnergy(system, parameters) {
  const topology = system.topology;
  let chargeSquaredSum = 0;
  let chargeSum = 0;
  for (let atom = 0; atom < topology.atomCount; atom += 1) {
    chargeSquaredSum += topology.charges[atom] * topology.charges[atom];
    chargeSum += topology.charges[atom];
  }
  const self = -ONE_4PI_EPS0 * parameters.alpha * chargeSquaredSum / Math.sqrt(Math.PI);
  // The neutralising-background term. Zero for a neutral system, which is what
  // tleap's addIons produces - but a user can drop a charged system, and then
  // this is the difference between a defensible energy and a meaningless one.
  const volume = boxVolume(system.boxVectors);
  const net = -ONE_4PI_EPS0 * Math.PI * chargeSum * chargeSum
    / (2 * parameters.alpha * parameters.alpha * volume);
  return { self, net, total: self + net };
}
