/**
 * Hold the WGSL FFT to a naive DFT, on random data, before anything uses it.
 *
 * 🔴 ISOLATED ON PURPOSE. The FFT sits in the middle of the PME pipeline, so a
 * wrong twiddle shows up as a wrong reciprocal energy - a single number, with
 * the spread, the convolution and the gather all able to produce the same
 * symptom. Testing the transform alone, against an O(n^2) DFT that shares no
 * code with it, means a PME disagreement afterwards cannot be the FFT.
 *
 * 🔴 AND ON RANDOM DATA, NOT ON A CHARGE GRID. A real charge grid is smooth and
 * dominated by low frequencies, so its high-frequency coefficients are tiny -
 * which is exactly where a transposed axis or a wrong output permutation hides.
 * Random data gives every coefficient the same expected magnitude.
 */
import { GpuAllocator } from "../../webgpu/runtime/allocator.js";
import { readBuffer, DeferredValidation } from "../../webgpu/runtime/execution.js";
import { encodeFft3d, fftStageCount } from "../../webgpu/kernels/pme-fft.js";
import { dftAlongAxis } from "../../cpu/forces/pme.js";

/** Grids chosen to exercise every radix and a non-cubic shape. */
const GRIDS = [
  [8, 8, 8],
  [24, 24, 24],
  [24, 32, 20],
  [27, 25, 16],
];

export async function main(device) {
  const allocator = new GpuAllocator(device);
  const results = [];
  let failures = 0;

  for (const grid of GRIDS) {
    const total = grid[0] * grid[1] * grid[2];
    const bytes = total * 2 * 4;

    // Deterministic pseudo-random complex data.
    const host = new Float64Array(total * 2);
    for (let index = 0; index < total * 2; index += 1) {
      host[index] = Math.sin(index * 1.61803398875 + 0.7) * Math.cos(index * 0.31831 - 0.2);
    }

    for (const sign of [1, -1]) {
      const reference = Float64Array.from(host);
      dftAlongAxis(reference, grid, 0, sign);
      dftAlongAxis(reference, grid, 1, sign);
      dftAlongAxis(reference, grid, 2, sign);

      const held = [];
      const keep = (allocation) => { held.push(allocation); return allocation; };
      try {
        const buffers = {
          a: keep(allocator.upload("fft.a", Float32Array.from(host),
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC)),
          b: keep(allocator.allocate("fft.b", bytes,
            GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC)),
        };
        const validation = new DeferredValidation(device, `fft/${grid.join("x")}`);
        validation.begin();
        const encoder = device.createCommandEncoder({ label: `fft/${grid.join("x")}` });
        const { current } = encodeFft3d(device, encoder, grid, buffers, sign);
        device.queue.submit([encoder.finish()]);
        validation.end("fft");
        await validation.settle();

        const got = await readBuffer(device, allocator, current, bytes, Float32Array);

        let worst = 0;
        let scale = 0;
        for (let index = 0; index < total; index += 1) {
          const magnitude = Math.hypot(reference[index * 2], reference[index * 2 + 1]);
          scale = Math.max(scale, magnitude);
          worst = Math.max(worst, Math.hypot(
            got[index * 2] - reference[index * 2],
            got[index * 2 + 1] - reference[index * 2 + 1],
          ));
        }
        const relative = worst / scale;
        // 🔴 THE BAR GROWS WITH THE TRANSFORM. A 3D FFT of side n accumulates
        // error over log2(n) stages per axis, three axes, in f32: roughly
        // sqrt(stages) * eps * the largest coefficient. For a 24^3 grid that is
        // 12 stages, so ~3.5 * 6e-8 = 2e-7, and the bar below is an order above
        // it - still far tighter than any wrong index map could survive.
        const stages = fftStageCount(grid);
        const bar = 20 * Math.sqrt(stages) * 2 ** -24;
        const ok = relative < bar;
        if (!ok) failures += 1;
        results.push({
          grid: grid.join("x"),
          direction: sign > 0 ? "forward" : "inverse",
          stages,
          relative,
          bar,
          ok,
        });
      } finally {
        for (let index = held.length - 1; index >= 0; index -= 1) held[index].release();
      }
    }
  }
  return { check: "WGSL mixed-radix FFT against a naive DFT", failures, results };
}
