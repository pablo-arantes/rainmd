/**
 * Adapter and device negotiation.
 *
 * 🔴 LIMITS AND FEATURES ARE NOT INHERITED FROM THE ADAPTER. A device gets the
 * SPEC DEFAULTS unless it asks, and the spec default for
 * maxStorageBufferBindingSize is 128 MiB while this Mac's adapter offers 9 GiB.
 * Measured on this machine during the port: the adapter advertises `shader-f16`
 * and the device created with no arguments reports it as FALSE. Nothing warns.
 * A solvated system's position buffer crossing 128 MiB fails validation
 * mid-run, which for MD means after minutes of simulation rather than at setup.
 *
 * Every limit is requested at exactly what the adapter reports, so the request
 * cannot be refused: requestDevice rejects a limit BETTER than the adapter's,
 * never one equal to it.
 */

/**
 * The limits this engine's kernels actually depend on.
 *
 * 🔴 maxComputeInvocationsPerWorkgroup IS LOAD-BEARING FOR THE FFT. A
 * radix-2 Stockham stage over a 512-point line wants 256 invocations and the
 * spec guarantees exactly 256, so the PME grid dimension is capped by this
 * limit rather than by memory. A conforming device reporting the minimum still
 * runs, but the FFT splits into more passes - which is why the kernel reads the
 * limit instead of assuming 1024 as this adapter offers.
 */
const RAISED_LIMITS = [
  "maxStorageBufferBindingSize",
  "maxBufferSize",
  "maxComputeWorkgroupStorageSize",
  "maxComputeInvocationsPerWorkgroup",
  "maxComputeWorkgroupSizeX",
  "maxComputeWorkgroupsPerDimension",
  "maxStorageBuffersPerShaderStage",
];

/**
 * Features worth having if the device has them, and that no kernel may use
 * without checking `device.features` first.
 *
 * 🔴 f16 IS NOT USED FOR ANYTHING IN THE FORCE PATH AND THAT IS DELIBERATE.
 * Half precision has ~3 decimal digits; a direct-space Coulomb sum cancels
 * numbers in the tens of thousands down to thousands, so f16 there is not an
 * optimisation but a different and much worse force field. It is requested
 * because the trajectory WRITER can use it for stored coordinates, where 3
 * digits of a nanometre is finer than any analysis resolves.
 */
const OPTIONAL_FEATURES = ["shader-f16", "timestamp-query", "subgroups"];

/** @returns {{adapter: GPUAdapter, device: GPUDevice, report: object}} */
export async function requestMdDevice(gpu, options = {}) {
  if (gpu === undefined || gpu === null) {
    throw new Error(
      "no WebGPU. This page needs Chrome or Edge 113+, or Safari 18+, with "
      + "hardware acceleration enabled. chrome://gpu will say whether WebGPU "
      + "is available; on a headless or server install it often needs "
      + "--enable-unsafe-webgpu or a Vulkan flag.",
    );
  }
  const adapter = await gpu.requestAdapter({
    powerPreference: options.powerPreference ?? "high-performance",
  });
  if (adapter === null) {
    throw new Error(
      "WebGPU reported no adapter. The browser has the API but cannot reach a "
      + "GPU - usually software rendering being disabled, or a headless "
      + "profile with no Vulkan/Metal backend.",
    );
  }
  const requiredFeatures = OPTIONAL_FEATURES.filter((name) => adapter.features.has(name));
  const requiredLimits = {};
  for (const name of RAISED_LIMITS) {
    const available = adapter.limits?.[name];
    if (typeof available === "number" && Number.isFinite(available)) {
      requiredLimits[name] = available;
    }
  }
  const device = await adapter.requestDevice({ requiredFeatures, requiredLimits });

  // 🔴 A LOST DEVICE MUST BE SURFACED, NOT POLLED FOR. An MD run is minutes to
  // hours of dispatches; if the driver resets - which an oversized allocation
  // or a long-running kernel can cause - every later submit silently does
  // nothing and the simulation appears to freeze at a step count. The promise
  // is the only notification there is.
  const report = { lost: null };
  device.lost.then((info) => {
    report.lost = { reason: info.reason, message: info.message };
  });
  device.addEventListener?.("uncapturederror", (event) => {
    report.lastUncapturedError = String(event.error?.message ?? event.error);
  });

  return {
    adapter,
    device,
    report: Object.assign(report, {
      vendor: adapter.info?.vendor,
      architecture: adapter.info?.architecture,
      description: adapter.info?.description,
      features: [...device.features].sort(),
      limits: Object.fromEntries(RAISED_LIMITS.map((name) => [name, device.limits?.[name]])),
      adapterOfferedF16: adapter.features.has("shader-f16"),
      deviceHasF16: device.features.has("shader-f16"),
    }),
  };
}
