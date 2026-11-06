/**
 * Dispatch, readback and deferred validation.
 */
import { pipelineCacheFor } from "./pipeline-cache.js";

/** WebGPU's own limit on a single dispatch dimension's workgroup count. */
export function dispatchGrid(workgroups, maxPerDimension) {
  if (workgroups <= maxPerDimension) return [workgroups, 1, 1];
  // 🔴 A LONG RUN OVERFLOWS ONE DIMENSION AND THE KERNEL MUST AGREE ABOUT THE
  // FOLD. maxComputeWorkgroupsPerDimension is 65535 on this adapter, so a
  // per-atom kernel at 64 threads covers 4.2M atoms in x alone - but the PME
  // grid and the neighbour list go wider. Every kernel that can be dispatched
  // this way reconstructs its linear index as `x + y * GRID_WIDTH * wgSize`,
  // with GRID_WIDTH baked into the source and into its cache key.
  const width = maxPerDimension;
  return [width, Math.ceil(workgroups / width), 1];
}

/**
 * Deferred WebGPU validation.
 *
 * 🔴 `await device.popErrorScope()` PER STEP COSTS A ROUND TRIP PER STEP. MD
 * submits a few dispatches and then wants to submit the next few; awaiting a
 * scope between them puts a host-device synchronisation in the middle of the
 * only loop that matters and removes all the pipelining. The pop itself is
 * synchronous - only its RESULT is a promise - so scopes can be opened and
 * closed every step and read once, at a boundary that already synchronises
 * (an energy readback, or the end of the run).
 *
 * The push must come BEFORE encoding: most validation errors are raised as
 * commands are encoded, not when the buffer is submitted.
 */
export class DeferredValidation {
  constructor(device, label) {
    this.device = device;
    this.label = label;
    this._pending = [];
  }

  begin() { this.device.pushErrorScope("validation"); }

  end(where) {
    this._pending.push(this.device.popErrorScope().then(
      (error) => (error === null ? null : `${where}: ${error.message}`),
    ));
  }

  async settle() {
    const pending = this._pending;
    this._pending = [];
    const failures = (await Promise.all(pending)).filter((message) => message !== null);
    if (failures.length > 0) {
      throw new Error(`WebGPU ${this.label} validation failed: ${failures.join("; ")}`);
    }
  }
}

/**
 * Read a storage buffer back to the host.
 *
 * 🔴 THE STAGING BUFFER IS ALLOCATED PER CALL AND NOT POOLED. A mapped buffer
 * cannot be reused until its map resolves, and pooling one by size hands a
 * still-mapped buffer to the next caller - "buffer is mapped" on a submit that
 * has nothing to do with the readback. Readbacks happen at reporting intervals,
 * not per step, so the allocation is not on any hot path.
 */
export async function readBuffer(device, allocator, source, byteLength, ArrayType = Float32Array) {
  const staging = allocator.allocate(
    `${source.label}.readback`, byteLength,
    GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  );
  try {
    const encoder = device.createCommandEncoder({ label: "rainmd.readback" });
    encoder.copyBufferToBuffer(source.buffer, 0, staging.buffer, 0, byteLength);
    device.queue.submit([encoder.finish()]);
    await staging.buffer.mapAsync(GPUMapMode.READ);
    const copy = new ArrayType(staging.buffer.getMappedRange().slice(0));
    staging.buffer.unmap();
    return copy;
  } finally {
    staging.release();
  }
}

/**
 * Encode one compute pass: bind group 0 from the given buffers, in order.
 */
export function encodePass(device, encoder, key, code, bindings, workgroups, label = key) {
  const pipeline = pipelineCacheFor(device).get(key, code);
  const pass = encoder.beginComputePass({ label });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: bindings.map((allocation, binding) => ({
      binding,
      resource: allocation.offset === undefined
        ? { buffer: allocation.buffer }
        : { buffer: allocation.buffer.buffer, offset: allocation.offset, size: allocation.size },
    })),
  }));
  const [x, y, z] = Array.isArray(workgroups) ? workgroups : [workgroups, 1, 1];
  pass.dispatchWorkgroups(x, y, z);
  pass.end();
}
