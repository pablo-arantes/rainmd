/**
 * Pooled GPU buffers, keyed by size and usage.
 *
 * 🔴 AN MD RUN ALLOCATES THE SAME SHAPES EVERY STEP, FOREVER. A fold runs its
 * graph a few times; a simulation runs it a million times. Creating and
 * destroying the force, neighbour and grid buffers per step is the difference
 * between a steady state and an allocator that the driver eventually refuses -
 * so the pool here is not an optimisation, it is what makes a long run possible.
 *
 * 🔴 AND A POOLED BUFFER KEEPS THE PREVIOUS STEP'S BYTES. Any kernel that reads
 * a region it did not write - a padded tail, an unfilled neighbour slot, a cell
 * with fewer atoms than its capacity - gives one answer on a fresh buffer and
 * another on a recycled one. That is deterministic and looks exactly like a
 * race. `setBufferPooling(false)` is the switch that tells the two apart: with
 * pooling off every allocation is zero-initialised, so a difference that
 * survives is a real race and one that vanishes is an uninitialised read.
 */
let poolingEnabled = true;
export function setBufferPooling(enabled) { poolingEnabled = enabled !== false; }
export function bufferPoolingEnabled() { return poolingEnabled; }

export class GpuBuffer {
  constructor(allocator, buffer, byteLength, usage, label) {
    this.buffer = buffer;
    this.byteLength = byteLength;
    this.usage = usage;
    this.label = label;
    this._allocator = allocator;
  }

  release() {
    const allocator = this._allocator;
    if (allocator === undefined) return;
    this._allocator = undefined;
    allocator.noteRelease(this);
  }
}

export class GpuAllocator {
  constructor(device, options = {}) {
    this.device = device;
    this.budgetBytes = options.budgetBytes ?? null;
    this._pool = new Map();
    this._liveBytes = 0;
    this._peakBytes = 0;
    this._allocations = 0;
    this._retired = [];
  }

  allocate(label, requestedBytes, usage) {
    if (!Number.isSafeInteger(requestedBytes) || requestedBytes <= 0) {
      throw new RangeError(`invalid allocation size ${requestedBytes} for ${label}`);
    }
    const byteLength = Math.ceil(requestedBytes / 4) * 4;
    // A pooled storage buffer carries both copy flags so that one size is one
    // key: otherwise STORAGE, STORAGE|COPY_SRC and STORAGE|COPY_DST are three
    // pools of identically sized buffers that can never serve each other.
    let effectiveUsage = usage;
    const mapped = GPUBufferUsage.MAP_READ | GPUBufferUsage.MAP_WRITE;
    if ((usage & GPUBufferUsage.STORAGE) !== 0 && (usage & mapped) === 0) {
      effectiveUsage |= GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
    }
    const key = `${byteLength}:${effectiveUsage}`;
    const pooled = poolingEnabled ? this._pool.get(key) : undefined;
    let buffer = pooled?.pop();
    if (pooled !== undefined && pooled.length === 0) this._pool.delete(key);
    if (buffer === undefined) {
      if (this.budgetBytes !== null && this._liveBytes + byteLength > this.budgetBytes) {
        // 🔴 DROP THE POOL BEFORE GIVING UP. Every released buffer still
        // occupies the device and still counts, so under pressure the cache is
        // what is failing. Freeing it turns "out of memory" into a slower run.
        this.destroyPooled();
        if (this._liveBytes + byteLength > this.budgetBytes) {
          throw new GpuBudgetError(label, byteLength, this._liveBytes, this.budgetBytes);
        }
      }
      buffer = this.device.createBuffer({ label, size: byteLength, usage: effectiveUsage });
    }
    this._liveBytes += byteLength;
    this._peakBytes = Math.max(this._peakBytes, this._liveBytes);
    this._allocations += 1;
    return new GpuBuffer(this, buffer, byteLength, effectiveUsage, label);
  }

  upload(label, data, usage) {
    const allocation = this.allocate(label, data.byteLength, usage | GPUBufferUsage.COPY_DST);
    this.write(allocation, data);
    return allocation;
  }

  write(allocation, data, offsetBytes = 0) {
    if (data.byteLength % 4 === 0) {
      this.device.queue.writeBuffer(
        allocation.buffer, offsetBytes, data.buffer, data.byteOffset, data.byteLength,
      );
    } else {
      const padded = new Uint8Array(Math.ceil(data.byteLength / 4) * 4);
      padded.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
      this.device.queue.writeBuffer(allocation.buffer, offsetBytes, padded);
    }
  }

  noteRelease(allocation) {
    this._liveBytes -= allocation.byteLength;
    if (this._liveBytes < 0) throw new Error("GPU allocator accounting underflow");
    if (!poolingEnabled) {
      // Held, never handed back, never destroyed: a released buffer may still
      // be named by an in-flight command buffer, and destroying it there is
      // "[Buffer] destroyed" on the next submit.
      this._retired.push(allocation.buffer);
      return;
    }
    const key = `${allocation.byteLength}:${allocation.usage}`;
    const pooled = this._pool.get(key) ?? [];
    pooled.push(allocation.buffer);
    this._pool.set(key, pooled);
  }

  destroyPooled() {
    for (const buffers of this._pool.values()) {
      for (const buffer of buffers) buffer.destroy();
    }
    this._pool.clear();
  }

  snapshot() {
    let pooledBytes = 0;
    for (const [key, buffers] of this._pool) {
      pooledBytes += Number(key.slice(0, key.indexOf(":"))) * buffers.length;
    }
    return {
      liveBytes: this._liveBytes,
      peakBytes: this._peakBytes,
      pooledBytes,
      allocations: this._allocations,
      retired: this._retired.length,
    };
  }
}

export class GpuBudgetError extends Error {
  constructor(label, requestedBytes, liveBytes, budgetBytes) {
    const mib = (value) => (value / (1024 * 1024)).toFixed(1);
    super(
      `${label} needs ${mib(requestedBytes)} MiB and the device budget is `
      + `${mib(budgetBytes)} MiB with ${mib(liveBytes)} MiB already held. `
      + "A smaller system, a shorter neighbour list, or a coarser PME grid "
      + "would fit.",
    );
    this.name = "GpuBudgetError";
    this.label = label;
    this.requestedBytes = requestedBytes;
  }
}
