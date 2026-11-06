/**
 * Compute pipelines, cached by key AND indexed by source.
 *
 * 🔴 THE SOURCE IS PART OF THE INDEX SO THAT A KEY WHICH DOES NOT NAME ITS
 * SHADER IS AN ERROR RATHER THAN A WRONG ANSWER. Every kernel here bakes
 * constants into its WGSL - atom counts, grid sizes, cell capacities, the
 * number of neighbour slots - and builds its key by hand from the few it
 * remembers. A key that omits one hands the caller somebody else's kernel,
 * which computes a plausible wrong force field and is essentially undebuggable.
 * This cache compares the text and throws with the FIRST DIFFERING LINE, which
 * is usually the constant that was left out of the key.
 *
 * 🔴 AND AN `enable` DIRECTIVE WITHOUT ITS FEATURE FAILS AS A WGSL PARSE ERROR
 * NAMING NO KEY. Dawn says "extension 'f16' is not allowed in the current
 * environment" against an uncaptured device error, with nothing to say which
 * of a dozen shader factories emitted it. This adapter advertises shader-f16
 * and a device created without asking for it does NOT have it, so that state
 * is reachable here and not hypothetical.
 */

const EXTENSION_FEATURES = [
  ["f16", "shader-f16"],
  ["subgroups", "subgroups"],
];

export class PipelineCache {
  constructor(device) {
    this.device = device;
    this._byKey = new Map();
    this._byContent = new Map();
    this.stats = { hits: 0, misses: 0, shared: 0 };
  }

  get(key, code, entryPoint = "main") {
    const cached = this._byKey.get(key);
    if (cached !== undefined) {
      if (cached.code !== code || cached.entryPoint !== entryPoint) {
        throw new Error(this._collisionMessage(key, cached, code, entryPoint));
      }
      this.stats.hits += 1;
      return cached.pipeline;
    }
    for (const [extension, feature] of EXTENSION_FEATURES) {
      if (!new RegExp(`(^|\\n)\\s*enable\\s+${extension}\\s*;`).test(code)) continue;
      if (this.device.features.has(feature)) continue;
      throw new Error(
        `${key} enables ${extension} on a device without ${feature}. The caller `
        + "must gate on device.features - see webgpu/runtime/device.js, which "
        + "requests it only when the adapter offers it.",
      );
    }
    const content = `${entryPoint}\u0000${code}`;
    const shared = this._byContent.get(content);
    if (shared !== undefined) {
      this.stats.shared += 1;
      this._byKey.set(key, { code, entryPoint, pipeline: shared });
      return shared;
    }
    this.stats.misses += 1;
    const pipeline = this.device.createComputePipeline({
      label: key,
      layout: "auto",
      compute: {
        module: this.device.createShaderModule({ label: `${key}.wgsl`, code }),
        entryPoint,
      },
    });
    this._byKey.set(key, { code, entryPoint, pipeline });
    this._byContent.set(content, pipeline);
    return pipeline;
  }

  _collisionMessage(key, cached, code, entryPoint) {
    if (cached.entryPoint !== entryPoint) {
      return `pipeline cache key collision for ${key} - entry point `
        + `${cached.entryPoint} against ${entryPoint}`;
    }
    const was = cached.code.split("\n");
    const now = String(code).split("\n");
    let line = 0;
    while (line < Math.max(was.length, now.length) && was[line] === now[line]) line += 1;
    return `pipeline cache key collision for ${key} - line ${line + 1} of `
      + `${now.length}: ${JSON.stringify((was[line] ?? "").trim().slice(0, 100))} `
      + `against ${JSON.stringify((now[line] ?? "").trim().slice(0, 100))}. `
      + "The key does not name everything the source embeds.";
  }
}

const cachesByDevice = new WeakMap();
export function pipelineCacheFor(device) {
  let cache = cachesByDevice.get(device);
  if (cache === undefined) {
    cache = new PipelineCache(device);
    cachesByDevice.set(device, cache);
  }
  return cache;
}
