/**
 * Run a GPU module against a real device, in Node, through Dawn.
 *
 *   node tools/gpu-node.mjs tools/gpu/check-bonded.mjs [--flags]
 *
 * The module exports `main(device, args)` and whatever it returns is printed as
 * JSON. The same module shape runs in the browser through tools/gpu-page.mjs,
 * so a kernel check is written once and can be run in either lane.
 *
 * 🔴 THE DAWN BINDING'S PREBUILT BINARY IS PINNED FOR A REASON. webgpu@0.6.0
 * and 0.5.0 both fail to load on this machine with "Symbol not found:
 * _OBJC_CLASS_$_MTLLogStateDescriptor" - they want a Metal from macOS 15 and
 * this is 14.5. 0.3.0 loads and reports a working Metal adapter. A newer
 * machine may want a newer pin; the failure is a dlopen error at import, not a
 * WebGPU error, so it is unmistakable.
 *
 * 🔴 AND THIS LANE IS NOT THE SHIPPING TARGET. Dawn in Node and Chrome's WebGPU
 * are different implementations of the same spec and they disagree about things
 * that matter - out-of-range stores, feature availability, timing. A kernel
 * that passes here must also pass in the browser lane before it is believed.
 */
import { pathToFileURL } from "node:url";

const args = process.argv.slice(2);
const modulePath = args.find((argument) => !argument.startsWith("--"));
if (modulePath === undefined) {
  console.error("usage: node tools/gpu-node.mjs <module.mjs> [--flags]");
  process.exit(2);
}

const { create, globals } = await import("webgpu");
Object.assign(globalThis, globals);
const gpu = create([]);

const { requestMdDevice } = await import("../webgpu/runtime/device.js");
const { device, report } = await requestMdDevice(gpu);

const module = await import(pathToFileURL(modulePath).href);
if (typeof module.main !== "function") {
  console.error(`${modulePath} does not export main(device, args)`);
  process.exit(2);
}

try {
  const result = await module.main(device, args);
  console.log(JSON.stringify({ adapter: report.description, ...result }, null, 2));
  process.exit(result && result.failures > 0 ? 1 : 0);
} catch (error) {
  console.error(`${modulePath} threw: ${error.stack}`);
  process.exit(1);
}
