/**
 * Compile the ribbon WGSL alone and print what the driver says about it.
 *
 * 🔴 `createRenderPipeline` REPORTS "invalid due to a previous error" AND
 * NOTHING ELSE. The real message is on the shader module's compilation info,
 * which nothing reads unless asked - so a one-character WGSL mistake surfaces
 * as a pipeline that cannot be used, named, at draw time, with no line number.
 */
export async function main(device) {
  // Imported rather than scraped, so this runs in the Dawn lane too - the one
  // without an HTTP server behind it.
  const { RIBBON_SHADER: code } = await import("../../web/viewer.js");
  device.pushErrorScope("validation");
  const module = device.createShaderModule({ code, label: "ribbon" });
  const info = await module.getCompilationInfo();
  const error = await device.popErrorScope();
  return {
    check: "ribbon WGSL compilation",
    failures: info.messages.filter((m) => m.type === "error").length,
    lines: code.split("\n").length,
    messages: info.messages.map((m) => ({
      type: m.type, line: m.lineNum, column: m.linePos, message: m.message,
      text: code.split("\n")[m.lineNum - 1],
    })),
    scopeError: error === null ? null : error.message,
  };
}
