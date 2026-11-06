/**
 * Read a test fixture in either lane.
 *
 * 🔴 A CHECK MODULE MUST RUN IN NODE AND IN THE BROWSER FROM ONE SOURCE. Dawn
 * and Chrome's WebGPU are different implementations of the same spec and they
 * disagree about things that matter - out-of-range stores, which optional
 * features exist, the reported limits. So every kernel check has to run in
 * both. Keeping two copies of a check is how they drift until the browser arm
 * quietly stops covering what the Node arm does; the only difference between
 * the lanes should be how the bytes arrive.
 */
export async function readFixture(name, importMetaUrl) {
  const url = new URL(`../../test/fixtures/${name}`, importMetaUrl);
  if (typeof process !== "undefined" && process.versions?.node !== undefined) {
    const { readFileSync } = await import("node:fs");
    return readFileSync(url, "utf8");
  }
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`could not fetch fixture ${name}: ${response.status}`);
  }
  return response.text();
}
