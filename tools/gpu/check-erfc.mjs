/**
 * Measure the f32 `erfc` floor on the device, and prove the measurement can
 * fail.
 *
 * 🔴 THIS CHECK EXISTS BECAUSE A TOLERANCE HAD NO COMMAND BEHIND IT. The
 * nonbonded differential test's Coulomb bar is four times the relative error of
 * the WGSL erfc; that error was a literal, `1.19e-6`, and the tree held three
 * different values for it at once. AGENTS.md requires a tolerance to have a
 * stated origin, and "it was measured once by something no longer present" is
 * not one. check-nonbonded.mjs now takes its bar from this probe.
 *
 * 🔴 AND THE BOUND IS COMPUTED, NOT CHOSEN. Holding the measurement to a number
 * typed in by hand would move the unexplained literal rather than remove it.
 * `argumentUlpBound` derives it: `exp` turns an absolute error in its argument
 * into a relative error in its result, the argument is about -z^2, and one f32
 * ulp of a number of that size is the floor. Four roundings - z*z, the add,
 * exp's own result, the multiply by t - is the allowance.
 *
 * Arm 4 is the one that makes the other three mean anything: it shifts one
 * shipped coefficient by 1e-5 and requires the measurement to report 1e-5,
 * which is the predicted answer rather than merely a bigger number, because a
 * shift of d in the constant term moves ln(erfc) by d everywhere.
 */
import { alphaFromTolerance } from "../../shared/pme/parameters.js";
import { erfc } from "../../shared/erf.js";
import { GpuAllocator } from "../../webgpu/runtime/allocator.js";
import { measureErfcFloor, argumentUlpBound } from "./erfc-floor.js";

const TOLERANCE = 0.0005;

export async function main(device) {
  const allocator = new GpuAllocator(device);
  const results = {};
  let failures = 0;

  // ---- arm 1: the domain a 1.0 nm cutoff actually reaches ----------------
  // 🔴 alpha COMES FROM THE SHIPPED TOLERANCE, NOT FROM A ROUND NUMBER. The
  // direct-space kernel only ever evaluates erfc at alpha*r for r <= cutoff,
  // so this is the whole domain that bears on the Coulomb energy: 2.628 at the
  // notebooks' 1.0 nm and 5e-4. Writing 2.8 by hand would be a guess that
  // happens to be conservative.
  const alpha = alphaFromTolerance(1.0, TOLERANCE);
  {
    const measured = await measureErfcFloor(device, allocator, { zMax: alpha * 1.0 });
    const ok = measured.worst < measured.bound;
    if (!ok) failures += 1;
    results.cutoffDomain = {
      ok,
      alpha: Number(alpha.toFixed(6)),
      relative: measured.worst,
      bound: measured.bound,
      worstAt: Number(measured.worstAt.toFixed(4)),
      // What the same arithmetic gives on the host, and what the device adds
      // on top of it. If these diverge the limit is the backend's `exp`, which
      // WGSL specifies only to 3 ulp, and not the polynomial.
      hostEmulation: measured.worstEmulated,
      deviceVsEmulation: measured.deviceVsEmulation,
      note: "this is the number check-nonbonded.mjs derives its Coulomb bar from",
    };
  }

  // ---- arm 2: every z the kernel could be asked for ----------------------
  // 🔴 BEYOND THE CUTOFF'S OWN SLICE, BECAUSE alpha IS NOT FIXED. A wider
  // cutoff or a tighter tolerance moves alpha*r up, and a polynomial fitted to
  // 2.8 would degrade silently there. 6.0 is where erfc falls to 2.2e-17 and a
  // pair cannot contribute to an f32 sum at all.
  {
    const measured = await measureErfcFloor(device, allocator, { zMax: 6.0 });
    const ok = measured.worst < measured.bound;
    if (!ok) failures += 1;
    results.fullRange = {
      ok,
      relative: measured.worst,
      bound: measured.bound,
      worstAt: Number(measured.worstAt.toFixed(4)),
      hostEmulation: measured.worstEmulated,
      deviceVsEmulation: measured.deviceVsEmulation,
    };
  }

  // ---- arm 3: the negative branch -----------------------------------------
  // The exclusion correction subtracts erf, which is 1 - erfc, and a sign error
  // in the `select` would leave the positive side perfect.
  {
    const measured = await measureErfcFloor(device, allocator,
      { zMax: 6.0, negative: true, samples: 1 << 14 });
    // erfc(-z) approaches 2, so the relative error here is bounded by f32
    // epsilon on the subtraction and has nothing to do with the polynomial.
    const bound = 8 * 2 ** -24;
    const ok = measured.worst < bound;
    if (!ok) failures += 1;
    results.negativeBranch = {
      ok, relative: measured.worst, bound, worstAt: Number(measured.worstAt.toFixed(4)),
      reference: erfc(-1.0),
    };
  }

  // ---- arm 4: a wrong coefficient must be CAUGHT, at the predicted size ----
  {
    // The constant term shifts P(t) by the same amount at every t, and a shift
    // of d in P is a relative error of d in erfc - so 1e-5 here must read as
    // 1e-5 there, to within the floor measured in arm 1.
    const shift = 1e-5;
    const measured = await measureErfcFloor(device, allocator, {
      zMax: alpha * 1.0,
      perturb: { from: "p = -1.26551223 + t * p;", to: "p = -1.26550223 + t * p;" },
    });
    const predicted = shift;
    const agrees = Math.abs(measured.worst - predicted) / predicted < 0.2;
    const caught = measured.worst > argumentUlpBound(alpha);
    if (!agrees || !caught) failures += 1;
    results.perturbationCaught = {
      ok: agrees && caught,
      shiftedConstantTermBy: shift,
      predictedRelative: predicted,
      measuredRelative: measured.worst,
      caughtByTheBound: caught,
      note: "proves the probe reads the coefficients and not a cached pipeline, "
        + "and that the arm-1 bound is tight enough to fail on a bad fit",
    };
  }

  return {
    check: "the f32 erfc floor, measured on the device",
    failures,
    ...results,
  };
}
