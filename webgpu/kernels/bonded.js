/**
 * Bonds, angles and torsions, one thread per ATOM.
 *
 * The WGSL below is a transcription of cpu/forces/bonded.js, which is validated
 * against OpenMM to 1e-15 on energy and against its own numerical gradient to
 * 1e-9 on force. It must stay a transcription: the expressions, the guards and
 * the torsion's middle-atom coefficients are all things that were got wrong
 * once already and are only correct because a differential test says so.
 *
 * 🔴 EACH TERM IS RECOMPUTED ONCE PER ATOM IT TOUCHES, AND THE THREAD TAKES
 * ONLY ITS OWN COMPONENT. That is what removes the atomics (see
 * shared/topology/gather.js). It also means every expression has to be
 * evaluated in full even to extract one of four vectors, which is why the
 * torsion branch computes all four forces and selects - a dead-code eliminator
 * may remove the rest, and if it does not, bonded terms are still a rounding
 * error beside the nonbonded sum.
 */
import { pipelineCacheFor } from "../runtime/pipeline-cache.js";
import { dispatchGrid } from "../runtime/execution.js";

export const BONDED_WORKGROUP = 64;

/**
 * 🔴 THE SOURCE IS GENERATED WITH THE COUNTS BAKED IN, SO THE CACHE KEY MUST
 * NAME EVERY ONE OF THEM. Five constants reach this text, and a key that omits
 * one hands a second system the first system's kernel - which reads the wrong
 * offsets into the shared term arrays and produces a bonded energy that is
 * wrong and finite. See PipelineCache, which compares the source and refuses.
 */
export function bondedShader(layout, atomCount, gridWidth) {
  return `
const ATOM_COUNT: u32 = ${atomCount}u;
const ANGLE_INDEX_BASE: u32 = ${layout.angleIndexBase}u;
const TORSION_INDEX_BASE: u32 = ${layout.torsionIndexBase}u;
const ANGLE_PARAM_BASE: u32 = ${layout.angleParamBase}u;
const TORSION_PARAM_BASE: u32 = ${layout.torsionParamBase}u;
const GRID_WIDTH: u32 = ${gridWidth}u;

@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read_write> forces: array<f32>;
@group(0) @binding(2) var<storage, read_write> energies: array<f32>;
@group(0) @binding(3) var<storage, read> termIndices: array<u32>;
@group(0) @binding(4) var<storage, read> termParams: array<f32>;
@group(0) @binding(5) var<storage, read> gatherStart: array<u32>;
@group(0) @binding(6) var<storage, read> gatherList: array<u32>;

fn atomPosition(atom: u32) -> vec3<f32> {
  return vec3<f32>(positions[atom * 3u], positions[atom * 3u + 1u], positions[atom * 3u + 2u]);
}

@compute @workgroup_size(${BONDED_WORKGROUP})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let atom = id.x + id.y * GRID_WIDTH * ${BONDED_WORKGROUP}u;
  // 🔴 THE BOUNDS CHECK IS NOT OPTIONAL EVEN THOUGH ONE BACKEND HIDES IT.
  // Metal CLAMPS an out-of-range store onto the last element where Vulkan
  // DISCARDS it, so a missing guard corrupts atom N-1 on Apple and is
  // invisible on Linux. WebGPU permits either.
  if (atom >= ATOM_COUNT) { return; }

  var force = vec3<f32>(0.0, 0.0, 0.0);
  var energy = 0.0;
  let begin = gatherStart[atom];
  let end = gatherStart[atom + 1u];

  for (var cursor = begin; cursor < end; cursor = cursor + 1u) {
    let word = gatherList[cursor];
    let kind = word >> 30u;
    let role = (word >> 27u) & 3u;
    let term = word & 0x7ffffffu;

    if (kind == 0u) {
      // ---- harmonic bond: E = (1/2) k (r - r0)^2 ----
      let ia = termIndices[term * 2u];
      let ib = termIndices[term * 2u + 1u];
      let k = termParams[term * 2u];
      let r0 = termParams[term * 2u + 1u];
      let delta = atomPosition(ia) - atomPosition(ib);
      let r = length(delta);
      let stretch = r - r0;
      energy = energy + 0.5 * 0.5 * k * stretch * stretch;
      if (r > 0.0) {
        let scale = k * stretch / r;
        if (role == 0u) { force = force - scale * delta; }
        else { force = force + scale * delta; }
      }
    } else if (kind == 1u) {
      // ---- harmonic angle: E = (1/2) k (theta - theta0)^2 ----
      let ia = termIndices[ANGLE_INDEX_BASE + term * 3u];
      let ib = termIndices[ANGLE_INDEX_BASE + term * 3u + 1u];
      let ic = termIndices[ANGLE_INDEX_BASE + term * 3u + 2u];
      let k = termParams[ANGLE_PARAM_BASE + term * 2u];
      let theta0 = termParams[ANGLE_PARAM_BASE + term * 2u + 1u];
      let pb = atomPosition(ib);
      let v1 = atomPosition(ia) - pb;
      let v2 = atomPosition(ic) - pb;
      let n1 = dot(v1, v1);
      let n2 = dot(v2, v2);
      let normal = cross(v1, v2);
      let crossMagnitude = length(normal);
      let theta = atan2(crossMagnitude, dot(v1, v2));
      let deviation = theta - theta0;
      energy = energy + (1.0 / 3.0) * 0.5 * k * deviation * deviation;
      let dEdTheta = k * deviation;
      // The singular arrangement is a real configuration, not a corner case:
      // a water mid-constraint can pass through 180 degrees and 1/0 here is a
      // NaN that reaches every atom within two steps.
      let safeCross = max(crossMagnitude, 1e-12);
      let scale1 = -dEdTheta / (n1 * safeCross);
      let scale2 = dEdTheta / (n2 * safeCross);
      let fa = scale1 * cross(v1, normal);
      let fc = scale2 * cross(v2, normal);
      if (role == 0u) { force = force + fa; }
      else if (role == 2u) { force = force + fc; }
      else { force = force - (fa + fc); }
    } else {
      // ---- periodic torsion: E = k (1 + cos(n phi - phase)) ----
      let ia = termIndices[TORSION_INDEX_BASE + term * 4u];
      let ib = termIndices[TORSION_INDEX_BASE + term * 4u + 1u];
      let ic = termIndices[TORSION_INDEX_BASE + term * 4u + 2u];
      let idd = termIndices[TORSION_INDEX_BASE + term * 4u + 3u];
      let k = termParams[TORSION_PARAM_BASE + term * 3u];
      let periodicity = termParams[TORSION_PARAM_BASE + term * 3u + 1u];
      let phase = termParams[TORSION_PARAM_BASE + term * 3u + 2u];
      let pa = atomPosition(ia);
      let pb = atomPosition(ib);
      let pc = atomPosition(ic);
      let pd = atomPosition(idd);
      let b1 = pb - pa;
      let b2 = pc - pb;
      let b3 = pd - pc;
      let n1 = cross(b1, b2);
      let n2 = cross(b2, b3);
      let n1Squared = dot(n1, n1);
      let n2Squared = dot(n2, n2);
      let b2Length = length(b2);
      if (n1Squared >= 1e-24 && n2Squared >= 1e-24 && b2Length >= 1e-12) {
        let phi = atan2(dot(cross(n1, n2), b2) / b2Length, dot(n1, n2));
        let argument = periodicity * phi - phase;
        energy = energy + 0.25 * k * (1.0 + cos(argument));
        let dEdPhi = -k * periodicity * sin(argument);
        let fa = (dEdPhi * b2Length / n1Squared) * n1;
        let fd = (-dEdPhi * b2Length / n2Squared) * n2;
        let inverseB2Squared = 1.0 / (b2Length * b2Length);
        let t = dot(b1, b2) * inverseB2Squared;
        let u = dot(b3, b2) * inverseB2Squared;
        if (role == 0u) { force = force + fa; }
        else if (role == 1u) { force = force + (-(1.0 + t) * fa + u * fd); }
        else if (role == 2u) { force = force + (t * fa - (1.0 + u) * fd); }
        else { force = force + fd; }
      }
    }
  }

  // 🔴 ACCUMULATED, NOT ASSIGNED, SO FORCE TERMS COMPOSE - AND THEREFORE THE
  // CALLER MUST CLEAR THE BUFFER. The allocator pools buffers, so a fresh
  // allocation carries the previous step's bytes; a read-modify-write on one
  // that was not cleared adds this step's forces to a stale set. That is the
  // pooling hazard setBufferPooling(false) exists to diagnose, and the fix is
  // an explicit clearBuffer rather than trusting the pool.
  forces[atom * 3u] = forces[atom * 3u] + force.x;
  forces[atom * 3u + 1u] = forces[atom * 3u + 1u] + force.y;
  forces[atom * 3u + 2u] = forces[atom * 3u + 2u] + force.z;
  energies[atom] = energy;
}
`;
}

/**
 * Encode the bonded pass. Writes forces (does not accumulate) and a per-atom
 * energy share that the caller reduces.
 */
export function encodeBonded(device, encoder, resources, atomCount, layout) {
  const maxPerDimension = device.limits.maxComputeWorkgroupsPerDimension;
  const workgroups = Math.ceil(atomCount / BONDED_WORKGROUP);
  const [x, y, z] = dispatchGrid(workgroups, maxPerDimension);
  const gridWidth = x === workgroups ? workgroups : maxPerDimension;
  const code = bondedShader(layout, atomCount, gridWidth);
  const key = `bonded:${atomCount}:${layout.bondCount}:${layout.angleCount}:`
    + `${layout.torsionCount}:${gridWidth}`;
  const pipeline = pipelineCacheFor(device).get(key, code);
  const pass = encoder.beginComputePass({ label: key });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      resources.positions, resources.forces, resources.energies,
      resources.termIndices, resources.termParams,
      resources.gatherStart, resources.gatherList,
    ].map((allocation, binding) => ({ binding, resource: { buffer: allocation.buffer } })),
  }));
  pass.dispatchWorkgroups(x, y, z);
  pass.end();
}
