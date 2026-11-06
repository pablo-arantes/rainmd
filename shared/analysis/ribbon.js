/**
 * Cartoon ribbon guides: a frame per residue, for a shader to sweep.
 *
 * 🔴 THE HOST DOES THE SEQUENTIAL HALF AND THE GPU DOES THE PARALLEL HALF, and
 * the split is not arbitrary - one step of it genuinely cannot be parallelised.
 * A ribbon needs a coordinate frame at every residue, and the frame's "which
 * way is up" must be made CONTINUOUS along the chain by comparing each residue
 * with the one before it and flipping when they disagree. That is a scan, not a
 * map. It is also tiny: one iteration per residue, hundreds of them, against
 * the tens of thousands of vertices the sweep produces. So this builds guides
 * on the host and the vertex shader interpolates between them procedurally -
 * no vertex buffer, no index buffer, the same trick the sphere impostors use.
 *
 * 🔴 WITHOUT THE FLIP CORRECTION A BETA STRAND IS A CORKSCREW. The natural
 * frame comes from the local curvature - the direction from the chord between
 * a residue's neighbours to the residue itself - and in a strand the backbone
 * zigzags, so that direction REVERSES at every residue. Interpolating between
 * two opposed frames rotates the ribbon 180 degrees per residue, which renders
 * as a twisted tangle that looks like a bug in the spline rather than in the
 * frame. PyMOL and 3Dmol both carry the same correction; it is the single most
 * important line in this file.
 *
 * 🔴 AND A SEGMENT NEVER SPANS A CHAIN BREAK. The control points come from an
 * explicit segment list rather than from "guide i and guide i+1", because two
 * chain ends that happen to be adjacent in the residue order are not adjacent
 * in space - and a Catmull-Rom through them draws a ribbon across the gap,
 * joining two molecules that are not bonded.
 */
import { HELIX, SHEET, COIL } from "./secondary.js";
import { STRUCTURE_COLOURS } from "./selection.js";

/**
 * Cross-section half-widths and half-thicknesses, in nm.
 *
 * 🔴 A COIL IS A SQUARE ROD, NOT A FLAT BAND. Giving coil the same width as a
 * helix and only changing the colour makes a structure that is mostly loop look
 * like a mostly-helical one; making it as thin in width as it is in thickness
 * is what reads as "no secondary structure here". These are close to PyMOL's
 * defaults: a 0.24 nm wide helix ribbon, a 0.08 nm loop.
 */
const CROSS_SECTION = {
  [HELIX]: { halfWidth: 0.120, halfThickness: 0.030 },
  [SHEET]: { halfWidth: 0.110, halfThickness: 0.028 },
  [COIL]: { halfWidth: 0.040, halfThickness: 0.040 },
};

/** How wide a strand's arrowhead base is, relative to the strand itself. */
const ARROW_WIDTH = 1.9;

/**
 * The arrow's point, in nm. Not zero: a cross-section of exactly zero width
 * makes two of the box's four faces degenerate, and a degenerate triangle with
 * an undefined normal shades as a black sliver at the very tip.
 */
const ARROW_TIP = 0.012;

/** Floats per guide: (pos,halfWidth) (side,halfThickness) (colour,pad). */
export const GUIDE_FLOATS = 12;

const subtract = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const scale = (a, k) => [a[0] * k, a[1] * k, a[2] * k];

function normalise(v) {
  const length = Math.sqrt(dot(v, v));
  return length > 1e-9 ? scale(v, 1 / length) : null;
}

/** Any unit vector perpendicular to `v`, for the degenerate first frame. */
function anyPerpendicular(v) {
  // Cross with whichever axis `v` is least aligned to, so the cross product is
  // never near zero.
  const axis = Math.abs(v[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  return normalise(cross(v, axis));
}

/**
 * Split the alpha-carbon trace into runs that are continuous in the chain.
 * @returns {Array<{from: number, to: number}>} half-open ranges into the trace
 */
function runsOfTrace(traceLength, breaks) {
  const starts = new Set(breaks);
  const runs = [];
  let from = 0;
  for (let index = 1; index <= traceLength; index += 1) {
    if (index === traceLength || starts.has(index)) {
      if (index - from >= 2) runs.push({ from, to: index });
      from = index;
    }
  }
  return runs;
}

/**
 * @param {object} topology from parsePrmtop
 * @param {ArrayLike<number>} positions 3N, nm
 * @param {object} selections from buildSelections
 * @param {object} secondary from secondaryOfAtoms
 * @param {object} [options]
 * @param {number[]} [options.visible] per-atom visibility, 1 or 0; a residue
 *   whose alpha carbon is hidden is left out of the ribbon entirely.
 * @returns {{guides: Float32Array, segments: Uint32Array, segmentCount: number,
 *   guideCount: number, runCount: number}}
 */
export function buildRibbon(topology, positions, selections, secondary, options = {}) {
  const trace = secondary.trace;
  const kinds = secondary.perResidue;
  const visible = options.visible ?? null;

  // 🔴 A HIDDEN RESIDUE BREAKS THE RUN, IT DOES NOT SHORTEN IT. Showing
  // "protein only" while a ligand sits mid-chain must not draw a ribbon that
  // leaps the ligand; and showing one chain of a complex must not connect it to
  // the other. Visibility is therefore a source of chain breaks, not a filter
  // applied afterwards.
  const keptTrace = [];
  const keptKinds = [];
  const breaks = new Set(secondary.breaks);
  const keptBreaks = [];
  let previousKept = -1;
  for (let index = 0; index < trace.length; index += 1) {
    if (visible !== null && visible[trace[index]] === 0) continue;
    // Either the chain itself breaks here, or residues were skipped because
    // they are hidden - both mean the ribbon must not be continuous.
    const broke = keptTrace.length > 0
      && (breaks.has(index) || previousKept !== index - 1);
    if (broke) keptBreaks.push(keptTrace.length);
    keptTrace.push(trace[index]);
    keptKinds.push(kinds[index]);
    previousKept = index;
  }

  const runs = runsOfTrace(keptTrace.length, keptBreaks);
  const guides = new Float32Array(keptTrace.length * GUIDE_FLOATS);
  const segmentList = [];

  for (const run of runs) {
    const length = run.to - run.from;
    const point = (offset) => {
      // Clamped inside the run, so an end residue's frame uses the residue
      // itself rather than a neighbour from another chain.
      const index = Math.min(Math.max(offset, run.from), run.to - 1);
      const atom = keptTrace[index];
      return [positions[atom * 3], positions[atom * 3 + 1], positions[atom * 3 + 2]];
    };

    let previousSide = null;
    for (let index = run.from; index < run.to; index += 1) {
      const here = point(index);
      const before = point(index - 1);
      const after = point(index + 1);
      const tangent = normalise(subtract(after, before))
        ?? normalise(subtract(after, here))
        ?? [0, 0, 1];
      // The curvature direction: from the chord's midpoint out to the residue.
      const midpoint = scale([before[0] + after[0], before[1] + after[1],
        before[2] + after[2]], 0.5);
      let radial = subtract(here, midpoint);
      // Project out the tangent so the frame is orthogonal.
      radial = subtract(radial, scale(tangent, dot(radial, tangent)));
      let side = normalise(cross(tangent, radial));
      if (side === null) {
        // Collinear residues - a perfectly straight stretch has no curvature to
        // read. Carry the previous frame forward, re-orthogonalised against the
        // new tangent, which keeps a straight run from twisting at all.
        side = previousSide === null ? anyPerpendicular(tangent)
          : normalise(subtract(previousSide, scale(tangent, dot(previousSide, tangent))))
            ?? anyPerpendicular(tangent);
      }
      // 🔴 THE FLIP CORRECTION. See the header: without it a strand corkscrews.
      if (previousSide !== null && dot(side, previousSide) < 0) side = scale(side, -1);
      previousSide = side;

      const kind = keptKinds[index] ?? COIL;
      const section = CROSS_SECTION[kind] ?? CROSS_SECTION[COIL];
      let halfWidth = section.halfWidth;
      // 🔴 AN ARROWHEAD IS A WIDTH PROFILE, NOT A SHAPE - but it needs THREE
      // residues, not two. The first version widened a strand's last residue
      // and stopped there, which draws a blunt flag rather than an arrow: the
      // ribbon simply got wider and ended square. A real arrow has a wide BASE
      // that tapers to a POINT, so the base is the second-to-last residue and
      // the tip is the last one, and the sweep interpolates the triangle
      // between them for free. No second geometry path and no extra vertices.
      const nextIsSheet = keptKinds[index + 1] === SHEET;
      const afterThatIsSheet = keptKinds[index + 2] === SHEET;
      if (kind === SHEET && !nextIsSheet) {
        halfWidth = ARROW_TIP;
      } else if (kind === SHEET && nextIsSheet && !afterThatIsSheet) {
        halfWidth *= ARROW_WIDTH;
      }
      const colour = (STRUCTURE_COLOURS[kind] ?? STRUCTURE_COLOURS[COIL]).rgb;

      const base = index * GUIDE_FLOATS;
      guides[base] = here[0];
      guides[base + 1] = here[1];
      guides[base + 2] = here[2];
      guides[base + 3] = halfWidth;
      guides[base + 4] = side[0];
      guides[base + 5] = side[1];
      guides[base + 6] = side[2];
      guides[base + 7] = section.halfThickness;
      guides[base + 8] = colour[0];
      guides[base + 9] = colour[1];
      guides[base + 10] = colour[2];
      guides[base + 11] = 0;
    }

    // One segment per interior pair, with the four Catmull-Rom control points
    // clamped into this run.
    for (let index = run.from; index + 1 < run.to; index += 1) {
      segmentList.push(
        Math.max(index - 1, run.from),
        index,
        index + 1,
        Math.min(index + 2, run.to - 1),
      );
    }
  }

  return {
    guides,
    segments: Uint32Array.from(segmentList),
    segmentCount: segmentList.length / 4,
    guideCount: keptTrace.length,
    runCount: runs.length,
  };
}
