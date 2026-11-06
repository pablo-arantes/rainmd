/**
 * The page's wiring, and the analysis it draws.
 *
 * 🔴 THE FIRST TEST HERE IS THE ONE THAT PAYS FOR THE FILE. `web/app.js` is a
 * module with no build step and top-level side effects, so a single
 * `element("hide-solvent").addEventListener(...)` naming a control that
 * `index.html` no longer has throws DURING MODULE EVALUATION - which aborts the
 * rest of the file silently. Everything declared after the throw simply does
 * not exist: `window.rainmd` was never assigned, the three export buttons never
 * got handlers, and the page still LOOKED fine, because the device panel is
 * filled by an earlier statement. It cost a whole debugging session, and the
 * error that finally named it came from CDP rather than from anything here.
 *
 * It is a static check because it has to be: no DOM, no device, no fixtures,
 * runs in `npm test`, and it cannot pass by accident.
 */
import assert from "node:assert/strict";
import { test } from "node:test";
import { readFileSync, existsSync } from "node:fs";
import { kabschTransform, applyTransform, centroid } from "../shared/analysis/kabsch.js";
import {
  buildSelections, classifyResidue, VIEW_PRESETS, BODY_COLOURS, STRUCTURE_COLOURS,
} from "../shared/analysis/selection.js";
import {
  rmsdFitOn, radiusOfGyration, interfaceDistance, RunningRmsf, LiveAnalysis,
} from "../shared/analysis/metrics.js";
import { writePdb } from "../shared/traj/pdb.js";
import {
  assignSecondary, secondaryOfAtoms, HELIX, SHEET, COIL,
} from "../shared/analysis/secondary.js";
import { buildRibbon, GUIDE_FLOATS } from "../shared/analysis/ribbon.js";
import { parsePrmtop } from "../shared/amber/prmtop.js";
import { parseInpcrd } from "../shared/amber/inpcrd.js";
import { buildSystem } from "../shared/topology/system.js";

const source = (name) => readFileSync(new URL(`../${name}`, import.meta.url), "utf8");
const fixture = (name) => readFileSync(new URL(`fixtures/${name}`, import.meta.url), "utf8");

// --------------------------------------------------------- the page's wiring ----

test("every control web/app.js names exists in index.html", () => {
  const app = source("web/app.js");
  const html = source("index.html");
  const present = new Set([...html.matchAll(/id="([^"]+)"/g)].map((m) => m[1]));
  // Both spellings: the helper, and any direct call that bypasses it.
  const named = new Set([
    ...[...app.matchAll(/element\("([^"]+)"\)/g)].map((m) => m[1]),
    ...[...app.matchAll(/getElementById\("([^"]+)"\)/g)].map((m) => m[1]),
  ]);
  assert.ok(named.size > 20, `only ${named.size} controls found - did the helper change name?`);
  const missing = [...named].filter((id) => !present.has(id)).sort();
  assert.deepEqual(missing, [],
    `web/app.js throws on load: index.html has no ${missing.join(", ")}`);
});

test("every control index.html offers is wired, or deliberately inert", () => {
  const app = source("web/app.js");
  const html = source("index.html");
  // Canvases and read-only report targets are written to, not listened to.
  const passive = new Set(["viewer", "log", "live", "device-report", "system-report",
    "load-error", "status-panel", "run-panel", "view-counts", "panels",
    "plot-rmsd", "plot-rg", "plot-energy", "plot-rmsf", "plot-iface", "plot-temperature"]);
  const ids = [...html.matchAll(/id="([^"]+)"/g)].map((m) => m[1]);
  const unreferenced = ids.filter((id) => !passive.has(id)
    && !app.includes(`"${id}"`)).sort();
  assert.deepEqual(unreferenced, [],
    `index.html offers controls the page never reads: ${unreferenced.join(", ")}`);
});

test("every module the page imports is on disk", () => {
  // 🔴 A BARE RELATIVE SPECIFIER IS THE WHOLE DEPLOYMENT CONTRACT. There is no
  // bundler, so a path that does not resolve is a 404 at load time and a blank
  // page - with no error anywhere a user would look.
  for (const file of ["web/app.js", "web/viewer.js", "web/plot.js"]) {
    const text = source(file);
    for (const [, specifier] of text.matchAll(/^import[^"']*from\s+"([^"]+)";/gm)) {
      assert.ok(specifier.startsWith("."), `${file} imports a bare specifier ${specifier}`);
      const url = new URL(`../web/${specifier}`, import.meta.url);
      assert.ok(existsSync(url), `${file} imports ${specifier}, which is not on disk`);
    }
  }
});

test("the panels the page draws are the ones the HTML provides", () => {
  const app = source("web/app.js");
  const html = source("index.html");
  const canvases = [...html.matchAll(/id="(plot-[^"]+)"/g)].map((m) => m[1]).sort();
  const used = [...app.matchAll(/element\("(plot-[^"]+)"\)/g)].map((m) => m[1]).sort();
  assert.deepEqual(used, canvases, "a panel canvas with no Panel, or the reverse");
  assert.equal(canvases.length, 6, "six live panels, as JIGGLE shows");
});

test("the body palette is one declaration, and the page reads it", () => {
  // 🔴 THE PLOT AND THE STRUCTURE MUST AGREE ON WHAT ORANGE MEANS. A reader
  // maps a trace onto a molecule by colour; two literals in two files cannot
  // be relied on to stay the same hue, and nothing fails when they drift.
  for (const [name, body] of Object.entries({ ...BODY_COLOURS, ...STRUCTURE_COLOURS })) {
    assert.match(body.line, /^#[0-9a-f]{6}$/, `${name}.line is not a hex colour`);
    assert.equal(body.rgb.length, 3, `${name}.rgb is not a triple`);
    for (const channel of body.rgb) {
      assert.ok(channel >= 0 && channel <= 1, `${name}.rgb is not in 0..1`);
    }
    // The two forms are deliberately different brightnesses - a line on a light
    // panel against a lit sphere on a dark one - but they must be the same HUE,
    // or the mapping is broken rather than merely adjusted. Compare the channel
    // ORDERING, which is what hue is.
    const fromHex = [1, 3, 5].map((at) => parseInt(body.line.slice(at, at + 2), 16) / 255);
    const order = (triple) => [0, 1, 2]
      .sort((a, b) => triple[b] - triple[a]).join("");
    assert.equal(order(fromHex), order(body.rgb),
      `${name}: ${body.line} and [${body.rgb}] are not the same hue`);
  }
  // Both consumers must actually read it rather than restating a literal.
  const viewer = source("web/viewer.js");
  assert.ok(viewer.includes("BODY_COLOURS"), "web/viewer.js does not read the palette");
  assert.doesNotMatch(viewer.slice(viewer.indexOf("GROUP_COLOURS")),
    /GROUP_COLOURS = \{/, "web/viewer.js restates the palette as a literal");
  const app = source("web/app.js");
  assert.ok(app.includes("BODY_COLOURS.receptor.line"),
    "web/app.js does not colour the receptor trace from the palette");
});

// ------------------------------------------------------ secondary structure ----

/** CA positions of an ideal alpha helix: r 0.23 nm, rise 0.15 nm, 100 deg. */
function idealHelix(count) {
  const out = [];
  for (let index = 0; index < count; index += 1) {
    const turn = (index * 100 * Math.PI) / 180;
    out.push(0.23 * Math.cos(turn), 0.23 * Math.sin(turn), 0.15 * index);
  }
  return Float64Array.from(out);
}

/** CA positions of an ideal extended strand, with its real zigzag. */
function idealStrand(count) {
  const out = [];
  for (let index = 0; index < count; index += 1) {
    out.push(0.34 * index, (index % 2) * 0.08, 0);
  }
  return Float64Array.from(out);
}

test("the CA criterion separates an ideal helix from an ideal strand", () => {
  // 🔴 THE IDEAL FORMS ARE BUILT HERE RATHER THAN TAKEN FROM A FIXTURE, because
  // the constants in secondary.js were DERIVED from exactly this geometry. A
  // hand-labelled PDB would test the assignment against somebody's judgement;
  // this tests it against the arithmetic it claims to implement. The repository
  // also has no protein fixture with real secondary structure - both are capped
  // peptides - so a fixture could not do it anyway.
  const trace = Array.from({ length: 14 }, (unused, index) => index);
  const helix = assignSecondary(idealHelix(14), trace);
  const strand = assignSecondary(idealStrand(14), trace);
  assert.deepEqual(helix, new Array(14).fill(HELIX), `helix came back ${helix.join(" ")}`);
  assert.deepEqual(strand, new Array(14).fill(SHEET), `strand came back ${strand.join(" ")}`);

  // And the distances are the ones the header derives, so a future edit to the
  // windows is checked against the geometry rather than against these results.
  const span = (positions, from, to) => Math.hypot(
    positions[from * 3] - positions[to * 3],
    positions[from * 3 + 1] - positions[to * 3 + 1],
    positions[from * 3 + 2] - positions[to * 3 + 2]);
  const h = idealHelix(14);
  assert.ok(Math.abs(span(h, 0, 3) - 0.505) < 0.002, `helix i,i+3 is ${span(h, 0, 3)}`);
  assert.ok(Math.abs(span(h, 0, 4) - 0.620) < 0.002, `helix i,i+4 is ${span(h, 0, 4)}`);
  const t = idealStrand(14);
  assert.ok(span(t, 0, 3) > 0.9 && span(t, 0, 4) > 1.2, "strand spans must be ~2x the helix");
});

test("a short trace, a bent chain and a chain break all come back coil", () => {
  // Fewer than five residues has no window to measure.
  assert.deepEqual(assignSecondary(idealHelix(4), [0, 1, 2, 3]),
    [COIL, COIL, COIL, COIL]);
  // 🔴 A CHAIN BREAK MUST BE REFUSED, NOT MEASURED ACROSS, and showing that
  // needs a case where the SPANNING window is the only source of structure.
  // Two helices joined end to end will not do it - both halves are helices
  // whichever way the break is read, so declaring it changes nothing and the
  // test passes while the break list is inert. (That was the first version of
  // this test, and it failed for exactly that reason.)
  //
  // What separates them: ONE ideal helix with a break declared in its middle.
  // Every window that would cross index 4 is refused, so the residues before
  // it lose their assignment and come back coil, while the window starting at
  // the break itself still spans a real helix and keeps it.
  const continuous = idealHelix(9);
  const trace9 = Array.from({ length: 9 }, (unused, index) => index);
  const whole = assignSecondary(continuous, trace9, []);
  const split = assignSecondary(continuous, trace9, [4]);
  assert.deepEqual(whole, new Array(9).fill(HELIX),
    "with no break declared the whole thing is one helix");
  assert.deepEqual(split.slice(0, 4), new Array(4).fill(COIL),
    "residues before a declared break have no window that does not cross it");
  assert.deepEqual(split.slice(4), new Array(5).fill(HELIX),
    "the window starting at the break is still measurable");
});

test("a force field's own helix and strand are assigned correctly", () => {
  // 🔴 THE IDEAL-GEOMETRY TEST ABOVE CANNOT SEE THIS, and the difference is the
  // reason both exist. That one builds the coordinates from the same equations
  // the windows were derived from, so it checks the arithmetic against itself.
  // These two fixtures are built by tleap from ff14SB with `impose` setting
  // phi/psi, so every bond length, angle and sidechain is the force field's -
  // and the CA spacings land at 0.521/0.638 nm for the helix where the textbook
  // says 0.505/0.620, and 1.051/1.386 for the strand against 1.023/1.360. That
  // 3% gap between an ideal helix and a real one is what the windows have to be
  // wide enough to absorb, and nothing but a real structure measures it.
  for (const [name, expected, spans] of [
    ["helix", HELIX, [0.521, 0.638]],
    ["strand", SHEET, [1.051, 1.386]],
  ]) {
    const topology = parsePrmtop(fixture(`${name}.prmtop`));
    const coordinates = parseInpcrd(fixture(`${name}.inpcrd`),
      { expectedAtomCount: topology.atomCount });
    assert.equal(coordinates.box, null, `${name} is a vacuum fixture`);
    const system = buildSystem(topology, {
      constraints: "HBonds", rigidWater: true, cutoff: 0, method: "NoCutoff",
    });
    const selections = buildSelections(system);
    assert.equal(selections.groups.alphaCarbon.length, 15,
      `${name} must have 15 alpha carbons`);
    const result = secondaryOfAtoms(system, coordinates.positions, selections);
    assert.deepEqual(result.perResidue, new Array(15).fill(expected),
      `${name} came back ${result.perResidue.join(" ")}`);
    assert.equal(result.breaks.length, 0, "one chain, no break");
    // And the spacings themselves, so a future change to the windows is
    // measured against the structure rather than against this result.
    const ca = selections.groups.alphaCarbon;
    const span = (from, to) => Math.hypot(
      coordinates.positions[ca[from] * 3] - coordinates.positions[ca[to] * 3],
      coordinates.positions[ca[from] * 3 + 1] - coordinates.positions[ca[to] * 3 + 1],
      coordinates.positions[ca[from] * 3 + 2] - coordinates.positions[ca[to] * 3 + 2]);
    assert.ok(Math.abs(span(2, 5) - spans[0]) < 0.01,
      `${name} i,i+3 is ${span(2, 5).toFixed(3)}, expected ${spans[0]}`);
    assert.ok(Math.abs(span(2, 6) - spans[1]) < 0.01,
      `${name} i,i+4 is ${span(2, 6).toFixed(3)}, expected ${spans[1]}`);
  }
});

test("secondaryOfAtoms spreads a residue's assignment to all its atoms", () => {
  const { system, coordinates } = loadFixture("two-body");
  const selections = buildSelections(system);
  const result = secondaryOfAtoms(system, coordinates.positions, selections);
  assert.equal(result.perAtom.length, system.topology.atomCount);
  // Every water and ion atom must be coil: they have no alpha carbon, and a
  // ligand coloured as structure reads as a third secondary structure.
  for (let atom = 0; atom < system.topology.atomCount; atom += 1) {
    if (selections.kindOfAtom[atom] === "water" || selections.kindOfAtom[atom] === "ion") {
      assert.equal(result.perAtom[atom], COIL);
    }
  }
  // Atoms of one residue must agree with each other.
  const byResidue = new Map();
  for (let atom = 0; atom < system.topology.atomCount; atom += 1) {
    const residue = system.topology.residueOfAtom[atom];
    const seen = byResidue.get(residue);
    if (seen === undefined) byResidue.set(residue, result.perAtom[atom]);
    else assert.equal(result.perAtom[atom], seen, `residue ${residue} is split`);
  }
  // The two chains must produce a break, or windows span them.
  assert.equal(result.breaks.length, 1, "two peptide chains, one break");
  assert.equal(result.counts.helix + result.counts.sheet + result.counts.coil,
    result.perResidue.length);
});

// ------------------------------------------------------------------ ribbon ----

/** Load a structure fixture and build its ribbon. */
function ribbonOf(name, options = {}) {
  const topology = parsePrmtop(fixture(`${name}.prmtop`));
  const coordinates = parseInpcrd(fixture(`${name}.inpcrd`),
    { expectedAtomCount: topology.atomCount });
  const system = buildSystem(topology, {
    constraints: "HBonds", rigidWater: true, cutoff: 0, method: "NoCutoff",
  });
  const selections = buildSelections(system);
  const secondary = secondaryOfAtoms(system, coordinates.positions, selections);
  const ribbon = buildRibbon(topology, coordinates.positions, selections,
    secondary, options);
  return { topology, coordinates, system, selections, secondary, ribbon };
}

const guideField = (ribbon, index, offset) => [
  ribbon.guides[index * GUIDE_FLOATS + offset],
  ribbon.guides[index * GUIDE_FLOATS + offset + 1],
  ribbon.guides[index * GUIDE_FLOATS + offset + 2],
];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

test("a ribbon's frame is unit length and perpendicular to the chain", () => {
  for (const name of ["helix", "strand", "mixed"]) {
    const { ribbon } = ribbonOf(name);
    assert.ok(ribbon.guideCount > 10, `${name} produced ${ribbon.guideCount} guides`);
    assert.equal(ribbon.segmentCount, ribbon.guideCount - 1,
      "one segment per interior pair, in a single run");
    for (let index = 1; index + 1 < ribbon.guideCount; index += 1) {
      const side = guideField(ribbon, index, 4);
      assert.ok(Math.abs(Math.hypot(...side) - 1) < 1e-5,
        `${name} guide ${index} side vector is not a unit vector`);
      const before = guideField(ribbon, index - 1, 0);
      const after = guideField(ribbon, index + 1, 0);
      const tangent = [after[0] - before[0], after[1] - before[1], after[2] - before[2]];
      const length = Math.hypot(...tangent);
      // 🔴 THE WIDTH MUST BE PERPENDICULAR TO THE CHAIN. A side vector with a
      // component along the backbone shears the cross-section, so the ribbon's
      // edges ripple and its apparent width changes with the viewing angle -
      // which looks like a spline artefact and is a frame error.
      assert.ok(Math.abs(dot3(side, tangent) / length) < 1e-5,
        `${name} guide ${index}: side is not perpendicular to the chain`);
    }
  }
});

test("the ribbon frame never flips, which is what keeps a strand from twisting", () => {
  // 🔴 THIS IS THE ONE PROPERTY THAT CANNOT BE SEEN IN A STILL OF A HELIX. A
  // helix has smooth curvature so its natural frame is already continuous; a
  // strand zigzags, so the curvature direction REVERSES at every residue and
  // an uncorrected frame rotates 180 degrees per residue. Falsified by deleting
  // the flip correction in ribbon.js, which takes the strand's minimum
  // consecutive dot product from +0.83 to about -1.
  for (const name of ["helix", "strand", "mixed"]) {
    const { ribbon } = ribbonOf(name);
    let worst = 1;
    for (let index = 1; index < ribbon.guideCount; index += 1) {
      worst = Math.min(worst,
        dot3(guideField(ribbon, index, 4), guideField(ribbon, index - 1, 4)));
    }
    assert.ok(worst > 0, `${name}: consecutive side vectors oppose (min dot ${worst})`);
  }
});

test("a strand ends in an arrowhead and a helix does not", () => {
  const strand = ribbonOf("strand").ribbon;
  const widths = Array.from({ length: strand.guideCount },
    (unused, index) => strand.guides[index * GUIDE_FLOATS + 3]);
  const body = widths[0];
  // Wide base, then a point: three residues make an arrow, two make a flag.
  assert.ok(widths.at(-2) > body * 1.5,
    `the arrow base is ${widths.at(-2)}, not wider than the strand's ${body}`);
  assert.ok(widths.at(-1) < body * 0.25,
    `the arrow tip is ${widths.at(-1)}, which is not a point`);
  // 🔴 NOT ZERO, EITHER. A cross-section of exactly zero width makes two of the
  // box's four faces degenerate, and a triangle with no area has no normal -
  // it shades as a black sliver exactly at the tip the eye is drawn to.
  assert.ok(widths.at(-1) > 0, "a zero-width tip has degenerate faces");

  const helix = ribbonOf("helix").ribbon;
  const helixWidths = Array.from({ length: helix.guideCount },
    (unused, index) => helix.guides[index * GUIDE_FLOATS + 3]);
  assert.equal(new Set(helixWidths.map((w) => w.toFixed(4))).size, 1,
    "a helix has no arrowhead, so its width is constant");
});

test("a coil is a thin rod and a helix is a wide band", () => {
  const { ribbon, secondary } = ribbonOf("mixed");
  const widthOf = (kind) => {
    for (let index = 0; index < ribbon.guideCount; index += 1) {
      if (secondary.perResidue[index] === kind) {
        return ribbon.guides[index * GUIDE_FLOATS + 3];
      }
    }
    return null;
  };
  const helix = widthOf(HELIX);
  const coil = widthOf(COIL);
  assert.ok(helix !== null && coil !== null, "the mixed fixture must have both");
  // 🔴 A COIL AS WIDE AS A HELIX MAKES EVERY STRUCTURE LOOK HELICAL. The
  // difference in width is what carries the information, not the colour alone -
  // a reader who is colour-blind still sees the secondary structure.
  assert.ok(coil < helix * 0.5,
    `coil ${coil} is not clearly thinner than helix ${helix}`);
});

test("a ribbon never spans a chain break or a hidden residue", () => {
  const { ribbon, selections, secondary } = ribbonOf("two-body");
  // Two peptide chains: two runs, and no segment joining them.
  assert.equal(ribbon.runCount, 2, `two chains gave ${ribbon.runCount} run(s)`);
  assert.equal(ribbon.segmentCount, ribbon.guideCount - 2,
    "each run loses one segment at its end, so two runs lose two");

  // 🔴 AND A RUN OF FEWER THAN TWO GUIDES IS DROPPED, NOT DRAWN. two-body's
  // chains carry only three and two alpha carbons - ACE and NME are caps and
  // have none - so hiding one residue leaves fragments too short to span a
  // segment. That is correct, and it is why the hiding case below uses the
  // 24-guide fixture instead: on a 5-guide one, "the run count went up" is not
  // a property the geometry can have.
  const whole = ribbonOf("mixed");
  const visible = new Uint8Array(whole.topology.atomCount).fill(1);
  const middle = whole.secondary.trace[Math.floor(whole.secondary.trace.length / 2)];
  visible[middle] = 0;
  const split = ribbonOf("mixed", { visible }).ribbon;
  assert.equal(split.guideCount, whole.ribbon.guideCount - 1, "one fewer guide");
  assert.equal(whole.ribbon.runCount, 1, "the mixed fixture is one chain");
  assert.equal(split.runCount, 2,
    `hiding a mid-chain residue must split the run, got ${split.runCount}`);
  // And the two halves together must have two fewer segments than the whole:
  // one lost to the hidden residue, one to the new run's end.
  assert.equal(split.segmentCount, whole.ribbon.segmentCount - 2);
  assert.equal(secondary.trace.length, 5, "two-body has five alpha carbons");
});

test("every ribbon segment names four control guides that exist", () => {
  for (const name of ["helix", "mixed", "two-body"]) {
    const { ribbon } = ribbonOf(name);
    assert.equal(ribbon.segments.length, ribbon.segmentCount * 4);
    for (let segment = 0; segment < ribbon.segmentCount; segment += 1) {
      const control = ribbon.segments.slice(segment * 4, segment * 4 + 4);
      for (const index of control) {
        assert.ok(index < ribbon.guideCount,
          `${name} segment ${segment} names guide ${index} of ${ribbon.guideCount}`);
      }
      // 🔴 THE MIDDLE PAIR MUST BE ADJACENT, and the outer pair clamped INTO
      // the same run. A Catmull-Rom whose control points come from two
      // different chains draws a ribbon across the gap between them.
      assert.equal(control[2] - control[1], 1,
        `${name} segment ${segment} does not span adjacent guides`);
      assert.ok(control[0] <= control[1] && control[3] >= control[2],
        "control points must bracket the span");
    }
  }
});

// ------------------------------------------------------------------ Kabsch ----

test("Kabsch recovers a known rotation and never a reflection", () => {
  // A 30 degree turn about z, applied to a deliberately non-planar set: a
  // planar one is degenerate and can be fitted by a reflection too.
  const angle = Math.PI / 6;
  const [c, s] = [Math.cos(angle), Math.sin(angle)];
  const reference = Float64Array.from([
    0.1, 0.2, 0.3, 1.4, 0.1, -0.2, -0.7, 1.1, 0.5, 0.3, -1.2, 0.9, 1.1, 0.8, -1.3,
  ]);
  const mobile = new Float64Array(reference.length);
  for (let atom = 0; atom < reference.length / 3; atom += 1) {
    const [x, y, z] = [reference[atom * 3], reference[atom * 3 + 1], reference[atom * 3 + 2]];
    mobile[atom * 3] = c * x - s * y + 5;
    mobile[atom * 3 + 1] = s * x + c * y - 3;
    mobile[atom * 3 + 2] = z + 2;
  }
  const indices = Int32Array.from([0, 1, 2, 3, 4]);
  const transform = kabschTransform(mobile, reference, indices);
  const out = new Float64Array(3);
  let worst = 0;
  for (const atom of indices) {
    applyTransform(transform, mobile, atom, out);
    for (let k = 0; k < 3; k += 1) {
      worst = Math.max(worst, Math.abs(out[k] - reference[atom * 3 + k]));
    }
  }
  assert.ok(worst < 1e-12, `recovered the rotation to ${worst}, wanted < 1e-12`);
  // 🔴 THE DETERMINANT IS THE ASSERTION, NOT A SANITY CHECK. An SVD Kabsch has
  // to correct the reflected branch by hand (`diag(1,1,d)`); the quaternion
  // form here cannot express a reflection at all, and this is what says so. A
  // reflected fit returns a plausible, smaller RMSD for a mirror-image pose.
  const r = transform.rotation;
  const det = r[0] * (r[4] * r[8] - r[5] * r[7])
    - r[1] * (r[3] * r[8] - r[5] * r[6])
    + r[2] * (r[3] * r[7] - r[4] * r[6]);
  assert.ok(Math.abs(det - 1) < 1e-12, `determinant ${det}, wanted +1`);
});

test("a centroid is the plain mean over the named atoms only", () => {
  const positions = Float64Array.from([0, 0, 0, 2, 0, 0, 0, 4, 0, 99, 99, 99]);
  const mean = centroid(positions, Int32Array.from([0, 1, 2]));
  const want = [2 / 3, 4 / 3, 0];
  for (let k = 0; k < 3; k += 1) {
    assert.ok(Math.abs(mean[k] - want[k]) < 1e-15,
      `component ${k} came back ${mean[k]}, wanted ${want[k]}`
      + " - the fourth atom must not be in the average");
  }
});

// --------------------------------------------------------------- selections ----

test("residues are classified by name, prefixes and termini stripped", () => {
  assert.equal(classifyResidue("ALA"), "protein");
  assert.equal(classifyResidue("NALA"), "protein", "an N-terminal prefix is still alanine");
  assert.equal(classifyResidue("CLYS"), "protein", "a C-terminal prefix too");
  assert.equal(classifyResidue("DG5"), "nucleic", "a 5' terminal nucleotide");
  assert.equal(classifyResidue("WAT"), "water");
  assert.equal(classifyResidue("HOH"), "water");
  assert.equal(classifyResidue("Na+"), "ion");
  assert.equal(classifyResidue("MOL"), "ligand", "anything unrecognised is a ligand");
});

function loadFixture(stem) {
  const topology = parsePrmtop(fixture(`${stem}.prmtop`));
  const coordinates = parseInpcrd(fixture(`${stem}.inpcrd`), topology.atomCount);
  const system = buildSystem(topology, {
    constraints: "HBonds", rigidWater: true, cutoff: 0.8,
    box: coordinates.box, method: "PME",
  });
  return { system, coordinates };
}

test("the two-body fixture splits into a receptor and a binder", () => {
  const { system } = loadFixture("two-body");
  const selections = buildSelections(system);
  // 🔴 THE SPLIT COMES FROM THE MOLECULE PARTITION, BECAUSE A PRMTOP HAS NO
  // CHAIN IDS. JIGGLE asks the user for receptor and binder chain letters;
  // AMBER throws chain information away, so the only thing left that says
  // "these atoms are one body" is which atoms are bonded to which.
  assert.equal(selections.soluteMoleculeCount, 2,
    "two peptide chains must come back as two solute molecules");
  assert.ok(selections.receptor.length > 0 && selections.binder.length > 0);
  assert.ok(selections.receptor.length >= selections.binder.length,
    "the larger body is the receptor");
  // Solvent is in no body and in no view but the last.
  const soluteSet = new Set(selections.groups.solute);
  for (const atom of selections.receptor) assert.ok(soluteSet.has(atom));
  for (const atom of selections.binder) assert.ok(soluteSet.has(atom));
  assert.equal(selections.counts.water + selections.counts.solute
    + selections.counts.ion, system.topology.atomCount);
  assert.ok(selections.counts.water > 0, "the fixture is solvated");
});

test("the receptor/binder split can be overridden, and says when it was", () => {
  const { system } = loadFixture("two-body");
  const auto = buildSelections(system);
  // 🔴 THE DERIVED SPLIT CANNOT EXPRESS A TWO-CHAIN RECEPTOR, which is why an
  // override exists at all. A prmtop has no chain IDs, so what is offered is
  // the molecule list - and a caller must be able to tell a chosen split from
  // a guessed one, or a plot legend claims more than it knows.
  assert.equal(auto.splitWasChosen, false);
  assert.equal(auto.soluteMolecules.length, 2);
  for (const molecule of auto.soluteMolecules) {
    assert.ok(molecule.atomCount > 0);
    assert.ok(molecule.label.length > 0, "a molecule must carry a readable label");
    assert.ok(Number.isInteger(molecule.molecule));
  }
  // Largest first, so "auto" is always the head of the list.
  assert.ok(auto.soluteMolecules[0].atomCount >= auto.soluteMolecules[1].atomCount);
  assert.equal(auto.receptor.length, auto.soluteMolecules[0].atomCount);

  // Naming the largest molecule as the binder must swap the two sets exactly.
  const flipped = buildSelections(system,
    { binderMolecules: [auto.soluteMolecules[0].molecule] });
  assert.equal(flipped.splitWasChosen, true);
  assert.equal(flipped.receptor.length, auto.binder.length);
  assert.equal(flipped.binder.length, auto.receptor.length);
  assert.deepEqual([...flipped.binder], [...auto.receptor]);

  // An empty override is not an override - it must fall back, not produce an
  // empty receptor. The `?? []` shape of this kind of option makes that the
  // easy mistake.
  const empty = buildSelections(system, { binderMolecules: [] });
  assert.equal(empty.receptor.length, auto.receptor.length);
  assert.equal(empty.splitWasChosen, false);

  // A residue-level split, for a boundary that is not a molecule boundary.
  const residues = [0, 1];
  const byResidue = buildSelections(system, { binderResidues: residues });
  assert.equal(byResidue.splitWasChosen, true);
  assert.ok(byResidue.binder.length > 0 && byResidue.binder.length < auto.receptor.length,
    "a two-residue binder must be smaller than a whole molecule");
  for (const atom of byResidue.binder) {
    assert.ok(residues.includes(system.topology.residueOfAtom[atom]));
  }
  // Whatever the split, every solute atom lands in exactly one body.
  for (const selections of [auto, flipped, byResidue]) {
    assert.equal(selections.receptor.length + selections.binder.length,
      selections.groups.solute.length, "the split must partition the solute");
    const both = new Set([...selections.receptor, ...selections.binder]);
    assert.equal(both.size, selections.groups.solute.length, "no atom in both bodies");
  }
});

test("every view preset names groups that exist", () => {
  const { system } = loadFixture("two-body");
  const selections = buildSelections(system);
  for (const preset of VIEW_PRESETS) {
    for (const name of preset.groups) {
      assert.ok(Object.hasOwn(selections.groups, name),
        `preset ${preset.id} names a group ${name} that buildSelections does not build`);
    }
  }
  assert.equal(selections.groups.all.length, system.topology.atomCount);
});

// ------------------------------------------------------------------ metrics ----

test("fit-on-receptor RMSD sees a binder displacement a whole-complex RMSD hides", () => {
  // 🔴 THIS IS JIGGLE'S OWN EXPERIMENT AND THE REASON THE SPLIT EXISTS. Move
  // the binder rigidly and nothing else. A whole-complex RMSD averages that
  // displacement over every receptor atom that did not move, so a pose coming
  // apart reads as a small number; fitting on the receptor and measuring the
  // binder reports the displacement itself.
  const { system, coordinates } = loadFixture("two-body");
  const selections = buildSelections(system);
  const reference = Float64Array.from(coordinates.positions);
  const moved = Float64Array.from(reference);
  const shift = 0.5;
  for (const atom of selections.binder) moved[atom * 3] += shift;

  const heavy = new Set(selections.groups.heavy);
  const fit = Int32Array.from([...selections.groups.solute].filter((a) => heavy.has(a)));
  const receptorFit = Int32Array.from([...selections.receptor].filter((a) => heavy.has(a)));
  const binderFit = Int32Array.from([...selections.binder].filter((a) => heavy.has(a)));

  const whole = rmsdFitOn(moved, reference, fit, fit);
  const onReceptor = rmsdFitOn(moved, reference, receptorFit, binderFit);
  // A rigid translation, with the receptor held still, IS the translation.
  assert.ok(Math.abs(onReceptor - shift) < 1e-9,
    `fit-on-receptor RMSD ${onReceptor.toFixed(6)} nm for a ${shift} nm shift`);
  // And the whole-complex number is substantially smaller - which is the point.
  assert.ok(whole < shift * 0.75,
    `whole-complex RMSD ${whole.toFixed(4)} nm did not hide the displacement`);
  // The binder fitted on ITSELF has not changed shape at all.
  const binderAlone = rmsdFitOn(moved, reference, binderFit, binderFit);
  assert.ok(binderAlone < 1e-9,
    `a rigid shift changed the binder's internal RMSD by ${binderAlone}`);
});

test("a rigid translation of everything leaves every metric unmoved", () => {
  const { system, coordinates } = loadFixture("two-body");
  const selections = buildSelections(system);
  const reference = Float64Array.from(coordinates.positions);
  const moved = Float64Array.from(reference);
  for (let atom = 0; atom < system.topology.atomCount; atom += 1) {
    moved[atom * 3] += 1.25;
    moved[atom * 3 + 2] -= 0.75;
  }
  const heavy = new Set(selections.groups.heavy);
  const fit = Int32Array.from([...selections.groups.solute].filter((a) => heavy.has(a)));
  assert.ok(rmsdFitOn(moved, reference, fit, fit) < 1e-9, "superposition must remove it");
  const masses = system.topology.masses;
  assert.ok(Math.abs(radiusOfGyration(moved, masses, fit)
    - radiusOfGyration(reference, masses, fit)) < 1e-12, "Rg is translation invariant");
  const receptorFit = Int32Array.from([...selections.receptor].filter((a) => heavy.has(a)));
  const binderFit = Int32Array.from([...selections.binder].filter((a) => heavy.has(a)));
  assert.ok(Math.abs(interfaceDistance(moved, receptorFit, binderFit)
    - interfaceDistance(reference, receptorFit, binderFit)) < 1e-12);
});

test("RunningRmsf is exact against the two-pass formula, and silent on one frame", () => {
  const reference = Float64Array.from([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]);
  const residues = Int32Array.from([0, 1, 2, 3]);
  const fit = Int32Array.from([0, 1, 2, 3]);
  const running = new RunningRmsf(reference, residues, fit);
  // 🔴 ZEROS UNTIL TWO FRAMES. One frame has no fluctuation to report, and a
  // number there would be meaningless rather than merely imprecise.
  running.addFrame(reference);
  assert.deepEqual([...running.values()], [0, 0, 0, 0], "one frame must report nothing");

  const frames = [reference];
  for (let step = 1; step < 12; step += 1) {
    const frame = Float64Array.from(reference);
    // Move one residue much more than the others, so a mistake is visible.
    frame[3] += 0.03 * Math.sin(step);
    frame[4] += 0.002 * Math.cos(step);
    frame[9] += 0.0005 * step;
    frames.push(frame);
    running.addFrame(frame);
  }
  assert.equal(running.frames, frames.length);
  const live = running.values();

  // The two-pass computation, over the same aligned frames.
  const aligned = frames.map((frame) => {
    const transform = kabschTransform(frame, reference, fit);
    const out = new Float64Array(residues.length * 3);
    const scratch = new Float64Array(3);
    residues.forEach((atom, index) => {
      applyTransform(transform, frame, atom, scratch);
      out.set(scratch, index * 3);
    });
    return out;
  });
  for (let index = 0; index < residues.length; index += 1) {
    let mean = [0, 0, 0];
    for (const frame of aligned) {
      for (let k = 0; k < 3; k += 1) mean[k] += frame[index * 3 + k] / aligned.length;
    }
    let variance = 0;
    for (const frame of aligned) {
      for (let k = 0; k < 3; k += 1) {
        variance += (frame[index * 3 + k] - mean[k]) ** 2 / aligned.length;
      }
    }
    const expected = Math.sqrt(variance);
    assert.ok(Math.abs(live[index] - expected) < 1e-9,
      `residue ${index}: running ${live[index]} against two-pass ${expected}`);
  }
  assert.ok(live[1] > live[0], "the residue that moved must have the larger RMSF");
});

test("LiveAnalysis rows carry every column the panels and the CSV read", () => {
  const { system, coordinates } = loadFixture("two-body");
  const selections = buildSelections(system);
  const analysis = new LiveAnalysis(system, selections, coordinates.positions);
  assert.ok(analysis.hasBinder, "the two-body fixture must take the split path");
  const row = analysis.addFrame(coordinates.positions,
    { step: 0, time: 0, potential: -1, kinetic: 2, total: 1, temperature: 298, volume: 30 });
  for (const key of ["step", "time", "potential", "kinetic", "total", "temperature",
    "volume", "rmsd", "radiusOfGyration", "rmsdReceptor", "rmsdBinder",
    "rmsdBinderOnReceptor", "interfaceDistance"]) {
    assert.ok(row[key] !== undefined, `the row has no ${key}`);
  }
  assert.ok(row.rmsd < 1e-9, "frame one against itself is zero by construction");
  // 🔴 A COLUMN KEY NO ROW CARRIES WRITES AN EMPTY FIELD RATHER THAN THROWING.
  // `metricsCsv` keeps two parallel lists - the row keys and the unit-bearing
  // header names - and its cell guard turns an undefined lookup into "". So a
  // key renamed on one side only produces a CSV that is the right shape, opens
  // in any spreadsheet, and has one column blank for every frame of the run.
  // This row has every quantity present, so no field may be empty.
  const csv = analysis.metricsCsv().trim().split("\n");
  const header = csv[0].split(",");
  const values = csv[1].split(",");
  assert.equal(values.length, header.length,
    "the header and the row have drifted apart, so every later column is shifted");
  const blank = header.filter((name, index) => values[index] === "");
  assert.deepEqual(blank, [],
    `metricsCsv wrote empty columns: ${blank.join(", ")} - a renamed row key`);
  // The units are in the header, because a bare "time" is not a measurement.
  assert.ok(header.includes("time_ps") && header.includes("rmsd_nm")
    && header.includes("rmsd_binder_on_receptor_nm"), header.join(","));
  assert.equal(analysis.rmsfCsv().split("\n").length - 1,
    selections.residueLabels.length + 1, "one RMSF row per residue, plus the header");
});

// ---------------------------------------------------------------------- PDB ----

test("a PDB record puts its fields in the columns a reader expects", () => {
  const { system, coordinates } = loadFixture("two-body");
  const text = writePdb(system, coordinates.positions, {
    box: coordinates.box, remarks: ["RainMD test"],
  });
  const lines = text.split("\n");
  assert.ok(lines[0].startsWith("REMARK "), "remarks come first");
  const cryst = lines.find((line) => line.startsWith("CRYST1"));
  assert.ok(cryst !== undefined, "a periodic system must carry CRYST1");
  const atoms = lines.filter((line) => line.startsWith("ATOM  ")
    || line.startsWith("HETATM"));
  assert.equal(atoms.length, system.topology.atomCount);
  for (const line of atoms.slice(0, 200)) {
    // Fixed columns, 1-indexed in the spec: serial 7-11, name 13-16,
    // residue 18-20, chain 22, residue number 23-26, x/y/z 31-38/39-46/47-54.
    assert.ok(line.length >= 54, `record is ${line.length} columns, needs 54`);
    assert.ok(/^\s*\d+$/.test(line.slice(6, 11)), `bad serial in ${JSON.stringify(line)}`);
    for (const [from, to] of [[30, 38], [38, 46], [46, 54]]) {
      const value = Number(line.slice(from, to));
      assert.ok(Number.isFinite(value), `bad coordinate in ${JSON.stringify(line)}`);
    }
  }
  // 🔴 A ONE-LETTER ELEMENT STARTS IN COLUMN 14 AND A TWO-LETTER ONE IN 13.
  // ` CA ` is an alpha carbon and `CA  ` is a calcium ion; every viewer and
  // every parser in the field reads the distinction, and getting it wrong
  // turns a protein backbone into a row of metal ions.
  const alphas = atoms.filter((line) => line.slice(12, 16) === " CA ");
  assert.ok(alphas.length > 0, "no alpha carbon was written in columns 13-16");
  assert.equal(atoms.filter((line) => line.slice(12, 16) === "CA  ").length, 0,
    "an alpha carbon was written where a calcium ion belongs");
  // One TER per molecule boundary that was closed.
  assert.ok(lines.filter((line) => line.startsWith("TER")).length >= 2,
    "two solute molecules must be separated by TER");
  assert.equal(lines.at(-1) === "" ? lines.at(-2) : lines.at(-1), "END");
});

test("writing a subset writes only that subset, renumbered from one", () => {
  const { system, coordinates } = loadFixture("two-body");
  const selections = buildSelections(system);
  const indices = Int32Array.from(selections.groups.solute);
  const text = writePdb(system, coordinates.positions, { indices });
  const atoms = text.split("\n").filter((line) => line.startsWith("ATOM  ")
    || line.startsWith("HETATM"));
  assert.equal(atoms.length, indices.length,
    "the solute view must write exactly the solute");
  assert.equal(Number(atoms[0].slice(6, 11)), 1, "serials restart at one");
  for (const line of atoms) {
    assert.notEqual(line.slice(17, 20).trim(), "WAT",
      "a solute-only file must contain no water");
  }
});
