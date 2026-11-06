/**
 * A WebGPU structure viewer that reads the simulation's own position buffer.
 *
 * 🔴 NO READBACK, WHICH IS THE WHOLE REASON TO WRITE THIS RATHER THAN VENDOR A
 * VIEWER. The coordinates are already in a GPU buffer because the integrator
 * put them there, so a frame is drawn from a device-to-device copy: no readback,
 * no upload, and no host work proportional to the atom count. A JavaScript
 * viewer - 3Dmol, NGL, py3Dmol - would need every position copied to the host
 * and back on every frame, which for a hundred thousand atoms is more traffic
 * than the simulation itself.
 *
 * 🔴 AND THE VIEWER OWNS ITS COPY RATHER THAN BINDING THE SIMULATION'S BUFFER.
 * Binding it directly is one copy cheaper and was the first design; it also
 * couples the viewer's lifetime to the simulation's, and releasing a simulation
 * after a run then leaves the viewer's bind group pointing at a destroyed
 * buffer. Caught by a validation scope around a redraw in the browser lane:
 * `[Buffer "positions"] used in submit while destroyed`. One `copyBufferToBuffer`
 * per redraw - at report intervals, not per step - buys a viewer that outlives
 * the run that produced it.
 *
 * 🔴 SPHERE IMPOSTORS, NOT SPHERE MESHES. Each atom is one quad whose fragment
 * shader solves for the sphere's surface, so an atom costs 6 vertices instead
 * of the few hundred a tessellated sphere needs, and the silhouette is exact at
 * any zoom. The depth is written from the solved surface, so impostors
 * intersect each other correctly.
 */

import { BODY_COLOURS, STRUCTURE_COLOURS } from "../shared/analysis/selection.js";

/** CPK-ish colours and van der Waals radii, by atomic number. */
const ELEMENTS = {
  1: { colour: [0.95, 0.95, 0.95], radius: 0.110 },
  6: { colour: [0.35, 0.37, 0.40], radius: 0.170 },
  7: { colour: [0.22, 0.38, 0.85], radius: 0.155 },
  8: { colour: [0.88, 0.25, 0.20], radius: 0.152 },
  11: { colour: [0.55, 0.35, 0.85], radius: 0.227 },
  12: { colour: [0.20, 0.70, 0.30], radius: 0.173 },
  15: { colour: [0.95, 0.60, 0.15], radius: 0.180 },
  16: { colour: [0.92, 0.85, 0.25], radius: 0.180 },
  17: { colour: [0.25, 0.80, 0.35], radius: 0.175 },
};
const DEFAULT_ELEMENT = { colour: [0.75, 0.45, 0.85], radius: 0.160 };

/**
 * Colours for the group mode.
 *
 * 🔴 RECEPTOR AND BINDER GET DISTINCT COLOURS BECAUSE THE SPLIT IS THE POINT.
 * JIGGLE gives them user-settable pickers for the same reason: when the
 * analysis panel is reporting four separate RMSDs, the viewer has to make it
 * obvious which body each one refers to. Element colouring is the default for a
 * single molecule, where the split carries no information.
 */
// 🔴 DERIVED, NOT RESTATED - see BODY_COLOURS in shared/analysis/selection.js.
// These were a second literal that happened to resemble the plot's, which is
// how "the orange trace is the orange molecule" stops being true in silence.
const GROUP_COLOURS = Object.fromEntries(
  Object.entries(BODY_COLOURS).map(([name, body]) => [name, body.rgb]));
const SECONDARY_COLOURS = Object.fromEntries(
  Object.entries(STRUCTURE_COLOURS).map(([name, kind]) => [name, kind.rgb]));

/**
 * The cartoon ribbon, swept procedurally from one guide frame per residue.
 *
 * 🔴 NO VERTEX BUFFER AND NO INDEX BUFFER, the same as the sphere impostors.
 * Every vertex derives its segment, its position along that segment and which
 * corner of the cross-section it is from `vertex_index` alone, then reads the
 * four Catmull-Rom control guides it needs out of a storage buffer. So changing
 * the structure means rewriting a few hundred guides, never rebuilding tens of
 * thousands of vertices on the host and uploading them.
 *
 * 🔴 NO BACKTICK MAY APPEAR ANYWHERE BELOW, COMMENT OR NOT. The WGSL is a JS
 * template literal, so a backtick ends the string and the file stops parsing at
 * a word inside a shader comment. Fourth time in this repository.
 *
 * 🔴 AND THE CROSS-SECTION IS A BOX, NOT A FLAT BAND. A zero-thickness ribbon
 * vanishes when seen edge-on, which happens constantly as a helix turns - the
 * band appears to break into disconnected pieces. Four faces per step is a
 * solid with a real silhouette, and it costs 24 vertices per step against 6.
 */
/** Spline samples per residue segment. Must match SUBDIV in RIBBON_SHADER. */
export const RIBBON_SUBDIV = 8;

export const RIBBON_SHADER = `
const SUBDIV: u32 = 8u;

struct Camera {
  viewProjection: mat4x4<f32>,
  cameraRight: vec4<f32>,
  cameraUp: vec4<f32>,
  cameraForward: vec4<f32>,
  centre: vec4<f32>,
};
struct Guide {
  posWidth: vec4<f32>,
  sideThick: vec4<f32>,
  colour: vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> guides: array<Guide>;
@group(0) @binding(2) var<storage, read> segments: array<vec4<u32>>;

struct Fragment {
  @builtin(position) clip: vec4<f32>,
  @location(0) colour: vec3<f32>,
  @location(1) normal: vec3<f32>,
};

/** Uniform Catmull-Rom: passes through p1 and p2, tangent from p0 and p3. */
fn catmull(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>, t: f32) -> vec3<f32> {
  let t2 = t * t;
  let t3 = t2 * t;
  return 0.5 * ((2.0 * p1)
    + (-p0 + p2) * t
    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);
}

/** The cross-section corner 'k' as an offset from the centre line. */
fn corner(side: vec3<f32>, up: vec3<f32>, halfWidth: f32, halfThick: f32,
          k: u32) -> vec3<f32> {
  let w = select(-halfWidth, halfWidth, k == 0u || k == 3u);
  let h = select(-halfThick, halfThick, k == 0u || k == 1u);
  return side * w + up * h;
}

struct Sample {
  centre: vec3<f32>,
  side: vec3<f32>,
  up: vec3<f32>,
  halfWidth: f32,
  halfThick: f32,
  colour: vec3<f32>,
};

fn sampleAt(control: vec4<u32>, t: f32) -> Sample {
  let g0 = guides[control.x];
  let g1 = guides[control.y];
  let g2 = guides[control.z];
  let g3 = guides[control.w];
  var out: Sample;
  out.centre = catmull(g0.posWidth.xyz, g1.posWidth.xyz,
    g2.posWidth.xyz, g3.posWidth.xyz, t);
  // The tangent is the curve's own derivative, taken as a short difference so
  // one expression covers both ends of the segment.
  let ahead = catmull(g0.posWidth.xyz, g1.posWidth.xyz,
    g2.posWidth.xyz, g3.posWidth.xyz, min(t + 0.01, 1.0));
  let behind = catmull(g0.posWidth.xyz, g1.posWidth.xyz,
    g2.posWidth.xyz, g3.posWidth.xyz, max(t - 0.01, 0.0));
  var tangent = ahead - behind;
  if (length(tangent) < 1e-7) { tangent = vec3<f32>(0.0, 0.0, 1.0); }
  tangent = normalize(tangent);
  // 🔴 RE-ORTHOGONALISED AGAINST THE SPLINE'S TANGENT, not used as uploaded.
  // The host makes each guide's side vector perpendicular to the straight line
  // through its neighbours; the spline's tangent between two guides is not that
  // line, so a lerped side vector is slightly out of plane. Left uncorrected
  // the cross-section shears and the ribbon's edges ripple.
  var side = normalize(mix(g1.sideThick.xyz, g2.sideThick.xyz, t));
  side = side - tangent * dot(side, tangent);
  if (length(side) < 1e-6) { side = normalize(cross(tangent, vec3<f32>(0.0, 0.0, 1.0))); }
  out.side = normalize(side);
  out.up = normalize(cross(tangent, out.side));
  out.halfWidth = mix(g1.posWidth.w, g2.posWidth.w, t);
  out.halfThick = mix(g1.sideThick.w, g2.sideThick.w, t);
  out.colour = mix(g1.colour.xyz, g2.colour.xyz, t);
  return out;
}

@vertex
fn vertexMain(@builtin(vertex_index) vertex: u32) -> Fragment {
  let quad = vertex / 24u;
  let within = vertex % 24u;
  let face = within / 6u;
  let which = within % 6u;
  let segment = quad / SUBDIV;
  let step = quad % SUBDIV;
  let control = segments[segment];

  let t0 = f32(step) / f32(SUBDIV);
  let t1 = f32(step + 1u) / f32(SUBDIV);
  let a = sampleAt(control, t0);
  let b = sampleAt(control, t1);

  // Two triangles over the quad joining cross-section corner 'face' to
  // 'face + 1', between the two samples.
  let farEnd = which == 2u || which == 3u || which == 5u;
  let nextCorner = which == 1u || which == 4u || which == 5u;
  let k = (face + select(0u, 1u, nextCorner)) % 4u;
  // 🔴 NOT select(): WGSL's select takes scalars and vectors only, never a
  // struct. The error it gives is "no matching call", which lists three
  // candidates and does not say that a struct can never match any of them.
  var at = a;
  if (farEnd) { at = b; }
  let offset = corner(at.side, at.up, at.halfWidth, at.halfThick, k);

  // The face's outward direction is the bisector of its two corners, which for
  // a box gives exactly +/-side or +/-up without a special case per face.
  let k0 = face % 4u;
  let k1 = (face + 1u) % 4u;
  let outward = corner(a.side, a.up, a.halfWidth, a.halfThick, k0)
    + corner(a.side, a.up, a.halfWidth, a.halfThick, k1);

  var out: Fragment;
  let world = at.centre + offset - camera.centre.xyz;
  out.clip = camera.viewProjection * vec4<f32>(world, 1.0);
  out.colour = at.colour;
  out.normal = select(normalize(outward), a.up, length(outward) < 1e-9);
  return out;
}

@fragment
fn fragmentMain(input: Fragment) -> @location(0) vec4<f32> {
  let lightDirection = normalize(vec3<f32>(0.35, 0.55, 0.78));
  // 🔴 LIT ON BOTH SIDES, BECAUSE NOTHING IS CULLED. A ribbon is an open
  // surface at its ends and the pipeline draws back faces, so a one-sided term
  // leaves the inside of every turn black - which reads as a hole in the
  // cartoon rather than as a lighting choice.
  let facing = abs(dot(normalize(input.normal), lightDirection));
  let shade = 0.26 + 0.74 * facing;
  return vec4<f32>(input.colour * shade, 1.0);
}
`;

const SHADER = `
struct Camera {
  viewProjection: mat4x4<f32>,
  cameraRight: vec4<f32>,
  cameraUp: vec4<f32>,
  cameraForward: vec4<f32>,
  centre: vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> positions: array<f32>;
@group(0) @binding(2) var<storage, read> styles: array<vec4<f32>>;

struct Fragment {
  @builtin(position) clip: vec4<f32>,
  @location(0) offset: vec2<f32>,
  @location(1) colour: vec3<f32>,
  @location(2) viewCentre: vec3<f32>,
  @location(3) radius: f32,
};

@vertex
fn vertexMain(@builtin(vertex_index) vertex: u32,
              @builtin(instance_index) atom: u32) -> Fragment {
  // Two triangles over the quad, from the vertex index alone - no index buffer
  // and no vertex buffer, so the only per-atom data is the position already on
  // the device.
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0));
  let corner = corners[vertex];
  let style = styles[atom];
  let radius = style.w;

  var out: Fragment;
  out.offset = corner;
  out.colour = style.xyz;
  out.radius = radius;
  let centre = vec3<f32>(
    positions[atom * 3u], positions[atom * 3u + 1u], positions[atom * 3u + 2u])
    - camera.centre.xyz;
  // 🔴 A ZERO RADIUS COLLAPSES THE QUAD RATHER THAN BRANCHING. Hiding the
  // solvent is a per-atom radius of zero, and a degenerate triangle is
  // discarded by the rasteriser for free - cheaper than a conditional draw and
  // it keeps the instance indices aligned with the atom indices.
  let world = centre + radius * (corner.x * camera.cameraRight.xyz
    + corner.y * camera.cameraUp.xyz);
  out.clip = camera.viewProjection * vec4<f32>(world, 1.0);
  out.viewCentre = centre;
  return out;
}

struct Shaded {
  @location(0) colour: vec4<f32>,
  @builtin(frag_depth) depth: f32,
};

@fragment
fn fragmentMain(input: Fragment) -> Shaded {
  let squared = dot(input.offset, input.offset);
  // Outside the inscribed circle there is no sphere, so the fragment is not
  // part of the atom at all.
  if (squared > 1.0) { discard; }
  let height = sqrt(1.0 - squared);
  let normal = vec3<f32>(input.offset, height);
  // 🔴 THE DEPTH COMES FROM THE SOLVED SURFACE, NOT FROM THE QUAD. Writing the
  // quad's own depth makes every atom a flat disc at one distance, so two
  // overlapping atoms show a straight edge where they cross instead of a curve
  // - the giveaway that a viewer is drawing billboards rather than spheres.
  let surface = input.viewCentre + input.radius * (normal.x * camera.cameraRight.xyz
    + normal.y * camera.cameraUp.xyz + normal.z * camera.cameraForward.xyz);
  let clip = camera.viewProjection * vec4<f32>(surface, 1.0);
  let lightDirection = normalize(vec3<f32>(0.35, 0.55, 0.78));
  let diffuse = max(dot(normal, lightDirection), 0.0);
  let rim = pow(1.0 - height, 2.5);
  let shade = 0.22 + 0.78 * diffuse;
  var out: Shaded;
  out.colour = vec4<f32>(input.colour * shade + vec3<f32>(0.10) * rim, 1.0);
  out.depth = clip.z / clip.w;
  return out;
}
`;

function perspective(fieldOfView, aspect, near, far) {
  const f = 1 / Math.tan(fieldOfView / 2);
  const range = 1 / (near - far);
  // Column-major, WebGPU clip space (z in [0, 1]).
  return [
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, far * range, -1,
    0, 0, near * far * range, 0,
  ];
}

function multiply(a, b) {
  const out = new Float32Array(16);
  for (let column = 0; column < 4; column += 1) {
    for (let row = 0; row < 4; row += 1) {
      let sum = 0;
      for (let k = 0; k < 4; k += 1) sum += a[k * 4 + row] * b[column * 4 + k];
      out[column * 4 + row] = sum;
    }
  }
  return out;
}

export class StructureViewer {
  constructor(device, canvas) {
    this.device = device;
    this.canvas = canvas;
    this.context = canvas.getContext("webgpu");
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({ device, format: this.format, alphaMode: "opaque" });
    this.azimuth = 0.6;
    this.elevation = 0.35;
    this.distance = 6;
    this.centre = [0, 0, 0];
    this.atomCount = 0;
    this.depth = null;
    this.uniform = device.createBuffer({
      size: 36 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.pipeline = device.createRenderPipeline({
      label: "rainmd.viewer",
      layout: "auto",
      vertex: { module: device.createShaderModule({ code: SHADER }), entryPoint: "vertexMain" },
      fragment: {
        module: device.createShaderModule({ code: SHADER }),
        entryPoint: "fragmentMain",
        targets: [{ format: this.format }],
      },
      primitive: { topology: "triangle-list" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });
    // 🔴 A SECOND PIPELINE, NOT A SECOND VIEWER. The ribbon shares the camera
    // uniform and the depth buffer with the spheres, so the two can be drawn in
    // one render pass and depth-test against each other - which is what makes
    // "both" a usable representation rather than two overlapping pictures.
    this.ribbonPipeline = device.createRenderPipeline({
      label: "rainmd.viewer.ribbon",
      layout: "auto",
      vertex: {
        module: device.createShaderModule({ code: RIBBON_SHADER }),
        entryPoint: "vertexMain",
      },
      fragment: {
        module: device.createShaderModule({ code: RIBBON_SHADER }),
        entryPoint: "fragmentMain",
        targets: [{ format: this.format }],
      },
      primitive: { topology: "triangle-list" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
    });
    this.ribbonSegments = 0;
    this.representation = "spheres";
    this.#attachControls();
  }

  /**
   * Upload a ribbon built by `shared/analysis/ribbon.js`.
   *
   * 🔴 THE BUFFERS ARE REALLOCATED ONLY WHEN THEY HAVE TO GROW. A run rebuilds
   * the ribbon at every report, and destroying two buffers a frame is how a
   * viewer ends up allocating more than the simulation it is watching.
   */
  setRibbon(ribbon) {
    this.ribbonSegments = ribbon.segmentCount;
    if (ribbon.segmentCount === 0) return;
    const guideBytes = Math.max(ribbon.guides.byteLength, 48);
    if (this.guideBytes === undefined || this.guideBytes < guideBytes) {
      this.guideBuffer?.destroy();
      this.guideBuffer = this.device.createBuffer({
        label: "viewer.ribbon.guides",
        size: guideBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.guideBytes = guideBytes;
      this.ribbonBindGroup = undefined;
    }
    const segmentBytes = Math.max(ribbon.segments.byteLength, 16);
    if (this.segmentBytes === undefined || this.segmentBytes < segmentBytes) {
      this.segmentBuffer?.destroy();
      this.segmentBuffer = this.device.createBuffer({
        label: "viewer.ribbon.segments",
        size: segmentBytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.segmentBytes = segmentBytes;
      this.ribbonBindGroup = undefined;
    }
    this.device.queue.writeBuffer(this.guideBuffer, 0, ribbon.guides.buffer,
      ribbon.guides.byteOffset, ribbon.guides.byteLength);
    this.device.queue.writeBuffer(this.segmentBuffer, 0, ribbon.segments.buffer,
      ribbon.segments.byteOffset, ribbon.segments.byteLength);
    if (this.ribbonBindGroup === undefined) {
      this.ribbonBindGroup = this.device.createBindGroup({
        layout: this.ribbonPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.uniform } },
          { binding: 1, resource: { buffer: this.guideBuffer } },
          { binding: 2, resource: { buffer: this.segmentBuffer } },
        ],
      });
    }
  }

  /**
   * Per-atom colour and radius, and a positions buffer of this viewer's own.
   *
   * @param {object} topology from parsePrmtop
   * @param {object} options.selections from buildSelections
   * @param {string[]} options.groups which selection groups are visible
   * @param {boolean} [options.heavyOnly] hide hydrogens
   * @param {"element"|"group"} [options.colourBy]
   */
  setStructure(topology, options = {}) {
    this.atomCount = topology.atomCount;
    this.topology = topology;
    this.lastOptions = options;
    const bytes = this.atomCount * 12;
    if (this.positionBytes !== bytes) {
      this.positions?.destroy();
      this.positions = this.device.createBuffer({
        label: "viewer.positions",
        size: bytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.positionBytes = bytes;
    }
    const scale = options.radiusScale ?? 0.65;
    const selections = options.selections ?? null;
    const colourBy = options.colourBy ?? "element";
    const heavyOnly = options.heavyOnly === true;

    // 🔴 VISIBILITY IS A SET OF SELECTION GROUPS, NOT A SINGLE FLAG. "hide
    // solvent" was one boolean and could not express "protein and ligand but
    // not the second chain" or "backbone only" - which are the questions
    // someone watching a complex actually asks. The presets in
    // shared/analysis/selection.js name the combinations; this turns one into a
    // per-atom radius.
    const visible = new Uint8Array(this.atomCount);
    if (selections === null) {
      visible.fill(1);
    } else {
      for (const name of options.groups ?? ["solute"]) {
        const group = selections.groups[name];
        if (group === undefined) continue;
        for (const atom of group) visible[atom] = 1;
      }
    }
    const receptor = selections === null ? new Set() : new Set(selections.receptor);
    const styles = new Float32Array(this.atomCount * 4);
    for (let atom = 0; atom < this.atomCount; atom += 1) {
      const element = ELEMENTS[topology.atomicNumbers[atom]] ?? DEFAULT_ELEMENT;
      let colour = element.colour;
      if (colourBy === "structure" && options.secondary !== undefined) {
        // 🔴 AN ATOM WITH NO ASSIGNMENT TAKES COIL, NOT ITS ELEMENT COLOUR. A
        // ligand drawn in CPK beside a protein drawn by structure reads as a
        // third secondary structure; drawn in the neutral coil grey it reads as
        // what it is, which is "this is not a polymer".
        colour = SECONDARY_COLOURS[options.secondary[atom]] ?? SECONDARY_COLOURS.coil;
      } else if (colourBy === "group" && selections !== null) {
        const kind = selections.kindOfAtom[atom];
        if (kind === "water") colour = GROUP_COLOURS.water;
        else if (kind === "ion") colour = GROUP_COLOURS.ion;
        else if (receptor.has(atom)) colour = GROUP_COLOURS.receptor;
        else colour = GROUP_COLOURS.binder;
      }
      const hidden = visible[atom] === 0
        || (heavyOnly && topology.atomicNumbers[atom] === 1);
      styles[atom * 4] = colour[0];
      styles[atom * 4 + 1] = colour[1];
      styles[atom * 4 + 2] = colour[2];
      styles[atom * 4 + 3] = hidden ? 0 : element.radius * scale;
    }
    this.visibleCount = styles.reduce(
      (total, value, index) => (index % 4 === 3 && value > 0 ? total + 1 : total), 0);
    // 🔴 EXPOSED, BECAUSE THE RIBBON HAS TO AGREE WITH THE SPHERES ABOUT WHAT
    // IS ON SCREEN. A cartoon drawn through residues whose atoms are hidden
    // would run across the gap the preset just opened - "protein only" would
    // put a ribbon straight through a ligand. This is the same array the styles
    // buffer was built from, so the two cannot disagree.
    //
    // It carries the HEAVY-ONLY decision too: an alpha carbon is never a
    // hydrogen, so a heavy-atom preset leaves the trace intact.
    this.visibleAtoms = new Uint8Array(this.atomCount);
    for (let atom = 0; atom < this.atomCount; atom += 1) {
      this.visibleAtoms[atom] = styles[atom * 4 + 3] > 0 ? 1 : 0;
    }
    this.styles?.destroy();
    this.styles = this.device.createBuffer({
      size: styles.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(this.styles, 0, styles);
    this.bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uniform } },
        { binding: 1, resource: { buffer: this.positions } },
        { binding: 2, resource: { buffer: this.styles } },
      ],
    });
    // 🔴 FRAME ON WHAT IS VISIBLE, NOT ON THE BOX. Framing from the box put a
    // 0.76 nm solute, with the solvent hidden, 5.6 nm from the camera - about
    // four pixels per atom in a 340-pixel-tall canvas. The render was correct
    // and the screenshot was 98.9% background, which is indistinguishable from
    // a viewer that draws nothing. Fitting the VISIBLE atoms' bounding box is
    // what makes "hide solvent" useful rather than a way to lose the molecule.
    if (options.positions !== undefined && options.positions !== null) {
      this.frame(options.positions, styles);
    } else if (options.box !== undefined && options.box !== null) {
      this.centre = options.box.lengths.map((length) => length / 2);
      this.distance = Math.max(...options.box.lengths) * 1.9;
    }
    this.draw();
  }

  /**
   * Point the camera at the visible atoms and pull back far enough to see them.
   *
   * 🔴 THE VERTICAL FIELD OF VIEW IS WHAT BOUNDS THE FIT. The projection uses
   * `f` for y and `f/aspect` for x, so a wide canvas sees more horizontally
   * than vertically - fitting against the larger of the two leaves the
   * structure cropped top and bottom on any landscape canvas.
   */
  frame(positions, styles) {
    const low = [Infinity, Infinity, Infinity];
    const high = [-Infinity, -Infinity, -Infinity];
    let visible = 0;
    for (let atom = 0; atom < this.atomCount; atom += 1) {
      if (styles[atom * 4 + 3] <= 0) continue;
      visible += 1;
      for (let axis = 0; axis < 3; axis += 1) {
        const value = positions[atom * 3 + axis];
        if (value < low[axis]) low[axis] = value;
        if (value > high[axis]) high[axis] = value;
      }
    }
    if (visible === 0) return;
    this.centre = [0, 1, 2].map((axis) => (low[axis] + high[axis]) / 2);
    const extent = Math.max(
      high[0] - low[0], high[1] - low[1], high[2] - low[2], 0.5);
    // Half the extent must fit inside half the vertical field of view, with a
    // margin so the outermost atoms are not clipped by their own radius.
    this.distance = (0.5 * extent * 1.45) / Math.tan(0.45);
  }

  /**
   * Copy the current coordinates in from a simulation, on the device.
   *
   * @param {object} source a GpuBuffer holding 3N f32 positions
   */
  updateFrom(source) {
    if (this.positions === undefined) return;
    const encoder = this.device.createCommandEncoder({ label: "rainmd.viewer.copy" });
    encoder.copyBufferToBuffer(source.buffer, 0, this.positions, 0, this.positionBytes);
    this.device.queue.submit([encoder.finish()]);
  }

  /**
   * Fill the viewer's own buffer from host coordinates.
   *
   * 🔴 FOR THE CASE WHERE THERE IS NO SIMULATION YET. `updateFrom` is the hot
   * path and copies device-to-device, which needs a device buffer to copy from;
   * a file that has just been dropped has only a host array. Without this the
   * viewer drew a black rectangle until the reader pressed Minimise - the
   * structure was framed, styled and bound, and the positions buffer was all
   * zeros, so every atom was drawn at the origin.
   *
   * @param {ArrayLike<number>} positions 3N, nm
   */
  upload(positions) {
    if (this.positions === undefined) return;
    const data = positions instanceof Float32Array
      ? positions : Float32Array.from(positions);
    this.device.queue.writeBuffer(this.positions, 0, data.buffer,
      data.byteOffset, Math.min(data.byteLength, this.positionBytes));
  }

  #attachControls() {
    let dragging = false;
    let lastX = 0;
    let lastY = 0;
    this.canvas.addEventListener("pointerdown", (event) => {
      dragging = true;
      lastX = event.clientX;
      lastY = event.clientY;
      this.canvas.setPointerCapture(event.pointerId);
    });
    this.canvas.addEventListener("pointerup", (event) => {
      dragging = false;
      this.canvas.releasePointerCapture(event.pointerId);
    });
    this.canvas.addEventListener("pointermove", (event) => {
      if (!dragging) return;
      this.azimuth -= (event.clientX - lastX) * 0.008;
      this.elevation = Math.max(-1.5, Math.min(1.5,
        this.elevation + (event.clientY - lastY) * 0.008));
      lastX = event.clientX;
      lastY = event.clientY;
      this.draw();
    });
    this.canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      this.distance = Math.max(0.5, this.distance * Math.exp(event.deltaY * 0.001));
      this.draw();
    }, { passive: false });
  }

  draw() {
    if (this.atomCount === 0 || this.bindGroup === undefined) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(this.canvas.clientWidth * ratio));
    const height = Math.max(1, Math.round(this.canvas.clientHeight * ratio));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
      // 🔴 THE DEPTH TEXTURE MUST BE RECREATED WITH THE CANVAS. A depth
      // attachment whose size does not match the colour attachment fails
      // validation, and the canvas resizes whenever the window does or the
      // page moves between displays of different pixel ratios.
      this.depth?.destroy();
      this.depth = this.device.createTexture({
        size: [width, height],
        format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      });
    }

    const cosElevation = Math.cos(this.elevation);
    const eye = [
      this.distance * cosElevation * Math.sin(this.azimuth),
      this.distance * Math.sin(this.elevation),
      this.distance * cosElevation * Math.cos(this.azimuth),
    ];
    const forward = eye.map((value) => value / this.distance);
    const worldUp = [0, 1, 0];
    let right = [
      worldUp[1] * forward[2] - worldUp[2] * forward[1],
      worldUp[2] * forward[0] - worldUp[0] * forward[2],
      worldUp[0] * forward[1] - worldUp[1] * forward[0],
    ];
    const rightLength = Math.hypot(...right) || 1;
    right = right.map((value) => value / rightLength);
    const up = [
      forward[1] * right[2] - forward[2] * right[1],
      forward[2] * right[0] - forward[0] * right[2],
      forward[0] * right[1] - forward[1] * right[0],
    ];
    const view = new Float32Array([
      right[0], up[0], forward[0], 0,
      right[1], up[1], forward[1], 0,
      right[2], up[2], forward[2], 0,
      -(right[0] * eye[0] + right[1] * eye[1] + right[2] * eye[2]),
      -(up[0] * eye[0] + up[1] * eye[1] + up[2] * eye[2]),
      -(forward[0] * eye[0] + forward[1] * eye[1] + forward[2] * eye[2]),
      1,
    ]);
    const projection = perspective(0.9, width / height, 0.05, this.distance * 6 + 20);
    const viewProjection = multiply(projection, view);

    const uniform = new Float32Array(36);
    uniform.set(viewProjection, 0);
    uniform.set([right[0], right[1], right[2], 0], 16);
    uniform.set([up[0], up[1], up[2], 0], 20);
    uniform.set([forward[0], forward[1], forward[2], 0], 24);
    uniform.set([this.centre[0], this.centre[1], this.centre[2], 0], 28);
    this.device.queue.writeBuffer(this.uniform, 0, uniform);

    const encoder = this.device.createCommandEncoder({ label: "rainmd.viewer" });
    const pass = encoder.beginRenderPass({
      colorAttachments: [{
        view: this.context.getCurrentTexture().createView(),
        clearValue: { r: 0.071, g: 0.078, b: 0.086, a: 1 },
        loadOp: "clear", storeOp: "store",
      }],
      depthStencilAttachment: {
        view: this.depth.createView(),
        depthClearValue: 1, depthLoadOp: "clear", depthStoreOp: "store",
      },
    });
    const wantsSpheres = this.representation !== "ribbon";
    const wantsRibbon = this.representation !== "spheres"
      && this.ribbonSegments > 0 && this.ribbonBindGroup !== undefined;
    if (wantsSpheres) {
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, this.bindGroup);
      pass.draw(6, this.atomCount);
    }
    if (wantsRibbon) {
      pass.setPipeline(this.ribbonPipeline);
      pass.setBindGroup(0, this.ribbonBindGroup);
      // 24 vertices a step - four faces of the cross-section, two triangles
      // each - and SUBDIV steps per segment, matching the shader's constant.
      pass.draw(this.ribbonSegments * RIBBON_SUBDIV * 24);
    }
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }
}
