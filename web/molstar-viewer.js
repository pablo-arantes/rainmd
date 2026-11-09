/**
 * The structure viewer, on Mol* — RCSB's own renderer.
 *
 * 🔴 IT COSTS 1.47 MB GZIPPED AGAINST 3Dmol's 149 KB, which is the whole of the
 * decision. Everything else in this repository — the parser, the force field,
 * PME, the integrator, the barostat, the analysis and the page — is 159 KB
 * gzipped, so the viewer is now nine times the program it is a viewer for, and
 * a first visit goes from about 310 KB to about 1.63 MB. That was taken
 * deliberately: the drawing is what a reader judges a molecular tool by, and
 * Mol* is the best there is. The numbers are here so nobody has to rediscover
 * them to reverse the decision.
 *
 * 🔴 VENDORED, NOT FETCHED FROM A CDN. MIT, so redistribution is fine with the
 * notice, and the page keeps working with no network and sends no request to
 * anyone when it opens — which is the claim this whole project rests on.
 *
 * 🔴 EVERY PIECE OF MOL*'s OWN CHROME IS TURNED OFF. It ships as an
 * application: a left panel, a sequence strip, a log, a state tree and a
 * vertical icon toolbar in the viewport. Left on, the page has two sets of
 * controls that do not agree with each other. The constructor turns the lot off
 * and the axes helper is disabled separately, because it is a canvas3d prop
 * rather than a layout option and survives every `layoutShow*: false`.
 *
 * 🔴 AND A FRAME UPDATE IS A RELOAD, WHICH IS THE ONE THING 3Dmol DID BETTER.
 * 3Dmol exposes `model.atoms` as a plain mutable array, so an animation frame
 * was three writes per atom. Mol* holds coordinates inside an immutable state
 * tree, and reaching in to mutate them means driving its transaction API; the
 * honest implementation is to clear and load. That is why the live animation
 * stays opt-in and why it is measured rather than assumed - see
 * `docs/STATUS.md`.
 */

/** Mol*'s own theme name for secondary structure, plus the two alternatives. */
const THEMES = {
  structure: "secondary-structure",
  chain: "chain-id",
  element: "element-symbol",
};

export class MolstarViewer {
  constructor(container) {
    if (typeof window.molstar === "undefined") {
      throw new Error(
        "Mol* did not load - web/vendor/molstar.js is missing or was blocked",
      );
    }
    this.container = container;
    this.viewer = null;
    this.ready = null;
    this.atomCount = 0;
    this.visibleCount = 0;
    this.chains = [];
    this.lastOptions = {};
    /**
     * 🔴 ONE RELOAD AT A TIME, AND A FRAME ARRIVING DURING ONE IS DROPPED.
     * A frame update is a clear-and-load, so two overlapping ones have a
     * `plugin.clear()` racing a load - the viewer goes blank, or ends up with
     * the state tree half built. Reports can arrive faster than Mol* rebuilds,
     * so the choice is drop or queue, and dropping is right: a queue would
     * render frames the simulation has already left behind and fall further
     * behind the longer the run went.
     */
    this.busy = false;
    this.dropped = 0;
  }

  /**
   * Mol*'s creation is asynchronous, so everything else awaits it.
   *
   * 🔴 ONE PROMISE, CACHED. `Viewer.create` builds a React application and a
   * WebGL context; calling it twice leaves two viewers fighting over the same
   * container, and the second one's canvas covers the first. Callers may race -
   * a checkpoint draw and a colour change can arrive together - so the promise
   * is the thing that is shared.
   */
  init() {
    if (this.ready !== null) return this.ready;
    this.ready = window.molstar.Viewer.create(this.container, {
      layoutIsExpanded: false,
      layoutShowControls: false,
      layoutShowSequence: false,
      layoutShowLog: false,
      layoutShowLeftPanel: false,
      layoutShowRemoteState: false,
      viewportShowExpand: false,
      viewportShowControls: false,
      viewportShowSettings: false,
      viewportShowSelectionMode: false,
      viewportShowAnimation: false,
      viewportShowTrajectoryControls: false,
      pdbProvider: "rcsb",
      emdbProvider: "pdbe",
    }).then((viewer) => {
      this.viewer = viewer;
      // The axes gizmo and the background are canvas3d props, not layout
      // options, and are unaffected by everything above.
      viewer.plugin.canvas3d?.setProps({
        camera: { helper: { axes: { name: "off", params: {} } } },
        renderer: { backgroundColor: 0xffffff },
      });
      return viewer;
    });
    return this.ready;
  }

  /**
   * Build from a PDB block. Call when the atom SET changes.
   *
   * @param {string} pdb
   * @param {object} [options]
   * @param {"structure"|"chain"|"element"} [options.colourBy]
   * @param {"cartoon"|"both"|"spheres"} [options.representation]
   */
  async setStructure(pdb, options = {}) {
    const viewer = await this.init();
    this.lastOptions = options;
    // 🔴 THE CAMERA SURVIVES A REBUILD. Without this every frame of an
    // animation re-ran `zoomTo`, so a reader who had rotated or zoomed was
    // snapped back to a default view several times a second - which looks
    // exactly like a viewer that is broken rather than one that is updating.
    // The snapshot is taken before the clear and put back after the load.
    const camera = options.keepCamera === true
      ? viewer.plugin.canvas3d?.camera?.getSnapshot() : null;
    const declared = (pdb.match(/^(ATOM {2}|HETATM)/gm) ?? []).length;
    // 🔴 CLEARED FIRST, OR EVERY LOAD ADDS ANOTHER STRUCTURE. Mol*'s
    // `loadStructureFromData` appends to the state tree; without this the
    // viewer accumulates one copy of the system per checkpoint, all drawn on
    // top of each other, and the atom count below would keep growing.
    await viewer.plugin.clear();
    await viewer.loadStructureFromData(pdb, "pdb");
    const structures = viewer.plugin.managers.structure.hierarchy.current.structures;
    this.atomCount = structures.reduce(
      (total, entry) => total + (entry.cell.obj?.data?.elementCount ?? 0), 0);
    this.chains = this.#chainList(structures);
    if (this.atomCount !== declared && declared > 0) {
      // Reported rather than thrown: Mol* drops nothing silently the way
      // 3Dmol's parser does, but a mismatch would still mean the drawing is
      // not the system, so it must not pass unremarked.
      this.parseWarning = `Mol* built ${this.atomCount} atoms from a block `
        + `declaring ${declared}`;
    } else {
      this.parseWarning = null;
    }
    await this.#paint(options);
    if (camera !== null && camera !== undefined) {
      try { viewer.plugin.canvas3d?.camera?.setState(camera, 0); } catch { /* ignore */ }
    }
    this.visibleCount = this.atomCount;
    return this.atomCount;
  }

  #chainList(structures) {
    const chains = new Set();
    for (const entry of structures) {
      const units = entry.cell.obj?.data?.units ?? [];
      for (const unit of units) {
        const label = unit.model?.atomicHierarchy?.chains?.label_asym_id;
        if (label === undefined) continue;
        for (let index = 0; index < label.rowCount; index += 1) {
          chains.add(label.value(index));
        }
      }
    }
    return [...chains].sort();
  }

  async #paint(options) {
    const viewer = this.viewer;
    if (viewer === null) return;
    const plugin = viewer.plugin;
    const structures = plugin.managers.structure.hierarchy.current.structures;
    const components = structures.flatMap((entry) => entry.components);
    if (components.length === 0) return;
    const theme = THEMES[options.colourBy ?? "structure"] ?? THEMES.structure;
    await plugin.managers.structure.component.updateRepresentationsTheme(
      components, { color: theme });
    // Mol*'s default preset is already a cartoon with ligands as ball-and-stick,
    // which is the representation this page wants; "spheres" is the one that
    // needs saying, and "both" adds sticks over the cartoon.
    const representation = options.representation ?? "cartoon";
    if (representation === "spheres") {
      await plugin.managers.structure.component.updateRepresentations(
        components, { type: "spacefill" });
    } else if (representation === "both") {
      // 🔴 A STRING, NOT `{ type }`. `addRepresentation(components, type)` takes
      // the type's NAME, where the sibling `updateRepresentations` takes a
      // params object - and passing the object throws
      // `e.endsWith is not a function` from inside Mol*, which names nothing a
      // caller can act on. The two calls next to each other take different
      // shapes; probed rather than assumed.
      await plugin.managers.structure.component.addRepresentation(
        components, "ball-and-stick");
    }
    if (options.keepCamera !== true) this.#frameCamera();
  }

  /**
   * 🔴 MOL*'s OWN FIT LEAVES A LOT OF ROOM, and on a thin solute in a wide
   * viewport that reads as a viewer that failed to focus. Shrinking the camera
   * radius is the one-line version of a tighter fit; it is applied after the
   * representation exists, because the radius is computed from what is drawn.
   */
  #frameCamera() {
    const camera = this.viewer?.plugin?.canvas3d?.camera;
    if (camera === undefined || camera === null) return;
    try {
      const radius = camera.state?.radius;
      if (typeof radius === "number" && radius > 0) {
        camera.setState({ radius: radius * 0.72 }, 0);
      }
    } catch { /* a camera that will not take a radius is still a camera */ }
  }

  /**
   * New coordinates for the same atoms: a reload, for the reason in the header.
   * @param {string} pdb
   */
  async setFrame(pdb) {
    if (this.ready === null) return;
    if (this.busy) { this.dropped += 1; return; }
    this.busy = true;
    try {
      await this.setStructure(pdb, { ...this.lastOptions, keepCamera: true });
    } finally {
      this.busy = false;
    }
  }

  /** Re-colour without rebuilding. */
  async recolour(options) {
    this.lastOptions = { ...this.lastOptions, ...options };
    if (this.viewer === null) return;
    await this.#paint(this.lastOptions);
  }

  resize() {
    try { this.viewer?.handleResize?.(); } catch { /* not ready */ }
  }
}
