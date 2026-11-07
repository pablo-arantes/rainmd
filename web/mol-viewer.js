/**
 * The structure viewer, on 3Dmol.js — the cartoon LocalFold draws.
 *
 * 🔴 WHY A LIBRARY HERE AND NOT THE WGSL RENDERER BESIDE IT. This port grew its
 * own sphere-impostor and ribbon renderer precisely because the WebGPU viewer
 * can bind the integrator's position buffer and draw a frame with no readback -
 * which matters when a frame is drawn every report. It stopped mattering when
 * the structure moved to three checkpoints: as loaded, after minimisation, and
 * at the end of a stage. Three PDB strings over a whole run is nothing, and in
 * exchange the cartoon is the one every structural biologist already reads.
 * `web/viewer.js` is still here and still gated; the page uses this.
 *
 * 🔴 3Dmol, NOT py2Dmol, AND THEY DRAW THE SAME PICTURE. py2Dmol is a 3Dmol
 * fork; LocalFold vendors its `full` build, which is the whole py2Dmol website
 * plus the embed API - 920 KB minified, 297 KB gzipped. Upstream 3Dmol alone is
 * 512 KB and 149 KB gzipped for the same renderer, it is BSD-3-Clause so it can
 * be redistributed with its notice, and it is vendored rather than loaded from
 * a CDN so the page keeps working with no network and sends no request anywhere
 * when someone opens it. The representation below is LocalFold's: a Richardson
 * cartoon with the sheets' pleat left in.
 *
 * 🔴 `keepH: true` IS MANDATORY AND IS NOT COSMETIC. 3Dmol's PDB parser
 * computes `stripH = !options.keepH` and then skips every atom whose element is
 * H. With no options object it silently discards all of them, so a hydrogenated
 * structure becomes a heavy-atom model - and any later coordinate update writes
 * the first 3N values of a hydrogenated array into N heavy atoms, giving each
 * atom some other atom's position and scrambling the backbone. That is
 * JIGGLE's own bug, recorded in its viewer template, and it is why the atom
 * count is asserted on every update here.
 *
 * 🔴 AND EVERY ATOM IS STYLED BEFORE ANYTHING IS LAYERED ON TOP. JIGGLE's other
 * recorded bug: styling only the chains it knew about left every other atom at
 * 3Dmol's default wireframe, which reads as a broken structure rather than as
 * an unstyled one. A baseline first, then the overrides.
 */

/** 3Dmol's own secondary-structure colours, which readers already know. */
const BODY_PALETTE = ["#4b8fd6", "#b07d2a", "#3f7f4f", "#7a4fa3", "#b3443a", "#2a6b6b"];

export class MolViewer {
  /**
   * @param {HTMLElement} container emptied and given a 3Dmol canvas
   */
  constructor(container) {
    if (typeof window.$3Dmol === "undefined") {
      throw new Error(
        "3Dmol did not load - web/vendor/3Dmol-min.js is missing or was blocked",
      );
    }
    this.container = container;
    // 🔴 WHITE, LIKE THE REFERENCE AND LIKE EVERY PUBLISHED FIGURE. A dark
    // viewer looked right against the old dark page and wrong against a light
    // one, and a cartoon that will be screenshotted into a paper wants the
    // background it will be printed on.
    this.viewer = window.$3Dmol.createViewer(container, {
      backgroundColor: "#ffffff",
      antialias: true,
    });
    this.model = null;
    this.atomCount = 0;
    this.visibleCount = 0;
    this.chains = [];
  }

  /**
   * Build the model. Call when the atom SET changes, not when it moves.
   *
   * @param {string} pdb a PDB block
   * @param {object} [options]
   * @param {"structure"|"chain"|"element"} [options.colourBy]
   * @param {"cartoon"|"both"|"spheres"} [options.representation]
   */
  setStructure(pdb, options = {}) {
    this.viewer.clear();
    // 🔴 keepH - see the header. Without it every hydrogen is dropped at parse
    // time and the atom count silently disagrees with the simulation's.
    this.model = this.viewer.addModel(pdb, "pdb", { keepH: true });
    const atoms = this.model.selectedAtoms({});
    this.atomCount = atoms.length;
    const declared = (pdb.match(/^(ATOM {2}|HETATM)/gm) ?? []).length;
    if (this.atomCount !== declared) {
      throw new Error(
        `3Dmol parsed ${this.atomCount} atoms from a block declaring ${declared}. `
        + "The usual cause is a missing {keepH: true}, which drops every "
        + "hydrogen - do not trust the drawing.",
      );
    }
    this.chains = [...new Set(atoms.map((atom) => atom.chain))].sort();
    this.lastOptions = options;
    this.#paint(options);
    this.viewer.zoomTo();
    // 🔴 zoomTo LEAVES A LOT OF SLACK. Its default framing puts a small solute
    // in the middle of a mostly empty canvas, which on a 460 px viewer reads as
    // a viewer that failed to fit the structure. 1.35 fills it without clipping
    // the outermost residue.
    this.viewer.zoom(1.6);
    this.viewer.render();
    this.visibleCount = this.atomCount;
  }

  #paint(options) {
    const colourBy = options.colourBy ?? "structure";
    const representation = options.representation ?? "cartoon";
    const viewer = this.viewer;

    // A baseline every atom gets, so nothing can be left at the default
    // wireframe - JIGGLE's recorded bug.
    viewer.setStyle({}, { cartoon: { color: "#8f9296" } });

    if (representation === "spheres") {
      viewer.setStyle({}, { sphere: { scale: 0.32, colorscheme: "default" } });
      return;
    }

    // LocalFold's representation: a Richardson cartoon, sheets left pleated.
    // `arrows` is what makes a strand read as directional; `thickness` and
    // `opacity` are 3Dmol's defaults restated so a future change is visible.
    const cartoon = { arrows: true, thickness: 0.4, opacity: 1 };
    if (colourBy === "structure") {
      // 3Dmol assigns secondary structure itself from the coordinates, so this
      // is its own assignment rather than the alpha-carbon criterion in
      // shared/analysis/secondary.js - a better one, and the reason to be here.
      //
      // 🔴 AND IT DISAGREES WITH THAT CRITERION ON AN ISOLATED STRAND, CORRECTLY.
      // On the mixed fixture 3Dmol reports 97 helix and 137 coil and NO sheet,
      // where the alpha-carbon criterion calls the extended stretch a sheet from
      // its geometry alone. A beta sheet is defined by backbone hydrogen bonds
      // between two strands; one strand on its own is extended coil, which is
      // what DSSP would say too. The geometric criterion cannot know that.
      // 🔴 `ssPyMol`, NOT `ssPyMOL`. 3Dmol does not validate a colorscheme
      // name: an unknown one falls through to the default and the whole cartoon
      // comes out flat grey, with no error and nothing in the console. The
      // builtin list is ssPyMol, ssJmol, Jmol, amino, shapely, nucleic, chain,
      // rasmol, default and the per-element carbon variants.
      viewer.setStyle({}, { cartoon: { ...cartoon, colorscheme: "ssPyMol" } });
    } else if (colourBy === "chain") {
      // 🔴 EXPLICIT PER-CHAIN COLOURS RATHER THAN THE `chain` SCHEME, because a
      // prmtop's chains are this port's own invention - one letter per molecule
      // from the bond graph - so a solvated box has hundreds of them and
      // 3Dmol's scheme hashes the letter to a colour. Naming them in order
      // keeps the receptor and the binder the same two colours the analysis
      // panels use.
      this.chains.forEach((chain, index) => {
        viewer.setStyle({ chain }, {
          cartoon: { ...cartoon, color: BODY_PALETTE[index % BODY_PALETTE.length] },
        });
      });
    } else {
      viewer.setStyle({}, { cartoon: { ...cartoon, colorscheme: "default" } });
    }

    // 🔴 A LIGAND HAS NO BACKBONE, SO A CARTOON DRAWS NOTHING FOR IT. Anything
    // that is not a polymer residue gets sticks, and ions get spheres - without
    // this a protein-ligand complex shows the protein and an empty space where
    // the thing being studied is.
    viewer.setStyle({ hetflag: true }, { stick: { radius: 0.18, colorscheme: "default" } });
    viewer.addStyle({ resn: ["WAT", "HOH", "SOL", "TIP3", "T3P"] },
      { line: { opacity: 0.35 } });
    for (const ion of ["NA", "NA+", "CL", "CL-", "K", "K+", "MG", "ZN", "CA2"]) {
      viewer.addStyle({ resn: ion }, { sphere: { scale: 0.28 } });
    }
    if (representation === "both") {
      viewer.addStyle({}, { stick: { radius: 0.08, colorscheme: "default" } });
    }
  }

  /**
   * New coordinates for the SAME atoms, in the same order.
   *
   * 🔴 THE COUNT IS CHECKED EVERY TIME, which is the whole of JIGGLE's hard-won
   * lesson. A mismatch here does not throw on its own: it writes the first 3N
   * values into N atoms and draws a plausible, wrong structure.
   */
  setFrame(flatAngstrom) {
    if (this.model === null) return;
    const atoms = this.model.selectedAtoms({});
    const expected = atoms.length * 3;
    if (flatAngstrom.length !== expected) {
      throw new Error(
        `coordinate count mismatch: ${flatAngstrom.length} values for `
        + `${atoms.length} atoms (expected ${expected}) - the viewer's atom list `
        + "and the simulation's have diverged",
      );
    }
    for (let index = 0; index < atoms.length; index += 1) {
      atoms[index].x = flatAngstrom[index * 3];
      atoms[index].y = flatAngstrom[index * 3 + 1];
      atoms[index].z = flatAngstrom[index * 3 + 2];
    }
    // GLModel rebuilds when molObj is null. Do NOT pass {regen:true} to
    // render(): its second parameter is the WebGL capability object, and
    // handing it an options bag silently disables the accelerated geometry.
    this.model.molObj = null;
    this.viewer.render();
  }

  /** Re-colour without rebuilding, for a change of colour mode. */
  recolour(options) {
    if (this.model === null) return;
    this.lastOptions = { ...this.lastOptions, ...options };
    this.#paint(this.lastOptions);
    this.viewer.render();
  }

  resize() {
    try { this.viewer.resize(); this.viewer.render(); } catch { /* not ready */ }
  }
}
