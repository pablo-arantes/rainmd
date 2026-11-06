/**
 * Cell-list geometry: how many cells, which neighbours to visit, how big the
 * Verlet list has to be.
 *
 * All host-side, computed once per box. The shader and this file must agree
 * about every number here or the neighbour list silently misses pairs - which
 * is the worst failure mode in the engine, because a missing pair lowers the
 * energy smoothly and reads as a slightly different force field rather than as
 * a bug.
 */

/**
 * 🔴 THE STENCIL MUST VISIT EACH CELL AT MOST ONCE, AND A SMALL BOX BREAKS THE
 * OBVIOUS 3x3x3. With two cells along an axis, offsets -1, 0, +1 reduce mod 2
 * to {1, 0, 1} - cell 1 is visited twice, every atom in it is added to the
 * neighbour list twice, and every interaction with it is counted twice. The
 * solvated test fixture is exactly this case: a 2.72 nm box with a 1.1 nm cell
 * target gives two cells per axis. So the offsets are deduplicated per axis:
 * three cells or more gets {-1,0,1}, two gets {0,1}, one gets {0}.
 *
 * This is safe because the cell size is at least the cutoff plus the skin by
 * construction (cells = floor(length / target)), so a one-deep stencil always
 * covers the interaction sphere.
 */
function axisOffsets(cellCount) {
  if (cellCount >= 3) return [-1, 0, 1];
  if (cellCount === 2) return [0, 1];
  return [0];
}

/** Boltzmann's constant, kJ/mol/K - the same value units.js uses. */
const BOLTZMANN = 0.008314462618;

/** The lightest atom that moves, in g/mol. Hydrogen unless the system is odd. */
function lightestMovingMass(system) {
  let lightest = Infinity;
  for (const mass of system.topology.masses) {
    if (mass > 0 && mass < lightest) lightest = mass;
  }
  return Number.isFinite(lightest) ? lightest : 1.008;
}

/**
 * How wide the Verlet skin has to be, from the timestep and the temperature.
 *
 * 🔴 THE SKIN IS A DISTANCE BUDGET AND IT WAS A CONSTANT. `0.2` nm was the
 * default for every system, timestep and temperature, and MEASURED on the
 * solvated fixture at 2 fs and 298 K with a 20-step rebuild interval it needed
 * **0.342 nm** - so the default was violated by 1.7x and the forces were
 * quietly missing pairs inside the cutoff. Nothing could see it: the energy
 * drift across five skin/interval settings was -1.3 to +3.7 kJ/mol/ns/atom with
 * no ordering, which is what `docs/STATUS.md` recorded as "fluctuation rather
 * than staleness". It was both - the drift test simply cannot resolve a handful
 * of missing pairs, and only the geometric criterion can.
 *
 * 🔴 AND IT IS ANCHORED ON THAT MEASUREMENT RATHER THAN DERIVED, BECAUSE THE
 * DERIVATION DOES NOT CLOSE. The free-flight bound `v * n * dt` is right only
 * while motion is ballistic; measured against it the worst displacement is
 * 3.17x at 2 steps, 4.16x at 5, 2.71x at 10 and 1.57x at 20, because the fast
 * atom's direction decorrelates and the motion crosses over to diffusive. A
 * bound using the worst of those ratios asks for 0.87 nm of skin at 20 steps,
 * which is more than half the box on the test system; fitting the crossover to
 * twenty blocks of one run would be fitting noise. So the anchor is measured
 * and only the two dependences that ARE solid physics are applied:
 *
 *   - linear in the timestep, since a displacement is a speed times a time;
 *   - as sqrt(T), since that is how a thermal speed scales;
 *   - as sqrt(1/m) on the lightest atom, for the same reason - a system whose
 *     lightest particle is a united-atom carbon moves far less than one with
 *     bare hydrogens.
 *
 * It is NOT scaled by the rebuild interval, deliberately: a shorter interval
 * only adds margin, and a longer one removes margin that the device-side
 * displacement check then catches and corrects. The guard is what makes a
 * default safe to ship; this is what keeps it from firing on every run.
 */
export function skinFor(system, options = {}) {
  const timestep = options.timestep ?? 0.002;
  const temperature = options.temperature ?? 298;
  const mass = lightestMovingMass(system);
  // The anchor: 0.40 nm covers the measured 0.342 nm requirement with 15% to
  // spare, at 2 fs, 298 K, hydrogen, and a 20-step interval.
  const anchor = 0.40;
  const speedRatio = Math.sqrt((temperature / 298) * (1.008 / mass));
  const scaled = anchor * (timestep / 0.002) * speedRatio;
  // 🔴 AND IT IS CAPPED BY THE BOX, NOT JUST FLOORED. `cutoff + skin` must stay
  // under half the smallest box width or the minimum-image convention breaks -
  // the same rule buildSystem applies to the cutoff alone. A hot, long-timestep
  // run in a tight box would otherwise ask for a skin that cannot exist, and
  // the refusal would come from somewhere far away.
  const widest = halfSmallestWidth(options.boxVectors ?? system.boxVectors);
  const room = widest === null ? Infinity : widest - system.options.cutoff;
  return Math.max(0.1, Math.min(scaled, room > 0 ? room * 0.9 : 0.1));
}

/** Half the smallest perpendicular width of a reduced box, or null. */
function halfSmallestWidth(box) {
  if (box === null || box === undefined) return null;
  const widths = [box[0], box[4], box[8]].map(Math.abs);
  return Math.min(...widths) / 2;
}

/**
 * @param {object} system from buildSystem
 * @param {object} [options]
 * @param {number} [options.skin] Verlet skin in nm. The list holds pairs out to
 *   cutoff + skin so it survives a few steps of motion before a rebuild.
 * @param {number} [options.neighbourSlack] multiplier on the estimated
 *   neighbours per atom.
 */
export function planCellList(system, options = {}) {
  const skin = options.skin ?? 0.1;
  const cutoff = system.options.cutoff;
  const listCutoff = cutoff + skin;
  // 🔴 THE BOX CAN BE OVERRIDDEN, BECAUSE A BAROSTAT CHANGES IT. The cell
  // count is derived from the box width, so an accepted volume move can change
  // it - and a plan built from the system's original box would then describe
  // cells that no longer tile the cell.
  const box = options.boxVectors ?? system.boxVectors;
  if (box === null) {
    throw new Error(
      "a cell list needs a periodic box. A non-periodic system uses the "
      + "all-pairs path instead; see directSpaceEnergy with cutoff Infinity.",
    );
  }

  // Reduced upper-triangular box, so the axis widths are the diagonal. For a
  // sheared cell the perpendicular width is smaller than the vector length,
  // and using the length would make the cells too thin to cover the cutoff.
  const widths = [box[0], box[4], box[8]];
  const counts = widths.map((width) => Math.max(1, Math.floor(Math.abs(width) / listCutoff)));
  const cellCount = counts[0] * counts[1] * counts[2];

  const offsets = [axisOffsets(counts[0]), axisOffsets(counts[1]), axisOffsets(counts[2])];
  const stencil = [];
  for (const dx of offsets[0]) {
    for (const dy of offsets[1]) {
      for (const dz of offsets[2]) stencil.push([dx, dy, dz]);
    }
  }

  // 🔴 THE CAPACITY IS AN ESTIMATE, THE KERNEL DETECTS OVERFLOW, AND THE
  // CALLER GROWS AND REBUILDS. All three parts are needed, and the first run of
  // the differential test proved it: a mean-density estimate of 323 neighbours
  // with 1.6x slack gave a capacity of 518, and the densest atom in this
  // 1190-atom fixture actually has **526**. So the maximum is 1.63x the mean -
  // more than the ~1.17x that Poisson counting fluctuation would give, because
  // atoms are structurally correlated: bonded neighbours, hydration shells, and
  // a protein interior denser than the bulk water around it.
  //
  // The slack is 2.0 now, which covers this fixture, and that is still only a
  // guess about the next system. What makes it safe is that the kernel reports
  // the true requirement through an atomicMax and `grownPlan` below re-plans
  // from the measurement - so a wrong estimate costs one wasted build rather
  // than a wrong force field.
  const volume = Math.abs(box[0] * box[4] * box[8]);
  const density = system.topology.atomCount / volume;
  const sphere = (4 / 3) * Math.PI * listCutoff ** 3;
  const estimated = density * sphere;
  const slack = options.neighbourSlack ?? 2.0;
  const maxNeighbours = options.maxNeighbours ?? Math.max(32, Math.ceil(estimated * slack));

  // The sort kernel is one thread per cell doing an insertion sort of that
  // cell's slice, so its serial depth is the occupancy squared - independent of
  // the system size, which is why it stays cheap as the box grows.
  const meanOccupancy = system.topology.atomCount / cellCount;

  return {
    skin,
    cutoff,
    listCutoff,
    options,
    counts,
    cellCount,
    stencil,
    maxNeighbours,
    estimatedNeighbours: estimated,
    meanOccupancy,
    widths,
  };
}

/**
 * The excluded partners of every atom, both directions, sorted.
 *
 * 🔴 EXCLUSION IS APPLIED WHEN THE LIST IS BUILT, NOT WHEN IT IS WALKED. The
 * alternative - testing each neighbour against the exclusion set every step -
 * is a scan of ~15 entries per neighbour per atom per step, which is more work
 * than the interaction it is deciding about. Excluding at build time pays it
 * once per rebuild instead, which is every twenty steps or so.
 *
 * 🔴 AND IT MUST BE SYMMETRIC, WHICH AMBER'S LIST IS NOT. The topology stores,
 * for each atom i, only the excluded partners with j > i. A gather kernel asks
 * "is j excluded from i" for every ordered pair, so atom j needs to know about
 * atom i as well. Using AMBER's one-sided list directly excludes each pair from
 * exactly one of its two atoms, so the interaction is dropped from one force
 * and kept in the other - a non-conservative force field, which drifts.
 *
 * Sorted because the kernel binary-searches it.
 */
export function buildExclusionGather(system) {
  const atomCount = system.topology.atomCount;
  const exclusions = system.topology.exclusions;
  const counts = new Int32Array(atomCount);
  for (let atom = 0; atom < atomCount; atom += 1) {
    const begin = exclusions.start[atom];
    const end = exclusions.start[atom + 1];
    counts[atom] += end - begin;
    for (let cursor = begin; cursor < end; cursor += 1) {
      counts[exclusions.partners[cursor]] += 1;
    }
  }
  const start = new Uint32Array(atomCount + 1);
  for (let atom = 0; atom < atomCount; atom += 1) start[atom + 1] = start[atom] + counts[atom];
  const partners = new Uint32Array(start[atomCount]);
  const cursor = start.slice(0, atomCount);
  for (let atom = 0; atom < atomCount; atom += 1) {
    const begin = exclusions.start[atom];
    const end = exclusions.start[atom + 1];
    for (let index = begin; index < end; index += 1) {
      const partner = exclusions.partners[index];
      partners[cursor[atom]] = partner;
      cursor[atom] += 1;
      partners[cursor[partner]] = atom;
      cursor[partner] += 1;
    }
  }
  for (let atom = 0; atom < atomCount; atom += 1) {
    const slice = partners.subarray(start[atom], start[atom + 1]);
    slice.sort();
  }
  let widest = 0;
  for (let atom = 0; atom < atomCount; atom += 1) {
    widest = Math.max(widest, start[atom + 1] - start[atom]);
  }
  return { start, partners, widest };
}

/**
 * The exception pairs, per atom, with their parameters inlined.
 *
 * The parameters are duplicated for both atoms of each pair rather than shared
 * through an index, because an exception costs three floats and a system has
 * about one per atom - so the duplication is kilobytes, and it saves the kernel
 * a binding and an indirection.
 */
export function buildExceptionGather(system) {
  const atomCount = system.topology.atomCount;
  const exceptions = system.exceptions;
  const counts = new Int32Array(atomCount);
  for (let index = 0; index < exceptions.count; index += 1) {
    counts[exceptions.atomA[index]] += 1;
    counts[exceptions.atomB[index]] += 1;
  }
  const start = new Uint32Array(atomCount + 1);
  for (let atom = 0; atom < atomCount; atom += 1) start[atom + 1] = start[atom] + counts[atom];
  const total = start[atomCount];
  const partners = new Uint32Array(total);
  // chargeProduct, ljA, ljB, and the partner's own charge product term for the
  // Ewald correction - which uses the atoms' FULL charges, not the scaled ones.
  const parameters = new Float32Array(total * 4);
  const cursor = start.slice(0, atomCount);
  const place = (atom, partner, index) => {
    const slot = cursor[atom];
    partners[slot] = partner;
    parameters[slot * 4] = exceptions.chargeProduct[index];
    parameters[slot * 4 + 1] = exceptions.ljA[index];
    parameters[slot * 4 + 2] = exceptions.ljB[index];
    parameters[slot * 4 + 3] = system.topology.charges[atom] * system.topology.charges[partner];
    cursor[atom] += 1;
  };
  for (let index = 0; index < exceptions.count; index += 1) {
    place(exceptions.atomA[index], exceptions.atomB[index], index);
    place(exceptions.atomB[index], exceptions.atomA[index], index);
  }
  let widest = 0;
  for (let atom = 0; atom < atomCount; atom += 1) {
    widest = Math.max(widest, start[atom + 1] - start[atom]);
  }
  return { start, partners, parameters, count: total, widest };
}

/**
 * The same plan with a capacity that fits a measured requirement.
 *
 * 🔴 THE MARGIN MATTERS BECAUSE THE REQUIREMENT GROWS BETWEEN REBUILDS. The
 * number the kernel reported is the maximum at the positions it just saw; atoms
 * keep moving, and the next build on a slightly compressed region needs a few
 * more. Growing to exactly the observed figure would overflow again almost
 * immediately and rebuild in a loop, so it grows past it.
 */
export function grownPlan(plan, required) {
  return {
    ...plan,
    maxNeighbours: Math.max(plan.maxNeighbours + 32, Math.ceil(required * 1.25)),
    grownFrom: plan.maxNeighbours,
  };
}
