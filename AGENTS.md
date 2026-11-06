# RainMD engineering invariants

- **Never loosen a tolerance to make a check pass.** A tolerance must have a
  stated origin — a measured grid floor, an f32 bound, a machine epsilon. Two of
  the three in this repository are derived from first principles and say so.
- **Every force kernel is held to two independent oracles**: OpenMM for the
  energy, and a central difference of its own energy for the force. Both are
  required. Three terms have already shipped with an exactly correct energy and
  a wrong gradient; no single-point energy comparison can see that, and the
  production symptom is a simulation that runs and silently heats up.
- **Compare per term, never on the total.** A total-energy match once hid a
  reciprocal sum that was 2.9e57 kJ/mol, compensated by the self energy, and
  separately hid two cancelling harness errors.
- **A reference must share no code with the thing it validates.** The PME
  reference uses a naive per-axis DFT precisely so that it cannot agree with a
  broken FFT. A cell list validated against another cell list agrees about the
  pairs they both drop.
- **No atomics in the force path.** One thread per atom, gathering. WGSL has no
  `atomicAdd` on `f32`, and fixed point would make the result device-dependent
  in its last bits, which destroys the differential tests. The one exception is
  the PME charge grid, where the accumulation is bounded and integer addition is
  exact.
- **Units are converted exactly once, at parse time**, into nm / kJ/mol / ps /
  amu / e. Nothing downstream may multiply by 4.184.
- **Never wrap positions into the box during a run.** Bonded terms are not
  periodic, so a molecule must stay whole in the coordinate array. Imaging
  happens when a frame is written.
- **Bounds-check every store.** Metal clamps an out-of-range write where Vulkan
  discards it, so a missing guard corrupts the last atom on one backend and is
  invisible on the other. WebGPU permits either.
- **Keep `shared/` and `cpu/` loadable by the browser as written** — plain ES
  modules, relative specifiers with `.js`, no build step, no bundler. A change
  that reintroduces a compile stage between the source and the page is the thing
  this layout exists to prevent.
- **Refuse by name rather than approximating.** An unsupported constraint mode,
  a legacy force-field term, a NetCDF restart, a cutoff that does not fit the
  box — each gets a message saying what to do instead. A silent approximation in
  a scientific tool is worse than a refusal.
- **A cutoff over half the smallest box width is an error, not a warning.** It
  breaks the minimum-image convention and nothing in WebGPU will say so.
- **State the degrees of freedom.** `3N - constraints - 3`. Using `3N` on a
  constrained system reads the temperature a third low, and the obvious fix
  makes it genuinely overheat.
