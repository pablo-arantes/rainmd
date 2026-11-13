#!/usr/bin/env python3
"""
Fit the ten coefficients of the WGSL `erfcApprox` in webgpu/kernels/wgsl-common.js.

🔴 WHAT IS BEING FITTED, AND WHY IT IS THE LOGARITHM. The kernel evaluates

    erfc(z) ~= t * exp(-z^2 + P(t)),      t = 1/(1 + z/2)

so the exact P is

    P(t) = ln(erfc(z)) + z^2 - ln(t),     z = 2(1 - t)/t

and because d(ln erfc) = dP, an ABSOLUTE error in P is a RELATIVE error in
erfc. Fitting P therefore minimises exactly the quantity the Ewald split cares
about. Fitting erfc itself would minimise absolute error, which is useless
here: erfc at a 1.0 nm cutoff is 2.0e-4, so 1e-7 of absolute accuracy is 5e-4
of relative accuracy on precisely the long-range terms the split exists to get
right.

🔴 AND THE DOMAIN IS THE WHOLE OF z >= 0, NOT THE CUTOFF'S SLICE. t in (0, 1]
covers every z, so the result is a drop-in with no domain caveat: alpha depends
on the cutoff and the tolerance, and a fit valid only to alpha*r = 2.8 would
become a silent approximation the first time someone widened the cutoff.
t -> 0 is z -> infinity, where P(0) = -ln(2*sqrt(pi)) exactly, so the endpoint
is pinned by the function rather than by the fit.

🔴 LAWSON REWEIGHTING, BECAUSE THE BAR IS A WORST CASE. Plain least squares
minimises the 2-norm of the error while every tolerance derived from this is a
maximum. Lawson's algorithm - reweight by |error| and refit - converges to the
minimax solution, and the best iterate is kept rather than the last.

Reports the f32 error as the shader would evaluate it, which is a LOWER BOUND:
a GPU's `exp` need not be correctly rounded. tools/gpu/check-erfc.mjs measures
the real device and is what the documented floor comes from.

Needs mpmath (`pip install mpmath`; it ships with sympy). Writes nothing - the
coefficients are pasted into the shader by hand, so that the shipped constants
are visible in the source rather than generated at build time.
"""
import argparse
import math
import struct

from mpmath import mp

DEGREE = 9


def f32(value):
    """Round to f32, the way every WGSL operation does."""
    return struct.unpack("f", struct.pack("f", float(value)))[0]


def targets(count):
    """P(t) at `count` points of (0, 1], in 40 digits."""
    mp.dps = 40
    nodes, values = [], []
    for index in range(count):
        # Uniform in t, plus a geometric tail: the error equioscillates in t,
        # but t -> 0 is an endpoint the fit has to be held to.
        t = mp.mpf(index + 1) / count
        nodes.append(t)
        z = 2 * (1 - t) / t
        values.append(mp.log(mp.erfc(z)) + z * z - mp.log(t))
    for exponent in range(2, 9):
        t = mp.mpf(10) ** (-exponent)
        z = 2 * (1 - t) / t
        nodes.append(t)
        values.append(mp.log(mp.erfc(z)) + z * z - mp.log(t))
    return nodes, values


def weighted_fit(nodes, values, weights):
    """Normal equations for a weighted polynomial least squares, in mpmath."""
    size = DEGREE + 1
    powers = [[t ** k for k in range(size)] for t in nodes]
    normal = mp.matrix(size, size)
    rhs = mp.matrix(size, 1)
    for row, (basis, value, weight) in enumerate(zip(powers, values, weights)):
        for i in range(size):
            wb = weight * basis[i]
            for j in range(i, size):
                normal[i, j] += wb * basis[j]
            rhs[i] += wb * value
    for i in range(size):
        for j in range(i):
            normal[i, j] = normal[j, i]
    return mp.lu_solve(normal, rhs)


def worst_error(coefficients, nodes, values):
    worst = mp.mpf(0)
    errors = []
    for t, value in zip(nodes, values):
        approximation = mp.mpf(0)
        for coefficient in reversed(coefficients):
            approximation = approximation * t + coefficient
        error = abs(approximation - value)
        errors.append(error)
        worst = max(worst, error)
    return worst, errors


def shader_relative_error(coefficients, z_max, samples):
    """Max relative error of erfc as the SHADER evaluates it, in f32."""
    constants = [f32(c) for c in coefficients]
    worst, at = 0.0, 0.0
    for index in range(1, samples + 1):
        z = f32(z_max * index / samples)
        t = f32(1.0 / f32(1.0 + f32(0.5 * z)))
        p = constants[DEGREE]
        for k in range(DEGREE - 1, -1, -1):
            p = f32(constants[k] + f32(t * p))
        argument = f32(-f32(z * z) + p)
        value = f32(t * f32(math.exp(argument)))
        reference = math.erfc(z)
        relative = abs(value - reference) / reference
        if relative > worst:
            worst, at = relative, z
    return worst, at


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=700)
    parser.add_argument("--passes", type=int, default=40)
    parser.add_argument("--report-at", type=float, default=2.8,
                        help="alpha*r the shipped cutoff reaches, for the f32 report")
    arguments = parser.parse_args()

    nodes, values = targets(arguments.points)
    weights = [mp.mpf(1)] * len(nodes)
    best = None
    for _ in range(arguments.passes):
        coefficients = weighted_fit(nodes, values, weights)
        worst, errors = worst_error(coefficients, nodes, values)
        if best is None or worst < best[1]:
            best = ([c for c in coefficients], worst)
        total = mp.mpf(0)
        for index, error in enumerate(errors):
            # A floor on the weight, or a point with near-zero error is removed
            # from the problem for good and the iteration stalls.
            weights[index] = weights[index] * (error + worst / 1000)
            total += weights[index]
        for index in range(len(weights)):
            weights[index] = weights[index] / total * len(weights)
    coefficients, worst = best

    print(f"max |dP| over t in (0,1]:  {float(worst):.4e}"
          "   (= max relative error of erfc in exact arithmetic)")
    for z_max in (arguments.report_at, 6.0):
        error, at = shader_relative_error(coefficients, z_max, 20000)
        print(f"f32 as the shader evaluates it, z <= {z_max}:  {error:.4e}  (worst at z = {at:.4f})")
    print()
    print("paste into ERFC_FUNCTION, Horner order (last line first):")
    for index, coefficient in enumerate(coefficients):
        print(f"  a{index} = {f32(coefficient):+.9g}")


if __name__ == "__main__":
    main()
