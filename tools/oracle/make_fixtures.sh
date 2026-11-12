#!/usr/bin/env bash
# Build the AMBER topologies the differential tests read.
#
# These are produced by tleap with the SAME leaprc lines the Making-it-rain
# notebooks use, so a fixture is a small instance of exactly what a user will
# drop on the page. Two systems, because they answer different questions:
#
#   ala-vac  alanine dipeptide in vacuum, 22 atoms. No box, no water, no
#            constraints - every bonded term, the 1-4 pairs and the full
#            exclusion list, small enough to compare atom by atom by eye.
#
#   ala-wat  the same solute in a TIP3P box with Na+/Cl-, a few thousand atoms.
#            This is the one that exercises PME, rigid water, the molecule
#            partition the barostat needs, and a periodic box.
#
# Needs AmberTools on PATH. Writes into test/fixtures/.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="$here/../../test/fixtures"
mkdir -p "$out"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cd "$work"

cat > build.leap <<'LEAP'
source leaprc.protein.ff14SB
source leaprc.water.tip3p

sys = sequence { ACE ALA NME }
saveamberparm sys ala-vac.prmtop ala-vac.inpcrd
savepdb sys ala-vac.pdb

solvateBox sys TIP3PBOX 8.0 0.7
addIonsRand sys Na+ 2 Cl- 2
saveamberparm sys ala-wat.prmtop ala-wat.inpcrd
savepdb sys ala-wat.pdb

# The same solute in a 12 A box, which is what a notebook user's padding looks
# like - 2854 atoms against ala-wat's 1190. It is the fixture the throughput
# table's 74 ns/day row is measured on, and the README quotes that number, so it
# had to become regenerable rather than be deleted.
#
# 🔴 THE CLOSENESS IS 1.0, NOT THE 0.7 ala-wat USES, AND IT IS NOT A GUESS. The
# recipe was never recorded, so it was recovered by sweeping: 12.0/0.7 gives
# 2884 atoms, 12.0/0.75 gives 2881, and 12.0/1.0 gives 2854 with the box
# 3.5010 x 3.7796 x 3.3088 nm - matching the committed file to four decimals.
# With no ions there is nothing random in it, and the rebuild is byte-identical
# to what is committed apart from the %VERSION date stamp.
big = sequence { ACE ALA NME }
solvateBox big TIP3PBOX 12.0 1.0
saveamberparm big ala-wat-big.prmtop ala-wat-big.inpcrd

# 🔴 A TRUNCATED OCTAHEDRON, BECAUSE EVERY OTHER FIXTURE HERE IS RECTANGULAR
# AND THAT HID A WRONG RECIPROCAL SUM. `solvateBox` gives a diagonal inverse
# box, for which the transpose of a matrix equals the matrix - so a PME kernel
# that built its wave vectors from the columns instead of the rows agreed with
# OpenMM to 1.4e-6 on every system this repository tested, and was 91 kJ/mol
# wrong on the first `solvateOct` box anyone tried. The notebooks offer
# solvateOct, so this is a configuration users reach.
#
# 🔴 8.0 A OF PADDING, NOT 6.0, AND THE CUTOFF IS WHY. For an AMBER truncated
# octahedron c_z = 0.8165 * L, so a 6.0 A pad gives L = 23.80 A and a smallest
# half-width of 0.97 nm - under the 1.0 nm cutoff, which `buildSystem` correctly
# refuses. 8.0 A gives L = 28.01 A, half-widths 1.40 / 1.32 / 1.14 nm, and 1315
# atoms.
oct = sequence { ACE ALA NME }
solvateOct oct TIP3PBOX 8.0
saveamberparm oct ala-oct.prmtop ala-oct.inpcrd

# Two 16-residue poly-alanines forced onto ideal backbone torsions, in vacuum.
#
# These exist for ONE reason: nothing else here has secondary structure. Both
# of the other fixtures are capped peptides that are correctly assigned coil, so
# the alpha-carbon criterion in shared/analysis/secondary.js could be validated
# against analytically ideal coordinates and against nothing a force field had
# ever touched. `impose` sets phi/psi directly, so ff14SB builds the rest - and
# the resulting CA spacings come out at 0.521/0.638 nm for the helix against the
# textbook 0.505/0.620, and 1.051/1.386 for the strand against 1.023/1.360. That
# gap between an ideal helix and a real one is exactly what the windows in
# secondary.js have to be wide enough to cover.
#
# Vacuum and no box, deliberately: they are for the viewer and the assignment,
# not for dynamics, and 162 atoms keeps them small enough to commit.
helix = sequence { ACE ALA ALA ALA ALA ALA ALA ALA ALA ALA ALA ALA ALA ALA ALA ALA NME }
impose helix { 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 } { { "N" "CA" "C" "N" -47.0 } { "C" "N" "CA" "C" -57.0 } }
saveamberparm helix helix.prmtop helix.inpcrd

strand = sequence { ACE ALA ALA ALA ALA ALA ALA ALA ALA ALA ALA ALA ALA ALA ALA ALA NME }
impose strand { 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 } { { "N" "CA" "C" "N" 135.0 } { "C" "N" "CA" "C" -139.0 } }
saveamberparm strand strand.prmtop strand.inpcrd

# Helix, loop and strand in one chain - the fixture the ribbon renderer needs,
# because a cartoon is mostly about the TRANSITIONS between elements.
#
# 🔴 THE LOOP TORSIONS ALTERNATE, AND THEY HAVE TO. A single regular setting in
# the bridge region reads as sheet - (-75, 150) is the beta region - and a loop
# of only three residues is absorbed whatever its torsions are, because the
# assignment votes over a five-residue window and the neighbouring elements
# cover it. Six residues with torsions alternating between (60, 60) and
# (-140, -80) leaves no uniform window, which is what coil actually is.
mixed = sequence { ACE ALA ALA ALA ALA ALA ALA ALA ALA ALA GLY GLY GLY GLY GLY GLY ALA ALA ALA ALA ALA ALA ALA ALA ALA NME }
impose mixed { 2 3 4 5 6 7 8 9 10 } { { "N" "CA" "C" "N" -47.0 } { "C" "N" "CA" "C" -57.0 } }
impose mixed { 11 13 15 } { { "N" "CA" "C" "N" 60.0 } { "C" "N" "CA" "C" 60.0 } }
impose mixed { 12 14 16 } { { "N" "CA" "C" "N" -140.0 } { "C" "N" "CA" "C" -80.0 } }
impose mixed { 17 18 19 20 21 22 23 24 25 } { { "N" "CA" "C" "N" 135.0 } { "C" "N" "CA" "C" -139.0 } }
saveamberparm mixed mixed.prmtop mixed.inpcrd
quit
LEAP

tleap -f build.leap > tleap.log 2>&1 || { cat tleap.log; exit 1; }
grep -iE "^(Error|Warning)" tleap.log || true

for name in ala-vac ala-wat; do
  cp "$name.prmtop" "$name.inpcrd" "$name.pdb" "$out/"
done
for name in helix strand mixed ala-oct ala-wat-big; do
  cp "$name.prmtop" "$name.inpcrd" "$out/"
done

# 🔴 two-body IS NOT BUILT HERE, AND THAT IS A KNOWN GAP RATHER THAN AN
# OVERSIGHT. Its recipe was never recorded and could not be recovered. What is
# known, from the committed file: two chains, ACE-ALA-ALA-ALA-NME and
# ACE-GLY-GLY-NME, 825 waters, 2 Na+ and 2 Cl-, 2547 atoms, box
# 4.1545 x 3.5779 x 2.6192 nm. Reconstructing it as
#
#   a = sequence { ACE ALA ALA ALA NME }
#   b = sequence { ACE GLY GLY NME }
#   translate b { 11.0 0.0 0.0 }
#   u = combine { a b }
#   solvateBox u TIP3PBOX 8.0 0.7
#   addIonsRand u Na+ 2 Cl- 2
#
# reproduces the y and z box edges exactly and gets to 2523 atoms with
# box x 4.1123 against the committed 2547 and 4.1545 - so the original used some
# further geometry (a rotation, or a different separation) that is not
# recoverable from the output. Writing the near-miss here under the same name
# would silently replace the fixture the page offers visitors and the several
# measured numbers that depend on it, so it is deliberately not written.
# test/amber-parse.test.js holds this list, so the README's claim cannot drift
# from it again.

echo "wrote:"
cd "$out" && ls -la ala-vac.* ala-wat.* ala-wat-big.* ala-oct.* helix.* strand.* mixed.*
