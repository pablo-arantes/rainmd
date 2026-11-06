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
for name in helix strand mixed; do
  cp "$name.prmtop" "$name.inpcrd" "$out/"
done

echo "wrote:"
cd "$out" && ls -la ala-vac.* ala-wat.* helix.* strand.* mixed.*
