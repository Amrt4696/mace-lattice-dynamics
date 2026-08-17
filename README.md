# MACE Lattice Dynamics: MgO and Bridgmanite

Phonon dispersion and thermal conductivity from trained MACE potentials, via phonopy/phono3py.

## Setup

./setup.sh
source .venv/bin/activate

or:

conda env create -f environment.yml
conda activate mace-lattice-dynamics

Needs phonopy/phono3py v4+ (phonopy-init / phono3py-init for the setup steps -- see the two workflow READMEs).

## Contents

models/                  trained MACE potentials (MgO, bridgmanite, 0-135 GPa)
scripts/                 MACE forces for phonopy/phono3py
phonon-dispersion/       see phonon-dispersion/README.md
thermal-conductivity/    see thermal-conductivity/README.md

## Structures

Ambient conditions only for now (0 GPa, 300 K). More P-T points may be added later.
