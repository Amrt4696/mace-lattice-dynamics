# Phonon dispersion

MgO: `mgo/`, bridgmanite: `bridgmanite/`

cd phonon-dispersion/mgo
phonopy-init -d --dim="2 2 2" -c POSCAR_MgO_0GPa_300K
python ../../scripts/make_force_sets.py --model ../../models/mgo_mace_p0-135gpa_v1.model --yaml phonopy_disp.yaml --sposcar SPOSCAR --output FORCE_SETS
phonopy band.conf

Same for bridgmanite, swap POSCAR/model paths. band.conf needs a band path -- write your own per material.

BORN file in the folder = NAC auto-applied. --nonac to disable.
