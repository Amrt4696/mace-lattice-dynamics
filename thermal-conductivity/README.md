# Thermal conductivity

MgO:

cd thermal-conductivity/mgo
phono3py-init -d --dim="4 4 4" -c POSCAR_MgO_0GPa_300K
python ../../scripts/make_forces_fc3.py --model ../../models/mgo_mace_p0-135gpa_v1.model --sposcar SPOSCAR --output FORCES_FC3
phono3py phono3py_disp.yaml --mesh="24 24 24" --lbte --ts="300"

Bridgmanite (cutoff-pair keeps the displacement count sane):

cd thermal-conductivity/bridgmanite
phono3py-init -d --dim="2 2 2" -c POSCAR_bridg_0GPa_300K --cutoff-pair 3
python ../../scripts/make_forces_fc3.py --model ../../models/bridgmanite_mace_p0-135gpa_v1.model --sposcar SPOSCAR --output FORCES_FC3
phono3py phono3py_disp.yaml --mesh="24 24 24" --lbte --ts="300"

Writes fc2.hdf5, fc3.hdf5, kappa-m242424.hdf5. BORN file in the folder = NAC auto-applied.

Read it with:
import h5py
f = h5py.File("kappa-m242424.hdf5")
f["kappa"][:]       # LBTE
f["kappa_RTA"][:]   # RTA reference
