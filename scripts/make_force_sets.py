#!/usr/bin/env python
"""
make_force_sets.py

Generates a phonopy Type-1 FORCE_SETS file by evaluating forces on
phonopy-generated displaced supercells with a MACE interatomic potential.

Workflow this script fits into:
  1. phonopy -d --dim="a b c" -c POSCAR-unitcell     -> POSCAR-001, POSCAR-002, ..., SPOSCAR, phonopy_disp.yaml
  2. python make_force_sets.py ...                    -> FORCE_SETS   (this script)
  3. phonopy -s band.conf -c POSCAR-unitcell           -> band.yaml (phonon dispersion)

Usage:
  python make_force_sets.py --model my.model --yaml phonopy_disp.yaml --sposcar SPOSCAR
  python make_force_sets.py --foundation small --yaml phonopy_disp.yaml --sposcar SPOSCAR
"""

import argparse
import glob
import os
import re

import numpy as np
import yaml
from ase.io import read


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def load_phonopy_displacements(yaml_path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    if "displacements" not in data:
        raise RuntimeError(f"'{yaml_path}' does not contain 'displacements'.")
    disp_data = data["displacements"]
    if not isinstance(disp_data, list) or len(disp_data) == 0:
        raise RuntimeError(f"No displacements found in '{yaml_path}'.")
    parsed = []
    for i, entry in enumerate(disp_data, start=1):
        if isinstance(entry, dict) and "atom" in entry and "displacement" in entry:
            parsed.append(
                {
                    "index": i,
                    "atom": int(entry["atom"]),
                    "displacement": np.array(entry["displacement"], dtype=float),
                }
            )
        else:
            raise RuntimeError(
                "This phonopy_disp.yaml is not the standard single-displacement "
                "dataset produced by 'phonopy -d'. Type-1 FORCE_SETS needs exactly "
                "one displaced atom per supercell."
            )
    return parsed


def find_disp_files(pattern, expected_n):
    files = sorted(glob.glob(pattern), key=natural_key)
    if len(files) != expected_n:
        raise RuntimeError(
            f"Expected {expected_n} displaced supercell files from the YAML, "
            f"but found {len(files)} files matching '{pattern}'."
        )
    return files


def check_structure_compatibility(atoms_ref, atoms, fname, cell_tol):
    if len(atoms) != len(atoms_ref):
        raise RuntimeError(f"{fname}: natom mismatch ({len(atoms)} != {len(atoms_ref)})")
    if atoms.get_chemical_symbols() != atoms_ref.get_chemical_symbols():
        raise RuntimeError(f"{fname}: chemical symbol order mismatch with SPOSCAR")
    if not np.allclose(atoms.cell.array, atoms_ref.cell.array, atol=cell_tol):
        raise RuntimeError(f"{fname}: cell mismatch with SPOSCAR")


def get_actual_displacement(atoms_ref, atoms, disp_tol):
    s_ref = atoms_ref.get_scaled_positions(wrap=True)
    s_new = atoms.get_scaled_positions(wrap=True)
    ds = s_new - s_ref
    ds -= np.round(ds)
    dR = ds @ atoms_ref.cell.array
    norms = np.linalg.norm(dR, axis=1)
    moved = np.where(norms > disp_tol)[0]
    return dR, moved


def build_calculator(args):
    """Return an ASE calculator, either a locally trained MACE model or a
    pretrained MACE-MP foundation model."""
    if args.model and args.foundation:
        raise SystemExit("Pass either --model or --foundation, not both.")
    if not args.model and not args.foundation:
        raise SystemExit("You must pass either --model <path> or --foundation <small|medium|large>.")

    if args.model:
        if not os.path.exists(args.model):
            raise FileNotFoundError(f"Missing model file: {args.model}")
        from mace.calculators import MACECalculator

        print(f"Loading local MACE model: {args.model}")
        return MACECalculator(model_paths=args.model, default_dtype=args.dtype, device=args.device)
    else:
        from mace.calculators import mace_mp

        print(f"Loading MACE-MP foundation model ('{args.foundation}') -- "
              f"this may download weights on first use.")
        return mace_mp(model=args.foundation, default_dtype=args.dtype, device=args.device)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="Path to a locally trained MACE .model file")
    p.add_argument("--foundation", choices=["small", "medium", "large"],
                   help="Use a pretrained MACE-MP foundation model instead of a local model")
    p.add_argument("--yaml", default="phonopy_disp.yaml", help="phonopy displacement YAML (default: phonopy_disp.yaml)")
    p.add_argument("--sposcar", default="SPOSCAR", help="Reference (undisplaced) supercell (default: SPOSCAR)")
    p.add_argument("--disp-glob", default="POSCAR-*", help="Glob pattern for displaced supercell files (default: POSCAR-*)")
    p.add_argument("--output", default="FORCE_SETS", help="Output FORCE_SETS path (default: FORCE_SETS)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Torch device (default: cpu)")
    p.add_argument("--dtype", default="float64", choices=["float32", "float64"], help="Torch dtype (default: float64)")
    p.add_argument("--no-subtract-residual", action="store_true",
                   help="Do not subtract the SPOSCAR's own (non-zero) forces from each displaced-cell force set")
    p.add_argument("--cell-tol", type=float, default=1e-8, help="Cell comparison tolerance")
    p.add_argument("--disp-tol", type=float, default=1e-6, help="Displacement detection tolerance, Angstrom")
    args = p.parse_args()

    if not os.path.exists(args.yaml):
        raise FileNotFoundError(f"Missing {args.yaml}")
    if not os.path.exists(args.sposcar):
        raise FileNotFoundError(f"Missing {args.sposcar}")

    import torch
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("OMP_NUM_THREADS", "1"))))

    calc = build_calculator(args)

    print(f"Reading reference supercell: {args.sposcar}")
    atoms_ref = read(args.sposcar)
    natom = len(atoms_ref)

    print(f"Reading displacement dataset from: {args.yaml}")
    disp_entries = load_phonopy_displacements(args.yaml)
    ndisp = len(disp_entries)
    disp_files = find_disp_files(args.disp_glob, ndisp)

    print(f"natom = {natom}, ndisplacements = {ndisp}")

    residual_forces = None
    if not args.no_subtract_residual:
        print("Calculating residual forces on SPOSCAR (subtracted from every displaced-cell force set)...")
        atoms_ref.calc = calc
        residual_forces = atoms_ref.get_forces()
        print(f"Max |residual force| on SPOSCAR = {np.abs(residual_forces).max():.6e} eV/Ang")

    with open(args.output, "w") as fout:
        fout.write(f"{natom}\n")
        fout.write(f"{ndisp}\n\n")
        for i, (entry, disp_file) in enumerate(zip(disp_entries, disp_files), start=1):
            atoms = read(disp_file)
            check_structure_compatibility(atoms_ref, atoms, disp_file, args.cell_tol)

            yaml_atom = entry["atom"]
            yaml_disp = entry["displacement"]
            dR, moved = get_actual_displacement(atoms_ref, atoms, args.disp_tol)

            if len(moved) == 0:
                raise RuntimeError(f"{disp_file}: no displacement detected relative to SPOSCAR")
            if len(moved) != 1:
                moved_1based = [m + 1 for m in moved]
                raise RuntimeError(
                    f"{disp_file}: detected {len(moved)} displaced atoms {moved_1based}. "
                    "Type-1 FORCE_SETS expects exactly one displaced atom per supercell."
                )

            actual_atom = moved[0] + 1
            actual_disp = dR[moved[0]]
            if actual_atom != yaml_atom:
                raise RuntimeError(
                    f"{disp_file}: displaced atom mismatch. "
                    f"YAML says atom {yaml_atom}, structure comparison says atom {actual_atom}"
                )
            if not np.allclose(actual_disp, yaml_disp, atol=5e-6):
                raise RuntimeError(
                    f"{disp_file}: displacement mismatch.\nYAML  : {yaml_disp}\nActual: {actual_disp}"
                )

            atoms.calc = calc
            forces = atoms.get_forces()
            if residual_forces is not None:
                forces = forces - residual_forces

            fout.write(f"{yaml_atom}\n")
            fout.write(f"{yaml_disp[0]:22.16f} {yaml_disp[1]:22.16f} {yaml_disp[2]:22.16f}\n")
            for fx, fy, fz in forces:
                fout.write(f"{fx:22.16f} {fy:22.16f} {fz:22.16f}\n")
            if i != ndisp:
                fout.write("\n")

            print(f"[{i}/{ndisp}] OK  {disp_file}  atom={yaml_atom}  "
                  f"disp=({yaml_disp[0]:.6f}, {yaml_disp[1]:.6f}, {yaml_disp[2]:.6f})")

    print(f"\nDone. Wrote {args.output}")


if __name__ == "__main__":
    main()
