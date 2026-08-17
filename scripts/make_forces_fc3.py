#!/usr/bin/env python
"""
make_forces_fc3.py

Generates a phono3py FORCES_FC3 file by evaluating forces on phono3py's
displaced supercells (POSCAR-##### files produced by `phono3py -d`) with a
MACE interatomic potential.

Workflow this script fits into:
  1. phono3py -d --dim="a b c" -c POSCAR    -> SPOSCAR, POSCAR-00001, ..., phono3py_disp.yaml
  2. python make_forces_fc3.py ...           -> FORCES_FC3   (this script)
  3. phono3py phono3py_disp.yaml --mesh="M M M" --br [--nac]
                                              -> kappa-mMxMxM.hdf5

Each displaced supercell has 0 (rare, zero net displacement relative to
SPOSCAR), 1 (the "first atom" displacements, also used for the fc2 part), or
2 (paired "first+second atom" displacements, for fc3) atoms moved relative to
SPOSCAR -- this is phono3py's standard FC3 displacement scheme. More than 2
moved atoms means the file doesn't match SPOSCAR and something is wrong.

Usage:
  python make_forces_fc3.py --model my.model --sposcar SPOSCAR
  python make_forces_fc3.py --foundation small --sposcar SPOSCAR
"""

import argparse
import glob
import os
import re

import numpy as np
from ase.io import read


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def find_displacement_files(pattern):
    files = [
        f for f in glob.glob(pattern)
        if os.path.isfile(f) and re.fullmatch(r"POSCAR-\d+", os.path.basename(f))
    ]
    files = sorted(files, key=natural_key)
    if not files:
        raise RuntimeError(f"No displaced files found matching '{pattern}'")

    numbers = [int(re.fullmatch(r"POSCAR-(\d+)", os.path.basename(f)).group(1)) for f in files]
    expected = list(range(numbers[0], numbers[0] + len(files)))
    if numbers != expected:
        missing = sorted(set(expected) - set(numbers))
        raise RuntimeError(
            f"POSCAR numbering is not contiguous. First={numbers[0]}, "
            f"last={numbers[-1]}, missing={missing[:20]}"
        )
    if numbers[0] != 1:
        raise RuntimeError(f"First displaced structure is not POSCAR-00001 (found {files[0]}).")
    return files


def check_structure_compatibility(atoms_ref, atoms, fname, cell_tol):
    if len(atoms) != len(atoms_ref):
        raise RuntimeError(f"{fname}: atom-count mismatch ({len(atoms)} != {len(atoms_ref)})")
    if atoms.get_chemical_symbols() != atoms_ref.get_chemical_symbols():
        raise RuntimeError(f"{fname}: chemical-symbol order differs from the reference")
    if not np.allclose(atoms.cell.array, atoms_ref.cell.array, atol=cell_tol):
        raise RuntimeError(f"{fname}: cell differs from the reference")


def get_displacements(atoms_ref, atoms, disp_tol):
    s_ref = atoms_ref.get_scaled_positions(wrap=True)
    s_new = atoms.get_scaled_positions(wrap=True)
    ds = s_new - s_ref
    ds -= np.round(ds)
    dR = ds @ atoms_ref.cell.array
    norms = np.linalg.norm(dR, axis=1)
    moved = np.sort(np.where(norms > disp_tol)[0])
    return dR, moved


def build_calculator(args):
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
                   help="Use a pretrained MACE-MP foundation model instead")
    p.add_argument("--sposcar", default="SPOSCAR", help="Reference undisplaced supercell (default: SPOSCAR)")
    p.add_argument("--disp-glob", default="POSCAR-*", help="Glob for displaced supercell files (default: POSCAR-*)")
    p.add_argument("--output", default="FORCES_FC3", help="Output path (default: FORCES_FC3)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Torch device (default: cpu)")
    p.add_argument("--dtype", default="float64", choices=["float32", "float64"], help="Torch dtype (default: float64)")
    p.add_argument("--no-subtract-residual", action="store_true",
                   help="Do not subtract SPOSCAR's own (non-zero) forces from each result")
    p.add_argument("--cell-tol", type=float, default=1e-8)
    p.add_argument("--disp-tol", type=float, default=1e-6, help="Displacement detection tolerance, Angstrom")
    args = p.parse_args()

    if not os.path.exists(args.sposcar):
        raise FileNotFoundError(f"Missing {args.sposcar}")

    import torch
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("OMP_NUM_THREADS", "1"))))

    calc = build_calculator(args)

    print(f"Reading reference supercell: {args.sposcar}")
    atoms_ref = read(args.sposcar)
    natom = len(atoms_ref)

    disp_files = find_displacement_files(args.disp_glob)
    ndisp = len(disp_files)
    print(f"natom = {natom}, ndisplacements = {ndisp}")

    residual_forces = None
    if not args.no_subtract_residual:
        print("Calculating residual forces on the reference supercell...")
        ref = atoms_ref.copy()
        ref.calc = calc
        residual_forces = ref.get_forces()
        print(f"Max |residual force| = {np.abs(residual_forces).max():.6e} eV/Ang")

    with open(args.output, "w") as fout:
        for i, disp_file in enumerate(disp_files, start=1):
            atoms = read(disp_file)
            check_structure_compatibility(atoms_ref, atoms, disp_file, args.cell_tol)

            dR, moved = get_displacements(atoms_ref, atoms, args.disp_tol)
            if len(moved) > 2:
                raise RuntimeError(
                    f"{disp_file}: detected {len(moved)} displaced atoms. "
                    "Expected 0, 1, or 2 for a normal phono3py FC3 displacement dataset."
                )

            atoms.calc = calc
            forces = atoms.get_forces()
            if residual_forces is not None:
                forces = forces - residual_forces

            for fx, fy, fz in forces:
                fout.write(f" {fx: .15f} {fy: .15f} {fz: .15f}\n")

            if len(moved) == 0:
                moved_msg = "zero net displacement relative to reference"
            else:
                moved_msg = ", ".join(f"atom={a + 1} disp=({dR[a][0]: .6f},{dR[a][1]: .6f},{dR[a][2]: .6f})" for a in moved)
            print(f"[{i}/{ndisp}] OK  {disp_file}  nmoved={len(moved)}  {moved_msg}")

    print(f"\nDone. Wrote {args.output}  ({ndisp} supercells x {natom} atoms)")


if __name__ == "__main__":
    main()
