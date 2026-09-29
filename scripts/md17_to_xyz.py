"""Download the MD17 aspirin trajectory and convert it to a multi-frame xyz file.

MD17 (Chmiela et al., Sci. Adv. 3, e1603015, 2017) is a public benchmark of
ab initio molecular dynamics trajectories (PBE+TS, 0.5 fs between frames).
The aspirin set has 211,762 time-ordered frames of 21 atoms, in the same atom
order that the ConfAna aspirin mappings use.

Run from the project root:
    python scripts/md17_to_xyz.py                 # download + convert
    python scripts/md17_to_xyz.py --npz FILE.npz  # convert an existing download
    python scripts/md17_to_xyz.py --help

The raw data is not redistributed with ConfAna; this script fetches it from the
sGDML project. Cite the paper above if you publish results made from it.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

import numpy as np

MD17_ASPIRIN_URL = "http://quantum-machine.org/gdml/data/npz/md17_aspirin.npz"
MD17_ASPIRIN_SHA256 = "af4ce531bf8cf07610c0bd876cb5b321f3c301d2365720967d98d45e1db1f748"

_ELEMENTS = {1: "H", 6: "C", 7: "N", 8: "O"}


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_md17(dest: Path, url: str = MD17_ASPIRIN_URL, sha256: str | None = MD17_ASPIRIN_SHA256) -> Path:
    """Download the MD17 npz to ``dest`` (skipped if present) and verify its checksum.

    Raises
    ------
    ValueError
        If the file's SHA-256 does not match ``sha256``.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"Downloading {url} -> {dest}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    if sha256 is not None:
        actual = sha256_of(dest)
        if actual != sha256:
            raise ValueError(
                f"{dest}: SHA-256 {actual} does not match the expected {sha256}. "
                "Delete the file and download again."
            )
    return dest


def md17_npz_to_xyz(npz_path: Path, xyz_path: Path, stride: int = 1) -> int:
    """Write an MD17 npz (keys ``z``, ``R``, ``E``) as a multi-frame xyz file.

    Each comment line records the MD17 step index and energy, e.g.
    ``MD17 aspirin  Step: 42  energy_kcal_mol=-406757.591264``.
    Positions are in Angstrom, as in the source file.

    Parameters
    ----------
    npz_path:
        MD17 ``.npz`` file.
    xyz_path:
        Output xyz path (overwritten).
    stride:
        Keep every ``stride``-th frame. The step index in the comment line is
        always the original MD17 frame index.

    Returns
    -------
    int
        Number of frames written.

    Raises
    ------
    ValueError
        On a missing key, an unknown element, or inconsistent array shapes.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    data = np.load(npz_path)
    for key in ("z", "R"):
        if key not in data:
            raise ValueError(f"{npz_path}: missing key {key!r}")
    z = np.asarray(data["z"]).astype(int)
    R = np.asarray(data["R"])
    if R.ndim != 3 or R.shape[1:] != (len(z), 3):
        raise ValueError(f"{npz_path}: R has shape {R.shape}, expected (n_frames, {len(z)}, 3)")
    unknown = sorted(set(z.tolist()) - set(_ELEMENTS))
    if unknown:
        raise ValueError(f"{npz_path}: unknown atomic numbers {unknown}")
    E = np.asarray(data["E"]).reshape(-1) if "E" in data else None
    if E is not None and len(E) != len(R):
        raise ValueError(f"{npz_path}: E has {len(E)} entries but R has {len(R)} frames")
    name = str(np.asarray(data["name"]).item().decode()) if "name" in data else "md17"

    symbols = [_ELEMENTS[n] for n in z]
    header = f"{len(z)}\n"
    xyz_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with open(xyz_path, "w") as fh:
        for i in range(0, len(R), stride):
            comment = f"MD17 {name}  Step: {i}"
            if E is not None:
                comment += f"  energy_kcal_mol={E[i]:.6f}"
            lines = [header, comment, "\n"]
            lines.extend(
                f"{s:>2s} {x:15.8f} {y:15.8f} {zc:15.8f}\n" for s, (x, y, zc) in zip(symbols, R[i])
            )
            fh.write("".join(lines))
            n_written += 1
    return n_written


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--npz", type=Path, default=Path("data/md17/md17_aspirin.npz"),
                        help="npz path; downloaded here if missing (default: %(default)s)")
    parser.add_argument("--out", type=Path, default=Path("data/md17/md17_aspirin.xyz"),
                        help="output xyz path (default: %(default)s)")
    parser.add_argument("--stride", type=int, default=1, help="keep every N-th frame (default: 1)")
    parser.add_argument("--no-download", action="store_true", help="fail instead of downloading")
    args = parser.parse_args(argv)

    if args.no_download:
        if not args.npz.exists():
            parser.error(f"{args.npz} does not exist")
    else:
        download_md17(args.npz)
    n = md17_npz_to_xyz(args.npz, args.out, stride=args.stride)
    print(f"Wrote {n} frames to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
