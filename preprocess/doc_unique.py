import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np


# -------------------------
# utils
# -------------------------
def to_str(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    return str(x)


def build_keep_first_occurrence(docids: np.ndarray) -> np.ndarray:
    """Return indices of first occurrence for each docid (stable)."""
    seen = set()
    keep = []
    for i, d in enumerate(docids):
        if d in seen:
            continue
        seen.add(d)
        keep.append(i)
    return np.asarray(keep, dtype=np.int64)


def slice_doc_axis_keys(
    z: np.lib.npyio.NpzFile,
    keep: np.ndarray,
    n_docs_full: int,
    doc_axis_keys: List[str],
) -> Dict[str, Any]:
    """
    Slice arrays whose first axis matches n_docs_full AND key is in doc_axis_keys.
    Copy everything else as-is.
    """
    out: Dict[str, Any] = {}
    doc_axis_set = set(doc_axis_keys)

    for k in z.files:
        arr = z[k]
        # candidate: non-scalar and first axis == n_docs_full
        if (k in doc_axis_set) and getattr(arr, "ndim", 0) > 0 and arr.shape[0] == n_docs_full:
            out[k] = arr[keep]
        else:
            out[k] = arr
    return out


def make_docidx_2_docid(docid_unique: np.ndarray) -> Dict[str, str]:
    return {str(i): to_str(docid_unique[i]) for i in range(len(docid_unique))}


def save_npz(path: str, data: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def sanity_check_unique(z_path: str) -> None:
    z = np.load(z_path, allow_pickle=True)
    n_docid = len(z["docid"]) if "docid" in z.files else None
    n_docs = z["documents"].shape[0] if "documents" in z.files else None
    m = z["docidx_2_docid"].item() if "docidx_2_docid" in z.files else None
    ex0 = m.get("0", None) if isinstance(m, dict) else None
    print(f"[CHECK] {z_path}")
    print(f"  docid: {n_docid} | documents: {n_docs} | map: {len(m) if isinstance(m, dict) else None} | ex0: {ex0}")


# -------------------------
# main ops
# -------------------------
def build_raw_unique(
    raw_full_npz: str,
    raw_unique_out: str,
    doc_axis_keys: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create raw_unique npz from raw_full using first-occurrence keep indices.
    Returns (keep, docid_unique).
    """
    z_full = np.load(raw_full_npz, allow_pickle=True)

    if "docid" not in z_full.files:
        raise ValueError(f"[ERROR] raw_full has no 'docid': {raw_full_npz}")

    docid_full = np.array([to_str(x) for x in z_full["docid"]], dtype=object)
    keep = build_keep_first_occurrence(docid_full)

    print(f"[INFO] raw_full docs={len(docid_full)} -> unique(first-occ)={len(keep)}")

    out_raw = slice_doc_axis_keys(z_full, keep, len(docid_full), doc_axis_keys)

    # override docid to string-clean unique version
    docid_unique = docid_full[keep]
    out_raw["docid"] = docid_unique
    out_raw["docidx_2_docid"] = make_docidx_2_docid(docid_unique)

    save_npz(raw_unique_out, out_raw)
    print(f"[DONE] raw_unique saved: {raw_unique_out}")

    return keep, docid_unique


def build_in_npz_unique(
    in_npz: str,
    out_npz: str,
    keep: np.ndarray,
    docid_unique: np.ndarray,
    n_docs_full: int,
    doc_axis_keys: List[str],
) -> None:
    """
    Create unique version of in_npz using the same keep indices.
    Requires in_npz's doc axis length == n_docs_full.
    """
    zin = np.load(in_npz, allow_pickle=True)

    if "documents" not in zin.files:
        raise ValueError(f"[ERROR] in_npz has no 'documents': {in_npz}")

    n_in = zin["documents"].shape[0]
    if n_in != n_docs_full:
        raise ValueError(
            f"[ERROR] in_npz doc count mismatch with raw_full: in={n_in} vs raw_full={n_docs_full}\n"
            "=> in_npz가 raw_full 인덱스 체계가 아닐 수 있음. (이 경우 keep을 그대로 적용하면 안 됨)"
        )

    out = slice_doc_axis_keys(zin, keep, n_docs_full, doc_axis_keys)

    # enforce metadata same as raw_unique
    out["docid"] = docid_unique
    out["docidx_2_docid"] = make_docidx_2_docid(docid_unique)

    save_npz(out_npz, out)
    print(f"[DONE] in_npz unique saved: {out_npz}")


def parse_args():
    p = argparse.ArgumentParser(description="Make unique(docid) npz by keeping first occurrence indices.")
    p.add_argument("--raw_full", type=str, required=True, help="raw_full npz (may contain duplicate docid)")
    p.add_argument("--raw_unique_out", type=str, required=True, help="output path for raw_unique npz")

    p.add_argument("--in_npz", type=str, default=None, help="(optional) npz to be aligned with raw_full indices (e.g., S3E_init mf25)")
    p.add_argument("--out_npz", type=str, default=None, help="(optional) output path for unique version of in_npz")

    p.add_argument(
        "--doc_axis_keys",
        type=str,
        default="docid,documents,doc_attnmask,doc_imgmask,attention",
        help="comma-separated keys treated as doc-axis arrays (sliced by keep if shape[0]==N_docs_full)",
    )
    p.add_argument("--sanity", action="store_true", help="print sanity check for saved files")

    return p.parse_args()


def main():
    args = parse_args()
    doc_axis_keys = [x.strip() for x in args.doc_axis_keys.split(",") if x.strip()]

    keep, docid_unique = build_raw_unique(
        raw_full_npz=args.raw_full,
        raw_unique_out=args.raw_unique_out,
        doc_axis_keys=doc_axis_keys,
    )

    # optional: also convert in_npz
    if (args.in_npz is None) ^ (args.out_npz is None):
        raise ValueError("[ERROR] --in_npz and --out_npz must be provided together (or both omitted).")

    if args.in_npz is not None:
        # need full doc count from raw_full (use keep + len(unique) relation)
        z_full = np.load(args.raw_full, allow_pickle=True)
        n_docs_full = len(z_full["docid"])
        build_in_npz_unique(
            in_npz=args.in_npz,
            out_npz=args.out_npz,
            keep=keep,
            docid_unique=docid_unique,
            n_docs_full=n_docs_full,
            doc_axis_keys=doc_axis_keys,
        )

    if args.sanity:
        sanity_check_unique(args.raw_unique_out)
        if args.out_npz is not None:
            sanity_check_unique(args.out_npz)


if __name__ == "__main__":
    main()