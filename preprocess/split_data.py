import argparse
from pathlib import Path
import numpy as np


def _parse_relevant_docs(z):
    v = z["relevant_docs"]
    if isinstance(v, np.ndarray) and v.shape == ():
        return v.item()
    if isinstance(v, dict):
        return v
    return v.item()


def split_object_query_npz(
    in_npz: str,
    out_dir: str,
    test_ratio: float = 0.2,
    shuffle: bool = False,
    seed: int = 42,
    drop_attention: bool = True,
):
    in_npz = Path(in_npz)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    z = np.load(in_npz, allow_pickle=True)

    required = [
        "task", "model",
        "documents", "doc_attnmask", "doc_imgmask",
        "query", "query_attnmask",
        "docid", "qid",
        "relevant_docs",
        "docidx_2_docid", "qsidx_2_query",
    ]
    for k in required:
        if k not in z.files:
            raise KeyError(f"Missing key '{k}'. Available keys: {z.files}")

    Q = z["query"]
    qid = z["qid"]
    qam = z["query_attnmask"]

    if not (isinstance(Q, np.ndarray) and Q.dtype == object):
        raise ValueError(f"Expected query as object array (Nq,), got dtype={Q.dtype}, shape={Q.shape}")
    if len(Q) != len(qid) or len(Q) != len(qam):
        raise ValueError(
            f"Length mismatch: len(query)={len(Q)}, len(qid)={len(qid)}, len(query_attnmask)={len(qam)}"
        )

    Nq = len(qid)
    n_test = int(Nq * test_ratio)
    if n_test <= 0 or n_test >= Nq:
        raise ValueError(f"Bad test_ratio={test_ratio}. Nq={Nq} -> n_test={n_test}")

    idx = np.arange(Nq, dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    def take(arr, indices):
        return arr[indices]

    # --- relevant_docs split (qid subset) ---
    rel_all = _parse_relevant_docs(z)
    qid_train = np.array([str(x) for x in take(qid, train_idx)], dtype=object)
    qid_test  = np.array([str(x) for x in take(qid, test_idx)], dtype=object)

    rel_train = {q: rel_all[str(q)] for q in qid_train}
    rel_test  = {q: rel_all[str(q)] for q in qid_test}

    # --- base: 문서쪽은 그대로 유지 + docidx_2_docid 포함 ---
    base = {
        "task": z["task"],
        "model": z["model"],
        "documents": z["documents"],
        "doc_attnmask": z["doc_attnmask"],
        "doc_imgmask": z["doc_imgmask"],
        "docid": z["docid"],
        "docidx_2_docid": z["docidx_2_docid"],  # ✅ 반드시 포함(그대로)
    }

    train_pack = dict(base)
    test_pack = dict(base)

    # ✅ qsidx_2_query는 split 크기에 맞게 슬라이스해서 저장해야
    #    eval 코드의 qkey = qsidx_2_query[qi] 를 그대로 쓸 수 있음
    train_pack.update({
        "query": take(z["query"], train_idx),
        "query_attnmask": take(z["query_attnmask"], train_idx),
        "qid": take(z["qid"], train_idx),
        "relevant_docs": np.array(rel_train, dtype=object),
        "qsidx": train_idx,
        "qsidx_2_query": take(z["qsidx_2_query"], train_idx), 
    })

    test_pack.update({
        "query": take(z["query"], test_idx),
        "query_attnmask": take(z["query_attnmask"], test_idx),
        "qid": take(z["qid"], test_idx),
        "relevant_docs": np.array(rel_test, dtype=object),
        "qsidx": test_idx,
        "qsidx_2_query": take(z["qsidx_2_query"], test_idx), 
    })

    if (not drop_attention) and ("attention" in z.files):
        # attention은 문서쪽이면 보통 query마다 반복 저장이라 그대로 둬도 됨
        train_pack["attention"] = z["attention"]
        test_pack["attention"] = z["attention"]

    stem = in_npz.stem
    stem = stem.replace("_dump_new", "").replace("_test_dump_new", "_test")
    train_path = out_dir / f"{stem}_train.npz"
    test_path  = out_dir / f"{stem}_test.npz"
    idx_path   = out_dir / f"{stem}_split_idx.npz"

    np.savez_compressed(train_path, **train_pack)
    np.savez_compressed(test_path, **test_pack)
    np.savez_compressed(
        idx_path,
        train_idx=train_idx,
        test_idx=test_idx,
        shuffle=shuffle,
        seed=seed,
        test_ratio=test_ratio,
        in_npz=str(in_npz),
        Nq=Nq,
    )

    print("saved:")
    print("  train:", train_path)
    print("  test :", test_path)
    print("  idx  :", idx_path)
    print(f"Nq={Nq}, train_q={len(train_idx)}, test_q={len(test_idx)}, shuffle={shuffle}")

    return str(train_path), str(test_path), str(idx_path)


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--in_npz", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep_attention", action="store_true")
    return p


def main():
    args = build_argparser().parse_args()
    split_object_query_npz(
        in_npz=args.in_npz,
        out_dir=args.out_dir,
        test_ratio=args.test_ratio,
        shuffle=args.shuffle,
        seed=args.seed,
        drop_attention=(not args.keep_attention),
    )


if __name__ == "__main__":
    main()