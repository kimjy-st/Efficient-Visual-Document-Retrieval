import argparse
import json
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn

from utils.utils import set_seed, set_optimizer, get_logger, align_by_docid, save_compressed_npz
from utils.preprocess_data import l2_normalize, load_train_payload, load_test_payload, load_init_payload, _as_object_array, preprocess_docs, preprocess_queries
from utils.mapping import DATASETMAP
from evaluator.retrieval import score_multi_vector_masked, CustomRetrievalEvaluator


# =============================================================================
# args
# =============================================================================
def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", type=str, nargs="+", required=True)
    p.add_argument("--data_root", type=str, default="data/split_features/colqwen")
    p.add_argument("--init_root", type=str, default="/home/hyshin/Project/EVDR/SIGIR/features2/colqwen")
    p.add_argument("--mfs", type=int, nargs="+", required=True)

    p.add_argument("--out_root", type=str, default="results")
    p.add_argument("--name", type=str, default="spl_train")

    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--opt", type=str, default="adamw")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--print_every", type=int, default=10)
    p.add_argument("--save_period", type=int, default = 3)

    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)

    return p


# =============================================================================
# train / eval / save
# =============================================================================
def train_epoch(
    P_teacher_norm: torch.Tensor,     # (N, Lt, D)
    pmask_teacher: torch.Tensor,      # (N, Lt)
    Q_all_norm: torch.Tensor,         # (Q, Lq, D)
    qmask: torch.Tensor,              # (Q, Lq)
    Pbar_param: nn.Parameter,         # (N, Ls, D) raw
    pmask_student: torch.Tensor,      # (N, Ls)
    opt: torch.optim.Optimizer,
    args,
    epoch: int,
    logger,
    tb=None
) -> Dict[str, float]:
    t0 = time.time()

    # student normalize
    Psb = l2_normalize(Pbar_param)     # (N, Ls, D)

    # teacher score (전체 Q x 전체 P)
    with torch.no_grad():
        sc_t = score_multi_vector_masked(Q_all_norm, P_teacher_norm, qmask, pmask_teacher)

    # student score (전체 Q x 전체 P)
    sc_s = score_multi_vector_masked(Q_all_norm, Psb, qmask, pmask_student)
    loss = 0.5 * (sc_t - sc_s).pow(2).mean()

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    dt = time.time() - t0

    # logging
    if args.print_every:
        logger.info({"epoch": epoch, "step": 1, "loss": float(loss.item()), "time_sec": dt})
        print(f"[train] ep={epoch} loss={loss.item():.6f} t={dt:.1f}s")
    step = epoch
    tb.add_scalar("train/loss", float(loss.item()), step)
    tb.add_scalar("train/time_sec", float(dt), step)
    return {
        "loss": float(loss.item()),
        "last_loss": float(loss.item()),
        "global_step": 1,
        "time_sec": float(dt),
    }


@torch.no_grad()
def eval_retrieval_from_tensors(
    test_payload: Dict[str, Any],
    P_norm: torch.Tensor,            # (N, Lp, D) normalized
    pmask: torch.Tensor,             # (N, Lp)
    device: torch.device,
    eval_batch_size: int,
) -> Dict[str, Any]:
    evaluator = CustomRetrievalEvaluator()

    Q_obj = test_payload["query"]
    q_attn = test_payload["query_attnmask"]
    relevant_docs = test_payload["relevant_docs"]
    docidx_2_docid = test_payload["docidx_2_docid"]
    qsidx_2_query = test_payload["qsidx_2_query"]

    Q_norm, qmask = preprocess_queries(Q_obj, q_attn, device=device)
    with torch.no_grad():
        scores = score_multi_vector_masked(Q_norm, P_norm, qmask, pmask)  # (Nq, Np)

    results = {}
    for qi in range(scores.shape[0]):
        qkey = str(qsidx_2_query[qi]) if qsidx_2_query is not None else str(qi)
        results[qkey] = {docidx_2_docid[str(di)]: float(scores[qi, di].item()) for di in range(scores.shape[1])}

    metrics = evaluator.compute_mteb_metrics(relevant_docs, results)
    return metrics




def tokens_to_object(P_pad_np: np.ndarray, pmask_np: np.ndarray) -> np.ndarray:
    """
    P_pad_np: (N,L,D) float32
    pmask_np: (N,L) bool
    -> object array (N,), each (Li,D) using pmask True positions
    """
    N = P_pad_np.shape[0]
    out = np.empty(N, dtype=object)
    for i in range(N):
        idx = np.where(pmask_np[i])[0]
        out[i] = P_pad_np[i, idx, :].astype(np.float32)
    return out


# =============================================================================
# main
def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )

    for dataset in args.datasets:

        paths = DATASETMAP[dataset]
        train_npz = f'{args.data_root}/{paths["train"]}'
        test_npz = f'{args.data_root}/{paths["test"]}'

        train_payload = load_train_payload(train_npz)
        test_payload = load_test_payload(test_npz)

        # ---------- TEACHER (train split) ----------
        docid_tr = train_payload["docid"]
        P_teacher_obj = train_payload["documents"]
        doc_attn_tr = train_payload["doc_attnmask"]
        doc_img_tr = train_payload["doc_imgmask"]

        Q_train_obj = train_payload["query"]
        q_attn_tr = train_payload["query_attnmask"]

        # preprocess teacher docs / train queries 
        _, P_teacher_norm, pmask_teacher, _valid_teacher = preprocess_docs(
            P_teacher_obj, doc_attn_tr, device=device
        )
        Q_train_norm, qmask = preprocess_queries(Q_train_obj, q_attn_tr, device=device)

        N = P_teacher_norm.shape[0]

        for mf in args.mfs:
            key = f"mf{mf}"
            if key not in paths:
                raise ValueError(f"Missing mapping for {dataset}:{key}")

            init_npz = f'{args.init_root}/{paths[key]}'
            init_payload = load_init_payload(init_npz)

            # ---------- STUDENT INIT ----------
            Pbar_obj = init_payload["documents"]
            doc_attn_in = init_payload["doc_attnmask"]
            doc_img_in = init_payload["doc_imgmask"]

            # optional align by docid
            docid_in = init_payload["docid"]
            if docid_in is not None:
                (Pbar_obj, doc_attn_in, doc_img_in), ok = align_by_docid(
                    _as_object_array(docid_tr),
                    _as_object_array(docid_in),
                    Pbar_obj,
                    doc_attn_in,
                    doc_img_in,
                )
                if ok:
                    print(f"[align] {dataset} mf{mf}: init matched by docid")
            Pbar_raw, _Pbar_norm0, pmask_student, _valid_student = preprocess_docs(
                Pbar_obj, doc_attn_in, device=device
            )
            if Pbar_raw.shape[0] != N:
                raise ValueError(f"init doc count mismatch: got {Pbar_raw.shape[0]} vs teacher {N}")

            Pbar_param = nn.Parameter(Pbar_raw)  # raw param
            opt = set_optimizer(args.opt, Pbar_param, args.lr, args.weight_decay)

            out_dir = Path(args.out_root) / args.name / f"mf{mf}" / dataset
            out_dir.mkdir(parents=True, exist_ok=True)
            logger, tb = get_logger(out_dir)

            cfg_path = out_dir / "config.json"
            if not cfg_path.exists():
                cfg_path.write_text(
                    json.dumps({"dataset": dataset, "mf": mf, **vars(args)}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            print(f"\n[run] out_dir={out_dir}")

            for epoch in range(1, args.epochs + 1):
                # ---------------- train ----------------
                # NOTE: train_epoch가 아래 키들을 같이 반환하도록 수정되어 있어야 함:
                # stats = {"loss":..., "global_step":..., "last_loss":..., "time_sec":...}
                stats = train_epoch(
                    P_teacher_norm=P_teacher_norm,
                    pmask_teacher=pmask_teacher,
                    Q_all_norm=Q_train_norm,
                    qmask=qmask,
                    Pbar_param=Pbar_param,
                    pmask_student=pmask_student,
                    opt=opt,
                    args=args,
                    epoch=epoch,
                    logger=logger,
                    tb=tb
                )

                # ---------------- eval ----------------
                Pbar_now_norm = l2_normalize(Pbar_param.detach())
                metrics = eval_retrieval_from_tensors(
                    test_payload=test_payload,
                    P_norm=Pbar_now_norm,
                    pmask=pmask_student,
                    device=device,
                    eval_batch_size=1,
                )
                step = epoch
                if tb is not None:
                    tb.add_scalar("eval/Recall@1", float(metrics["Recall"]["Recall@1"]), step)
                    tb.add_scalar("eval/NDCG@5", float(metrics["NDCG"]["NDCG@5"]), step)

                r1 = float(metrics["Recall"]["Recall@1"]) * 100.0
                nd5 = float(metrics["NDCG"]["NDCG@5"]) * 100.0
                print("[evaluator metrics]")
                print(f"Recall@1 = {r1:.2f}")
                print(f"nDCG@5    = {nd5:.2f}")

                # ---------------- log (main-level, 안전하게) ----------------
                log_obj = {
                    "dataset": dataset,
                    "mf": mf,
                    "epoch": int(epoch),
                    "train/avg_loss": float(stats.get("loss", 0.0)),
                    "train/last_loss": float(stats.get("last_loss", 0.0)),
                    "train/step": int(stats.get("global_step", 0)),
                    "train/time_sec": float(stats.get("time_sec", 0.0)),
                    "eval/Recall@1": float(metrics["Recall"]["Recall@1"]),
                    "eval/NDCG@5": float(metrics["NDCG"]["NDCG@5"]),
                }
                logger.info(json.dumps(log_obj, ensure_ascii=False))

                # ---------------- save ----------------
                Pbar_now_np = Pbar_now_norm.detach().cpu().numpy().astype(np.float32)
                pmask_np = pmask_student.detach().cpu().numpy().astype(bool)
                docs_obj = tokens_to_object(Pbar_now_np, pmask_np)

                out_npz = out_dir / f"compressed_ep{epoch}.npz"
                if epoch % args.save_period == 0:
                    save_compressed_npz(
                        save_path=out_npz,
                        docid=_as_object_array(docid_tr),
                        documents_obj=docs_obj,
                        doc_attnmask_obj=doc_attn_in,  # 원본(학생) 마스크 그대로
                        doc_imgmask_obj=doc_img_in,
                        meta={
                            "dataset": dataset,
                            "mf": mf,
                            "epoch": epoch,
                            "avg_loss": stats,
                            "eval": {
                                "Recall@1": float(metrics["Recall"]["Recall@1"]),
                                "NDCG@5": float(metrics["NDCG"]["NDCG@5"]),
                            },
                            "lr": args.lr,
                            "loss": "0.5*(SPL(mask=valid&attn&img))^2",
                        },
                    )

            print(f"[done] {dataset} mf{mf} -> {out_dir}")
            if tb is not None:
                tb.flush()
                tb.close()


if __name__ == "__main__":
    main()