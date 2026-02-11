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
    Psb = l2_normalize(Pbar_param * pmask_student.unsqueeze(-1))     # (N, Ls, D)

    # teacher score (전체 Q x 전체 P)
    with torch.no_grad():
        sc_t = score_multi_vector_masked(Q_all_norm, P_teacher_norm, qmask, pmask_teacher, 64)

    # student score (전체 Q x 전체 P)
    sc_s = score_multi_vector_masked(Q_all_norm, Psb, qmask, pmask_student)
    loss = 0.5 * (sc_t - sc_s).pow(2).mean()

    opt.zero_grad(set_to_none=True)
    loss.backward()
    with torch.no_grad():
        g = Pbar_param.grad                      # (N,L,D)
        g_abs = g.abs().amax(dim=-1)             # (N,L)
        m2 = pmask_student                       # (N,L) bool

        g_valid_max   = g_abs[m2].max().item() if g_abs[m2].numel() else 0.0
        g_invalid_max = g_abs[~m2].max().item() if g_abs[~m2].numel() else 0.0
        print(f"[grad] valid max={g_valid_max:.3e} | invalid max={g_invalid_max:.3e}")
    opt.step()
    with torch.no_grad():
        p_abs = Pbar_param.detach().abs().amax(dim=-1)  # (N,L)
        m2 = pmask_student
        p_invalid_max = p_abs[~m2].max().item() if p_abs[~m2].numel() else 0.0
        print(f"[param] invalid abs max after step={p_invalid_max:.3e}")
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
    Q_norm: torch.Tensor,            # (Nq, Lq, D) normalized
    qmask: torch.Tensor,             # (Nq, Lq) bool
    P_norm: torch.Tensor,            # (Np, Lp, D) normalized
    pmask: torch.Tensor,             # (Np, Lp) bool
    relevant_docs: Dict[str, Dict[str, int]],
    docidx_2_docid: Dict[str, str],
    qsidx_2_query,                   # list/np.ndarray or None
) -> Dict[str, Any]:
    evaluator = CustomRetrievalEvaluator()

    scores = score_multi_vector_masked(Q_norm, P_norm, qmask, pmask)  # (Nq, Np)

    results = {}
    for qi in range(scores.shape[0]):
        qkey = str(qsidx_2_query[qi]) if qsidx_2_query is not None else str(qi)
        results[qkey] = {docidx_2_docid[str(di)]: float(scores[qi, di].item())
                         for di in range(scores.shape[1])}

    metrics = evaluator.compute_mteb_metrics(relevant_docs, results)
    return metrics

@torch.no_grad()
def eval_spl_loss(
    P_teacher_norm: torch.Tensor,   # (N, Lt, D) masked+norm
    pmask_teacher: torch.Tensor,    # (N, Lt)
    Q_norm: torch.Tensor,           # (Q, Lq, D)
    qmask: torch.Tensor,            # (Q, Lq)
    Pbar_param: nn.Parameter,       # (N, Ls, D) raw
    pmask_student: torch.Tensor,    # (N, Ls)
    chunk_p: int = 64,
) -> float:
    # student masked+norm
    Psb = l2_normalize(Pbar_param * pmask_student.unsqueeze(-1))

    sc_t = score_multi_vector_masked(Q_norm, P_teacher_norm, qmask, pmask_teacher, chunk_p)
    sc_s = score_multi_vector_masked(Q_norm, Psb,           qmask, pmask_student, chunk_p)

    loss = 0.5 * (sc_t - sc_s).pow(2).mean()
    return float(loss.item())


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

        # ---------- TEACHER (train / test split) ----------
        docid_tr = train_payload["docid"]
        P_teacher_obj = train_payload["documents"]
        doc_attn_tr = train_payload["doc_attnmask"]
        doc_img_tr = train_payload["doc_imgmask"]

        Q_train_obj = train_payload["query"]
        q_attn_tr = train_payload["query_attnmask"] 
        Q_train_norm, qmask = preprocess_queries(Q_train_obj, q_attn_tr, device=device)

        Q_test_obj = test_payload["query"]
        q_attn_test = test_payload["query_attnmask"]
        Q_test_norm, qmask_test = preprocess_queries(Q_test_obj, q_attn_test, device=device)

        relevant_docs_test = test_payload["relevant_docs"]
        docidx_2_docid_test = test_payload["docidx_2_docid"]
        qsidx_2_query_test = test_payload["qsidx_2_query"]

        # preprocess teacher docs / train queries 
        P_teacher_raw, pmask_teacher, _valid_teacher = preprocess_docs(
            P_teacher_obj, doc_attn_tr, doc_img_tr, device=device
        )
        P_teacher_norm = l2_normalize(P_teacher_raw * pmask_teacher.unsqueeze(-1))

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
            Pbar_raw, pmask_student, _valid_student = preprocess_docs(
                Pbar_obj, doc_attn_in,doc_img_in,device=device
            )
            if Pbar_raw.shape[0] != N:
                raise ValueError(f"init doc count mismatch: got {Pbar_raw.shape[0]} vs teacher {N}")

            Pbar_param = nn.Parameter(Pbar_raw*pmask_student.unsqueeze(-1))  # 유효한 문서 토큰만 반영하기 위해 masking
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


            # =========================
            # eval @ step=0 (init Pbar)
            # =========================
            with torch.no_grad():
                Pbar_init_norm = l2_normalize(Pbar_param.detach() * pmask_student.unsqueeze(-1))

            metrics0 = eval_retrieval_from_tensors(
                Q_norm=Q_test_norm,
                qmask=qmask_test,
                P_norm=Pbar_init_norm,
                pmask=pmask_student,
                relevant_docs=relevant_docs_test,
                docidx_2_docid=docidx_2_docid_test,
                qsidx_2_query=qsidx_2_query_test,
            )

            eval_loss0 = eval_spl_loss(
                P_teacher_norm=P_teacher_norm,
                pmask_teacher=pmask_teacher,
                Q_norm=Q_test_norm,
                qmask=qmask_test,
                Pbar_param=Pbar_param,
                pmask_student=pmask_student,
                chunk_p=64,
            )

            step0 = 0
            if tb is not None:
                tb.add_scalar("eval/Recall@1", float(metrics0["Recall"]["Recall@1"]), step0)
                tb.add_scalar("eval/NDCG@5", float(metrics0["NDCG"]["NDCG@5"]), step0)
                tb.add_scalar("eval/loss", float(eval_loss0), step0)

            log0 = {
                "dataset": dataset,
                "mf": mf,
                "epoch": 0,
                "eval/loss": float(eval_loss0),
                "eval/Recall@1": float(metrics0["Recall"]["Recall@1"]),
                "eval/NDCG@5": float(metrics0["NDCG"]["NDCG@5"]),
                "note": "init Pbar before training",
            }
            logger.info(json.dumps(log0, ensure_ascii=False))

            r1_0 = float(metrics0["Recall"]["Recall@1"]) * 100.0
            nd5_0 = float(metrics0["NDCG"]["NDCG@5"]) * 100.0
            print("[evaluator metrics @ init]")
            print(f"Recall@1 = {r1_0:.2f}")
            print(f"nDCG@5    = {nd5_0:.2f}")
            print(f"SPL loss  = {eval_loss0:.6f}")

            
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
                Pbar_now_norm = l2_normalize(Pbar_param.detach() * pmask_student.unsqueeze(-1))
                metrics = eval_retrieval_from_tensors(
                    Q_norm=Q_test_norm,
                    qmask=qmask_test,
                    P_norm=Pbar_now_norm,
                    pmask=pmask_student,
                    relevant_docs=relevant_docs_test,
                    docidx_2_docid=docidx_2_docid_test,
                    qsidx_2_query=qsidx_2_query_test,
                )

                eval_loss = eval_spl_loss(
                    P_teacher_norm=P_teacher_norm,
                    pmask_teacher=pmask_teacher,
                    Q_norm=Q_test_norm,
                    qmask=qmask_test,
                    Pbar_param=Pbar_param,
                    pmask_student=pmask_student,
                    chunk_p=64,
                )
                step = epoch
                if tb is not None:
                    tb.add_scalar("eval/Recall@1", float(metrics["Recall"]["Recall@1"]), step)
                    tb.add_scalar("eval/NDCG@5", float(metrics["NDCG"]["NDCG@5"]), step)
                    tb.add_scalar("eval/loss", float(eval_loss), step)

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
                    "eval/loss": float(eval_loss),
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