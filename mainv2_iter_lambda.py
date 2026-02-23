import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

from utils.utils import (
    set_seed, set_optimizer, get_logger, align_by_docid,
    save_compressed_npz, tokens_to_object, log_json
)
from utils.preprocess_data import (
    l2_normalize, load_query_payload, load_payload, load_init_payload,
    _as_object_array, preprocess_docs, preprocess_queries
)
from utils.mapping import DATASETMAP
from evaluator.retrieval import score_multi_vector_masked, CustomRetrievalEvaluator
from Qdatasets.query_tensor_dataset import QueryTensorDataset
from torch.utils.data import DataLoader
from criterion import lambda_loss


# =============================================================================
# args
# =============================================================================
def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", type=str, nargs="+", required=True)
    p.add_argument("--query_root", type=str, default="ProxyQ/results0212/colqwen")
    p.add_argument("--teacher_root", type=str, default="/home/hyshin/Project/EVDR/data/vidore_test/0221_features/colqwen")
    p.add_argument("--init_root", type=str, default="/home/hyshin/Project/EVDR/data/vidore_test/0221_features/colqwen/S3E_init")
    p.add_argument("--mfs", type=int, nargs="+", default=[5, 10, 25, 50])
    p.add_argument("--out_root", type=str, default="results")
    p.add_argument("--name", type=str, default="lambda_loss")

    p.add_argument("--max_steps", type=int, default=23460)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--q_batch", type=int, default=32)
    p.add_argument("--opt", type=str, default="adamw")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--print_every", type=int, default=20)

    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    return p

# =============================================================================
# main
# =============================================================================
def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)

    for dataset in args.datasets:
        paths = DATASETMAP[dataset]
        train_query_npz = f"{args.query_root}/{paths['pseudoQ']}"
        teacher_npz = f"{args.teacher_root}/{paths['split_before']}"

        train_query_payload = load_query_payload(train_query_npz)
        teacher_payload = load_payload(teacher_npz)

        # teacher
        docid_tr = teacher_payload["docid"]
        P_teacher_obj = teacher_payload["documents"]
        doc_attn_tr = teacher_payload["doc_attnmask"]
        doc_img_tr = teacher_payload["doc_imgmask"]

        # train queries
        Q_train_obj = train_query_payload["query"]
        q_attn_tr = train_query_payload["query_attnmask"]
        Q_train_norm, qmask = preprocess_queries(Q_train_obj, q_attn_tr, device="cpu")

        train_ds = QueryTensorDataset(Q_train_norm, qmask)
        train_dl = DataLoader(train_ds, batch_size=args.q_batch, shuffle=True, drop_last=False, num_workers=0, pin_memory=True)

        # test queries
        Q_test_obj = teacher_payload["query"]
        q_attn_test = teacher_payload["query_attnmask"]
        Q_test_norm, qmask_test = preprocess_queries(Q_test_obj, q_attn_test, device=device)

        relevant_docs_test = teacher_payload["relevant_docs"]
        docidx_2_docid_test = teacher_payload["docidx_2_docid"]
        qsidx_2_query_test = teacher_payload["qsidx_2_query"]

        # teacher docs -> masked+norm
        P_teacher_raw, pmask_teacher, _ = preprocess_docs(P_teacher_obj, doc_attn_tr, doc_img_tr, device=device)
        P_teacher_norm = l2_normalize(P_teacher_raw * pmask_teacher.unsqueeze(-1)).detach()
        N = P_teacher_norm.shape[0]

        eval_every = args.eval_every if args.eval_every and args.eval_every > 0 else len(train_dl)
        eval_every = max(int(eval_every), 1)

        for mf in args.mfs:
            key = f"mf{mf}"
            if key not in paths:
                raise ValueError(f"Missing mapping for {dataset}:{key}")

            init_npz = f"{args.init_root}/{paths[key]}"
            init_payload = load_init_payload(init_npz)

            # student init
            Pbar_obj = init_payload["documents"]
            doc_attn_in = init_payload["doc_attnmask"]
            doc_img_in = init_payload["doc_imgmask"]

            docid_in = init_payload.get("docid", None)
            if docid_in is not None:
                (Pbar_obj, doc_attn_in, doc_img_in), ok = align_by_docid(
                    _as_object_array(docid_tr), _as_object_array(docid_in),
                    Pbar_obj, doc_attn_in, doc_img_in,
                )
                if ok:
                    print(f"[align] {dataset} mf{mf}: init matched by docid")

            Pbar_raw, pmask_student, _ = preprocess_docs(Pbar_obj, doc_attn_in, doc_img_in, device=device)
            if Pbar_raw.shape[0] != N:
                raise ValueError(f"init doc count mismatch: got {Pbar_raw.shape[0]} vs teacher {N}")

            Pbar_param = nn.Parameter(Pbar_raw * pmask_student.unsqueeze(-1))
            opt = set_optimizer(args.opt, Pbar_param, args.lr, args.weight_decay)

            out_dir = Path(args.out_root) / args.name / f"mf{mf}" / dataset
            out_dir.mkdir(parents=True, exist_ok=True)
            logger, tb = get_logger(out_dir)

            cfg_path = out_dir / "config.json"
            if not cfg_path.exists():
                cfg_path.write_text(json.dumps({"dataset": dataset, "mf": mf, **vars(args)}, ensure_ascii=False, indent=2), encoding="utf-8")

            evaluator = CustomRetrievalEvaluator()

            # -----------------------------
            # step=0 eval
            # -----------------------------
            metrics0 = eval_retrieval(
                evaluator=evaluator,
                Q_test_norm=Q_test_norm, qmask_test=qmask_test,
                Pbar_param=Pbar_param, pmask_student=pmask_student,
                relevant_docs_test=relevant_docs_test,
                docidx_2_docid_test=docidx_2_docid_test,
                qsidx_2_query_test=qsidx_2_query_test,
                chunk_p=64,
            )
            loss0 = evaluation_loss(
                Q_test_norm=Q_test_norm, qmask_test=qmask_test,
                P_teacher_norm=P_teacher_norm, pmask_teacher=pmask_teacher,
                Pbar_param=Pbar_param, pmask_student=pmask_student,
                alpha=args.alpha, eps=args.eps, chunk_p=64,
            )
            log_eval(logger, tb, dataset=dataset, mf=mf, step=0, metrics=metrics0, loss=loss0)
            log_json(logger, {"dataset": dataset, "mf": mf, "step": 0, "note": "init Pbar before training"})

            best_r1, _ = update_best(None, metrics0, 0, "r1")
            best_nd5, _ = update_best(None, metrics0, 0, "nd5")

            last_metrics = metrics0

            # -----------------------------
            # training loop
            # -----------------------------
            train_iter = iter(train_dl)
            t0 = time.time()
            loss_sum, loss_cnt = 0.0, 0

            for step in range(1, args.max_steps + 1):
                # dataloader가 끝나면 다시 처음부터(“max_steps” 기반 학습이니까)
                try:
                    Qb, qmb = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_dl)
                    Qb, qmb = next(train_iter)

                loss_val = train_one_step(
                    Qb, qmb,
                    P_teacher_norm, pmask_teacher,
                    Pbar_param, pmask_student,
                    opt,
                    alpha=args.alpha, eps=args.eps,
                    chunk_p=64,
                )

                loss_sum += loss_val
                loss_cnt += 1

                if tb is not None:
                    tb.add_scalar("train/loss", float(loss_val), step)

                if args.print_every and (step % args.print_every == 0):
                    dt = time.time() - t0
                    avg_loss = loss_sum / max(loss_cnt, 1)
                    log_json(logger, {
                        "dataset": dataset, "mf": mf, "step": step,
                        "train/loss": float(loss_val),
                        "train/avg_loss": float(avg_loss),
                        "time_sec": float(dt),
                    })
                    print(f"[train] step={step}/{args.max_steps} loss={loss_val:.6f} avg={avg_loss:.6f} t={dt:.1f}s")

                if (step % eval_every == 0) or (step == args.max_steps):
                    metrics = eval_retrieval(
                        evaluator=evaluator,
                        Q_test_norm=Q_test_norm, qmask_test=qmask_test,
                        Pbar_param=Pbar_param, pmask_student=pmask_student,
                        relevant_docs_test=relevant_docs_test,
                        docidx_2_docid_test=docidx_2_docid_test,
                        qsidx_2_query_test=qsidx_2_query_test,
                        chunk_p=64,
                    )
                    ev_loss = evaluation_loss(
                        Q_test_norm=Q_test_norm, qmask_test=qmask_test,
                        P_teacher_norm=P_teacher_norm, pmask_teacher=pmask_teacher,
                        Pbar_param=Pbar_param, pmask_student=pmask_student,
                        alpha=args.alpha, eps=args.eps, chunk_p=64,
                    )
                    log_eval(logger, tb, dataset=dataset, mf=mf, step=step, metrics=metrics, loss=ev_loss)
                    last_metrics = metrics
                    best_r1, upd_r1 = update_best(best_r1, metrics, step, "r1")
                    best_nd5, upd_nd5 = update_best(best_nd5, metrics, step, "nd5")

                    if upd_r1:
                        logger.info(f"best recall step| {step} | nDCG@5={best_r1['NDCG@5']:.5f} | "
                                    f"Recall@1={best_r1['Recall@1']:.5f} | Latency {metrics['latency']:.5f}")
                        save_best_npz(
                            out_dir=out_dir, fname="best_recall.npz",
                            dataset=dataset, mf=mf, step=step,
                            best=best_r1, metrics=metrics,
                            Pbar_param=Pbar_param, pmask_student=pmask_student,
                            docid_tr=docid_tr, doc_attn_in=doc_attn_in, doc_img_in=doc_img_in,
                            args=args,
                        )

                    if upd_nd5:
                        logger.info(f"best nDCG@5 step| {step} | nDCG@5={best_nd5['NDCG@5']:.5f} | "
                                    f"Recall@1={best_nd5['Recall@1']:.5f} | Latency {metrics['latency']:.5f}")
                        save_best_npz(
                            out_dir=out_dir, fname="best_ndcg5.npz",
                            dataset=dataset, mf=mf, step=step,
                            best=best_nd5, metrics=metrics,
                            Pbar_param=Pbar_param, pmask_student=pmask_student,
                            docid_tr=docid_tr, doc_attn_in=doc_attn_in, doc_img_in=doc_img_in,
                            args=args,
                        )

            # -----------------------------
            # final summary line
            # -----------------------------
            log_json(logger, {
                "summary/latency": float(last_metrics.get("latency", 0.0)),
                "summary/best_recall": best_r1,
                "summary/best_ndcg5": best_nd5,
                "note": "training finished",
            })

            print(f"[done] {dataset} mf{mf} -> {out_dir}")
            if tb is not None:
                tb.flush()
                tb.close()

# =============================================================================
# train primitive
# =============================================================================
def train_one_step(
    Qb: torch.Tensor,qmb: torch.Tensor,P_teacher_norm: torch.Tensor,pmask_teacher: torch.Tensor,Pbar_param: nn.Parameter,
    pmask_student: torch.Tensor,opt: torch.optim.Optimizer,alpha: float,eps: float,chunk_p: int = 64,
) -> float:
    """1 step 업데이트 후 loss 반환"""
    device = Pbar_param.device
    Qb = Qb.to(device, non_blocking=True)
    qmb = qmb.to(device, non_blocking=True)

    # student (masked + normalized)
    Psb = l2_normalize(Pbar_param * pmask_student.unsqueeze(-1))

    # teacher score (no grad)
    with torch.no_grad():
        sc_t = score_multi_vector_masked(Qb, P_teacher_norm, qmb, pmask_teacher, chunk_p)

    # student score
    sc_s = score_multi_vector_masked(Qb, Psb, qmb, pmask_student, chunk_p)
    loss = lambda_loss(sc_s, sc_t, alpha=alpha, eps=eps)

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return float(loss.item())


# =============================================================================
# eval primitives (ctx 제거)
# =============================================================================
@torch.no_grad()
def eval_retrieval(
    evaluator: CustomRetrievalEvaluator,Q_test_norm: torch.Tensor,qmask_test: torch.Tensor,Pbar_param: nn.Parameter,pmask_student: torch.Tensor,
    relevant_docs_test: Dict[str, Dict[str, int]],docidx_2_docid_test: Dict[str, str],
    qsidx_2_query_test,chunk_p: int = 64,
) -> Dict[str, Any]:
    """retrieval metrics (Recall/NDCG/latency)"""
    P_now = l2_normalize(Pbar_param.detach() * pmask_student.unsqueeze(-1))

    t0 = time.time()
    scores = score_multi_vector_masked(Q_test_norm, P_now, qmask_test, pmask_student, chunk_p=chunk_p)
    latency_ms = (time.time() - t0) * 1000 / max(Q_test_norm.shape[0], 1)

    results = {}
    for qi in range(scores.shape[0]):
        qkey = str(qsidx_2_query_test[qi]) if qsidx_2_query_test is not None else str(qi)
        results[qkey] = {
            docidx_2_docid_test[str(di)]: float(scores[qi, di].item())
            for di in range(scores.shape[1])
        }

    metrics = evaluator.compute_mteb_metrics(relevant_docs_test, results)
    metrics["latency"] = float(latency_ms)
    return metrics


@torch.no_grad()
def evaluation_loss(
    Q_test_norm: torch.Tensor,
    qmask_test: torch.Tensor,
    P_teacher_norm: torch.Tensor,
    pmask_teacher: torch.Tensor,
    Pbar_param: nn.Parameter,
    pmask_student: torch.Tensor,
    alpha: float,
    eps: float,
    chunk_p: int = 64,
) -> float:
    """lambda loss on test queries"""
    Psb = l2_normalize(Pbar_param * pmask_student.unsqueeze(-1))

    sc_t = score_multi_vector_masked(Q_test_norm, P_teacher_norm, qmask_test, pmask_teacher, chunk_p=chunk_p)
    torch.cuda.empty_cache()
    sc_s = score_multi_vector_masked(Q_test_norm, Psb, qmask_test, pmask_student, chunk_p=chunk_p)
    torch.cuda.empty_cache()

    loss = lambda_loss(sc_s, sc_t, alpha=alpha, eps=eps)
    return float(loss.item())


def log_eval(
    logger,
    tb,
    *,
    dataset: str,
    mf: int,
    step: int,
    metrics: Dict[str, Any],
    loss: float,
):
    if tb is not None:
        tb.add_scalar("eval/Recall@1", float(metrics["Recall"]["Recall@1"]), step)
        tb.add_scalar("eval/NDCG@5", float(metrics["NDCG"]["NDCG@5"]), step)
        tb.add_scalar("eval/loss", float(loss), step)

    log_json(logger, {
        "dataset": dataset,
        "mf": mf,
        "step": int(step),
        "eval/loss": float(loss),
        "eval/Recall@1": float(metrics["Recall"]["Recall@1"]),
        "eval/NDCG@5": float(metrics["NDCG"]["NDCG@5"]),
        "eval/latency": float(metrics["latency"]),
    })


def update_best(best: Optional[Dict[str, Any]],metrics: Dict[str, Any],step: int, kind: str) -> Tuple[Dict[str, Any], bool]:
    """
    kind='r1' or 'nd5'
    return: (new_best, updated?)
    """
    cur_r1 = float(metrics["Recall"]["Recall@1"])
    cur_nd5 = float(metrics["NDCG"]["NDCG@5"])

    if best is None:
        return {"step": step, "Recall@1": cur_r1, "NDCG@5": cur_nd5}, True

    if kind == "r1":
        updated = (cur_r1 > best["Recall@1"]) or (cur_r1 == best["Recall@1"] and cur_nd5 > best["NDCG@5"])
    else:  # 'nd5'
        updated = (cur_nd5 > best["NDCG@5"]) or (cur_nd5 == best["NDCG@5"] and cur_r1 > best["Recall@1"])

    if not updated:
        return best, False

    return {"step": step, "Recall@1": cur_r1, "NDCG@5": cur_nd5}, True

def save_best_npz(
    *,
    out_dir: Path,fname: str,dataset: str,mf: int,step: int,best: Dict[str, Any],metrics: Dict[str, Any],
    Pbar_param: nn.Parameter,pmask_student: torch.Tensor,docid_tr,doc_attn_in,doc_img_in,
    args,
):
    P_now = Pbar_param.detach() * pmask_student.unsqueeze(-1)
    P_np = P_now.cpu().numpy().astype(np.float32)
    pmask_np = pmask_student.detach().cpu().numpy().astype(bool)
    docs_obj = tokens_to_object(P_np, pmask_np)

    save_compressed_npz(
        save_path=out_dir / fname,
        docid=_as_object_array(docid_tr),
        documents_obj=docs_obj,
        doc_attnmask_obj=doc_attn_in,
        doc_imgmask_obj=doc_img_in,
        meta={
            "dataset": dataset,
            "mf": mf,
            "step": int(step),
            "best_type": "Recall@1" if fname == "best_recall.npz" else "NDCG@5",
            "best": best,
            "eval": {
                "Recall@1": float(metrics["Recall"]["Recall@1"]),
                "NDCG@5": float(metrics["NDCG"]["NDCG@5"]),
            },
            "latency": float(metrics["latency"]),
            "loss": "lambda loss",
            "alpha": args.alpha,
            "eps": args.eps,
            "lr": args.lr,
        },
    )



if __name__ == "__main__":
    main()