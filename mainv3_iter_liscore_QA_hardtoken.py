# mainv3_iter_liscore_QA_hardtoken.py
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

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
from criterion import listwise_distillation_loss, score_preserving_loss


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
    p.add_argument("--name", type=str, default="liscroe loss ")

    p.add_argument("--max_steps", type=int, default=23460)
    p.add_argument("--eval_every", type=int, default=200)
    p.add_argument("--q_batch", type=int, default=32)
    p.add_argument("--opt", type=str, default="adamw")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--lambda_list", type=float, default=1.0)
    p.add_argument("--lambda_score", type=float, default=0.01)
    p.add_argument("--virt_noise_std", type=float, default=0.1)
    p.add_argument("--lambda_aux", type=float, default=0.3,
                   help="Aux loss weight. total = main + lambda_aux * aux (aux uses same lambda_list/lambda_score).")
    p.add_argument("--aux_docs", type=int, default=4, help="How many top-gap docs to use for virtual-query aux per step.")
    p.add_argument("--k", type=int, default=40)
    p.add_argument("--temp", type=float, default=0.1)
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
        train_dl = DataLoader(
            train_ds, batch_size=args.q_batch, shuffle=True, drop_last=False, num_workers=0, pin_memory=True
        )

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
                cfg_path.write_text(
                    json.dumps({"dataset": dataset, "mf": mf, **vars(args)}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

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
                chunk_p=32,
            )
            loss0 = evaluation_loss(
                Q_test_norm=Q_test_norm, qmask_test=qmask_test,
                P_teacher_norm=P_teacher_norm, pmask_teacher=pmask_teacher,
                Pbar_param=Pbar_param, pmask_student=pmask_student,
                k=args.k, temp=args.temp, chunk_p=32,
                lambda_list=args.lambda_list,
                lambda_score=args.lambda_score,
            )
            log_eval(logger, tb, dataset=dataset, mf=mf, step=0, metrics=metrics0, loss=loss0)
            log_json(logger, {"dataset": dataset, "mf": mf, "step": 0, "note": "init Pbar before training"})

            best_r1, _ = update_best(None, metrics0, 0, "r1")
            best_nd5, _ = update_best(None, metrics0, 0, "nd5")
            last_metrics = metrics0

            # -----------------------------
            # gaplog (하드코딩: 500 step / top10)
            # -----------------------------
            GAPLOG_EVERY = 500
            GAPLOG_TOPK = 10
            gap_doc_sum = defaultdict(float)
            gap_doc_cnt = defaultdict(int)

            # -----------------------------
            # training loop
            # -----------------------------
            train_iter = iter(train_dl)
            t0 = time.time()
            loss_sum, loss_cnt = 0.0, 0

            for step in range(1, args.max_steps + 1):
                try:
                    Qb, qmb = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_dl)
                    Qb, qmb = next(train_iter)

                loss_dict = train_one_step(
                    Qb, qmb,
                    P_teacher_norm, pmask_teacher,
                    Pbar_param, pmask_student,
                    opt,
                    k=args.k, temp=args.temp,
                    chunk_p=32,
                    lambda_list=args.lambda_list,
                    lambda_score=args.lambda_score,
                    lambda_aux=args.lambda_aux,
                    virt_noise_std = args.virt_noise_std,
                    aux_docs=args.aux_docs
                )

                loss_val = float(loss_dict["total_loss"])
                loss_sum += loss_val
                loss_cnt += 1

                # gap 누적(top20만) + 500 step마다 top10 로깅
                for di, gv in zip(loss_dict.get("gap_top_idx", []), loss_dict.get("gap_top_val", [])):
                    di = int(di)
                    gap_doc_sum[di] += float(gv)
                    gap_doc_cnt[di] += 1

                if step % GAPLOG_EVERY == 0:
                    top = sorted(gap_doc_sum.items(), key=lambda x: x[1], reverse=True)[:GAPLOG_TOPK]
                    top_list = []
                    for di, s in top:
                        docid = docidx_2_docid_test.get(str(di), str(di))
                        top_list.append({"doc_idx": di, "docid": docid, "gap_sum": float(s), "seen": int(gap_doc_cnt[di])})
                    log_json(logger, {"dataset": dataset, "mf": mf, "step": step, "gaplog/top_docs": top_list})

                if tb is not None:
                    tb.add_scalar("train/loss", float(loss_dict["total_loss"]), step)
                    tb.add_scalar("train/loss_list", float(loss_dict["loss_list"]), step)
                    tb.add_scalar("train/loss_score", float(loss_dict["loss_score"]), step)
                    tb.add_scalar("train/loss_list_aux", float(loss_dict["loss_list_aux"]), step)
                    tb.add_scalar("train/loss_score_aux", float(loss_dict["loss_score_aux"]), step)
                    tb.add_scalar("train/loss_main", float(loss_dict["loss_main"]), step)
                    tb.add_scalar("train/loss_aux", float(loss_dict["loss_aux"]), step)

                if args.print_every and (step % args.print_every == 0):
                    dt = time.time() - t0
                    avg_loss = loss_sum / max(loss_cnt, 1)
                    log_json(logger, {
                        "dataset": dataset, "mf": mf, "step": step,
                        "train/total_loss": float(loss_dict["total_loss"]),
                        "train/loss_list": float(loss_dict["loss_list"]),
                        "train/loss_score": float(loss_dict["loss_score"]),
                        "train/loss_list_aux": float(loss_dict["loss_list_aux"]),
                        "train/loss_score_aux": float(loss_dict["loss_score_aux"]),
                        "train/loss_main": float(loss_dict["loss_main"]),
                        "train/loss_aux": float(loss_dict["loss_aux"]),
                        "train/avg_total_loss": float(avg_loss),
                        "time_sec": float(dt),
                    })
                    print(
                        f"[train] step={step}/{args.max_steps} "
                        f"total={float(loss_dict['total_loss']):.6f} "
                        f"main={float(loss_dict['loss_main']):.6f} aux={float(loss_dict['loss_aux']):.6f} "
                        f"list={float(loss_dict['loss_list']):.6f} score={float(loss_dict['loss_score']):.6f} "
                        f"alist={float(loss_dict['loss_list_aux']):.6f} ascore={float(loss_dict['loss_score_aux']):.6f} "
                        f"t={dt:.1f}s"
                    )

                if (step % eval_every == 0) or (step == args.max_steps):
                    metrics = eval_retrieval(
                        evaluator=evaluator,
                        Q_test_norm=Q_test_norm, qmask_test=qmask_test,
                        Pbar_param=Pbar_param, pmask_student=pmask_student,
                        relevant_docs_test=relevant_docs_test,
                        docidx_2_docid_test=docidx_2_docid_test,
                        qsidx_2_query_test=qsidx_2_query_test,
                        chunk_p=32,
                    )
                    ev_loss = evaluation_loss(
                        Q_test_norm=Q_test_norm, qmask_test=qmask_test,
                        P_teacher_norm=P_teacher_norm, pmask_teacher=pmask_teacher,
                        Pbar_param=Pbar_param, pmask_student=pmask_student,
                        k=args.k, temp=args.temp, chunk_p=32,
                        lambda_list=args.lambda_list,
                        lambda_score=args.lambda_score,
                    )
                    log_eval(logger, tb, dataset=dataset, mf=mf, step=step, metrics=metrics, loss=ev_loss)
                    last_metrics = metrics
                    best_r1, upd_r1 = update_best(best_r1, metrics, step, "r1")
                    best_nd5, upd_nd5 = update_best(best_nd5, metrics, step, "nd5")

                    if upd_r1:
                        logger.info(
                            f"best recall step| {step} | nDCG@5={best_r1['NDCG@5']:.5f} | "
                            f"Recall@1={best_r1['Recall@1']:.5f} | Latency {metrics['latency']:.5f}"
                        )
                        save_best_npz(
                            out_dir=out_dir, fname="best_recall.npz",
                            dataset=dataset, mf=mf, step=step,
                            best=best_r1, metrics=metrics,
                            Pbar_param=Pbar_param, pmask_student=pmask_student,
                            docid_tr=docid_tr, doc_attn_in=doc_attn_in, doc_img_in=doc_img_in,
                            args=args,
                        )

                    if upd_nd5:
                        logger.info(
                            f"best nDCG@5 step| {step} | nDCG@5={best_nd5['NDCG@5']:.5f} | "
                            f"Recall@1={best_nd5['Recall@1']:.5f} | Latency {metrics['latency']:.5f}"
                        )
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
# train primitive (virtual query + aux)
# =============================================================================
def train_one_step(
    Qb: torch.Tensor, qmb: torch.Tensor,
    P_teacher_norm: torch.Tensor, pmask_teacher: torch.Tensor,
    Pbar_param: nn.Parameter, pmask_student: torch.Tensor,
    opt: torch.optim.Optimizer,
    k: int, temp: float, chunk_p: int = 64,
    lambda_list: float = 1.0, lambda_score: float = 1.0,
    lambda_aux: float = 0.3, virt_noise_std: float=0.0,
    aux_docs: int = 4,
) -> Dict:
    """
    total = main + lambda_aux * aux
    main = lambda_list*L_list + lambda_score*L_score  (Qb)
    aux  = lambda_list*L_list_aux + lambda_score*L_score_aux (virtual query)
    """
    GAP_TOPK_RETURN = 20

    device = Pbar_param.device
    Qb = Qb.to(device, non_blocking=True)
    qmb = qmb.to(device, non_blocking=True)

    # student (masked + normalized)
    Psb = l2_normalize(Pbar_param * pmask_student.unsqueeze(-1))

    # teacher/student scores (main)
    with torch.no_grad():
        sc_t = score_multi_vector_masked(Qb, P_teacher_norm, qmb, pmask_teacher, chunk_p)
    sc_s = score_multi_vector_masked(Qb, Psb, qmb, pmask_student, chunk_p)

    # rank-gap + topK for logging + pick hard (doc*, q*)
    with torch.no_grad():
        rank_t = torch.argsort(torch.argsort(sc_t, dim=-1, descending=True), dim=-1)  # (B,N)
        rank_s = torch.argsort(torch.argsort(sc_s, dim=-1, descending=True), dim=-1)  # (B,N)
        gap = (rank_t.float() - rank_s.float())  # (B,N)
        G = gap.abs().sum(dim=0)                       # (N,)

        k_top = min(int(GAP_TOPK_RETURN), G.numel())
        gap_top_val, gap_top_idx = torch.topk(G, k=k_top)

        kk = min(int(k), sc_t.shape[1])
        topk_idx = torch.topk(sc_t, k=kk, dim=-1, largest=True).indices
        gap_topk = gap.gather(1, topk_idx).abs()
        a = min(int(aux_docs), kk)
        if a > 0:
            aux_pos = torch.topk(gap_topk, k=a, dim=1, largest=True).indices
            aux_doc_idx_q = topk_idx.gather(1, aux_pos)
        else:
            aux_doc_idx_q = torch.empty((Qb.shape[0], 0), device=device, dtype=torch.long)

    # main losses
    loss_list = listwise_distillation_loss(sc_s, sc_t, k=k, temperature=temp)
    loss_score = score_preserving_loss(sc_s, sc_t)
    loss_main = lambda_list * loss_list + lambda_score * loss_score

    # aux defaults
    loss_list_aux = 0.0 * loss_list
    loss_score_aux = 0.0 * loss_score
    aux_used = 0

    # hard docs 있을 때만 virtual queries 만들고 aux 계산
    if aux_doc_idx_q.numel() > 0:
        q_virtual_list = []
        with torch.no_grad():
            B = Qb.shape[0]
            for q_i in range(B):
                q_tokens = Qb[q_i]
                q_mask = qmb[q_i].bool()
                q_tokens = q_tokens[q_mask]

                for doc_i in aux_doc_idx_q[q_i].tolist():
                    doc_i = int(doc_i)

                    doc_tok = P_teacher_norm[doc_i]
                    doc_mask = pmask_teacher[doc_i].bool()

                    sim = q_tokens @ doc_tok.T
                    sim[:, ~doc_mask] = float("-inf")

                    max_over_q = sim.max(dim=0).values
                    best_tok_idx = torch.argmax(max_over_q)
                    hard_tok = doc_tok[best_tok_idx]       # (D,)

                    # virtual query = hard token (+noise)
                    qv = hard_tok
                    if virt_noise_std and virt_noise_std > 0:
                        qv = qv + torch.randn_like(qv) * virt_noise_std
                    qv = l2_normalize(qv).view(1, 1, -1)
                    q_virtual_list.append(qv)

        if len(q_virtual_list) > 0:
            q_virtual = torch.cat(q_virtual_list, dim=0)
            qmask_v = torch.ones(q_virtual.shape[0], 1, device=device, dtype=torch.bool)
            aux_used = int(q_virtual.shape[0])

            with torch.no_grad():
                sc_t_v = score_multi_vector_masked(q_virtual, P_teacher_norm, qmask_v, pmask_teacher, chunk_p)
            sc_s_v = score_multi_vector_masked(q_virtual, Psb, qmask_v, pmask_student, chunk_p)

            loss_list_aux = listwise_distillation_loss(sc_s_v, sc_t_v, k=k, temperature=temp)
            loss_score_aux = score_preserving_loss(sc_s_v, sc_t_v)

    loss_aux = lambda_list * loss_list_aux + lambda_score * loss_score_aux
    total_loss = loss_main + lambda_aux * loss_aux

    opt.zero_grad(set_to_none=True)
    total_loss.backward()
    opt.step()

    return {
        "total_loss": float(total_loss.item()),
        "loss_list": float(loss_list.item()),
        "loss_score": float(loss_score.item()),
        "loss_list_aux": float(loss_list_aux.item()),
        "loss_score_aux": float(loss_score_aux.item()),
        "loss_main": float(loss_main.item()),
        "loss_aux": float(loss_aux.item()),
        "aux_used": aux_used,
        "gap_top_idx": gap_top_idx.detach().cpu().tolist(),
        "gap_top_val": gap_top_val.detach().cpu().tolist(),
    }


# =============================================================================
# eval primitives (ctx 제거)
# =============================================================================
@torch.no_grad()
def eval_retrieval(
    evaluator: CustomRetrievalEvaluator, Q_test_norm: torch.Tensor, qmask_test: torch.Tensor,
    Pbar_param: nn.Parameter, pmask_student: torch.Tensor,
    relevant_docs_test: Dict[str, Dict[str, int]], docidx_2_docid_test: Dict[str, str],
    qsidx_2_query_test, chunk_p: int = 64,
) -> Dict[str, Any]:
    P_now = l2_normalize(Pbar_param.detach() * pmask_student.unsqueeze(-1))

    t0 = time.time()
    scores = score_multi_vector_masked(Q_test_norm, P_now, qmask_test, pmask_student, chunk_p=chunk_p)
    latency_ms = (time.time() - t0) * 1000 / max(Q_test_norm.shape[0], 1)

    results = {}
    for qi in range(scores.shape[0]):
        qkey = str(qsidx_2_query_test[qi]) if qsidx_2_query_test is not None else str(qi)
        results[qkey] = {docidx_2_docid_test[str(di)]: float(scores[qi, di].item())
                         for di in range(scores.shape[1])}

    metrics = evaluator.compute_mteb_metrics(relevant_docs_test, results)
    metrics["latency"] = float(latency_ms)
    return metrics


@torch.no_grad()
def evaluation_loss(
    Q_test_norm: torch.Tensor, qmask_test: torch.Tensor,
    P_teacher_norm: torch.Tensor, pmask_teacher: torch.Tensor,
    Pbar_param: nn.Parameter, pmask_student: torch.Tensor,
    k: int, temp: float, chunk_p: int = 64,
    lambda_list: float = 1.0, lambda_score: float = 1.0
) -> Dict:
    Psb = l2_normalize(Pbar_param * pmask_student.unsqueeze(-1))

    sc_t = score_multi_vector_masked(Q_test_norm, P_teacher_norm, qmask_test, pmask_teacher, chunk_p=chunk_p)
    torch.cuda.empty_cache()
    sc_s = score_multi_vector_masked(Q_test_norm, Psb, qmask_test, pmask_student, chunk_p=chunk_p)
    torch.cuda.empty_cache()

    loss_list = listwise_distillation_loss(sc_s, sc_t, k=k, temperature=temp)
    loss_score = score_preserving_loss(sc_s, sc_t)

    total_loss = lambda_list * loss_list + lambda_score * loss_score
    return {"total_loss": total_loss.item(), "loss_list": loss_list.item(), "loss_score": loss_score.item()}


def log_eval(
    logger, tb, *, dataset: str, mf: int, step: int,
    metrics: Dict[str, Any], loss: Dict[str, float],
):
    total = float(loss["total_loss"])
    l_list = float(loss["loss_list"])
    l_score = float(loss["loss_score"])

    if tb is not None:
        tb.add_scalar("eval/Recall@1", float(metrics["Recall"]["Recall@1"]), step)
        tb.add_scalar("eval/NDCG@5", float(metrics["NDCG"]["NDCG@5"]), step)
        tb.add_scalar("eval/loss", total, step)
        tb.add_scalar("eval/loss_list", l_list, step)
        tb.add_scalar("eval/loss_score", l_score, step)

    log_json(logger, {
        "dataset": dataset, "mf": mf, "step": int(step),
        "eval/eval loss": total,
        "eval/loss_list": l_list,
        "eval/loss_score": l_score,
        "eval/Recall@1": float(metrics["Recall"]["Recall@1"]),
        "eval/NDCG@5": float(metrics["NDCG"]["NDCG@5"]),
        "eval/latency": float(metrics["latency"]),
    })


def update_best(best: Optional[Dict[str, Any]], metrics: Dict[str, Any], step: int, kind: str) -> Tuple[Dict[str, Any], bool]:
    cur_r1 = float(metrics["Recall"]["Recall@1"])
    cur_nd5 = float(metrics["NDCG"]["NDCG@5"])

    if best is None:
        return {"step": step, "Recall@1": cur_r1, "NDCG@5": cur_nd5}, True

    if kind == "r1":
        updated = (cur_r1 > best["Recall@1"]) or (cur_r1 == best["Recall@1"] and cur_nd5 > best["NDCG@5"])
    else:
        updated = (cur_nd5 > best["NDCG@5"]) or (cur_nd5 == best["NDCG@5"] and cur_r1 > best["Recall@1"])

    if not updated:
        return best, False

    return {"step": step, "Recall@1": cur_r1, "NDCG@5": cur_nd5}, True


def save_best_npz(
    *,
    out_dir: Path, fname: str, dataset: str, mf: int, step: int,
    best: Dict[str, Any], metrics: Dict[str, Any],
    Pbar_param: nn.Parameter, pmask_student: torch.Tensor,
    docid_tr, doc_attn_in, doc_img_in,
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
            "eval": {"Recall@1": float(metrics["Recall"]["Recall@1"]), "NDCG@5": float(metrics["NDCG"]["NDCG@5"])},
            "latency": float(metrics["latency"]),
            "loss": "liscore loss",
            "k": args.k,
            "temp": args.temp,
            "lambda list": args.lambda_list,
            "lambda score": args.lambda_score,
            "lambda aux": args.lambda_aux,
            "lr": args.lr,
        },
    )


if __name__ == "__main__":
    main()