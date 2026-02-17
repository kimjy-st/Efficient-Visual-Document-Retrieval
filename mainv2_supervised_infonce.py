import argparse
import json
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import tqdm
from utils.utils import set_seed, set_optimizer, get_logger, align_by_docid, save_compressed_npz
from utils.preprocess_data import l2_normalize, load_query_payload, load_payload, load_init_payload, _as_object_array, preprocess_docs, preprocess_queries
from utils.mapping import DATASETMAP
from evaluator.retrieval import score_multi_vector_masked, CustomRetrievalEvaluator
from Qdatasets.query_tensor_dataset import QueryTensorDataset_gtdocs
from torch.utils.data import DataLoader
from criterion import infonce_supervised_loss
# =============================================================================
# args
# =============================================================================
def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", type=str, nargs="+", required=True)
    p.add_argument("--query_root", type=str, default="ProxyQ/results0212/colqwen")
    p.add_argument("--teacher_root", type=str, default="/home/hyshin/Project/EVDR/data/vidore_test/0213_features/all")
    p.add_argument("--init_root", type=str, default="/home/hyshin/Project/EVDR/data/vidore_test/0213_features/S3E_init/colqwen")
    p.add_argument("--mfs", type=int, nargs="+", required=True)

    p.add_argument("--out_root", type=str, default="results")
    p.add_argument("--name", type=str, default="spl_train")

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--q_batch", type=int, default=32)
    p.add_argument("--opt", type=str, default="adamw")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--temp", type=float, default=1e-2)
    p.add_argument("--print_every", type=int, default=10)

    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)

    return p


# =============================================================================
# train / eval / save
# =============================================================================
def train_epoch(
    train_dl,
    Pbar_param: torch.nn.Parameter,
    pmask_student: torch.Tensor,
    opt: torch.optim.Optimizer,
    args,
    epoch: int,
    logger,
    tb=None
):
    t0_epoch = time.time()
    total_iter = len(train_dl)

    loss_sum = 0.0
    last_loss = 0.0

    for it, (Qb, qmb, pos_idx) in enumerate(tqdm.tqdm(train_dl, desc=f"train ep{epoch}", leave=False)):
        gstep = (epoch - 1) * total_iter + it

        Qb = Qb.to(Pbar_param.device, non_blocking=True)
        qmb = qmb.to(Pbar_param.device, non_blocking=True)
        pos_idx = pos_idx.to(Pbar_param.device, non_blocking=True)  # (B,)

        # iter마다 최신 student로 normalize
        Psb = l2_normalize(Pbar_param * pmask_student.unsqueeze(-1))

        # student score: (B, N)
        sc_s = score_multi_vector_masked(Qb, Psb, qmb, pmask_student)

        # InfoNCE = CE over docs
        loss = infonce_supervised_loss(sc_s, pos_idx, temperature=args.temp)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        loss_val = float(loss.item())
        loss_sum += loss_val
        last_loss = loss_val

        if args.print_every and (it % args.print_every == 0):
            dt = time.time() - t0_epoch
            logger.info({"iter": gstep, "epoch": epoch, "it": it, "loss": loss_val, "time_sec": dt})
            print(f"[train] ep={epoch} it={it}/{total_iter} loss={loss_val:.6f} t={dt:.1f}s")

        if tb is not None:
            tb.add_scalar("train/loss", loss_val, gstep)

    dt_epoch = time.time() - t0_epoch
    return {
        "loss": loss_sum / total_iter,
        "last_loss": last_loss,
        "global_step": epoch * total_iter,
        "time_sec": dt_epoch,
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
def eval(
    eval_dl,
    Pbar_param: nn.Parameter,
    pmask_student: torch.Tensor,
    temperature: float,
) -> float:
    device = Pbar_param.device
    pmask_student = pmask_student.to(device, non_blocking=True)

    # (N, L, D) masked+norm
    Psb = l2_normalize(Pbar_param * pmask_student.unsqueeze(-1))

    loss_sum, cnt = 0.0, 0
    for Qb, qmb, pos_idx in eval_dl:
        Qb = Qb.to(device, non_blocking=True)
        qmb = qmb.to(device, non_blocking=True)
        pos_idx = pos_idx.to(device, non_blocking=True)

        sc = score_multi_vector_masked(Qb, Psb, qmb, pmask_student)  # (B, N)
        loss = infonce_supervised_loss(sc, pos_idx, temperature=temperature)

        bsz = Qb.shape[0]
        loss_sum += float(loss.item()) * bsz
        cnt += bsz

    return loss_sum / max(cnt, 1)


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
        train_query_npz = f'{args.query_root}/{paths["pseudoQ"]}'
        teacher_npz = f'{args.teacher_root}/{paths["split_before"]}'

        train_query_payload = load_query_payload(train_query_npz)
        teacher_payload = load_payload(teacher_npz)

        # ---------- TEACHER (train / test split) ----------
        # Teacher doc embeddings
        docid_tr = teacher_payload["docid"]
        P_teacher_obj = teacher_payload["documents"]
        doc_attn_tr = teacher_payload["doc_attnmask"]
        doc_img_tr = teacher_payload["doc_imgmask"]
        # Train query embeddings
        Q_train_obj = train_query_payload["query"]
        q_attn_tr = train_query_payload["query_attnmask"] 
        Q_train_norm, qmask = preprocess_queries(Q_train_obj, q_attn_tr, device=device)

        train_ds = QueryTensorDataset_gtdocs(
            Q_train_norm, qmask,
            qid=train_query_payload["qid"],
            relevant_docs=train_query_payload["relevant_docs"],
            docidx_2_docid=teacher_payload["docidx_2_docid"],
        )
        train_dl = DataLoader(
            train_ds,
            batch_size=args.q_batch,
            shuffle=True,
            drop_last=False,
            num_workers=0,        # 텐서 이미 메모리에 있으니 보통 0이 제일 깔끔
            pin_memory=False,
        )
        # Test query embeddings
        Q_test_obj = teacher_payload["query"]
        q_attn_test = teacher_payload["query_attnmask"]
        Q_test_norm, qmask_test = preprocess_queries(Q_test_obj, q_attn_test, device=device)

        relevant_docs_test = teacher_payload["relevant_docs"]
        docidx_2_docid_test = teacher_payload["docidx_2_docid"]
        qsidx_2_query_test = teacher_payload["qsidx_2_query"]

        eval_ds = QueryTensorDataset_gtdocs(
            Q_test_norm, qmask_test,
            qid=teacher_payload['qid'],
            relevant_docs=relevant_docs_test,
            docidx_2_docid=docidx_2_docid_test,   # teacher 기준 doc index -> docid
        )

        eval_dl = DataLoader(
            eval_ds,
            batch_size=args.q_batch,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=False,
        )

        # preprocess teacher docs / train queries 
        P_teacher_raw, pmask_teacher, _valid_teacher = preprocess_docs(
            P_teacher_obj, doc_attn_tr, doc_img_tr, device=device
        )
        P_teacher_norm = l2_normalize(P_teacher_raw * pmask_teacher.unsqueeze(-1)).detach()
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

            eval_loss0 = eval(
                eval_dl=eval_dl,
                Pbar_param=Pbar_param,
                pmask_student=pmask_student,
                temperature=args.temp,
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

            best_r1 = {
                "epoch": 0,
                "NDCG@5": float(metrics0["NDCG"]["NDCG@5"]),
                "Recall@1": float(metrics0["Recall"]["Recall@1"]),
            }
            best_nd5 = {
                "epoch": 0,
                "NDCG@5": float(metrics0["NDCG"]["NDCG@5"]),
                "Recall@1": float(metrics0["Recall"]["Recall@1"]),
            }

            logger.info(f"best recall epoch| 0 | nDCG@5={float(best_r1['NDCG@5']):.5f} | Recall@1={float(best_r1['Recall@1']):.5f}")
            logger.info(f"best nDCG@5 epoch| 0 | nDCG@5={best_nd5['NDCG@5']:.5f} | Recall@1={best_nd5['Recall@1']:.5f}")


            r1_0 = float(metrics0["Recall"]["Recall@1"]) * 100.0
            nd5_0 = float(metrics0["NDCG"]["NDCG@5"]) * 100.0
            print("[evaluator metrics @ init]")
            print(f"Recall@1 = {r1_0:.5f}")
            print(f"nDCG@5    = {nd5_0:.5f}")
            print(f"eval loss  = {eval_loss0:.6f}")

            for epoch in range(1, args.epochs + 1):
                # ---------------- train ----------------
                # NOTE: train_epoch가 아래 키들을 같이 반환하도록 수정되어 있어야 함:
                # stats = {"loss":..., "global_step":..., "last_loss":..., "time_sec":...}
                stats = train_epoch(
                    train_dl=train_dl,
                    Pbar_param=Pbar_param,
                    pmask_student=pmask_student,
                    opt=opt,
                    args=args,
                    epoch=epoch,
                    logger=logger,
                    tb=tb,
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

                eval_loss = eval(
                    eval_dl=eval_dl,
                    Pbar_param=Pbar_param,
                    pmask_student=pmask_student,
                    temperature=args.temp,
                )
                step = epoch * len(train_dl) 
                if tb is not None:
                    tb.add_scalar("eval/Recall@1", float(metrics["Recall"]["Recall@1"]), step)
                    tb.add_scalar("eval/NDCG@5", float(metrics["NDCG"]["NDCG@5"]), step)
                    tb.add_scalar("eval/loss", float(eval_loss), step)

                r1 = float(metrics["Recall"]["Recall@1"]) * 100.0
                nd5 = float(metrics["NDCG"]["NDCG@5"]) * 100.0
                print("[evaluator metrics]")
                print(f"Recall@1 = {r1:.5f}")
                print(f"nDCG@5    = {nd5:.5f}")

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
                # ---------------- best update ----------------
                cur_r1 = float(metrics["Recall"]["Recall@1"])
                cur_nd5 = float(metrics["NDCG"]["NDCG@5"])

                updated_r1 = False
                updated_nd5 = False

                # best Recall@1 갱신 (tie-break: NDCG@5)
                if (cur_r1 > best_r1["Recall@1"]) or (cur_r1 == best_r1["Recall@1"] and cur_nd5 > best_r1["NDCG@5"]):
                    best_r1 = {"epoch": int(epoch), "NDCG@5": cur_nd5, "Recall@1": cur_r1}
                    updated_r1 = True

                # best NDCG@5 갱신 (tie-break: Recall@1)
                if (cur_nd5 > best_nd5["NDCG@5"]) or (cur_nd5 == best_nd5["NDCG@5"] and cur_r1 > best_nd5["Recall@1"]):
                    best_nd5 = {"epoch": int(epoch), "NDCG@5": cur_nd5, "Recall@1": cur_r1}
                    updated_nd5 = True

                if updated_r1:
                    logger.info(
                        f"best recall epoch| {best_r1['epoch']} | nDCG@5={best_r1['NDCG@5']:.5f} | Recall@1={best_r1['Recall@1']:.5f}"
                    )
                if updated_nd5:
                    logger.info(
                        f"best nDCG@5 epoch| {best_nd5['epoch']} | nDCG@5={best_nd5['NDCG@5']:.5f} | Recall@1={best_nd5['Recall@1']:.5f}"
                    )

                # ---------------- save (only when best updates) ----------------
                if updated_r1 or updated_nd5:
                    Pbar_now_np = Pbar_now_norm.detach().cpu().numpy().astype(np.float32)
                    pmask_np = pmask_student.detach().cpu().numpy().astype(bool)
                    docs_obj = tokens_to_object(Pbar_now_np, pmask_np)

                if updated_r1:
                    out_npz = out_dir / "best_recall.npz"
                    save_compressed_npz(
                        save_path=out_npz,
                        docid=_as_object_array(docid_tr),
                        documents_obj=docs_obj,
                        doc_attnmask_obj=doc_attn_in,
                        doc_imgmask_obj=doc_img_in,
                        meta={
                            "dataset": dataset,
                            "mf": mf,
                            "epoch": int(epoch),
                            "best_type": "Recall@1",
                            "best": best_r1,
                            "eval": {
                                "Recall@1": float(metrics["Recall"]["Recall@1"]),
                                "NDCG@5": float(metrics["NDCG"]["NDCG@5"]),
                            },
                            "lr": args.lr,
                            "loss": "inffonce_loss",
                            "temp":args.temp
                        },
                    )

                if updated_nd5:
                    out_npz = out_dir / "best_ndcg5.npz"
                    save_compressed_npz(
                        save_path=out_npz,
                        docid=_as_object_array(docid_tr),
                        documents_obj=docs_obj,
                        doc_attnmask_obj=doc_attn_in,
                        doc_imgmask_obj=doc_img_in,
                        meta={
                            "dataset": dataset,
                            "mf": mf,
                            "epoch": int(epoch),
                            "best_type": "NDCG@5",
                            "best": best_nd5,
                            "eval": {
                                "Recall@1": float(metrics["Recall"]["Recall@1"]),
                                "NDCG@5": float(metrics["NDCG"]["NDCG@5"]),
                            },
                            "lr": args.lr,
                            "loss": "infonce_loss",
                            "temperature": args.temp,
                        },
                    )
            logger.info(json.dumps({
                    "summary/best_recall": best_r1,
                    "summary/best_ndcg5": best_nd5,
                    "note": "training finished",
                }, ensure_ascii=False))
            print(f"[done] {dataset} mf{mf} -> {out_dir}")
            if tb is not None:
                tb.flush()
                tb.close()


if __name__ == "__main__":
    main()