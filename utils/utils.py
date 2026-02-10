
import torch 
import numpy as np
import json 
from utils.preprocess_data import _as_object_array
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import logging, os
from torch.utils.tensorboard import SummaryWriter



def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_logger(save_dir: str, name: str = "run", verbosity: int = 1, use_tb: bool = True):
    """
    returns:
      logger: logging.Logger
      tb: SummaryWriter or None
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    level = level_dict.get(verbosity, logging.INFO)

    logger = logging.getLogger(f"{name}@{save_dir}")
    logger.setLevel(level)
    logger.propagate = False  # 중복 출력 방지

    # 핸들러 중복 방지 (같은 logger를 여러 번 만들 때)
    if len(logger.handlers) == 0:
        fmt = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")

        fh = logging.FileHandler(save_dir / "train.log", mode="a")
        fh.setFormatter(fmt)
        fh.setLevel(level)
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(level)
        logger.addHandler(sh)

    tb = SummaryWriter(log_dir=str(save_dir)) if use_tb else None
    return logger, tb


def log_dict(logger, tb: Optional[SummaryWriter], scalars: Dict[str, Any], step: int):
    """
    scalars: {"loss": 0.1, "Recall@1": 0.23, ...}
    """
    # console/file
    logger.info(json.dumps({"step": step, **scalars}, ensure_ascii=False))

    # tensorboard
    if tb is not None:
        for k, v in scalars.items():
            if isinstance(v, (int, float)):
                tb.add_scalar(k, v, step)
        tb.flush()
    return logger


def set_optimizer(name, param, lr, wd):
    if name == 'adamw':
        return torch.optim.AdamW([param], lr=lr, weight_decay=wd)
    

def save_compressed_npz(
    save_path: Path,
    docid: np.ndarray,
    documents_obj: np.ndarray,          # object (N,) each (Li,D)
    doc_attnmask_obj: Optional[np.ndarray],
    doc_imgmask_obj: Optional[np.ndarray],
    meta: Dict[str, Any],
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "docid": _as_object_array(docid),
        "documents": _as_object_array(documents_obj),
        "doc_attnmask": _as_object_array(doc_attnmask_obj) if doc_attnmask_obj is not None else None,
        "doc_imgmask": _as_object_array(doc_imgmask_obj) if doc_imgmask_obj is not None else None,
        "meta": np.array(meta, dtype=object),
    }
    payload = {k: v for k, v in payload.items() if v is not None}
    np.savez_compressed(str(save_path), **payload)
    print(f"[save] {save_path}")


def align_by_docid(
    docid_ref: np.ndarray,
    docid_other: Optional[np.ndarray],
    *arrays_to_perm: Optional[np.ndarray],
) -> Tuple[Tuple[Optional[np.ndarray], ...], bool]:
    """
    Permute 'other' arrays to match ref docid order (only if possible).
    Works for object arrays only (which is your case).
    """
    if docid_other is None:
        return arrays_to_perm, False

    docid_ref = _as_object_array(docid_ref)
    docid_other = _as_object_array(docid_other)
    if len(docid_other) != len(docid_ref):
        return arrays_to_perm, False

    map_other = {str(docid_other[i]): i for i in range(len(docid_other))}
    perm = []
    for i in range(len(docid_ref)):
        did = str(docid_ref[i])
        if did not in map_other:
            return arrays_to_perm, False
        perm.append(map_other[did])
    perm = np.array(perm, dtype=np.int64)

    out = []
    for arr in arrays_to_perm:
        if arr is None:
            out.append(None)
        else:
            out.append(_as_object_array(arr)[perm])
    return tuple(out), True