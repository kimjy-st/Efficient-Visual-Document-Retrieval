import numpy as np
from typing import Dict, List, Tuple, Optional
import torch

def load_npz(path: str):
    return np.load(path, allow_pickle=True)

def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def parse_relevant_docs(z) -> Dict[str, dict]:
    v = z["relevant_docs"]
    # shape=() object -> dict
    if isinstance(v, np.ndarray) and v.shape == ():
        return v.item()
    if isinstance(v, dict):
        return v
    return v.item()


def _as_object_array(x):
    return x.astype(object) if isinstance(x, np.ndarray) else np.array(x, dtype=object)


def _to_bool_1d(arr) -> Optional[np.ndarray]:
    if arr is None:
        return None
    a = np.array(arr)
    if a.dtype == object:
        a = np.array(a.tolist())
    a = a.astype(bool)
    if a.ndim == 2 and a.shape[-1] == 1:
        a = a.squeeze(-1)
    return a


def pad_tokens_object(tok_list: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    tok_list: object array (N,), each (Li, D) float
    returns:
      pad:   (N, Lmax, D) float32
      valid: (N, Lmax) bool
    """
    tok_list = _as_object_array(tok_list)
    N = len(tok_list)
    lens = np.array([int(tok_list[i].shape[0]) for i in range(N)], dtype=np.int64)
    D = int(tok_list[0].shape[1])
    L = int(lens.max())

    pad = np.zeros((N, L, D), dtype=np.float32)
    valid = np.zeros((N, L), dtype=bool)
    for i in range(N):
        Li = int(lens[i])
        pad[i, :Li] = tok_list[i].astype(np.float32)
        valid[i, :Li] = True
    return pad, valid


def pad_mask_object(mask_list: Optional[np.ndarray], L: int, N: int, valid: np.ndarray) -> np.ndarray:
    """
    mask_list: object array (N,), each (Li,) bool-like OR None
    return: (N, L) bool, padded False.
    If mask_list is None -> all True on valid positions.
    """
    if mask_list is None:
        return valid.copy()

    mask_list = _as_object_array(mask_list)
    out = np.zeros((N, L), dtype=bool)
    for i in range(N):
        mi = _to_bool_1d(mask_list[i])
        if mi is None:
            out[i] = valid[i]
        else:
            Li = min(L, mi.shape[0])
            out[i, :Li] = mi[:Li]
    return out


def preprocess_docs(
    documents_obj: np.ndarray,
    doc_attnmask_obj: Optional[np.ndarray],
    doc_imgmask_obj: Optional[np.ndarray],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Returns:
      P_raw:  (N, L, D) torch.float32 (NOT normalized)
      P_norm: (N, L, D) torch.float32 (L2 normalized)
      pmask:  (N, L)    torch.bool = valid & attn & img
      valid:  (N, L)    np.bool_   (debug)
    """
    # zero pad 
    P_pad, valid = pad_tokens_object(documents_obj)  # (N,L,D), (N,L)
    N, L, _ = P_pad.shape

    # attn mask (없으면 all-True로)
    am = pad_mask_object(doc_attnmask_obj, L=L, N=N, valid=valid)  # (N,L) np.bool_
    # img mask (없으면 all-True로)
    im = pad_mask_object(doc_imgmask_obj, L=L, N=N, valid=valid)   # (N,L) np.bool_
    pmask_np = valid & am & im

    P_raw = torch.from_numpy(P_pad).to(device=device, dtype=torch.float32)
    pmask = torch.from_numpy(pmask_np).to(device=device, dtype=torch.bool)
    return P_raw, pmask, valid


def preprocess_queries(
    query_obj: np.ndarray,
    query_attnmask_obj: Optional[np.ndarray],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      Q_norm: (Q, Lq, D) torch.float32 (L2 normalized)
      qmask:  (Q, Lq)    torch.bool = valid & attn
    """
    Q_pad, valid = pad_tokens_object(query_obj)  # (Q,Lq,D), (Q,Lq)
    Qn, Lq, _ = Q_pad.shape
    qm = pad_mask_object(query_attnmask_obj, L=Lq, N=Qn, valid=valid)
    qmask_np = valid & qm

    Q = torch.from_numpy(Q_pad).to(device=device, dtype=torch.float32)
    Q = l2_normalize(Q)
    qmask = torch.from_numpy(qmask_np).to(device=device)
    return Q, qmask


def load_train_payload(train_npz: str):
    z = load_npz(train_npz)
    return {
        "docid": z["docid"],
        "documents": z["documents"],
        "doc_attnmask": z["doc_attnmask"] if "doc_attnmask" in z.files else None,
        "doc_imgmask": z["doc_imgmask"] if "doc_imgmask" in z.files else None,
        "query": z["query"],
        "query_attnmask": z["query_attnmask"] if "query_attnmask" in z.files else None,
        "relevant_docs": z["relevant_docs"].item() if "relevant_docs" in z.files else None,
        "docidx_2_docid": z["docidx_2_docid"].item() if "docidx_2_docid" in z.files else None,
        "qsidx_2_query": z["qsidx_2_query"] if "qsidx_2_query" in z.files else None,
    }

def load_test_payload(test_npz: str):
    """
    load_train_payload랑 역할은 동일함 그냥 구분해둔듯 ...... 
    걍 하나로 합칠게요 
    Inputs
        test_npz : 테스트 npz 경로 
    
    Return 
        docid, documents, doc_attnmask 등이 담긴 Dict
    """
    z = load_npz(test_npz)
    return {
        "docid": z["docid"],
        "documents": z["documents"],
        "doc_attnmask": z["doc_attnmask"] if "doc_attnmask" in z.files else None,
        "doc_imgmask": z["doc_imgmask"] if "doc_imgmask" in z.files else None,
        "query": z["query"],
        "query_attnmask": z["query_attnmask"] if "query_attnmask" in z.files else None,
        "relevant_docs": z["relevant_docs"].item() if "relevant_docs" in z.files else None,
        "docidx_2_docid": z["docidx_2_docid"].item() if "docidx_2_docid" in z.files else None,
        "qsidx_2_query": z["qsidx_2_query"] if "qsidx_2_query" in z.files else None,
    }

def load_init_payload(init_npz: str):
    z = load_npz(init_npz)
    return {
        "docid": z["docid"] if "docid" in z.files else None,
        "documents": z["documents"],
        "doc_attnmask": z["doc_attnmask"] if "doc_attnmask" in z.files else None,
        "doc_imgmask": z["doc_imgmask"] if "doc_imgmask" in z.files else None,
    }

def load_query_payload(npz_path: str):
    z = load_npz(npz_path)
    return {
        "query": z["query"],
        "query_attnmask": z["query_attnmask"] if "query_attnmask" in z.files else None,
        "qsidx_2_query": z["qsidx_2_query"] if "qsidx_2_query" in z.files else None,
        "relevant_docs": z["relevant_docs"].item() if "relevant_docs" in z.files else None,
    }

def load_payload(npz_path: str):
    """
    load_train_payload, load_test_payload와 동일한 역할 
    docid, documents, doc_attnmask, doc_imgmask, query, query_attnmask, relevant_docs,
    docidx_2_docid, qsidx_2_query 키를 로드한 뒤 Dict형태로 정리하여 반환함 
    
    Input 
        npz_path: npz 경로 
    Return
        Dicts
    """
    z = load_npz(npz_path)
    return {
        "docid": z["docid"],
        "documents": z["documents"] if "documents" in z.files else None,
        "doc_attnmask": z["doc_attnmask"] if "doc_attnmask" in z.files else None,
        "doc_imgmask": z["doc_imgmask"] if "doc_imgmask" in z.files else None,
        "query": z["query"] if "query" in z.files else None,
        "query_attnmask": z["query_attnmask"] if "query_attnmask" in z.files else None,
        "relevant_docs": z["relevant_docs"].item() if "relevant_docs" in z.files else None,
        "docidx_2_docid": z["docidx_2_docid"].item() if "docidx_2_docid" in z.files else None,
        "qsidx_2_query": z["qsidx_2_query"] if "qsidx_2_query" in z.files else None
    }