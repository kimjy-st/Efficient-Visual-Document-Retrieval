from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
import numpy as np

import time 
import torch
from PIL import Image
from transformers import BatchEncoding, BatchFeature

def get_torch_device(device: str = "auto") -> str:
    """
    Returns the device (string) to be used by PyTorch.

    `device` arg defaults to "auto" which will use:
    - "cuda:0" if available
    - else "mps" if available
    - else "cpu".
    """

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():  # for Apple Silicon
            device = "mps"
        else:
            device = "cpu"

    return device

def left_padding(sequences, batch_first=True, padding_value=0):
    # 1D -> 2D 변환 (e.g. [128] -> [1, 128]) #modified
    sequences = [seq.unsqueeze(0) if seq.ndim == 1 else seq for seq in sequences]

    max_len = max(seq.size(0) for seq in sequences)
    d = sequences[0].size(-1)
    padded = []

    for seq in sequences:
        seq = seq.to(device='cuda')
        pad_len = max_len - seq.size(0)
        pad = torch.full((pad_len,d), padding_value, device = 'cuda',dtype=seq.dtype)
        padded_seq = torch.cat([pad, seq], dim=0)
        padded.append(padded_seq)

    return torch.stack(padded) if batch_first else torch.stack(padded).transpose(0, 1)

class BaseVisualRetrieverProcessor(ABC):
    """
    Base class for visual retriever processors.
    """

    @abstractmethod
    def process_images(
        self,
        images: List[Image.Image],
    ) -> Union[BatchFeature, BatchEncoding]:
        pass

    @abstractmethod
    def process_queries(
        self,
        queries: List[str],
        max_length: int = 50,
        suffix: Optional[str] = None,
    ) -> Union[BatchFeature, BatchEncoding]:
        pass

    @abstractmethod
    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        pass

    @staticmethod
    def score_single_vector(
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:

        device = device or get_torch_device("auto")

        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")

        qs_stacked = torch.stack(qs).to(device)
        ps_stacked = torch.stack(ps).to(device)

        scores = torch.einsum("bd,cd->bc", qs_stacked, ps_stacked)
        assert scores.shape[0] == len(qs), f"Expected {len(qs)} scores, got {scores.shape[0]}"

        scores = scores.to(torch.float32)
        return scores

    @staticmethod
    def score_multi_vector(
        qs: Union[torch.Tensor, List[torch.Tensor]],
        ps: Union[torch.Tensor, List[torch.Tensor]],
        batch_size: int = 128,
        device: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:

        device = device or get_torch_device("auto")

        if len(qs) == 0:
            raise ValueError("No queries provided")
        if len(ps) == 0:
            raise ValueError("No passages provided")

        scores_list: List[torch.Tensor] = []
        maxtime_list: List[torch.Tensor] = []
        print('qs:',len(qs),'ps:',len(ps))

        for i in range(0, len(qs), batch_size):
            scores_batch = []
            qs_batch = left_padding(
                qs[i : i + batch_size], 
                batch_first=True, 
                padding_value=0).to(device)
            for j in range(0, len(ps), batch_size):
                ps_batch = left_padding(
                    ps[j : j + batch_size], 
                    batch_first=True, 
                    padding_value=0).to(device)
                # === [추가] dtype 정렬: einsum은 양쪽 dtype이 동일해야 함 ===(modified)
                if qs_batch.dtype != ps_batch.dtype or qs_batch.dtype in (torch.float16, torch.bfloat16):
                    qs_batch = qs_batch.to(torch.float32)
                    ps_batch = ps_batch.to(torch.float32)
                tmp = time.time()
                scores_batch.append(torch.einsum("bnd,csd->bcns", qs_batch, ps_batch).max(dim=3)[0].sum(dim=2))
                maxtime = time.time()-tmp
                maxtime_list.append(maxtime)

            
            scores_batch = torch.cat(scores_batch, dim=1).cpu()
            scores_list.append(scores_batch)
        print('len(maxtime_list):',len(maxtime_list))
        print('maxtime_list 평균',np.mean(maxtime_list))
        print('maxtime_list min, max',np.min(maxtime_list),np.max(maxtime_list) )
        scores = torch.cat(scores_list, dim=0)
        assert scores.shape[0] == len(qs), f"Expected {len(qs)} scores, got {scores.shape[0]}"

        scores = scores.to(torch.float32)
        return scores

    @abstractmethod
    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        patch_size: int = 14,
        *args,
        **kwargs,
    ) -> Tuple[int, int]:
        """
        Get the number of patches (n_patches_x, n_patches_y) that will be used to process an
        image of size (height, width) with the given patch size.
        """
        pass

def score_multi_vector_masked(
    Q: torch.Tensor,        # (Q, Lq, D)
    P: torch.Tensor,        # (P, Lp, D)
    qmask: torch.Tensor,    # (Q, Lq) bool
    pmask: torch.Tensor,    # (P, Lp) bool
    chunk_p: int = 128,
) -> torch.Tensor:
    Qn, Lq, D = Q.shape
    Pn, Lp, _ = P.shape

    Q = Q.to(torch.float32)
    P = P.to(torch.float32)

    qmask = qmask.bool()
    pmask = pmask.bool()

    qmask_f = qmask.to(torch.float32)

    out = []
    neg = -1e4  # torch.finfo.min 대신 큰 음수 (더 안전)

    for ps in range(0, Pn, chunk_p):
        Pc  = P[ps:ps+chunk_p]          # (C, Lp, D)
        pmc = pmask[ps:ps+chunk_p]      # (C, Lp)

        # 이 doc에 유효 토큰이 1개라도 있나? (C,)
        doc_has_token = pmc.any(dim=1)  # bool

        # (Q, C, Lq, Lp)
        sim = torch.einsum("qnd,cmd->qcnm", Q, Pc)

        # invalid doc tokens 제외
        sim = sim.masked_fill(~pmc[None, :, None, :], neg)

        # max over doc tokens -> (Q, C, Lq)
        mx = sim.max(dim=-1).values

        # ✅ doc 전체가 invalid면 mx를 0으로 (Q,C,Lq)
        mx = mx * doc_has_token[None, :, None].to(mx.dtype)

        # query invalid token 제외
        mx = mx * qmask_f[:, None, :]

        # sum over query tokens -> (Q, C)
        sc = mx.sum(dim=-1)
        out.append(sc)

    return torch.cat(out, dim=1)  # (Q, P)
"""
source: https://github.com/illuin-tech/colpali/blob/main/colpali_engine/trainer/eval_utils.py
"""
from typing import Dict, Any, List
from mteb.evaluation.evaluators.RetrievalEvaluator import RetrievalEvaluator

class CustomRetrievalEvaluator():
    def __init__(
        self,
        k_values : List[int] = [1, 3, 5, 10, 50,70,100],
        score_function: str = "cos_sim",
    ):
        super().__init__()
        self.k_values = k_values
        self.score_function = score_function

    def compute_mteb_metrics(
        self,
        relevant_docs : Dict[str, Dict[str, int]],
        results : Dict[str, Dict[str, float]],
        **kwargs,
    ) -> Dict[str, float]:
        """
        Compute the MTEB retrieval metrics.
        """
        ndcg, _map, recall, precision, naucs = RetrievalEvaluator.evaluate(
            relevant_docs,
            results,
            self.k_values,
            ignore_identical_ids=kwargs.get("ignore_identical_ids", False),
        )

        mrr = RetrievalEvaluator.evaluate_custom(relevant_docs, results, self.k_values, "mrr")[0]

        metrics = {}
        metrics["NDCG"] = ndcg 
        metrics["mAP"] = _map 
        metrics["Recall"] = recall 
        metrics["Precision"] = precision 
        metrics["mRR"] = mrr 

        return metrics 