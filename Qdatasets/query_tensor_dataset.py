from torch.utils.data import Dataset
import torch 
from pathlib import Path

class QueryTensorDataset(Dataset):
    def __init__(self, Q_norm: torch.Tensor, qmask: torch.Tensor):
        assert Q_norm.shape[:2] == qmask.shape[:2]
        self.Q = Q_norm
        self.m = qmask

    def __len__(self):
        return self.Q.shape[0]

    def __getitem__(self, i):
        return self.Q[i], self.m[i]
    


class QueryTensorDataset_gtdocs(Dataset):
    def __init__(self, Q: torch.Tensor, qmask: torch.Tensor,
                 qid, relevant_docs, docidx_2_docid,
                 filter_missing: bool = True):
        """
        Q: (M, Lq, D)
        qmask: (M, Lq)
        qid: array-like length M (str/int)
        relevant_docs: dict[str(qid)] -> dict[docid] -> rel
        docidx_2_docid: dict[str(docidx)] -> docid
        """
        self.Q = Q
        self.qmask = qmask

        # scalar object(npz)면 .item()
        if hasattr(relevant_docs, "shape") and relevant_docs.shape == ():
            relevant_docs = relevant_docs.item()
        if hasattr(docidx_2_docid, "shape") and docidx_2_docid.shape == ():
            docidx_2_docid = docidx_2_docid.item()

        # docid -> docidx
        docid2idx = {docid: int(di) for di, docid in docidx_2_docid.items()}

        # qid -> pos_idx
        pos = torch.empty(len(qid), dtype=torch.long)
        pos.fill_(-1)
        miss = 0
        for i, q in enumerate(qid):
            qk = str(q)
            gt = relevant_docs.get(qk, None)
            if gt is None or len(gt) == 0:
                miss += 1
                continue
            gt_docid = max(gt.items(), key=lambda kv: kv[1])[0]  # rel max
            di = docid2idx.get(gt_docid, None)
            if di is None:
                miss += 1
                continue
            pos[i] = di

        if filter_missing:
            valid = pos >= 0
            if miss > 0:
                print(f"[Dataset] missing gt mapping {miss}/{len(pos)} -> filtered")
            self.Q = self.Q[valid]
            self.qmask = self.qmask[valid]
            self.pos_idx = pos[valid]
        else:
            self.pos_idx = pos  # -1 포함 가능 (train에서 처리해야 함)

    def __len__(self):
        return self.Q.shape[0]

    def __getitem__(self, idx):
        return self.Q[idx], self.qmask[idx], self.pos_idx[idx]

# check 용 
# def _norm_docid(x: str) -> str:
#     # 절대/상대 경로 섞여있어도 파일명 기준으로 매칭
#     return Path(str(x)).name

# class QueryTensorDataset_gtdocs(Dataset):
#     def __init__(
#         self,
#         Q: torch.Tensor, qmask: torch.Tensor,
#         qid, relevant_docs, docidx_2_docid,
#         filter_missing: bool = True,
#         use_filename_match: bool = True,   # <- 추천
#     ):
#         """
#         Q: (M, Lq, D)
#         qmask: (M, Lq)
#         qid: array-like length M (str/int)
#         relevant_docs: dict[str(qid)] -> dict[docid] -> rel
#         docidx_2_docid: dict[str(docidx)] -> docid
#         """
#         self.Q = Q
#         self.qmask = qmask

#         # qid를 텐서로 들고가면 collate가 편함
#         # (문자 qid면 텐서화 불가라 리스트로 둠)
#         self.qid = list(qid)

#         # scalar object(npz)면 .item()
#         if hasattr(relevant_docs, "shape") and relevant_docs.shape == ():
#             relevant_docs = relevant_docs.item()
#         if hasattr(docidx_2_docid, "shape") and docidx_2_docid.shape == ():
#             docidx_2_docid = docidx_2_docid.item()

#         # docid -> docidx (경로 normalize 옵션)
#         if use_filename_match:
#             docid2idx = {_norm_docid(docid): int(di) for di, docid in docidx_2_docid.items()}
#         else:
#             docid2idx = {str(docid): int(di) for di, docid in docidx_2_docid.items()}

#         # qid -> pos_idx
#         pos = torch.full((len(self.qid),), -1, dtype=torch.long)
#         miss = 0

#         for i, q in enumerate(self.qid):
#             qk = str(q)
#             gt = relevant_docs.get(qk, None)
#             if not gt:   # None or empty dict
#                 miss += 1
#                 continue

#             # rel max docid 선택
#             gt_docid = max(gt.items(), key=lambda kv: kv[1])[0]
#             key = _norm_docid(gt_docid) if use_filename_match else str(gt_docid)

#             di = docid2idx.get(key, None)
#             if di is None:
#                 miss += 1
#                 continue
#             pos[i] = di

#         if filter_missing:
#             valid = pos >= 0
#             if miss > 0:
#                 print(f"[Dataset] missing gt mapping {miss}/{len(pos)} -> filtered")

#             self.Q = self.Q[valid]
#             self.qmask = self.qmask[valid]
#             self.pos_idx = pos[valid]
#             self.qid = [self.qid[i] for i in torch.nonzero(valid, as_tuple=False).squeeze(1).tolist()]
#         else:
#             self.pos_idx = pos  # -1 포함 가능 (train에서 처리해야 함)

#     def __len__(self):
#         return self.Q.shape[0]

#     def __getitem__(self, idx):
#         # qid까지 같이 반환!
#         return self.Q[idx], self.qmask[idx], self.pos_idx[idx], self.qid[idx]