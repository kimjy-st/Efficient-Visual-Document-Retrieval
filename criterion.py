import torch.nn.functional as F
import torch 


def pairwise_loss(sc_t: torch.Tensor, sc_s: torch.Tensor):
    """
    sc_t, sc_s: (B, N)  (teacher/student 점수)
    모든 pair (i,j)에 대해 teacher의 pairwise 확률 sigmoid(dt)을
    student의 sigmoid(ds)이 모사하도록 BCE를 계산.
    """
    # (B, N, 1) - (B, 1, N) -> (B, N, N)
    dt = (sc_t.unsqueeze(2) - sc_t.unsqueeze(1))
    ds = (sc_s.unsqueeze(2) - sc_s.unsqueeze(1))

    # 대각선 + (i,j) (j, i) 안 겹치도록 마스킹
    B, N, _ = dt.shape
    mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=sc_t.device), diagonal=1)
    mask = mask.unsqueeze(0).expand(B, -1, -1)

    # teacher 확률(soft target)
    pt = torch.sigmoid(dt).detach()

    # BCE with logits: logits=ds, target=pt
    loss = F.binary_cross_entropy_with_logits(ds[mask], pt[mask])
    return loss