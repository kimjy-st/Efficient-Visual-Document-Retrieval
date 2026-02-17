import torch.nn.functional as F
import torch 

# 이거 안 씀 
# def pairwise_loss(sc_t: torch.Tensor, sc_s: torch.Tensor):
#     """
#     sc_t, sc_s: (B, N)  (teacher/student 점수)
#     모든 pair (i,j)에 대해 teacher의 pairwise 확률 sigmoid(dt)을
#     student의 sigmoid(ds)이 모사하도록 BCE를 계산.
#     """
#     # (B, N, 1) - (B, 1, N) -> (B, N, N)
#     dt = (sc_t.unsqueeze(2) - sc_t.unsqueeze(1))
#     ds = (sc_s.unsqueeze(2) - sc_s.unsqueeze(1))

#     # 대각선 + (i,j) (j, i) 안 겹치도록 마스킹
#     B, N, _ = dt.shape
#     mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=sc_t.device), diagonal=1)
#     mask = mask.unsqueeze(0).expand(B, -1, -1)

#     # teacher 확률(soft target)
#     pt = torch.sigmoid(dt).detach()

#     # BCE with logits: logits=ds, target=pt
#     loss = F.binary_cross_entropy_with_logits(ds[mask], pt[mask])
#     return loss


# def infonce_loss(sc_s: torch.Tensor, pos_idx: torch.Tensor, temp: float = 1.0):
#     """
#     sc_s: (B, N)  student similarity logits
#     pos_idx: (B,)  정답 문서 인덱스 (0..N-1), dtype long
#     """
#     logits = sc_s / temp
#     return F.cross_entropy(logits, pos_idx.long())





# ----------------------------------------------------------------------
# InforNCE Loss Variants (Top-1 정답 레이블이 있는 경우 vs 없는 경우)
# ----------------------------------------------------------------------
def infonce_supervised_loss(
    score_s: torch.Tensor, 
    labels: torch.Tensor, 
    temperature: float = 0.07
) -> torch.Tensor:
    """
    명시적인 정답 레이블(Top-1)이 있는 경우
    """
    loss = F.cross_entropy(score_s / temperature, labels)

    return loss


def infonce_distillation_loss(
    score_s: torch.Tensor, 
    score_t: torch.Tensor, 
    temperature: float = 0.07
) -> torch.Tensor:
    """
    정답 없이 Teacher의 예측(Top-1)을 정답으로 삼는 경우 (Pseudo-labeling)
    """
    score_t = score_t.detach()
    targets = torch.argmax(score_t, dim=1)    
    loss = F.cross_entropy(score_s / temperature, targets)  
    
    return loss


# ----------------------------------------------------------------------
# Score-based Distillation Loss
# ----------------------------------------------------------------------
def score_preserving_loss(
    score_s: torch.Tensor, 
    score_t: torch.Tensor
) -> torch.Tensor:
    """
    Teacher와 Student의 점수(Logit) 값 자체를 비슷하게 맞추는 손실함수
    """
    score_t = score_t.detach()    
    loss = F.mse_loss(score_s, score_t)
    return loss


# ----------------------------------------------------------------------
# Pairwise Ranking Distillation Loss
# ----------------------------------------------------------------------
def pairwise_distillation_loss(
    score_s: torch.Tensor, 
    score_t: torch.Tensor
) -> torch.Tensor:
    """
    모든 문서 쌍(Pair)에 대해 Teacher의 우위 관계(A > B)를 Student가 따르도록 함
    """
    score_t = score_t.detach()

    # 1. 문서 간 점수 차이 행렬 생성
    diff_s = score_s.unsqueeze(2) - score_s.unsqueeze(1)
    diff_t = score_t.unsqueeze(2) - score_t.unsqueeze(1)
    
    # 2. Teacher의 점수 차이를 확률(0~1)로 변환 (Soft Target)
    target_probs = torch.sigmoid(diff_t)
    
    # 3. Binary Cross Entropy 계산
    loss = F.binary_cross_entropy_with_logits(diff_s, target_probs)

    return loss


# ----------------------------------------------------------------------
# Listwise Ranking Distillation Loss
# ----------------------------------------------------------------------
def listwise_distillation_loss(
    score_s: torch.Tensor, 
    score_t: torch.Tensor,
    k: int = 10,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    쿼리 하나에 대한 전체 문서 점수의 '확률 분포'를 KL-Divergence로 모방
    """
    score_t = score_t.detach()

    # 1. Softmax 계산
    log_prob_s = F.log_softmax(score_s / temperature, dim=1)
    prob_t = F.softmax(score_t / temperature, dim=1)
    
    # 2. Teacher 기준 Top-K 인덱스 추출
    _, topk_indices = torch.topk(prob_t, k, dim=1)
    
    # 3. 해당 인덱스의 확률 값만 선택
    selected_prob_t = torch.gather(prob_t, 1, topk_indices)    
    selected_log_prob_s = torch.gather(log_prob_s, 1, topk_indices)
    
    # 4. Partial Cross Entropy (부분 엔트로피) 계산
    loss = - (selected_prob_t * selected_log_prob_s).sum(dim=1).mean()

    # 5. Temperature Scaling 보정
    loss = loss * (temperature ** 2)
    
    return loss


# ----------------------------------------------------------------------
# Advanced Losses
# ----------------------------------------------------------------------
def lambda_loss(
    score_s: torch.Tensor, 
    score_t: torch.Tensor, 
    alpha: float = 1.0,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    LambdaLoss: (Listwise-aware Pairwise Loss)
    Teacher의 순위를 기준으로 NDCG 변화량(Delta)을 가중치로 사용하여 Student가 올바른 순위를 학습하도록 유도
    """
    score_t = score_t.detach()
    batch_size, n_docs = score_s.shape
    
    # 1. Teacher 기준 정렬 (Ideal Ranking)
    sorted_scores_t, sorted_idx = torch.sort(score_t, dim=1, descending=True)   
    score_s_sorted = torch.gather(score_s, 1, sorted_idx)
    
    # 2. Delta NDCG 가중치 계산: i번째 문서(상위권)와 j번째 문서(하위권)의 순위가 바뀌었을 때의 NDCG 손실량   
    # Discount Factor: 1 / log2(rank + 1)
    ranks = torch.arange(1, n_docs + 1, device=score_s.device, dtype=torch.float)
    discounts = 1.0 / torch.log2(ranks + 1.0)
    
    # Discount 차이 행렬: |1/log(i) - 1/log(j)|
    discounts_diff = torch.abs(discounts.view(1, -1, 1) - discounts.view(1, 1, -1))
    
    # Gain (Relevance) 차이 행렬: |2^Rel_i - 2^Rel_j|
    rel_t = torch.sigmoid(sorted_scores_t) # Relevance를 0~1 확률로 변환
    gain_diff = torch.abs(rel_t.unsqueeze(2) - rel_t.unsqueeze(1))
    
    # Lambda Weight (Delta NDCG)
    lambda_weight = (gain_diff * discounts_diff) * 10.0 # 스케일링
    
    # 3. Pairwise Loss 계산
    diff_s = score_s_sorted.unsqueeze(2) - score_s_sorted.unsqueeze(1)
    pairwise_loss = -F.logsigmoid(alpha * diff_s)    
    weighted_loss = lambda_weight * pairwise_loss
    
    # 4. Masking (i < j 인 상삼각행렬만 사용)
    mask = torch.triu(torch.ones(n_docs, n_docs, device=score_s.device), diagonal=1)    
    loss = (weighted_loss * mask).sum() / (mask.sum() + eps)
    
    return loss


def ranknce_loss(
    score_s: torch.Tensor, 
    score_t: torch.Tensor, 
    temperature: float = 1.0,
    lambda_weight: float = 1.0
) -> torch.Tensor:
    """
    RankNCE:
    InfoNCE + Ranking Regularization
    """
    score_t = score_t.detach()
    batch_size, n_docs = score_s.shape
    
    # 1. Hard Negative Mining & Sorting
    sorted_score_t, sorted_idx = torch.sort(score_t, dim=1, descending=True)
    sorted_score_s = torch.gather(score_s, 1, sorted_idx)
    
    # Part A: InfoNCE (Positive vs All Negatives)
    labels = torch.zeros(batch_size, dtype=torch.long, device=score_s.device)
    loss_infonce = F.cross_entropy(sorted_score_s / temperature, labels)
    
    # Part B: Ranking Regularization (Hard Neg vs Easy Neg)
    diff_s = sorted_score_s[:, :-1] - sorted_score_s[:, 1:]
    diff_t = sorted_score_t[:, :-1] - sorted_score_t[:, 1:]
    
    # Margin Ranking Loss Variant:
    # Teacher 차이가 클수록(weights), Student가 순서를 어기면(diff_s < 0) 페널티 강화
    # Margin을 0으로 두는 대신, softplus 등을 사용하여 순서 위반 시 Loss 부여    
    # 가중치: Teacher가 확실하게 구분한 순서일수록 중요함
    weights = torch.sigmoid(diff_t) 
    ranking_loss = (weights * F.softplus(-diff_s)).mean()

    loss = loss_infonce + lambda_weight * ranking_loss
    
    return loss