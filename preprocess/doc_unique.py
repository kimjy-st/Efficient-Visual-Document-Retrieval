import numpy as np
from pathlib import Path

# # 1) 원본 raw(중복 포함) — keep 인덱스 만들 용도
# raw_full_npz = "/home/hyshin/Project/EVDR/data/vidore_test/0213_features/all/tatdqa_test_dump_all.npz"

# # 2) raw unique (이미 만들어둔 파일) — docid/매핑 저장용
# raw_unique_npz = "/home/hyshin/Project/EVDR/data/vidore_test/0221_features/colqwen/tatdqa_test_dump_all.npz"

# # 3) in_npz (S3E_init mf5 등) — documents가 raw_full과 같은 인덱스 체계라고 가정
# in_npz_path = "/home/hyshin/Project/EVDR/data/vidore_test/0213_features/S3E_init/colqwen/mf50/tatdqa_test.npz"

# # 4) output
# out_path = "/home/hyshin/Project/EVDR/data/vidore_test/0221_features/colqwen/S3E_init/mf50/tatdqa_test.npz"
# Path(out_path).parent.mkdir(parents=True, exist_ok=True)

# def _to_str(x):
#     if isinstance(x, bytes):
#         return x.decode("utf-8", errors="ignore")
#     return str(x)

# # -------------------------
# # A) keep 인덱스 = raw_full에서 "첫 등장" 인덱스
# # -------------------------
# z_full = np.load(raw_full_npz, allow_pickle=True)
# docid_full = np.array([_to_str(x) for x in z_full["docid"]], dtype=object)

# seen = set()
# keep = []
# for i, d in enumerate(docid_full):
#     if d in seen:
#         continue
#     seen.add(d)
#     keep.append(i)
# keep = np.array(keep, dtype=np.int64)

# print(f"[INFO] raw_full docs={len(docid_full)} -> unique(first-occ)={len(keep)}")

# # -------------------------
# # B) raw_unique의 docid가 docid_full[keep]와 동일한지 검증
# # -------------------------
# z_uni = np.load(raw_unique_npz, allow_pickle=True)
# docid_unique = np.array([_to_str(x) for x in z_uni["docid"]], dtype=object)

# if len(docid_unique) != len(keep):
#     raise ValueError(f"[ERROR] unique length mismatch: raw_unique={len(docid_unique)} vs keep={len(keep)}")

# # 완전 동일 체크(중요)
# if not np.all(docid_unique == docid_full[keep]):
#     # 어디부터 다른지 조금만 출력
#     diff = np.where(docid_unique != docid_full[keep])[0]
#     j = int(diff[0])
#     raise ValueError(
#         "[ERROR] docid order mismatch between raw_unique and raw_full[keep]. "
#         f"first diff at {j}: unique={docid_unique[j]} vs full_keep={docid_full[keep][j]}"
#     )

# print("[INFO] ✅ docid_unique matches docid_full[keep]")

# # -------------------------
# # C) in_npz를 keep으로 정확히 선택
# # -------------------------
# zin = np.load(in_npz_path, allow_pickle=True)
# if "documents" not in zin.files:
#     raise ValueError("[ERROR] in_npz has no 'documents'")

# N_in = zin["documents"].shape[0]
# if N_in != len(docid_full):
#     raise ValueError(
#         f"[ERROR] in_npz doc count mismatch with raw_full: in={N_in} vs raw_full={len(docid_full)}. "
#         "이 경우엔 in_npz가 raw_full 인덱스 체계가 아닐 수 있음."
#     )

# out = {}
# out["documents"] = zin["documents"][keep]

# for k in ["doc_attnmask", "doc_imgmask", "attention"]:
#     if k in zin.files:
#         arr = zin[k]
#         if getattr(arr, "ndim", 0) == 0:
#             print(f"[WARN] skip key={k} (ndim=0)")
#             continue
#         if arr.shape[0] == N_in:
#             out[k] = arr[keep]
#         else:
#             print(f"[WARN] skip key={k} (shape0={arr.shape[0]} != N_in={N_in})")

# # doc metadata
# out["docid"] = docid_unique
# out["docidx_2_docid"] = {str(i): docid_unique[i] for i in range(len(docid_unique))}

# # 문서축 아닌 나머지 키 복사
# skip = set(out.keys()) | {"documents", "doc_attnmask", "doc_imgmask", "attention", "docid", "docidx_2_docid"}
# for k in zin.files:
#     if k in skip:
#         continue
#     out[k] = zin[k]

# np.savez_compressed(out_path, **out)
# print(f"[DONE] saved: {out_path}")

# # sanity
# z2 = np.load(out_path, allow_pickle=True)
# m = z2["docidx_2_docid"].item()
# print("[CHECK] documents:", z2["documents"].shape[0], "docid:", len(z2["docid"]), "map:", len(m), "ex0:", m["0"])




import numpy as np
from pathlib import Path

# 1) 원본 raw(중복 포함) — keep 인덱스 만들 용도
raw_full_npz = "/home/hyshin/Project/EVDR/data/vidore_test/0213_features/all/tabfquad_test_subsampled_dump_all.npz"

# 2) raw unique (이 스크립트에서 새로 생성)
raw_unique_npz = "/home/hyshin/Project/EVDR/data/vidore_test/0221_features/colqwen/tabfquad_test_subsampled_dump_all.npz"

# 3) in_npz (S3E_init mf50 등) — documents가 raw_full과 같은 인덱스 체계라고 가정
in_npz_path = "/home/hyshin/Project/EVDR/data/vidore_test/0213_features/S3E_init/colqwen/mf25/tabfquad_test_subsampled.npz"

# 4) output (S3E_init mf50 unique 버전)
out_path = "/home/hyshin/Project/EVDR/data/vidore_test/0221_features/colqwen/S3E_init/mf25/tabfquad_test_subsampled.npz"
Path(raw_unique_npz).parent.mkdir(parents=True, exist_ok=True)
Path(out_path).parent.mkdir(parents=True, exist_ok=True)

def _to_str(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    return str(x)

# -------------------------
# A) raw_full 로드 + keep 인덱스 = "첫 등장" 인덱스
# -------------------------
z_full = np.load(raw_full_npz, allow_pickle=True)

if "docid" not in z_full.files:
    raise ValueError("[ERROR] raw_full_npz has no 'docid'")

docid_full = np.array([_to_str(x) for x in z_full["docid"]], dtype=object)

seen = set()
keep = []
for i, d in enumerate(docid_full):
    if d in seen:
        continue
    seen.add(d)
    keep.append(i)
keep = np.array(keep, dtype=np.int64)

print(f"[INFO] raw_full docs={len(docid_full)} -> unique(first-occ)={len(keep)}")

# -------------------------
# B) raw_unique_npz 생성 (raw_full에서 keep으로 잘라 저장)
#    - docid, documents, (있으면) doc_attnmask/doc_imgmask/attention 등 doc-axis 키도 같이
#    - docidx_2_docid는 dict로 생성
# -------------------------
out_raw = {}

# doc-axis로 취급할 "후보 키"들 (있으면 같이 슬라이싱)
doc_axis_candidates = {"docid", "documents", "doc_attnmask", "doc_imgmask", "attention"}

for k in z_full.files:
    arr = z_full[k]

    # doc-axis 후보이고, 0-dim이 아니고, 첫 축 길이가 raw doc 수와 같으면 -> keep 적용
    if (k in doc_axis_candidates) and getattr(arr, "ndim", 0) > 0 and arr.shape[0] == len(docid_full):
        out_raw[k] = arr[keep]
    else:
        # 나머지는 그대로 복사
        out_raw[k] = arr

# docid는 문자열로 통일한 버전을 강제로 사용 (bytes 섞임 방지)
docid_unique = docid_full[keep]
out_raw["docid"] = docid_unique

# eval 코드 호환: dict[str, str]
out_raw["docidx_2_docid"] = {str(i): docid_unique[i] for i in range(len(docid_unique))}

np.savez_compressed(raw_unique_npz, **out_raw)
print(f"[DONE] raw_unique saved: {raw_unique_npz}")

# sanity (raw_unique)
z_uni = np.load(raw_unique_npz, allow_pickle=True)
m_uni = z_uni["docidx_2_docid"].item()
print("[CHECK raw_unique] docid/documents:",
      len(z_uni["docid"]),
      (z_uni["documents"].shape[0] if "documents" in z_uni.files else None),
      "map:", len(m_uni),
      "ex0:", m_uni.get("0", None))

# -------------------------
# C) in_npz를 keep으로 정확히 선택하여 out_path로 저장
# -------------------------
zin = np.load(in_npz_path, allow_pickle=True)
if "documents" not in zin.files:
    raise ValueError("[ERROR] in_npz has no 'documents'")

N_in = zin["documents"].shape[0]
if N_in != len(docid_full):
    raise ValueError(
        f"[ERROR] in_npz doc count mismatch with raw_full: in={N_in} vs raw_full={len(docid_full)}. "
        "이 경우엔 in_npz가 raw_full 인덱스 체계가 아닐 수 있음."
    )

out = {}
out["documents"] = zin["documents"][keep]

for k in ["doc_attnmask", "doc_imgmask", "attention"]:
    if k in zin.files:
        arr = zin[k]
        if getattr(arr, "ndim", 0) == 0:
            print(f"[WARN] skip key={k} (ndim=0)")
            continue
        if arr.shape[0] == N_in:
            out[k] = arr[keep]
        else:
            print(f"[WARN] skip key={k} (shape0={arr.shape[0]} != N_in={N_in})")

# doc metadata (raw_unique와 동일하게)
out["docid"] = docid_unique
out["docidx_2_docid"] = {str(i): docid_unique[i] for i in range(len(docid_unique))}

# 문서축 아닌 나머지 키 복사
skip = set(out.keys()) | {"documents", "doc_attnmask", "doc_imgmask", "attention", "docid", "docidx_2_docid"}
for k in zin.files:
    if k in skip:
        continue
    out[k] = zin[k]

np.savez_compressed(out_path, **out)
print(f"[DONE] S3E_init unique saved: {out_path}")

# sanity (out)
z2 = np.load(out_path, allow_pickle=True)
m = z2["docidx_2_docid"].item()
print("[CHECK out] documents:", z2["documents"].shape[0],
      "docid:", len(z2["docid"]),
      "map:", len(m),
      "ex0:", m["0"])