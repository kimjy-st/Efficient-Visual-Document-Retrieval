import os, json, argparse, base64
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
import time


SYSTEM_PROMPT = (
    "You are a dataset curator for document image QA. "
    "Generate diverse, non-redundant questions that are answerable ONLY from the given document image. "
    "Do not include questions that require external knowledge."
)

def to_data_url_jpg(p: Path) -> str:
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

def parse_questions_from_text(text: str, n_questions: int) -> list[str]:
    text = (text or "").strip()

    # 코드펜스 제거(있어도)
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.strip().endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # "1. ..." "2) ..." "- ..." 같은 접두 제거
    import re
    cleaned = []
    for ln in lines:
        ln = re.sub(r"^\s*(\d+[\.\)]\s*|[-*]\s+)", "", ln).strip()
        if ln:
            cleaned.append(ln)

    # 정확히 n_questions만 취함(초과하면 자르고, 부족하면 에러)
    if len(cleaned) < n_questions:
        raise ValueError(f"Too few lines: got {len(cleaned)}, expected {n_questions}")
    return cleaned[:n_questions]


def gen_questions(client: OpenAI, model: str, img_path: Path, n_questions: int, max_retries: int = 3):
    user_prompt = f"""
        Based on the document image, generate exactly {n_questions} questions that are answerable ONLY from the image.

        Rules:
        - Questions must be written in English.
        - Every question must be answerable using only the content visible in the document image (no external knowledge).
        - Minimize redundancy and near-duplicates.
        - Cover diverse aspects such as tables, charts/figures, equations, captions, headers/footers, layout/structure,
        numbering, units, legends, and footnotes.
        - Use varied forms (e.g., what/which/how many/how much/where/when/why/how).
        - Each line must be one question.
        - Do NOT use code fences. 
        - Do Not add any extra text before or after the {n_questions} lines.
        """
    data_url = to_data_url_jpg(img_path)

    last_err = None
    for _ in range(max_retries):
        try:
            resp = client.responses.create(
                model=model,
                temperature=0.0,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "input_text", "text": user_prompt},
                        {"type": "input_image", "image_url": data_url},
                    ]},
                ],
            )
            text = resp.output_text.strip()
            qs = parse_questions_from_text(text, n_questions)

            if len(qs) != n_questions:
                raise ValueError(f"Expected {n_questions}, got {len(qs)}.")
            qs = [q.strip() for q in qs]
            if any(not q for q in qs):
                raise ValueError("Empty question detected.")
            return qs
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Failed on {img_path}: {last_err}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True, type=str, help="dataset image directory (jpg only)")
    ap.add_argument("--out_json", required=True, type=str, help="output json path ; result.json")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--nq", type=int, default=100, help="number of questions per image")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--save_every", type=int, default=5)
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY env var.")

    client = OpenAI(api_key=api_key)

    img_dir = Path(args.img_dir)
    imgs = sorted(img_dir.glob("*.jpg"))

    out_path = Path(args.out_json)

    # -------------------------
    # Resume: load existing json
    # -------------------------
    data = {}
    processed_paths = set()
    next_id = 0

    if args.resume and out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # 1) 이미 처리된 image_path set 만들기 (중복 방지 핵심)
        for rec in data.values():
            p = rec.get("image_path", "")
            if p:
                processed_paths.add(p)

        # 2) 다음 id 계산: (숫자 key만 고려)
        existing_ids = []
        for k in data.keys():
            try:
                existing_ids.append(int(k))
            except:
                pass
        next_id = (max(existing_ids) + 1) if existing_ids else 0

    # -------------------------
    # Main loop
    # -------------------------
    save_cnt = 0
    for img_path in tqdm(imgs):
        # 이미 처리된 이미지면 스킵 (id 말고 image_path로 판단)
        img_path_str = str(img_path)
        if args.resume and img_path_str in processed_paths:
            continue

        sid = str(next_id)
        next_id += 1

        try:
            qs = gen_questions(client, args.model, img_path, args.nq)
            data[sid] = {"image_path": img_path_str, "Question": qs}
        except Exception as e:
            data[sid] = {"image_path": img_path_str, "Question": [], "error": str(e)}

        # 방금 처리한 것도 processed에 추가 (같은 run에서 중복 방지)
        processed_paths.add(img_path_str)

        save_cnt += 1
        if save_cnt % args.save_every == 0:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    # final save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved -> {out_path} (records={len(data)}, images={len(imgs)})")

if __name__ == "__main__":
    main()