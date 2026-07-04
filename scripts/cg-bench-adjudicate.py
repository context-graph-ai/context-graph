#!/usr/bin/env python3
"""Gray-zone adjudication (frozen method, criteria 99d99ee): each 0.55-0.75 cosine pair goes to
the FIXED judge prompt below (temperature 0, hosted). Output: input tsv + verdict column, for
counting AND for the owner's >=20% spot-check sample. The prompt is frozen for the run: editing
it invalidates prior adjudications."""
import json, os, sys, urllib.request, random

JUDGE_MODEL = "anthropic/claude-haiku-4.5"
PROMPT = (
    "You judge whether two statements from a software-engineering work session describe THE SAME "
    "decision/goal, allowing paraphrase, or different ones. Answer with exactly one word: SAME or "
    "DIFFERENT.\n\nStatement A: {a}\n\nStatement B: {b}\n\nAnswer:"
)

def judge(a: str, b: str) -> str:
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps({
            "model": JUDGE_MODEL,
            "temperature": 0,
            "max_tokens": 4,
            "messages": [{"role": "user", "content": PROMPT.format(a=a, b=b)}],
        }).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        reply = json.load(resp)["choices"][0]["message"]["content"].strip().upper()
    return "SAME" if reply.startswith("SAME") else "DIFFERENT"

def main(path_in: str, path_out: str) -> None:
    rows = []
    for line in open(path_in):
        if not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3:
            print(f"SKIP malformed row (needs 3 tab fields): {parts[:1]}", flush=True)
            continue
        # Statements containing literal tabs: first field is session, last is gold, middle rejoins.
        rows.append([parts[0], "\t".join(parts[1:-1]), parts[-1]])
    out, same = [], 0
    for i, row in enumerate(rows):
        session, extracted, gold = row[0], row[1], row[2]
        verdict = judge(extracted, gold)
        same += verdict == "SAME"
        out.append(f"{session}\t{verdict}\t{extracted}\t{gold}")
        print(f"[{i+1}/{len(rows)}] {verdict}", flush=True)
    open(path_out, "w").write("\n".join(out))
    # Owner sample: 20% random rows, seeded deterministically for reproducibility.
    random.seed(99)
    sample = random.sample(out, max(1, len(out) // 5))
    open(path_out + ".owner-sample", "w").write("\n".join(sample))
    print(f"DONE {path_in}: {same}/{len(rows)} SAME -> {path_out} (+.owner-sample)")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
