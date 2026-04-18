"""T-29 — Held-out scenario generation driver.

Calls the Cipher Agent API (Gemini 2.5 Pro) with the verbatim prompt from
``PREREGISTERED.md`` and writes 3 held-out scenarios encrypted with ``age``.

Plaintext files are NEVER committed — only ciphertext (``*.age``) ends up in
the repo. The age recipient(s) are read from ``--recipients-file``.

Usage:
    python benchmark/amb_v2/scripts/generate_held_out.py \
        --recipients-file held_out/recipients.txt \
        --out-dir benchmark/amb_v2/held_out
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

CIPHER_API = os.environ.get("CIPHER_API_URL", "http://100.109.132.104:7799/quick")
ROLE = "researcher"  # Gemini 2.5 Pro per prompts/CIPHER.md
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

PROMPT_TEMPLATE = """You are generating a held-out test scenario for AMB v2 (Agentic Memory
Benchmark v2). This scenario will be used to score memory adapters and MUST
NOT leak into training data.

Constraints (follow exactly):
1. Output strict JSON matching this schema:
   {{
     "scenario_id": "h0{n}",
     "name": "<short descriptive name>",
     "timeline": [ {{
        "id": "h0{n}-d<DDD>-<seq>",
        "day": <int 0-90>,
        "text": "<one sentence fact>",
        "type": "<fact|update|credential|preference|session>",
        "supersedes": "<optional id of an earlier event this overrides>"
     }} ],
     "queries": [ {{
        "query_id": "h0{n}-q<NN>",
        "question": "<natural-language question>",
        "expected_answer": "<short answer phrase>",
        "resolution_type": "<fact|contradiction|credential|preference>",
        "checkpoint_eligibility": [0, 7, 14, 30, 60, 90]
     }} ]
   }}
2. 20–30 timeline events across days 0–90, including AT LEAST 2 contradictions
   (later events with ``supersedes`` pointing to earlier ones) and AT LEAST 1
   credential.
3. 8–12 queries covering all resolution types. For contradictions, the
   expected_answer MUST be the NEW value (from the superseding event).
4. No real PII. Use fictional names, addresses, and phone numbers.
5. Output JSON only. No prose, no code fences.

Generate scenario {n} of 3 now."""


def _call_cipher(prompt: str, timeout: int = 180) -> str:
    body = json.dumps({"role": ROLE, "content": prompt}).encode("utf-8")
    req = urllib.request.Request(
        CIPHER_API, data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())
    return payload.get("response") or payload.get("answer") or ""


def _extract_json(text: str) -> dict:
    """Locate and parse the first JSON object in `text`.

    Tolerates markdown code fences, leading prose, and trailing commentary.
    """
    m = _JSON_OBJECT_RE.search(text)
    if not m:
        raise json.JSONDecodeError("no JSON object found in response", text, 0)
    return json.loads(m.group(0))


def _age_encrypt(plaintext: str, cipher_out: Path, recipients_file: Path) -> None:
    result = subprocess.run(
        ["age", "-R", str(recipients_file), "-o", str(cipher_out)],
        input=plaintext.encode("utf-8"),
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"age encrypt failed: {result.stderr.decode()}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--recipients-file", type=Path, required=True,
                   help="File containing age recipient public keys (one per line)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--count", type=int, default=3)
    args = p.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for n in range(1, args.count + 1):
        prompt = PROMPT_TEMPLATE.format(n=n)
        print(f"[held-out] asking Cipher for scenario {n}/{args.count}...")
        try:
            raw = _call_cipher(prompt)
        except urllib.error.URLError as e:
            print(f"[held-out] Cipher unreachable: {e}", file=sys.stderr)
            return 2
        try:
            obj = _extract_json(raw)
        except json.JSONDecodeError as e:
            print(f"[held-out] scenario {n}: JSON parse failed ({e}); raw output saved",
                  file=sys.stderr)
            (args.out_dir / f"h0{n}.rawerror.txt").write_text(raw)
            return 3
        plaintext = json.dumps(obj, indent=2, sort_keys=True)
        cipher_out = args.out_dir / f"h0{n}.json.age"
        _age_encrypt(plaintext, cipher_out, args.recipients_file)
        print(f"[held-out] wrote {cipher_out} ({cipher_out.stat().st_size} bytes)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
