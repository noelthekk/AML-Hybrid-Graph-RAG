"""Check the actual rate limits behind the project's OpenAI API key.

There's no "check my quota" endpoint on the standard (non-admin) OpenAI API - the
university-issued key is a project-scoped service-account key
(sk-svcacct-...), not an org-admin key, so the usage/billing endpoints
(/v1/organization/usage/...) aren't accessible with it. What IS available: every chat
completions response carries rate-limit headers for the calling key's actual tier,
regardless of key type. This script makes one minimal call and reads those headers -
the real answer to "how much usage does this key support," not a guess from OpenAI's
public tier tables (project-specific overrides are common with institutional keys).

Usage:
    uv run python scripts/check_api_limits.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")


def main() -> None:
    client = OpenAI()
    resp = client.chat.completions.with_raw_response.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        max_completion_tokens=10,
    )
    headers = resp.headers
    parsed = resp.parse()

    print(f"Model: {MODEL}")
    print(f"Reply: {parsed.choices[0].message.content!r}")
    print()
    print("Rate limits for this key (from response headers):")
    for key in [
        "x-ratelimit-limit-requests", "x-ratelimit-remaining-requests", "x-ratelimit-reset-requests",
        "x-ratelimit-limit-tokens", "x-ratelimit-remaining-tokens", "x-ratelimit-reset-tokens",
    ]:
        print(f"  {key}: {headers.get(key, '(not present)')}")

    limit_rpm = headers.get("x-ratelimit-limit-requests")
    limit_tpm = headers.get("x-ratelimit-limit-tokens")
    if limit_rpm and limit_tpm:
        print()
        print(f"Summary: {limit_rpm} requests/min, {limit_tpm} tokens/min for {MODEL} on this key.")
        print("This is the actual ceiling that matters for RAGAS batch sizing - not a")
        print("published tier table, which institutional/service-account keys often override.")


if __name__ == "__main__":
    main()
