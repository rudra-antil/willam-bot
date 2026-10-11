"""
Sync WILLAM docs from this repo into an Obsidian vault.

The Obsidian "Local REST API" plugin must be RUNNING (Obsidian open).
Usage:

    python docs/sync_to_obsidian.py --key <API_KEY>
    # or set OBSIDIAN_API_KEY in the environment
    python docs/sync_to_obsidian.py

Options:
    --host    default https://127.0.0.1:27124
    --folder  vault subfolder, default trading-bots/willam
    --key     Local REST API key (or env OBSIDIAN_API_KEY)

Writes each docs/vault/*.md into <vault>/<folder>/ using PUT (creates or updates).
"""

import argparse
import os
import sys

import urllib3
import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

VaultRoot = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(VaultRoot, "vault")


def probe(host, headers):
    try:
        r = requests.get(f"{host}/vault/", headers=headers, verify=False, timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="https://127.0.0.1:27124")
    ap.add_argument("--folder", default="trading-bots/willam")
    ap.add_argument("--key", default=os.environ.get("OBSIDIAN_API_KEY", ""))
    args = ap.parse_args()

    if not args.key:
        print("ERROR: no API key. Pass --key or set OBSIDIAN_API_KEY.")
        print("Find it in Obsidian → Settings → Community plugins → Local REST API.")
        return 1

    headers = {"Authorization": f"Bearer {args.key}", "Content-Type": "text/markdown"}

    if not probe(args.host, headers):
        print(f"ERROR: cannot reach Obsidian at {args.host}")
        print("  • Is Obsidian open?")
        print("  • Is the 'Local REST API' plugin enabled?")
        print("  • Is the HTTPS server enabled on the same port?")
        return 1

    if not os.path.isdir(SOURCE):
        print(f"ERROR: no docs/vault folder at {SOURCE}")
        return 1

    files = sorted(f for f in os.listdir(SOURCE) if f.endswith(".md"))
    if not files:
        print("ERROR: docs/vault has no .md files")
        return 1

    ok = fail = 0
    for name in files:
        with open(os.path.join(SOURCE, name), encoding="utf-8") as f:
            content = f.read()
        url = f"{args.host}/vault/{args.folder}/{name}"
        try:
            r = requests.put(url, headers=headers, data=content, verify=False, timeout=15)
            if r.status_code in (200, 201, 204):
                print(f"  ✓ {args.folder}/{name}")
                ok += 1
            else:
                print(f"  ✗ {args.folder}/{name} — HTTP {r.status_code}: {r.text[:120]}")
                fail += 1
        except Exception as e:
            print(f"  ✗ {args.folder}/{name} — {type(e).__name__}: {str(e)[:120]}")
            fail += 1

    print(f"\nSynced {ok} file(s), {fail} failed -> {args.folder}/")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
