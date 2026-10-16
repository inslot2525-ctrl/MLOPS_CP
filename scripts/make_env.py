"""`make env`: copy .env.example to .env if absent and generate local API keys and their SHA-256 hashes.

Existing real values are kept; only the obviously fake example key values are replaced.
"""

from __future__ import annotations

import hashlib
import secrets
import shutil
from pathlib import Path


def sha(k: str) -> str:
    return hashlib.sha256(k.encode()).hexdigest()


def main(env: Path = Path(".env"), example: Path = Path(".env.example")) -> None:
    if not env.exists():
        shutil.copyfile(example, env)
    lines = env.read_text(encoding="utf-8").splitlines()
    vals = dict(line.split("=", 1) for line in lines if "=" in line and not line.startswith("#"))
    if "not-real" not in vals.get("ADMIN_API_KEY", "not-real") and not set(vals.get("API_KEYS", "0")) <= set(
        "client:0,"
    ):
        print(".env already has generated keys; unchanged")
        return
    admin, sim = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    new = {
        "ADMIN_API_KEY": admin,
        "ADMIN_API_KEY_SHA256": sha(admin),
        "SIM_API_KEY": sim,
        "API_KEYS": f"simulator:{sha(sim)}",
    }
    out = [
        f"{line.split('=', 1)[0]}={new[line.split('=', 1)[0]]}" if line.split("=", 1)[0] in new else line
        for line in lines
    ]
    env.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("generated ADMIN_API_KEY, SIM_API_KEY and their hashes in .env (keys are not printed)")


if __name__ == "__main__":
    main()
