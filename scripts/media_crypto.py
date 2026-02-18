#!/usr/bin/env python3
from __future__ import annotations

import base64
import getpass
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

MAGIC = b"ODFM_MSG_V1\n"  # marker to locate appended payload


# -------------------------
# Crypto primitives
# -------------------------
def _kdf(key_str: str, salt: bytes) -> bytes:
    # scrypt -> 32-byte key
    kdf = Scrypt(salt=salt, length=32, n=2**14, r=8, p=1)
    return kdf.derive(key_str.encode("utf-8"))


@dataclass
class Payload:
    salt_b64: str
    nonce_b64: str
    ct_b64: str

    def to_bytes(self) -> bytes:
        return json.dumps(self.__dict__, separators=(",", ":")).encode("utf-8")

    @staticmethod
    def from_bytes(b: bytes) -> "Payload":
        d = json.loads(b.decode("utf-8"))
        return Payload(**d)


# -------------------------
# IO helpers / UX hardening
# -------------------------
def _clean_path(s: str) -> str:
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    return s.strip()


def resolve_out_path(media_in: Path, user_out: str) -> Path:
    """
    Make it hard to accidentally save a file named 'png'.

    Rules:
      - blank => auto next to input: <stem>_enc<suffix>
      - if user types 'png' or '.png' => <stem>_enc.png
      - if user provides a directory => put output in that dir with <stem>_enc<suffix>
      - if user provides a path with no suffix => keep input suffix
      - otherwise use exactly what user provided
    """
    user_out = _clean_path(user_out)

    if not user_out:
        return media_in.with_name(f"{media_in.stem}_enc{media_in.suffix}")

    ext_tokens = {"png", ".png", "jpg", ".jpg", "jpeg", ".jpeg", "gif", ".gif", "mp4", ".mp4", "mov", ".mov"}
    if user_out.lower() in ext_tokens:
        ext = user_out if user_out.startswith(".") else f".{user_out}"
        return media_in.with_name(f"{media_in.stem}_enc{ext}")

    p = Path(user_out).expanduser()

    if p.exists() and p.is_dir():
        return p / f"{media_in.stem}_enc{media_in.suffix}"

    if p.suffix == "":
        return p.with_suffix(media_in.suffix)

    return p


def confirm_overwrite(path: Path) -> None:
    if path.exists():
        ans = input(f"⚠️  Output exists: {path}\nOverwrite? [y/N]: ").strip().lower()
        if ans not in {"y", "yes"}:
            raise SystemExit("Aborted (did not overwrite).")


# -------------------------
# Core functions
# -------------------------
def encrypt_into_media(media_in: Path, media_out: Path, message: str, key: str) -> None:
    data = media_in.read_bytes()

    salt = os.urandom(16)
    nonce = os.urandom(12)

    k = _kdf(key, salt)
    aes = AESGCM(k)
    ct = aes.encrypt(nonce, message.encode("utf-8"), associated_data=None)

    payload = Payload(
        salt_b64=base64.b64encode(salt).decode("ascii"),
        nonce_b64=base64.b64encode(nonce).decode("ascii"),
        ct_b64=base64.b64encode(ct).decode("ascii"),
    ).to_bytes()

    out_bytes = data + MAGIC + payload + b"\n"
    media_out.write_bytes(out_bytes)


def _extract_payload(media_path: Path) -> Payload:
    b = media_path.read_bytes()
    i = b.rfind(MAGIC)
    if i < 0:
        # Helpful hint: user likely decrypted the original instead of the _enc file
        guess = media_path.with_name(f"{media_path.stem}_enc{media_path.suffix}")
        hint = ""
        if guess.exists():
            hint = f"\nHint: I found a likely encrypted file next to it:\n  {guess}"
        raise ValueError(f"No embedded payload found (MAGIC marker missing).{hint}")

    payload_bytes = b[i + len(MAGIC):].strip()
    if not payload_bytes:
        raise ValueError("Payload marker found, but payload is empty/corrupt.")
    return Payload.from_bytes(payload_bytes)


def decrypt_from_media(media_path: Path, key: str) -> str:
    payload = _extract_payload(media_path)
    salt = base64.b64decode(payload.salt_b64)
    nonce = base64.b64decode(payload.nonce_b64)
    ct = base64.b64decode(payload.ct_b64)

    k = _kdf(key, salt)
    aes = AESGCM(k)
    pt = aes.decrypt(nonce, ct, associated_data=None)
    return pt.decode("utf-8")


# -------------------------
# CLI
# -------------------------
def main() -> None:
    print("Choose mode:")
    print("  1) Encrypt (append encrypted message to media)")
    print("  2) Decrypt (extract message from media)")
    mode = input("Enter 1 or 2: ").strip()
    if mode not in {"1", "2"}:
        raise SystemExit("Invalid choice. Enter 1 or 2.")

    media_in = Path(_clean_path(input("Enter media file path (image / gif / video): "))).expanduser()
    if not media_in.exists():
        raise SystemExit(f"File not found: {media_in}")

    if mode == "1":
        message = input("Enter message to encrypt: ")

        key = getpass.getpass("Enter key (hidden): ")
        key2 = getpass.getpass("Confirm key (hidden): ")
        if key != key2:
            raise SystemExit("Keys do not match. Aborting to prevent a bad encrypt.")

        out_in = input("Output path (blank=auto, or type png/mp4 to keep format): ")
        media_out = resolve_out_path(media_in, out_in)
        confirm_overwrite(media_out)

        encrypt_into_media(media_in, media_out, message, key)

        # sanity check: marker exists in output
        try:
            _ = _extract_payload(media_out)
        except Exception as e:
            raise SystemExit(f"Wrote output but failed to re-locate payload. Something is wrong.\n{e}")

        print(f"✅ Encrypted message appended.\nWrote: {media_out}")

    else:
        key = getpass.getpass("Enter key (hidden): ")
        try:
            msg = decrypt_from_media(media_in, key)
        except ValueError as e:
            raise SystemExit(str(e))

        print("\n✅ Decrypted message:")
        print(msg)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)
