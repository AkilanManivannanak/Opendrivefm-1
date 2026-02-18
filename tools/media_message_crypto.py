#!/usr/bin/env python3
from __future__ import annotations

import base64
import getpass
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

MAGIC = b"ODFM_MSG_V1\n"  # marker to locate appended payload


def _kdf(key_str: str, salt: bytes) -> bytes:
    # scrypt -> 32 bytes key
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

    out = data + MAGIC + payload + b"\n"
    media_out.write_bytes(out)


def _extract_payload(media_path: Path) -> Payload:
    b = media_path.read_bytes()
    i = b.rfind(MAGIC)
    if i < 0:
        raise ValueError("No embedded payload found (MAGIC marker missing).")

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


def main() -> None:
    print("Choose mode:")
    print("  1) Encrypt (append encrypted message to media)")
    print("  2) Decrypt (extract message from media)")
    mode = input("Enter 1 or 2: ").strip()

    if mode not in {"1", "2"}:
        raise SystemExit("Invalid choice. Enter 1 or 2.")

    media_path = Path(input("Enter media file path (image / gif / video): ").strip()).expanduser()
    if not media_path.exists():
        raise SystemExit(f"File not found: {media_path}")

    if mode == "1":
        message = input("Enter message to encrypt: ")
        key = getpass.getpass("Enter key (hidden): ")
        out_path = Path(input("Output file path (e.g., out.mp4 / out.png): ").strip()).expanduser()
        encrypt_into_media(media_path, out_path, message, key)
        print(f"✅ Encrypted message appended. Wrote: {out_path}")

    else:
        key = getpass.getpass("Enter key (hidden): ")
        msg = decrypt_from_media(media_path, key)
        print("\n✅ Decrypted message:")
        print(msg)


if __name__ == "__main__":
    main()

