from __future__ import annotations

import argparse
import getpass
import os
import struct
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

MAGIC = b"ODFMENC1"  # 8 bytes
VERSION = 1


def _kdf(passphrase: str, salt: bytes) -> bytes:
    # Strong password->key derivation (tune N for your machine if needed)
    kdf = Scrypt(
        salt=salt,
        length=32,        # AES-256
        n=2**15,          # CPU/memory cost
        r=8,
        p=1,
    )
    return kdf.derive(passphrase.encode("utf-8"))


def encrypt_file(inp: Path, out: Path, passphrase: str) -> None:
    data = inp.read_bytes()
    salt = os.urandom(16)
    nonce = os.urandom(12)  # AESGCM standard nonce size
    key = _kdf(passphrase, salt)
    aes = AESGCM(key)

    # AAD binds metadata to ciphertext integrity
    aad = MAGIC + struct.pack(">B", VERSION)

    ct = aes.encrypt(nonce, data, aad)

    # file format:
    # MAGIC(8) | VER(1) | SALT(16) | NONCE(12) | CT(...)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(MAGIC + struct.pack(">B", VERSION) + salt + nonce + ct)


def decrypt_file(inp: Path, out: Path, passphrase: str) -> None:
    blob = inp.read_bytes()

    if len(blob) < 8 + 1 + 16 + 12:
        raise ValueError("Encrypted file too small / corrupt")

    magic = blob[:8]
    if magic != MAGIC:
        raise ValueError("Bad magic header (not an ODFM encrypted file)")

    ver = blob[8]
    if ver != VERSION:
        raise ValueError(f"Unsupported version: {ver}")

    salt = blob[9:25]
    nonce = blob[25:37]
    ct = blob[37:]

    key = _kdf(passphrase, salt)
    aes = AESGCM(key)
    aad = MAGIC + struct.pack(">B", VERSION)

    pt = aes.decrypt(nonce, ct, aad)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pt)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("encrypt")
    e.add_argument("--in", dest="inp", required=True)
    e.add_argument("--out", dest="out", required=True)
    e.add_argument("--pass_env", default="ODFM_PASSPHRASE")

    d = sub.add_parser("decrypt")
    d.add_argument("--in", dest="inp", required=True)
    d.add_argument("--out", dest="out", required=True)
    d.add_argument("--pass_env", default="ODFM_PASSPHRASE")

    args = ap.parse_args()

    passphrase = os.getenv(args.pass_env)
    if not passphrase:
        passphrase = getpass.getpass(f"Enter passphrase (or set {args.pass_env}): ")

    inp = Path(args.inp)
    out = Path(args.out)

    if args.cmd == "encrypt":
        encrypt_file(inp, out, passphrase)
        print(f"ENCRYPTED: {inp} -> {out}")
    else:
        decrypt_file(inp, out, passphrase)
        print(f"DECRYPTED: {inp} -> {out}")


if __name__ == "__main__":
    main()
