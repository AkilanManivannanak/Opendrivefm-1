from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from getpass import getpass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt


MAGIC = b"STEGCRYPTv1:"
VIDEO_META_MAX = 20_000  # characters for metadata comment (conservative)


# -------------------------
# Crypto
# -------------------------
def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("utf-8").rstrip("=")


def _b64d(s: str) -> bytes:
    pad = "=" * ((4 - (len(s) % 4)) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("utf-8"))


def derive_key(password: str, salt: bytes) -> bytes:
    kdf = Scrypt(salt=salt, length=32, n=2**15, r=8, p=1)
    return kdf.derive(password.encode("utf-8"))


def encrypt_message(message: str, password: str) -> bytes:
    salt = os.urandom(16)
    key = derive_key(password, salt)
    nonce = os.urandom(12)
    aes = AESGCM(key)
    ct = aes.encrypt(nonce, message.encode("utf-8"), associated_data=None)

    payload = {
        "v": 1,
        "alg": "AES-256-GCM",
        "kdf": "scrypt",
        "salt": _b64e(salt),
        "nonce": _b64e(nonce),
        "ct": _b64e(ct),
    }
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return raw


def decrypt_message(payload_raw: bytes, password: str) -> str:
    payload = json.loads(payload_raw.decode("utf-8"))
    salt = _b64d(payload["salt"])
    nonce = _b64d(payload["nonce"])
    ct = _b64d(payload["ct"])

    key = derive_key(password, salt)
    aes = AESGCM(key)
    pt = aes.decrypt(nonce, ct, associated_data=None)
    return pt.decode("utf-8")


def pack_payload(payload_raw: bytes) -> bytes:
    # MAGIC + base64(payload_raw)
    return MAGIC + _b64e(payload_raw).encode("utf-8")


def unpack_payload(blob: bytes) -> bytes:
    if not blob.startswith(MAGIC):
        raise ValueError("No STEGCRYPT payload found (bad magic).")
    b64 = blob[len(MAGIC):].decode("utf-8")
    return _b64d(b64)


# -------------------------
# PNG: LSB embed/extract (RGB)
# -------------------------
def png_capacity_bytes(img_rgb: Image.Image) -> int:
    w, h = img_rgb.size
    channels = w * h * 3
    usable_bits = channels
    usable_bytes = usable_bits // 8
    # we store 4-byte length header inside bits too
    return max(0, usable_bytes - 4)


def png_embed(in_path: Path, out_path: Path, payload: bytes) -> None:
    im = Image.open(in_path).convert("RGB")
    arr = np.array(im, dtype=np.uint8)
    flat = arr.reshape(-1)  # includes all channels

    data = payload
    n = len(data)
    cap = png_capacity_bytes(im)
    if n > cap:
        raise ValueError(
            f"PNG capacity too small. Need {n} bytes, capacity {cap} bytes. "
            f"Use a larger PNG or shorter message."
        )

    header = n.to_bytes(4, byteorder="big")
    full = header + data
    bits = np.unpackbits(np.frombuffer(full, dtype=np.uint8))

    # write bits into LSBs
    flat[: len(bits)] = (flat[: len(bits)] & 0xFE) | bits
    out_arr = flat.reshape(arr.shape)
    out = Image.fromarray(out_arr, mode="RGB")
    out.save(out_path, format="PNG")


def png_extract(in_path: Path) -> bytes:
    im = Image.open(in_path).convert("RGB")
    arr = np.array(im, dtype=np.uint8).reshape(-1)
    lsb = arr & 1

    # first 32 bits = length
    header_bits = lsb[: 32]
    header = np.packbits(header_bits).tobytes()
    n = int.from_bytes(header, byteorder="big")

    if n <= 0:
        raise ValueError("No payload length found (length<=0).")

    total_bits = (4 + n) * 8
    data_bits = lsb[: total_bits]
    full = np.packbits(data_bits).tobytes()
    return full[4: 4 + n]


# -------------------------
# GIF: comment metadata embed/extract
# -------------------------
def gif_embed(in_path: Path, out_path: Path, payload: bytes) -> None:
    im = Image.open(in_path)

    # Put payload into GIF comment extension via Pillow "comment"
    # NOTE: this is metadata-based, not pixel-based.
    comment = payload
    save_kwargs = {}

    if getattr(im, "is_animated", False):
        frames = []
        try:
            for i in range(im.n_frames):
                im.seek(i)
                frames.append(im.copy())
        except EOFError:
            pass

        durations = []
        try:
            for f in range(len(frames)):
                im.seek(f)
                durations.append(im.info.get("duration", 100))
        except Exception:
            durations = None

        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            loop=im.info.get("loop", 0),
            duration=durations if durations else im.info.get("duration", 100),
            disposal=im.info.get("disposal", 2),
            optimize=False,
            comment=comment,
        )
    else:
        im.save(out_path, format="GIF", comment=comment)


def gif_extract(in_path: Path) -> bytes:
    im = Image.open(in_path)
    c = im.info.get("comment", None)
    if c is None:
        raise ValueError("No GIF comment payload found.")
    # Pillow may give str or bytes
    if isinstance(c, str):
        c = c.encode("utf-8")
    return c


# -------------------------
# Video: ffmpeg metadata + sidecar fallback
# -------------------------
def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def video_sidecar_path(media_path: Path) -> Path:
    return media_path.with_suffix(media_path.suffix + ".steg")


def video_embed(in_path: Path, out_path: Path, payload: bytes) -> Tuple[bool, Optional[Path]]:
    """
    Returns (embedded_in_video_metadata, sidecar_path_if_written)
    """
    payload_str = payload.decode("utf-8", errors="strict")
    sidecar = None

    # Always write sidecar as a robust fallback
    sidecar = video_sidecar_path(out_path)
    sidecar.write_text(payload_str, encoding="utf-8")

    if not have_ffmpeg():
        # No ffmpeg: cannot embed metadata, only sidecar.
        shutil.copy2(in_path, out_path)
        return False, sidecar

    # If payload too long, skip metadata embedding
    if len(payload_str) > VIDEO_META_MAX:
        shutil.copy2(in_path, out_path)
        return False, sidecar

    # Embed as metadata comment without re-encoding (-c copy)
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(in_path),
        "-map", "0",
        "-c", "copy",
        "-metadata", f"comment={payload_str}",
        str(out_path),
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        # If ffmpeg fails, fall back to plain copy + sidecar
        shutil.copy2(in_path, out_path)
        return False, sidecar

    return True, sidecar


def video_extract(in_path: Path) -> bytes:
    # Prefer metadata if possible
    if have_ffmpeg():
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format_tags=comment",
            "-of", "json",
            str(in_path),
        ]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if r.returncode == 0 and r.stdout.strip():
            try:
                obj = json.loads(r.stdout)
                tags = (obj.get("format", {}) or {}).get("tags", {}) or {}
                c = tags.get("comment", None)
                if c:
                    return c.encode("utf-8")
            except Exception:
                pass

    # Sidecar fallback
    sidecar = video_sidecar_path(in_path)
    if sidecar.exists():
        return sidecar.read_text(encoding="utf-8").encode("utf-8")

    raise ValueError("No video payload found (no metadata comment and no .steg sidecar).")


# -------------------------
# UI / Routing
# -------------------------
def prompt_path(prompt: str) -> Path:
    p = input(prompt).strip().strip('"').strip("'")
    path = Path(p).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def detect_kind(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in [".png"]:
        return "png"
    if ext in [".gif"]:
        return "gif"
    if ext in [".mp4", ".mov", ".mkv"]:
        return "video"
    raise ValueError(f"Unsupported file type: {ext} (use .png, .gif, .mp4/.mov/.mkv)")


def out_name(in_path: Path, suffix: str) -> Path:
    return in_path.with_name(in_path.stem + suffix + in_path.suffix)


def do_encrypt() -> None:
    media = prompt_path("Enter path to IMAGE/GIF/VIDEO (drag-drop into terminal): ")
    kind = detect_kind(media)

    print("\nType the message to encrypt. Finish with Enter.\n")
    msg = input("Message: ")
    pw = getpass("Key/Password (hidden): ")
    if not pw:
        raise ValueError("Key cannot be empty.")

    payload_raw = encrypt_message(msg, pw)
    blob = pack_payload(payload_raw)

    outp = out_name(media, "_enc")

    if kind == "png":
        png_embed(media, outp, blob)
        print(f"\n✅ Encrypted+embedded into PNG: {outp}")
        print(f"   Payload bytes: {len(blob)}  | PNG capacity bytes: {png_capacity_bytes(Image.open(media).convert('RGB'))}")
        return

    if kind == "gif":
        gif_embed(media, outp, blob)
        print(f"\n✅ Encrypted+embedded into GIF (comment metadata): {outp}")
        print(f"   Payload bytes: {len(blob)}")
        return

    if kind == "video":
        embedded, sidecar = video_embed(media, outp, blob)
        print(f"\n✅ Video output: {outp}")
        if embedded:
            print("   Embedded in video metadata (comment tag) ✅")
        else:
            print("   Could NOT embed into metadata (ffmpeg missing / too large / failed). Using sidecar only ⚠️")
        if sidecar:
            print(f"   Sidecar saved (robust fallback): {sidecar}")
        print(f"   Payload bytes: {len(blob)} (metadata max ~{VIDEO_META_MAX} chars)")
        return


def do_decrypt() -> None:
    media = prompt_path("Enter path to encrypted IMAGE/GIF/VIDEO: ")
    kind = detect_kind(media)
    pw = getpass("Key/Password (hidden): ")
    if not pw:
        raise ValueError("Key cannot be empty.")

    if kind == "png":
        blob = png_extract(media)
    elif kind == "gif":
        blob = gif_extract(media)
    else:
        blob = video_extract(media)

    payload_raw = unpack_payload(blob)
    msg = decrypt_message(payload_raw, pw)
    print("\n✅ Decrypted message:\n")
    print(msg)


def main() -> None:
    print("\n=== StegCrypt (PNG LSB | GIF comment | Video metadata+sidecar) ===\n")
    print("1) Encrypt (hide message in media)")
    print("2) Decrypt (extract message from media)\n")
    choice = input("Choose 1 or 2: ").strip()

    try:
        if choice == "1":
            do_encrypt()
        elif choice == "2":
            do_decrypt()
        else:
            raise ValueError("Invalid choice. Enter 1 or 2.")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
