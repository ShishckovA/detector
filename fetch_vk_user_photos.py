#!/usr/bin/env python3

import argparse
import csv
import json
import mimetypes
import time
from pathlib import Path
from urllib.parse import urlparse

import cv2 as cv
import requests


API_VERSION = "5.199"


def chunked(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def api_call(session, method, token, params, retries=3):
    url = f"https://api.vk.com/method/{method}"
    payload = dict(params)
    payload["access_token"] = token
    payload["v"] = API_VERSION

    for attempt in range(retries):
        response = session.get(url, params=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        error = data.get("error")
        if not error:
            return data["response"]

        code = error.get("error_code")
        if code == 6 and attempt + 1 < retries:
            time.sleep(0.5 * (attempt + 1))
            continue
        raise RuntimeError(f"VK API {method} failed: {code} {error.get('error_msg')}")

    raise RuntimeError(f"VK API {method} failed after retries")


def best_size(sizes):
    if not sizes:
        return None
    return max(sizes, key=lambda item: int(item.get("width") or 0) * int(item.get("height") or 0))


def url_extension(url, content_type=None):
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return ".jpg" if suffix == ".jpeg" else suffix
    if content_type:
        guessed = mimetypes.guess_extension(content_type.split(";", 1)[0].strip())
        if guessed:
            return ".jpg" if guessed == ".jpe" else guessed
    return ".jpg"


def download(session, url, path_without_ext):
    response = session.get(url, timeout=60)
    response.raise_for_status()
    ext = url_extension(url, response.headers.get("content-type"))
    path = path_without_ext.with_suffix(ext)
    path.write_bytes(response.content)
    image = cv.imread(str(path))
    if image is None:
        return path, len(response.content), 0, 0
    height, width = image.shape[:2]
    return path, len(response.content), width, height


def load_ids(path):
    ids = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        uid = line.strip()
        if uid and uid not in seen:
            seen.add(uid)
            ids.append(uid)
    return ids


def parse_args():
    parser = argparse.ArgumentParser(description="Download VK avatars and previous profile avatars.")
    parser.add_argument("--ids", type=Path, default=Path("query_user_ids.txt"))
    parser.add_argument("--token", type=Path, default=Path(".token"))
    parser.add_argument("--out", type=Path, default=Path("vk_user_photos"))
    parser.add_argument("--sleep", type=float, default=0.34)
    parser.add_argument("--min-normal-size", type=int, default=241)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    token = args.token.read_text(encoding="utf-8").strip()
    ids = load_ids(args.ids)
    args.out.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    manifest_rows = []
    errors = []
    skipped_small = []

    for index, uid in enumerate(ids, 1):
        user_dir = args.out / uid
        user_dir.mkdir(parents=True, exist_ok=True)

        try:
            photos = api_call(
                session,
                "photos.get",
                token,
                {
                    "owner_id": uid,
                    # Public VK URLs show this album as album{owner_id}_0.
                    # The API exposes the same profile-photo album via this alias.
                    "album_id": "profile",
                    "rev": 1,
                    "count": 2,
                    "photo_sizes": 1,
                },
            )
            items = photos.get("items", [])
            for kind, item in (("avatar", items[0] if len(items) >= 1 else None), ("previous_avatar", items[1] if len(items) >= 2 else None)):
                size_info = best_size(item.get("sizes") if item else None)
                photo_url = size_info.get("url") if size_info else None
                if not photo_url:
                    continue
                target = user_dir / kind
                if not args.resume or not any(user_dir.glob(f"{kind}.*")):
                    path, byte_count, actual_width, actual_height = download(session, photo_url, target)
                else:
                    existing = next(user_dir.glob(f"{kind}.*"))
                    image = cv.imread(str(existing))
                    actual_height, actual_width = image.shape[:2] if image is not None else (0, 0)
                    path, byte_count = existing, existing.stat().st_size
                if actual_width < args.min_normal_size or actual_height < args.min_normal_size:
                    if path.exists():
                        path.unlink()
                    skipped_small.append(
                        {
                            "user_id": uid,
                            "kind": kind,
                            "photo_id": item.get("id"),
                            "width": actual_width,
                            "height": actual_height,
                            "source_url": photo_url,
                            "reason": f"smaller_than_{args.min_normal_size}",
                        }
                    )
                    continue
                manifest_rows.append(
                    {
                        "user_id": uid,
                        "kind": kind,
                        "photo_id": item.get("id"),
                        "source_url": photo_url,
                        "file_path": str(path),
                        "width": actual_width,
                        "height": actual_height,
                        "bytes": byte_count,
                    }
                )
        except Exception as exc:
            errors.append({"user_id": uid, "kind": "profile_album", "error": str(exc)})

        if index % 50 == 0 or index == len(ids):
            print(f"processed {index}/{len(ids)}, files={len(manifest_rows)}, errors={len(errors)}")
        time.sleep(args.sleep)

    with (args.out / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["user_id", "kind", "photo_id", "source_url", "file_path", "width", "height", "bytes"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    with (args.out / "errors.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["user_id", "kind", "error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(errors)

    with (args.out / "skipped_small.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["user_id", "kind", "photo_id", "width", "height", "source_url", "reason"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(skipped_small)

    summary = {
        "ids": len(ids),
        "files": len(manifest_rows),
        "errors": len(errors),
        "skipped_small": len(skipped_small),
        "min_normal_size": args.min_normal_size,
        "output": str(args.out),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
