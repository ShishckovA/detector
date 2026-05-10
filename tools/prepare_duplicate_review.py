#!/usr/bin/env python3

import argparse
import csv
import hashlib
import html
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.neighbors import NearestNeighbors


REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}


@dataclass(frozen=True)
class ImageItem:
    image_id: str
    path: Path
    rel_path: str
    source: str
    group: str
    width: int
    height: int
    bytes: int
    manifest: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build nearest-image duplicate review HTML from CNN embeddings.")
    parser.add_argument("--input", type=Path, default=Path("data/positives_original_search4faces_top20"))
    parser.add_argument("--out", type=Path, default=Path("reports/search4faces_duplicate_review"))
    parser.add_argument("--manifest", type=Path, default=None, help="Optional augmentation manifest.csv.")
    parser.add_argument("--top-pairs", type=int, default=800)
    parser.add_argument("--neighbors", type=int, default=12)
    parser.add_argument("--group-threshold", type=float, default=0.94)
    parser.add_argument("--thumb-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def rel_path(path: Path, base: Path = REPO_ROOT) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def rel_url(path: Path, base: Path) -> str:
    return html.escape(os.path.relpath(path.resolve(), base.resolve()))


def image_id(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]


def iter_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def read_manifest(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            output_path = row.get("output_path", "")
            if not output_path:
                continue
            path_obj = Path(output_path)
            if not path_obj.is_absolute():
                path_obj = REPO_ROOT / path_obj
            rows[str(path_obj.resolve())] = row
    return rows


def source_group(path: Path, root: Path, manifest_row: dict[str, str]) -> tuple[str, str]:
    if manifest_row.get("source") and manifest_row.get("group"):
        return manifest_row["source"], manifest_row["group"]
    try:
        rel = path.resolve().relative_to(root.resolve())
        if len(rel.parts) >= 3:
            return rel.parts[0], rel.parts[1]
    except ValueError:
        pass
    return "", ""


def collect_images(root: Path, manifest_rows: dict[str, dict[str, str]]) -> list[ImageItem]:
    items: list[ImageItem] = []
    for path in iter_images(root):
        try:
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image)
                width, height = image.size
        except (UnidentifiedImageError, OSError):
            continue
        manifest_row = manifest_rows.get(str(path.resolve()), {})
        source, group = source_group(path, root, manifest_row)
        items.append(
            ImageItem(
                image_id=image_id(path),
                path=path,
                rel_path=rel_path(path),
                source=source,
                group=group,
                width=width,
                height=height,
                bytes=path.stat().st_size,
                manifest=manifest_row,
            )
        )
    return items


def load_model(device: str) -> tuple[Any, Any, Any]:
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    return torch, model, weights.transforms()


def compute_embeddings(items: list[ImageItem], batch_size: int, device: str) -> tuple[list[ImageItem], np.ndarray]:
    torch, model, transform = load_model(device)
    embeddings: list[np.ndarray] = []
    batch = []
    batch_items: list[ImageItem] = []
    embedded_items: list[ImageItem] = []
    with torch.inference_mode():
        for index, item in enumerate(items, 1):
            try:
                with Image.open(item.path) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    batch.append(transform(image))
                    batch_items.append(item)
            except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
                print(f"skip missing/unreadable during embedding: {item.path} ({exc})", flush=True)
            if len(batch) == batch_size or index == len(items):
                if not batch:
                    continue
                tensor = torch.stack(batch).to(device)
                output = model(tensor).detach().cpu().numpy().astype(np.float32)
                embeddings.append(output)
                embedded_items.extend(batch_items)
                batch = []
                batch_items = []
                print(f"embedded {index}/{len(items)}", flush=True)
    matrix = np.concatenate(embeddings, axis=0)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embedded_items, matrix / norms


def find_pairs(embeddings: np.ndarray, neighbors: int) -> list[dict[str, Any]]:
    n_items = embeddings.shape[0]
    n_neighbors = min(neighbors + 1, n_items)
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(embeddings)
    distances, indices = nn.kneighbors(embeddings)
    seen: set[tuple[int, int]] = set()
    pairs: list[dict[str, Any]] = []
    for left_index in range(n_items):
        for distance, right_index in zip(distances[left_index][1:], indices[left_index][1:]):
            a, b = sorted((left_index, int(right_index)))
            if a == b or (a, b) in seen:
                continue
            seen.add((a, b))
            pairs.append({"left_index": a, "right_index": b, "similarity": 1.0 - float(distance)})
    pairs.sort(key=lambda row: row["similarity"], reverse=True)
    return pairs


def connected_components(item_count: int, pairs: list[dict[str, Any]], threshold: float) -> list[list[int]]:
    parent = list(range(item_count))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(a: int, b: int) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for pair in pairs:
        if pair["similarity"] < threshold:
            break
        union(int(pair["left_index"]), int(pair["right_index"]))

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(item_count):
        groups[find(index)].append(index)
    components = [sorted(values) for values in groups.values() if len(values) > 1]
    components.sort(key=lambda values: (-len(values), values[0]))
    return components


def write_thumbnails(items: list[ImageItem], thumb_dir: Path, thumb_size: int) -> dict[str, Path]:
    thumb_dir.mkdir(parents=True, exist_ok=True)
    thumbs: dict[str, Path] = {}
    for index, item in enumerate(items, 1):
        thumb_path = thumb_dir / f"{item.image_id}.jpg"
        thumbs[item.image_id] = thumb_path
        if thumb_path.exists() and thumb_path.stat().st_size > 0:
            continue
        try:
            with Image.open(item.path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (thumb_size, thumb_size), (238, 239, 236))
                offset = ((thumb_size - image.width) // 2, (thumb_size - image.height) // 2)
                canvas.paste(image, offset)
                canvas.save(thumb_path, format="JPEG", quality=86, optimize=True)
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            print(f"skip missing/unreadable during thumbnail: {item.path} ({exc})", flush=True)
            canvas = Image.new("RGB", (thumb_size, thumb_size), (245, 238, 238))
            canvas.save(thumb_path, format="JPEG", quality=80, optimize=True)
        if index % 200 == 0 or index == len(items):
            print(f"thumbs {index}/{len(items)}", flush=True)
    return thumbs


def write_images_csv(path: Path, items: list[ImageItem]) -> None:
    fieldnames = [
        "image_id",
        "path",
        "source",
        "group",
        "width",
        "height",
        "bytes",
        "seed_path",
        "rank",
        "similarity",
        "profile_key",
        "profile_url",
        "data_imgsrc_url",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "image_id": item.image_id,
                    "path": item.rel_path,
                    "source": item.source,
                    "group": item.group,
                    "width": item.width,
                    "height": item.height,
                    "bytes": item.bytes,
                    "seed_path": item.manifest.get("seed_path", ""),
                    "rank": item.manifest.get("rank", ""),
                    "similarity": item.manifest.get("similarity", ""),
                    "profile_key": item.manifest.get("profile_key", ""),
                    "profile_url": item.manifest.get("profile_url", ""),
                    "data_imgsrc_url": item.manifest.get("data_imgsrc_url", ""),
                }
            )


def write_pairs_csv(path: Path, items: list[ImageItem], pairs: list[dict[str, Any]]) -> None:
    fieldnames = [
        "pair_rank",
        "similarity",
        "left_id",
        "left_path",
        "left_source",
        "left_group",
        "right_id",
        "right_path",
        "right_source",
        "right_group",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for rank, pair in enumerate(pairs, 1):
            left = items[int(pair["left_index"])]
            right = items[int(pair["right_index"])]
            writer.writerow(
                {
                    "pair_rank": rank,
                    "similarity": f"{pair['similarity']:.9f}",
                    "left_id": left.image_id,
                    "left_path": left.rel_path,
                    "left_source": left.source,
                    "left_group": left.group,
                    "right_id": right.image_id,
                    "right_path": right.rel_path,
                    "right_source": right.source,
                    "right_group": right.group,
                }
            )


def card_html(item: ImageItem, thumb_path: Path, base_dir: Path) -> str:
    profile = item.manifest.get("profile_url", "")
    profile_link = f'<a href="{html.escape(profile)}" target="_blank">profile</a>' if profile else ""
    meta = " ".join(
        part
        for part in [
            html.escape(f"{item.source}/{item.group}" if item.source or item.group else ""),
            html.escape(f"{item.width}x{item.height}"),
            html.escape(f"rank {item.manifest.get('rank', '')}" if item.manifest.get("rank") else ""),
            html.escape(item.manifest.get("similarity", "")),
            profile_link,
        ]
        if part
    )
    return f"""
      <article class="image-card" data-image-id="{item.image_id}" data-path="{html.escape(item.rel_path)}">
        <a href="{rel_url(item.path, base_dir)}" target="_blank">
          <img src="{rel_url(thumb_path, base_dir)}" loading="lazy" alt="">
        </a>
        <div class="decision" role="group" aria-label="decision">
          <label><input type="radio" name="d-{item.image_id}" value="" checked> undecided</label>
          <label><input type="radio" name="d-{item.image_id}" value="keep"> keep</label>
          <label><input type="radio" name="d-{item.image_id}" value="delete"> delete</label>
        </div>
        <div class="meta">
          <strong>{html.escape(item.path.name)}</strong>
          <span>{meta}</span>
          <code>{html.escape(item.rel_path)}</code>
        </div>
      </article>
    """


def render_group_section(
    items: list[ImageItem],
    thumbs: dict[str, Path],
    components: list[list[int]],
    pairs: list[dict[str, Any]],
    base_dir: Path,
    max_groups: int = 250,
) -> str:
    pair_lookup: dict[tuple[int, int], float] = {}
    for pair in pairs:
        a, b = int(pair["left_index"]), int(pair["right_index"])
        pair_lookup[(a, b)] = pair["similarity"]
        pair_lookup[(b, a)] = pair["similarity"]
    parts = []
    for group_rank, component in enumerate(components[:max_groups], 1):
        best_similarity = 0.0
        for i, left in enumerate(component):
            for right in component[i + 1 :]:
                best_similarity = max(best_similarity, pair_lookup.get((left, right), 0.0))
        cards = "\n".join(card_html(items[index], thumbs[items[index].image_id], base_dir) for index in component)
        parts.append(
            f"""
            <section class="candidate-group" id="group-{group_rank}">
              <h2>Group {group_rank}: {len(component)} images, best similarity {best_similarity:.4f}</h2>
              <div class="group-tools">
                <button type="button" class="mark-first-button" onclick="markFirstKeepRestDelete(this)">Keep first, delete rest</button>
                <span class="muted">Tab/Shift+Tab moves between groups</span>
              </div>
              <div class="image-grid">{cards}</div>
            </section>
            """
        )
    return "\n".join(parts)


def render_pair_section(
    items: list[ImageItem],
    thumbs: dict[str, Path],
    pairs: list[dict[str, Any]],
    base_dir: Path,
    top_pairs: int,
) -> str:
    parts = []
    for rank, pair in enumerate(pairs[:top_pairs], 1):
        left = items[int(pair["left_index"])]
        right = items[int(pair["right_index"])]
        cards = "\n".join(
            [
                card_html(left, thumbs[left.image_id], base_dir),
                card_html(right, thumbs[right.image_id], base_dir),
            ]
        )
        parts.append(
            f"""
            <section class="pair" id="pair-{rank}">
              <h2>Pair {rank}: similarity {pair['similarity']:.4f}</h2>
              <div class="image-grid two">{cards}</div>
            </section>
            """
        )
    return "\n".join(parts)


def write_html(
    path: Path,
    items: list[ImageItem],
    thumbs: dict[str, Path],
    pairs: list[dict[str, Any]],
    components: list[list[int]],
    args: argparse.Namespace,
) -> None:
    base_dir = path.parent
    groups_html = render_group_section(items, thumbs, components, pairs, base_dir)
    pairs_html = render_pair_section(items, thumbs, pairs, base_dir, args.top_pairs)
    document = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Duplicate review</title>
  <style>
    body {{ margin: 0; font-family: system-ui, sans-serif; background: #f6f7f4; color: #171917; }}
    header {{ position: sticky; top: 0; z-index: 3; background: #fff; border-bottom: 1px solid #d8ddd5; padding: 12px 16px; }}
    h1 {{ margin: 0 0 6px; font-size: 20px; }}
    h2 {{ margin: 0 0 10px; font-size: 15px; }}
    p {{ margin: 0; color: #555; font-size: 13px; }}
    nav {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 10px; font-size: 13px; }}
    nav a, a {{ color: #1f6f4b; }}
    .tools {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; align-items: center; }}
    button {{ border: 1px solid #aeb8aa; background: #fff; border-radius: 6px; padding: 7px 10px; cursor: pointer; }}
    button:focus-visible {{ outline: 3px solid #4d8fcc; outline-offset: 2px; }}
    textarea {{ width: min(100%, 980px); height: 120px; margin-top: 10px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }}
    main {{ padding: 14px; }}
    .candidate-group, .pair {{ background: #fff; border: 1px solid #d8ddd5; border-radius: 8px; margin: 0 0 14px; padding: 12px; }}
    .group-tools {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin: 0 0 10px; }}
    .image-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 10px; }}
    .image-grid.two {{ grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }}
    .image-card {{ border: 1px solid #dfe4dc; border-radius: 6px; overflow: hidden; background: #fbfbfa; }}
    .image-card img {{ display: block; width: 100%; height: 230px; object-fit: contain; background: #eef0ec; }}
    .decision {{ display: flex; gap: 8px; flex-wrap: wrap; padding: 8px; border-top: 1px solid #e3e7e0; border-bottom: 1px solid #e3e7e0; font-size: 12px; }}
    .decision label {{ white-space: nowrap; }}
    .image-card[data-state="keep"] {{ outline: 3px solid #5aa05a; }}
    .image-card[data-state="delete"] {{ outline: 3px solid #c44e4e; opacity: 0.78; }}
    .meta {{ display: grid; gap: 4px; padding: 8px; font-size: 12px; overflow-wrap: anywhere; }}
    code {{ font-size: 11px; color: #555; }}
    .muted {{ color: #66706a; }}
  </style>
</head>
<body>
  <header>
    <h1>Duplicate review</h1>
    <p>{len(items)} current images, {len(pairs)} nearest pairs, {len(components)} candidate groups. Embedding: torchvision ResNet18 avgpool cosine.</p>
    <nav>
      <a href="#groups">Candidate groups</a>
      <a href="#pairs">Nearest pairs</a>
      <a href="pairs.csv">pairs.csv</a>
      <a href="images.csv">images.csv</a>
    </nav>
    <div class="tools">
      <button type="button" onclick="exportCsv()">Export decisions CSV</button>
      <button type="button" onclick="copyDeleteList()">Copy delete paths</button>
      <button type="button" onclick="clearDecisions()">Clear local decisions</button>
      <span class="muted" id="decision-counts"></span>
    </div>
    <textarea id="export" placeholder="Export appears here"></textarea>
  </header>
  <main>
    <h1 id="groups">Candidate groups</h1>
    {groups_html}
    <h1 id="pairs">Nearest pairs</h1>
    {pairs_html}
  </main>
  <script>
    const storageKey = "duplicate-review-decisions-v1";
    let decisions = JSON.parse(localStorage.getItem(storageKey) || "{{}}");
    function save() {{
      localStorage.setItem(storageKey, JSON.stringify(decisions));
      updateCounts();
    }}
    function applyCard(card) {{
      const id = card.dataset.imageId;
      const state = decisions[id] || "";
      card.dataset.state = state;
      card.querySelectorAll("input[type=radio]").forEach(input => {{
        input.checked = input.value === state;
      }});
    }}
    function applyAll() {{
      document.querySelectorAll(".image-card").forEach(applyCard);
      updateCounts();
    }}
    function setCardDecision(card, state) {{
      const id = card.dataset.imageId;
      if (state) decisions[id] = state;
      else delete decisions[id];
      document.querySelectorAll(`.image-card[data-image-id="${{id}}"]`).forEach(applyCard);
    }}
    function markFirstKeepRestDelete(button) {{
      const group = button.closest(".candidate-group");
      const cards = [...group.querySelectorAll(".image-card")];
      cards.forEach((card, index) => setCardDecision(card, index === 0 ? "keep" : "delete"));
      save();
    }}
    function groupButtons() {{
      return [...document.querySelectorAll(".mark-first-button")];
    }}
    function focusAdjacentGroupButton(currentButton, direction) {{
      const buttons = groupButtons();
      const currentIndex = buttons.indexOf(currentButton);
      if (currentIndex < 0 || buttons.length === 0) return;
      const nextIndex = (currentIndex + direction + buttons.length) % buttons.length;
      buttons[nextIndex].focus();
      buttons[nextIndex].closest(".candidate-group").scrollIntoView({{ block: "start", behavior: "smooth" }});
    }}
    document.addEventListener("change", event => {{
      const input = event.target;
      if (!input.matches(".decision input")) return;
      const card = input.closest(".image-card");
      setCardDecision(card, input.value);
      save();
    }});
    document.addEventListener("keydown", event => {{
      if (event.key !== "Tab") return;
      const button = event.target.closest?.(".mark-first-button");
      if (!button) return;
      event.preventDefault();
      focusAdjacentGroupButton(button, event.shiftKey ? -1 : 1);
    }});
    function uniqueCards() {{
      const map = new Map();
      document.querySelectorAll(".image-card").forEach(card => map.set(card.dataset.imageId, card));
      return [...map.values()];
    }}
    function decisionRows() {{
      return uniqueCards()
        .map(card => [card.dataset.imageId, decisions[card.dataset.imageId] || "", card.dataset.path])
        .filter(row => row[1]);
    }}
    function exportCsv() {{
      const rows = [["image_id","decision","path"], ...decisionRows()];
      document.getElementById("export").value = rows.map(row => row.map(cell => `"${{String(cell).replaceAll('"', '""')}}"`).join(",")).join("\\n");
    }}
    async function copyDeleteList() {{
      const text = decisionRows().filter(row => row[1] === "delete").map(row => row[2]).join("\\n");
      document.getElementById("export").value = text;
      try {{ await navigator.clipboard.writeText(text); }} catch (error) {{}}
    }}
    function clearDecisions() {{
      decisions = {{}};
      save();
      applyAll();
      document.getElementById("export").value = "";
    }}
    function updateCounts() {{
      let keep = 0, del = 0;
      for (const value of Object.values(decisions)) {{
        if (value === "keep") keep += 1;
        if (value === "delete") del += 1;
      }}
      document.getElementById("decision-counts").textContent = `keep: ${{keep}}, delete: ${{del}}`;
    }}
    applyAll();
  </script>
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.clean and args.out.exists():
        import shutil

        shutil.rmtree(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest
    if manifest_path is None:
        default_manifest = args.input / "manifest.csv"
        manifest_path = default_manifest if default_manifest.exists() else None

    manifest_rows = read_manifest(manifest_path)
    items = collect_images(args.input, manifest_rows)
    if len(items) < 2:
        raise ValueError(f"Need at least 2 readable images below {args.input}, found {len(items)}")

    items, embeddings = compute_embeddings(items, args.batch_size, args.device)
    if len(items) < 2:
        raise ValueError(f"Need at least 2 embeddable images below {args.input}, found {len(items)}")
    np.savez_compressed(
        args.out / "embeddings.npz",
        embeddings=embeddings,
        image_ids=np.array([item.image_id for item in items]),
        paths=np.array([item.rel_path for item in items]),
    )
    pairs = find_pairs(embeddings, args.neighbors)
    pairs = pairs[: max(args.top_pairs * 3, args.top_pairs)]
    components = connected_components(len(items), pairs, args.group_threshold)
    thumbs = write_thumbnails(items, args.out / "thumbs", args.thumb_size)

    top_pairs = pairs[: args.top_pairs]
    write_images_csv(args.out / "images.csv", items)
    write_pairs_csv(args.out / "pairs.csv", items, top_pairs)
    write_html(args.out / "index.html", items, thumbs, top_pairs, components, args)

    summary = {
        "input": rel_path(args.input),
        "out": rel_path(args.out),
        "manifest": rel_path(manifest_path) if manifest_path else None,
        "images": len(items),
        "embedding": "torchvision resnet18 avgpool, L2 normalized",
        "neighbors": args.neighbors,
        "top_pairs": len(top_pairs),
        "group_threshold": args.group_threshold,
        "candidate_groups": len(components),
        "max_similarity": top_pairs[0]["similarity"] if top_pairs else None,
        "min_top_pair_similarity": top_pairs[-1]["similarity"] if top_pairs else None,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
