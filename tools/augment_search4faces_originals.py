#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEARCH_URL = "https://search4faces.com/search_vkwall.html"
DEFAULT_SOURCES = ("akykla", "inter-escort")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}
PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/146.0.0.0 YaBrowser/26.4.0.0 Safari/537.36"
)

MANIFEST_FIELDS = [
    "source",
    "group",
    "group_key",
    "seed_index",
    "seed_path",
    "seed_name",
    "rank",
    "similarity",
    "profile_key",
    "profile_url",
    "result_name",
    "data_imgsrc_url",
    "data_imghref_url",
    "output_path",
    "bytes",
    "status",
]
ERROR_FIELDS = [
    "source",
    "group",
    "group_key",
    "seed_index",
    "seed_path",
    "seed_name",
    "stage",
    "profile_key",
    "data_imgsrc_url",
    "data_imghref_url",
    "output_path",
    "error",
]


@dataclass(frozen=True)
class GroupPlan:
    source: str
    group: str
    seed_paths: tuple[Path, ...]

    @property
    def group_key(self) -> str:
        return f"{self.source}/{self.group}"


@dataclass(frozen=True)
class SearchTask:
    source: str
    group: str
    seed_index: int
    seed_path: Path

    @property
    def group_key(self) -> str:
        return f"{self.source}/{self.group}"


@dataclass(frozen=True)
class SearchResult:
    rank: int
    data_imgsrc_url: str
    data_imghref_url: str
    profile_url: str
    profile_key: str
    similarity: str
    result_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Augment positive original images with top search4faces results from data-imgsrc."
    )
    parser.add_argument("--positive-dir", type=Path, default=Path("data/positives_original"))
    parser.add_argument("--out", type=Path, default=Path("data/positives_original_search4faces_top20"))
    parser.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    parser.add_argument(
        "--seeds-per-group",
        type=int,
        default=3,
        help="How many original images to use as new seeds per group.",
    )
    parser.add_argument(
        "--top-results",
        type=int,
        default=20,
        help="Only consider this many DOM-ordered search4faces results per seed.",
    )
    parser.add_argument(
        "--max-per-profile",
        type=int,
        default=2,
        help="Maximum newly downloaded images from one search4faces profile per group.",
    )
    parser.add_argument(
        "--per-group",
        type=int,
        default=0,
        help="Optional cap for total valid images per group in the output. 0 disables the cap.",
    )
    parser.add_argument("--results", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--search-url", default=DEFAULT_SEARCH_URL)
    parser.add_argument("--limit-groups", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--headed", action="store_true", help="Run browser with a visible window.")
    parser.add_argument(
        "--browser-channel",
        default=None,
        help="Optional Playwright browser channel, e.g. chrome, to use an installed system browser.",
    )
    parser.add_argument("--slow-mo-ms", type=int, default=0)
    parser.add_argument("--page-timeout-ms", type=int, default=60000)
    parser.add_argument("--download-timeout-ms", type=int, default=60000)
    parser.add_argument("--wait-after-upload-ms", type=int, default=1500)
    parser.add_argument("--wait-after-search-ms", type=int, default=1500)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--keep-proxy-env",
        action="store_true",
        help="Do not remove proxy environment variables before browser/network calls.",
    )
    return parser.parse_args()


def rel_path(path: Path, base: Path = REPO_ROOT) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except ValueError:
        return str(path)


def iter_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return [dict(row) for row in reader]


def seed_tokens(path: Path) -> set[str]:
    return {rel_path(path), str(path), str(path.resolve())}


def row_has_seed_index(row: dict[str, str]) -> bool:
    return str(row.get("seed_index", "")).strip() != ""


def seed_path_from_row(row: dict[str, str]) -> Path | None:
    seed_path = row.get("seed_path", "")
    if not seed_path:
        return None
    path = Path(seed_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def seed_plan_state(rows: list[dict[str, str]]) -> tuple[dict[str, set[str]], dict[str, list[Path]]]:
    old_seed_paths: dict[str, set[str]] = defaultdict(set)
    planned_seed_paths: dict[str, dict[str, Path]] = defaultdict(dict)
    for row in rows:
        group_key = row.get("group_key") or f"{row.get('source', '')}/{row.get('group', '')}"
        path = seed_path_from_row(row)
        if not group_key or path is None:
            continue
        if row_has_seed_index(row):
            planned_seed_paths[group_key][str(path.resolve())] = path
            continue
        old_seed_paths[group_key].update(seed_tokens(path))
    return old_seed_paths, {group_key: list(paths.values()) for group_key, paths in planned_seed_paths.items()}


def select_group_plans(
    positive_dir: Path,
    sources: list[str],
    seed: int,
    seeds_per_group: int,
    excluded_old_seed_paths: dict[str, set[str]],
    planned_seed_paths: dict[str, list[Path]],
) -> list[GroupPlan]:
    rng = random.Random(seed)
    plans: list[GroupPlan] = []
    for source in sources:
        source_dir = positive_dir / source
        if not source_dir.exists():
            continue
        for group_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
            images = iter_images(group_dir)
            if not images:
                continue
            group_key = f"{source}/{group_dir.name}"
            planned = [path for path in planned_seed_paths.get(group_key, []) if path.exists()]
            planned = planned[:seeds_per_group]
            excluded = set(excluded_old_seed_paths.get(group_key, set()))
            for path in planned:
                excluded.update(seed_tokens(path))
            candidates = [path for path in images if seed_tokens(path).isdisjoint(excluded)]
            if not candidates:
                if planned:
                    plans.append(GroupPlan(source=source, group=group_dir.name, seed_paths=tuple(planned)))
                continue
            selected_count = min(seeds_per_group - len(planned), len(candidates))
            plans.append(
                GroupPlan(
                    source=source,
                    group=group_dir.name,
                    seed_paths=tuple([*planned, *rng.sample(candidates, selected_count)]),
                )
            )
    return plans


def plans_to_tasks(plans: list[GroupPlan]) -> list[SearchTask]:
    tasks: list[SearchTask] = []
    for plan in plans:
        for seed_index, seed_path in enumerate(plan.seed_paths, 1):
            tasks.append(
                SearchTask(
                    source=plan.source,
                    group=plan.group,
                    seed_index=seed_index,
                    seed_path=seed_path,
                )
            )
    return tasks


def import_playwright() -> tuple[Any, Any]:
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit(
            "Playwright is required for network mode. Install it with: "
            "python -m pip install playwright && python -m playwright install chromium. "
            "If Chrome is already installed, use --browser-channel chrome instead of installing Chromium."
        ) from exc
    return sync_playwright, PlaywrightTimeoutError


def clear_proxy_env() -> dict[str, str]:
    removed = {}
    for name in PROXY_ENV_VARS:
        if name in os.environ:
            removed[name] = os.environ.pop(name)
    return removed


def output_file_exists(row: dict[str, str]) -> bool:
    raw_output_path = row.get("output_path", "")
    if not raw_output_path:
        return False
    output_path = Path(raw_output_path)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    return output_path.exists() and output_path.stat().st_size > 0


def valid_manifest_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("status") == "downloaded" and output_file_exists(row)]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_artifacts(
    out: Path,
    manifest_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    write_csv(out / "manifest.csv", MANIFEST_FIELDS, manifest_rows)
    write_csv(out / "errors.csv", ERROR_FIELDS, error_rows)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_unique_error(error_rows: list[dict[str, Any]], error_keys: set[tuple[str, str, str, str]], row: dict[str, Any]) -> None:
    key = (
        str(row.get("group_key", "")),
        str(row.get("seed_path", "")),
        str(row.get("stage", "")),
        str(row.get("data_imgsrc_url", "")),
    )
    if key in error_keys:
        return
    error_rows.append(row)
    error_keys.add(key)


def record_missing_manifest_rows(
    raw_manifest_rows: list[dict[str, str]],
    error_rows: list[dict[str, Any]],
    error_keys: set[tuple[str, str, str, str]],
) -> None:
    for row in raw_manifest_rows:
        if row.get("status") != "downloaded" or output_file_exists(row):
            continue
        append_unique_error(
            error_rows,
            error_keys,
            {
                "source": row.get("source", ""),
                "group": row.get("group", ""),
                "group_key": row.get("group_key", ""),
                "seed_index": row.get("seed_index", ""),
                "seed_path": row.get("seed_path", ""),
                "seed_name": row.get("seed_name", ""),
                "stage": "missing_existing_file",
                "profile_key": row.get("profile_key", ""),
                "data_imgsrc_url": row.get("data_imgsrc_url", ""),
                "data_imghref_url": row.get("data_imghref_url", ""),
                "output_path": row.get("output_path", ""),
                "error": "manifest row ignored because output file is missing",
            },
        )


def existing_success_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        if row.get("status") != "downloaded":
            continue
        group_key = row.get("group_key") or f"{row.get('source', '')}/{row.get('group', '')}"
        counts[group_key] = counts.get(group_key, 0) + 1
    return counts


def seen_urls_by_group(
    raw_manifest_rows: list[dict[str, str]],
    valid_rows: list[dict[str, str]],
    error_rows: list[dict[str, Any]],
) -> dict[str, set[str]]:
    seen: dict[str, set[str]] = defaultdict(set)
    for row in [*raw_manifest_rows, *valid_rows]:
        url = row.get("data_imgsrc_url", "")
        if not url:
            continue
        group_key = row.get("group_key") or f"{row.get('source', '')}/{row.get('group', '')}"
        seen[group_key].add(url)
    for row in error_rows:
        if row.get("stage") != "missing_existing_file":
            continue
        url = row.get("data_imgsrc_url", "")
        if not url:
            continue
        group_key = row.get("group_key") or f"{row.get('source', '')}/{row.get('group', '')}"
        seen[group_key].add(url)
    return seen


def profile_counts_by_group(rows: list[dict[str, str]]) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        profile_key = row.get("profile_key", "")
        if not profile_key:
            continue
        group_key = row.get("group_key") or f"{row.get('source', '')}/{row.get('group', '')}"
        counts[group_key][profile_key] += 1
    return counts


def set_results_count(page: Any, results: int) -> bool:
    return bool(
        page.evaluate(
            """
            (results) => {
              const target = String(results);
              for (const select of document.querySelectorAll('select')) {
                const option = Array.from(select.options).find((item) => {
                  return item.value === target || item.textContent.trim() === target;
                });
                if (option) {
                  select.value = option.value;
                  select.dispatchEvent(new Event('change', { bubbles: true }));
                  return true;
                }
              }
              for (const input of document.querySelectorAll('input[type="radio"]')) {
                if (input.value === target) {
                  input.checked = true;
                  input.dispatchEvent(new Event('change', { bubbles: true }));
                  return true;
                }
              }
              return false;
            }
            """,
            results,
        )
    )


def click_first(page: Any, selectors: list[str], timeout_ms: int = 5000, force: bool = False) -> bool:
    for selector in selectors:
        locator = page.locator(selector).first
        try:
            if locator.count() == 0:
                continue
            locator.click(timeout=timeout_ms, force=force)
            return True
        except Exception:
            try:
                locator.evaluate("(element) => element.click()")
                return True
            except Exception:
                continue
    return False


def profile_key_from_urls(profile_url: str, data_imghref_url: str, data_imgsrc_url: str) -> str:
    for url in (profile_url, data_imghref_url):
        parsed = urlparse(url)
        if parsed.netloc.lower().endswith("vk.com"):
            path = parsed.path.strip("/")
            if path:
                return f"vk:{path.lower()}"
    match = re.search(r"photo(-?\d+)_", data_imghref_url)
    if match:
        return f"vk:id{match.group(1)}"
    digest = hashlib.sha1(data_imgsrc_url.encode("utf-8")).hexdigest()[:12]
    return f"unknown:{digest}"


def search_results(page: Any, task: SearchTask, args: argparse.Namespace) -> list[SearchResult]:
    page.goto(args.search_url, wait_until="domcontentloaded", timeout=args.page_timeout_ms)
    set_results_count(page, args.results)

    click_first(page, ["#upload-button", ".uppload-button"], force=True)
    file_input = page.locator("input[type=file]").first
    file_input.set_input_files(str(task.seed_path), timeout=args.page_timeout_ms)
    page.wait_for_timeout(args.wait_after_upload_ms)

    upload_confirmed = click_first(
        page,
        [
            ".effects-continue--upload",
            "button:has-text('Upload')",
            "button:has-text('\\u0417\\u0430\\u0433\\u0440\\u0443\\u0437\\u0438\\u0442\\u044c')",
        ],
        force=True,
    )
    if upload_confirmed:
        page.wait_for_function(
            """
            () => {
              const button = document.querySelector('#search-button');
              return button && !button.disabled;
            }
            """,
            timeout=args.page_timeout_ms,
        )

    search_clicked = click_first(
        page,
        [
            "#search-button",
            "[id='search-button']",
            "button:has-text('Find')",
            "button:has-text('\\u041d\\u0430\\u0439\\u0442\\u0438')",
            "input[type='button'][value*='Find']",
            "input[type='button'][value*='\\u041d\\u0430\\u0439\\u0442\\u0438']",
        ],
        force=True,
    )
    if not search_clicked:
        page.keyboard.press("Enter")

    page.wait_for_timeout(args.wait_after_search_ms)
    try:
        page.wait_for_selector("[data-imgsrc]", timeout=args.page_timeout_ms)
    except Exception:
        return []

    raw_results = page.eval_on_selector_all(
        "[data-imgsrc]",
        """
        (items, topResults) => items.slice(0, topResults).map((item, idx) => {
          const card = item.closest('.card-vk01-body') || item.closest('.card') || item.parentElement?.parentElement;
          const links = Array.from((card || item).querySelectorAll('a'));
          const profileLink = links.find((link) => {
            const text = link.textContent.trim().toLowerCase();
            return text.includes('profile') || text.includes('\\u043f\\u0440\\u043e\\u0444\\u0438\\u043b');
          }) || links.find((link) => link.href && link.href.startsWith('https://vk.com/'));
          return {
            rank: idx + 1,
            data_imgsrc_url: item.getAttribute('data-imgsrc') || '',
            data_imghref_url: item.getAttribute('data-imghref') || '',
            profile_url: profileLink?.href || '',
            similarity: card?.querySelector('.score-label')?.textContent.trim() || '',
            result_name: card?.querySelector('.card-vk01-header')?.textContent.trim() || '',
          };
        }).filter((item) => item.data_imgsrc_url)
        """,
        args.top_results,
    )
    results: list[SearchResult] = []
    for raw in raw_results:
        data_imgsrc_url = urljoin(page.url, raw["data_imgsrc_url"])
        data_imghref_url = urljoin(page.url, raw.get("data_imghref_url", ""))
        profile_url = urljoin(page.url, raw.get("profile_url", ""))
        results.append(
            SearchResult(
                rank=int(raw["rank"]),
                data_imgsrc_url=data_imgsrc_url,
                data_imghref_url=data_imghref_url,
                profile_url=profile_url,
                profile_key=profile_key_from_urls(profile_url, data_imghref_url, data_imgsrc_url),
                similarity=raw.get("similarity", ""),
                result_name=raw.get("result_name", ""),
            )
        )
    return results


def target_path(out: Path, task: SearchTask, result: SearchResult) -> Path:
    digest = hashlib.sha1(result.data_imgsrc_url.encode("utf-8")).hexdigest()[:12]
    return (
        out
        / task.source
        / task.group
        / f"s4f_seed{task.seed_index:02d}_rank{result.rank:03d}_{digest}.jpg"
    )


def download_data_imgsrc(
    request_context: Any,
    page_url: str,
    url: str,
    output_path: Path,
    timeout_ms: int,
) -> int:
    response = request_context.get(
        url,
        headers={
            "Referer": page_url,
            "User-Agent": USER_AGENT,
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        },
        timeout=timeout_ms,
    )
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status}")
    body = response.body()
    if not body:
        raise RuntimeError("empty response body")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        written = tmp_path.write_bytes(body)
        if written != len(body):
            raise RuntimeError(f"incomplete write: {written} of {len(body)} bytes")
        tmp_path.replace(output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    if not output_path.exists() or output_path.stat().st_size != len(body):
        raise RuntimeError("downloaded file was not written correctly")
    return len(body)


def dry_run_payload(plans: list[GroupPlan], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "positive_dir": rel_path(args.positive_dir),
        "out": rel_path(args.out),
        "sources": args.sources,
        "seeds_per_group": args.seeds_per_group,
        "top_results": args.top_results,
        "max_per_profile": args.max_per_profile,
        "per_group": args.per_group,
        "results": args.results,
        "seed": args.seed,
        "groups": [group_plan_payload(plan) for plan in plans],
        "group_count": len(plans),
        "search_task_count": len(plans_to_tasks(plans)),
    }


def group_plan_payload(plan: GroupPlan) -> dict[str, Any]:
    return {
        "source": plan.source,
        "group": plan.group,
        "group_key": plan.group_key,
        "seed_paths": [rel_path(path) for path in plan.seed_paths],
        "seed_names": [path.name for path in plan.seed_paths],
    }


def task_error_row(task: SearchTask, stage: str, result: SearchResult | None, error: str) -> dict[str, str]:
    return {
        "source": task.source,
        "group": task.group,
        "group_key": task.group_key,
        "seed_index": str(task.seed_index),
        "seed_path": rel_path(task.seed_path),
        "seed_name": task.seed_path.name,
        "stage": stage,
        "profile_key": result.profile_key if result else "",
        "data_imgsrc_url": result.data_imgsrc_url if result else "",
        "data_imghref_url": result.data_imghref_url if result else "",
        "output_path": "",
        "error": error,
    }


def current_summary(
    args: argparse.Namespace,
    plans: list[GroupPlan],
    tasks: list[SearchTask],
    manifest_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    downloaded_by_group: dict[str, int] = {}
    downloaded_by_source: dict[str, int] = {}
    for row in manifest_rows:
        if row.get("status") != "downloaded":
            continue
        group_key = str(row.get("group_key", ""))
        source = str(row.get("source", ""))
        downloaded_by_group[group_key] = downloaded_by_group.get(group_key, 0) + 1
        downloaded_by_source[source] = downloaded_by_source.get(source, 0) + 1
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "positive_dir": rel_path(args.positive_dir),
        "out": rel_path(args.out),
        "search_url": args.search_url,
        "sources": args.sources,
        "seed": args.seed,
        "seeds_per_group": args.seeds_per_group,
        "top_results": args.top_results,
        "max_per_profile": args.max_per_profile,
        "per_group": args.per_group,
        "results": args.results,
        "eligible_groups": len(plans),
        "search_tasks": len(tasks),
        "downloaded": len([row for row in manifest_rows if row.get("status") == "downloaded"]),
        "groups_with_downloads": len(downloaded_by_group),
        "errors": len(error_rows),
        "downloaded_by_source": dict(sorted(downloaded_by_source.items())),
        "downloaded_by_group": dict(sorted(downloaded_by_group.items())),
    }


def launch_browser(playwright: Any, args: argparse.Namespace) -> Any:
    launch_kwargs: dict[str, Any] = {"headless": not args.headed, "slow_mo": args.slow_mo_ms}
    if args.browser_channel:
        launch_kwargs["channel"] = args.browser_channel
        return playwright.chromium.launch(**launch_kwargs)

    try:
        return playwright.chromium.launch(**launch_kwargs)
    except Exception:
        launch_kwargs["channel"] = "chrome"
        print("Bundled Playwright Chromium is unavailable; retrying with installed Chrome.", flush=True)
        return playwright.chromium.launch(**launch_kwargs)


def process_tasks(args: argparse.Namespace, plans: list[GroupPlan], tasks: list[SearchTask]) -> int:
    if not args.keep_proxy_env:
        clear_proxy_env()

    sync_playwright, PlaywrightTimeoutError = import_playwright()
    args.out.mkdir(parents=True, exist_ok=True)

    raw_manifest_rows = read_rows(args.out / "manifest.csv") if args.resume else []
    manifest_rows: list[dict[str, Any]] = valid_manifest_rows(raw_manifest_rows) if args.resume else []
    error_rows: list[dict[str, Any]] = read_rows(args.out / "errors.csv") if args.resume else []
    error_keys: set[tuple[str, str, str, str]] = {
        (
            str(row.get("group_key", "")),
            str(row.get("seed_path", "")),
            str(row.get("stage", "")),
            str(row.get("data_imgsrc_url", "")),
        )
        for row in error_rows
    }
    record_missing_manifest_rows(raw_manifest_rows, error_rows, error_keys)

    downloaded_counts = existing_success_counts(manifest_rows)
    seen_urls = seen_urls_by_group(raw_manifest_rows, manifest_rows, error_rows)
    profile_counts = profile_counts_by_group(manifest_rows)

    processed_tasks = 0
    skipped_tasks = 0

    def save_progress() -> None:
        manifest_rows[:] = valid_manifest_rows(manifest_rows)
        write_artifacts(args.out, manifest_rows, error_rows, current_summary(args, plans, tasks, manifest_rows, error_rows))

    save_progress()

    with sync_playwright() as playwright:
        browser = launch_browser(playwright, args)
        context = browser.new_context(
            user_agent=USER_AGENT,
            locale="ru-RU",
            extra_http_headers={"Accept-Language": "ru,en;q=0.9"},
        )
        page = context.new_page()

        try:
            for task_index, task in enumerate(tasks, 1):
                if args.per_group and downloaded_counts.get(task.group_key, 0) >= args.per_group:
                    skipped_tasks += 1
                    print(
                        f"[{task_index}/{len(tasks)}] skip {task.group_key} seed#{task.seed_index}: "
                        f"group has {downloaded_counts.get(task.group_key, 0)}",
                        flush=True,
                    )
                    continue

                print(
                    f"[{task_index}/{len(tasks)}] search {task.group_key} seed#{task.seed_index} "
                    f"seed={rel_path(task.seed_path)}",
                    flush=True,
                )
                try:
                    results = search_results(page, task, args)
                except PlaywrightTimeoutError as exc:
                    append_unique_error(error_rows, error_keys, task_error_row(task, "search", None, f"timeout: {exc}"))
                    save_progress()
                    continue
                except Exception as exc:
                    append_unique_error(error_rows, error_keys, task_error_row(task, "search", None, str(exc)))
                    save_progress()
                    continue

                if not results:
                    append_unique_error(error_rows, error_keys, task_error_row(task, "search", None, "no data-imgsrc results"))
                    save_progress()
                    continue

                downloaded_this_task = 0
                skipped_duplicate = 0
                skipped_profile = 0
                local_seen_urls: set[str] = set()

                for result in results:
                    if result.data_imgsrc_url in local_seen_urls:
                        skipped_duplicate += 1
                        continue
                    local_seen_urls.add(result.data_imgsrc_url)

                    if result.data_imgsrc_url in seen_urls.setdefault(task.group_key, set()):
                        skipped_duplicate += 1
                        continue
                    if (
                        args.max_per_profile
                        and profile_counts.setdefault(task.group_key, Counter())[result.profile_key] >= args.max_per_profile
                    ):
                        skipped_profile += 1
                        continue
                    if args.per_group and downloaded_counts.get(task.group_key, 0) >= args.per_group:
                        break

                    output_path = target_path(args.out, task, result)
                    try:
                        byte_count = download_data_imgsrc(
                            context.request,
                            page.url,
                            result.data_imgsrc_url,
                            output_path,
                            args.download_timeout_ms,
                        )
                    except Exception as exc:
                        error_row = task_error_row(task, "download", result, str(exc))
                        error_row["output_path"] = rel_path(output_path)
                        append_unique_error(error_rows, error_keys, error_row)
                        save_progress()
                        continue

                    manifest_rows.append(
                        {
                            "source": task.source,
                            "group": task.group,
                            "group_key": task.group_key,
                            "seed_index": task.seed_index,
                            "seed_path": rel_path(task.seed_path),
                            "seed_name": task.seed_path.name,
                            "rank": result.rank,
                            "similarity": result.similarity,
                            "profile_key": result.profile_key,
                            "profile_url": result.profile_url,
                            "result_name": result.result_name,
                            "data_imgsrc_url": result.data_imgsrc_url,
                            "data_imghref_url": result.data_imghref_url,
                            "output_path": rel_path(output_path),
                            "bytes": byte_count,
                            "status": "downloaded",
                        }
                    )
                    seen_urls[task.group_key].add(result.data_imgsrc_url)
                    profile_counts[task.group_key][result.profile_key] += 1
                    downloaded_counts[task.group_key] = downloaded_counts.get(task.group_key, 0) + 1
                    downloaded_this_task += 1

                processed_tasks += 1
                print(
                    f"[{task_index}/{len(tasks)}] done {task.group_key} seed#{task.seed_index}: "
                    f"downloaded={downloaded_this_task}, group_total={downloaded_counts.get(task.group_key, 0)}, "
                    f"skipped_duplicate={skipped_duplicate}, skipped_profile={skipped_profile}",
                    flush=True,
                )
                save_progress()
        finally:
            save_progress()
            context.close()
            browser.close()

    summary = current_summary(args, plans, tasks, manifest_rows, error_rows)
    summary["processed_tasks"] = processed_tasks
    summary["skipped_tasks"] = skipped_tasks
    write_artifacts(args.out, manifest_rows, error_rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = parse_args()
    if args.seeds_per_group <= 0:
        raise ValueError(f"--seeds-per-group must be positive, got {args.seeds_per_group}")
    if args.top_results <= 0:
        raise ValueError(f"--top-results must be positive, got {args.top_results}")
    if args.max_per_profile < 0:
        raise ValueError(f"--max-per-profile must be non-negative, got {args.max_per_profile}")
    if args.per_group < 0:
        raise ValueError(f"--per-group must be non-negative, got {args.per_group}")
    if args.results <= 0:
        raise ValueError(f"--results must be positive, got {args.results}")

    resume_rows = []
    if args.resume:
        resume_rows = [
            *read_rows(args.out / "manifest.csv"),
            *read_rows(args.out / "errors.csv"),
        ]
    excluded_old_seed_paths, planned_seed_paths = seed_plan_state(resume_rows)
    plans = select_group_plans(
        args.positive_dir,
        args.sources,
        args.seed,
        args.seeds_per_group,
        excluded_old_seed_paths,
        planned_seed_paths,
    )
    if args.limit_groups is not None:
        plans = plans[: args.limit_groups]
    tasks = plans_to_tasks(plans)
    if not tasks:
        raise FileNotFoundError(f"No eligible seed images found below {args.positive_dir} for sources {args.sources}")

    if args.dry_run:
        print(json.dumps(dry_run_payload(plans, args), ensure_ascii=False, indent=2))
        return 0

    return process_tasks(args, plans, tasks)


if __name__ == "__main__":
    raise SystemExit(main())
