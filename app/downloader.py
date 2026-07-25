"""Download the source video with yt-dlp (the only video-fetch network call).

We use the yt-dlp *Python API* rather than shelling out so failures surface as
exceptions we can translate into a clean 400. The server must never crash on a
bad or blocked URL.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Any, Callable, Optional, cast

import yt_dlp

from .models import InvalidVideoURLError
from .paths import DOWNLOADS_DIR

logger = logging.getLogger(__name__)

# Strip ANSI colour codes yt-dlp sometimes embeds in its error strings.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Browsers we'll borrow cookies from (in order) when YouTube throws up a
# sign-in / "confirm you're not a bot" wall. Override with env vars below.
_DEFAULT_BROWSERS = ["chrome", "edge", "brave", "firefox", "opera", "vivaldi"]
_COOKIE_FILE_ENV = "CLIPFORGE_COOKIES_FILE"       # path to a cookies.txt
_COOKIE_BROWSER_ENV = "CLIPFORGE_COOKIES_BROWSER"  # force one browser, e.g. "chrome"


def _needs_cookies(reason: str) -> bool:
    """True when a failure reason looks like a sign-in / bot / cookie wall —
    used to add a helpful cookie hint to the final error message.
    """
    r = (reason or "").lower()
    return any(k in r for k in (
        "sign in", "not a bot", "cookie", "log in", "login", "consent",
        "age", "members-only", "account", "authentication",
    ))


def _is_terminal(reason: str) -> bool:
    """True when no retry (other client / cookies) can possibly help, so we stop
    early instead of grinding through every fallback for a dead/blocked link.
    """
    r = (reason or "").lower()
    return any(k in r for k in (
        "unavailable", "been removed", "does not exist", "no longer",
        "unsupported url", "not available in your", "is not a valid",
        "deleted", "terminated",
    ))


# YouTube player clients to fall back through. Different clients are served by
# different endpoints, and the cookie-free mobile/TV ones frequently slip past
# the "confirm you're not a bot" wall the default web client trips.
_PLAYER_CLIENTS = ["android", "ios", "tv", "mweb"]


def _download_attempts(base_opts: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Ordered (label, ydl_opts) attempts.

    Order is cheapest-and-most-likely first:
      1. explicit cookies (only if the user configured a file/browser),
      2. the plain default pass (fast for public videos),
      3. cookie-free alternate YouTube player clients (dodge the bot wall),
      4. browser cookies (needs the browser closed on Windows) as a last resort.
    """
    cookie_file = os.environ.get(_COOKIE_FILE_ENV)
    forced = os.environ.get(_COOKIE_BROWSER_ENV)

    attempts: list[tuple[str, dict]] = []
    if cookie_file:
        attempts.append(("cookies file", {**base_opts, "cookiefile": cookie_file}))
    if forced:
        b = forced.strip().lower()
        attempts.append((f"{b} cookies", {**base_opts, "cookiesfrombrowser": (b,)}))

    attempts.append(("default", dict(base_opts)))

    for client in _PLAYER_CLIENTS:
        attempts.append((
            f"{client} client",
            {**base_opts, "extractor_args": {"youtube": {"player_client": [client]}}},
        ))

    if not forced:  # auto-try common browsers unless the user pinned one
        for b in _DEFAULT_BROWSERS:
            attempts.append((f"{b} cookies", {**base_opts, "cookiesfrombrowser": (b,)}))

    return attempts


def _clean_ydl_error(raw: str) -> str:
    """Turn a raw yt-dlp DownloadError string into one short, readable line.

    yt-dlp prefixes messages with ``ERROR:`` (sometimes coloured) and can append
    a hint about reporting bugs — we drop both and keep just the real reason so
    the UI can show *why* a video failed (private, age-gated, geo-blocked, etc.).
    """
    text = _ANSI_RE.sub("", raw or "").strip()
    # Keep only the first line — that's the human reason.
    line = text.splitlines()[0] if text else ""
    line = re.sub(r"^ERROR:\s*", "", line).strip()
    # Drop yt-dlp's "; please report this issue …" tail and extractor prefixes.
    line = re.split(r";\s*(please report|you might want)", line, maxsplit=1)[0].strip()
    line = re.sub(r"^\[[^\]]+\]\s*[^:]*:\s*", "", line)  # e.g. "[youtube] ID: "
    return line[:300]


def normalize_url(url: str) -> str:
    """Normalize video URLs (e.g. standardise YouTube links) so different URL formats
    of the same video produce identical cache keys.
    """
    clean = (url or "").strip()
    if not clean:
        return ""
    yt_match = re.search(r"(?:v=|\/shorts\/|\/embed\/|youtu\.be\/)([a-zA-Z0-9_-]{11})", clean)
    if yt_match:
        return f"https://www.youtube.com/watch?v={yt_match.group(1)}"
    return clean


def download_video(
    url: str, progress_hook: Optional[Callable[[dict], None]] = None
) -> Path:
    """Download `url` to downloads/<url_hash>.mp4 and return the file path (cached)."""
    if not url or not url.strip():
        raise InvalidVideoURLError("No video URL was provided.")

    url_clean = normalize_url(url)
    url_hash = hashlib.sha256(url_clean.encode("utf-8")).hexdigest()[:16]
    expected_path = DOWNLOADS_DIR / f"{url_hash}.mp4"

    # CACHE CHECK: If video already downloaded, reuse it immediately!
    if expected_path.exists() and expected_path.stat().st_size > 1024:
        logger.info("Reusing cached video download for %s: %s", url_clean, expected_path)
        if progress_hook:
            progress_hook({
                "status": "finished",
                "total_bytes": expected_path.stat().st_size,
                "downloaded_bytes": expected_path.stat().st_size
            })
        return expected_path

    existing = [
        p for p in sorted(DOWNLOADS_DIR.glob(f"{url_hash}.*"))
        if not p.name.endswith((".part", ".ytdl", ".temp", ".tmp")) and p.stat().st_size > 1024
    ]
    if existing:
        logger.info("Reusing cached video download for %s: %s", url_clean, existing[0])
        if progress_hook:
            progress_hook({
                "status": "finished",
                "total_bytes": existing[0].stat().st_size,
                "downloaded_bytes": existing[0].stat().st_size
            })
        return existing[0]

    out_template = str(DOWNLOADS_DIR / f"{url_hash}.%(ext)s")

    base_opts: dict[str, Any] = {
        # Prefer the best stream up to 1080p (plenty for shorts, avoids slow 4K
        # downloads). Container is normalised to mp4 by merge_output_format, so
        # we don't restrict by extension - that was too strict and could fall
        # back to a tiny stream when no progressive mp4 existed.
        "format": (
            "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best"
        ),
        "merge_output_format": "mp4",
        "outtmpl": out_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        # Be resilient: keep going if a single fragment hiccups.
        "ignoreerrors": False,
    }
    if progress_hook is not None:
        base_opts["progress_hooks"] = [progress_hook]

    # Try in order: explicit cookies (if set) → plain → browser cookies. We only
    # fall through to the cookie-based attempts when the failure looks like a
    # sign-in / bot wall, so public videos still download on the first (fast) try.
    last_reason = ""
    last_exc: Optional[Exception] = None
    ok = False
    for label, opts in _download_attempts(base_opts):
        try:
            with yt_dlp.YoutubeDL(cast(Any, opts)) as ydl:
                ydl.download([url.strip()])
            ok = True
            if label != "default":
                logger.info("Downloaded %s using %s", url, label)
            break
        except Exception as exc:  # noqa: BLE001 - never let the server crash here
            last_reason = _clean_ydl_error(str(exc))
            last_exc = exc
            logger.warning("yt-dlp [%s] failed for %s: %s", label, url, last_reason)
            if _is_terminal(last_reason):
                break  # dead/blocked link — no fallback can recover it

    if not ok:
        msg = ("Could not download that video. Check the URL is correct, public, "
               "and reachable from this machine.")
        if last_reason:
            msg += f"\nReason: {last_reason}"
        if _needs_cookies(last_reason):
            msg += (
                "\n\nThis video is behind YouTube's sign-in / bot check. Make sure "
                "you're logged into YouTube in your browser, then close the browser "
                "and try again. To pin a browser set CLIPFORGE_COOKIES_BROWSER "
                "(chrome/edge/firefox), or point CLIPFORGE_COOKIES_FILE at a cookies.txt."
            )
        raise InvalidVideoURLError(msg) from last_exc

    if expected_path.exists():
        return expected_path

    # Some sources may not produce exactly <url_hash>.mp4 (e.g. a different
    # container survived the merge). Fall back to any file with our url_hash prefix.
    candidates = sorted(DOWNLOADS_DIR.glob(f"{url_hash}.*"))
    if candidates:
        return candidates[0]

    raise InvalidVideoURLError(
        "The download completed but no output file was produced. The video may "
        "be unavailable or region-locked."
    )
