"""Fetch, unpack and validate training videos/images; write captions."""

from __future__ import annotations

import logging
import os
import re
import shutil
import tarfile
import zipfile
from dataclasses import dataclass, field
from typing import Any

import requests

from .secret_utils import redact

log = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
VIDEO_EXTS = (".mp4", ".webm", ".mov", ".mkv")
MEDIA_EXTS = IMAGE_EXTS + VIDEO_EXTS
MAX_DOWNLOAD_BYTES = 8 * 2**30
MAX_UNPACKED_BYTES = 8 * 2**30
MIN_IMAGE_SIDE = 256


class DatasetError(ValueError):
    pass


@dataclass
class DatasetReport:
    source: str
    videos_found: int = 0
    images_found: int = 0
    videos_used: int = 0
    images_used: int = 0
    skipped: list[str] = field(default_factory=list)
    captions_from_files: int = 0
    captions_trigger_only: int = 0
    example_captions: list[str] = field(default_factory=list)
    videos_dir: str = ""
    images_dir: str = ""
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["skipped"] = self.skipped[:20]
        return d

    @property
    def has_videos(self) -> bool:
        return self.videos_used > 0

    @property
    def has_images(self) -> bool:
        return self.images_used > 0


def _download(url: str, dest: str) -> str:
    log.info("downloading dataset archive")
    with requests.get(url, stream=True, timeout=(15, 300), allow_redirects=True) as r:
        if r.status_code != 200:
            raise DatasetError(f"dataset_url returned HTTP {r.status_code}")
        size = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                size += len(chunk)
                if size > MAX_DOWNLOAD_BYTES:
                    raise DatasetError("dataset archive is larger than 8 GiB")
                f.write(chunk)
    return dest


def _safe_member(base: str, name: str) -> str | None:
    # Zip-slip: refuse members that would escape dest (including names with .. or abs paths).
    target = os.path.realpath(os.path.join(base, name))
    root = os.path.realpath(base)
    if not target.startswith(root + os.sep) and target != root:
        return None
    return target


def _extract(archive: str, dest: str) -> None:
    os.makedirs(dest, exist_ok=True)
    total = 0
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for m in z.infolist():
                if m.is_dir():
                    continue
                total += m.file_size
                if total > MAX_UNPACKED_BYTES:
                    raise DatasetError("dataset archive unpacks to more than 8 GiB")
                target = _safe_member(dest, m.filename)
                if target is None:
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with z.open(m) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
        return
    try:
        with tarfile.open(archive) as t:
            for m in t.getmembers():
                if not m.isfile():
                    continue
                total += m.size
                if total > MAX_UNPACKED_BYTES:
                    raise DatasetError("dataset archive unpacks to more than 8 GiB")
                target = _safe_member(dest, m.name)
                if target is None:
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with t.extractfile(m) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
    except tarfile.ReadError as exc:
        raise DatasetError("dataset_url must point to a .zip or .tar(.gz) archive") from exc


def _is_junk(name: str) -> bool:
    base = os.path.basename(name)
    return base.startswith(".") or base.startswith("__MACOSX") or "/__MACOSX/" in name.replace("\\", "/")


def _caption_for(path: str) -> str | None:
    txt = os.path.splitext(path)[0] + ".txt"
    if os.path.isfile(txt):
        with open(txt, encoding="utf-8", errors="replace") as fh:
            cap = re.sub(r"\s+", " ", fh.read()).strip()
            return cap or None
    return None


def _collect(root: str) -> tuple[list[tuple[str, str | None]], list[tuple[str, str | None]]]:
    """Return (videos, images) as (path, caption) pairs. Honors videos/ and images/ folders if present."""
    videos_root = None
    images_root = None
    for dirpath, dirnames, _files in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith((".", "__MACOSX"))]
        lower = {d.lower(): os.path.join(dirpath, d) for d in dirnames}
        if "videos" in lower and videos_root is None:
            videos_root = lower["videos"]
        if "images" in lower and images_root is None:
            images_root = lower["images"]

    def walk(base: str, exts: tuple[str, ...]) -> list[tuple[str, str | None]]:
        out = []
        for dirpath, dirnames, files in os.walk(base):
            dirnames[:] = [d for d in dirnames if not d.startswith((".", "__MACOSX"))]
            for f in sorted(files):
                if _is_junk(f) or not f.lower().endswith(exts):
                    continue
                p = os.path.join(dirpath, f)
                out.append((p, _caption_for(p)))
        out.sort(key=lambda x: x[0])
        return out

    if videos_root or images_root:
        videos = walk(videos_root, VIDEO_EXTS) if videos_root else []
        images = walk(images_root, IMAGE_EXTS) if images_root else []
        return videos, images

    # Flat (or mixed) archive: classify by extension, skip other folders' junk.
    videos, images = [], []
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith((".", "__MACOSX"))]
        for f in sorted(files):
            if _is_junk(f):
                continue
            p = os.path.join(dirpath, f)
            low = f.lower()
            if low.endswith(VIDEO_EXTS):
                videos.append((p, _caption_for(p)))
            elif low.endswith(IMAGE_EXTS):
                images.append((p, _caption_for(p)))
    videos.sort(key=lambda x: x[0])
    images.sort(key=lambda x: x[0])
    return videos, images


def _base_caption(cfg) -> str:
    return " ".join(x for x in (cfg.trigger_word, cfg.class_word) if x).strip()


def _write_caption(media_path: str, text: str, cfg) -> None:
    text = (text or "").strip()
    tw = cfg.trigger_word
    if tw and cfg.prepend_trigger and not re.search(rf"(^|\W){re.escape(tw)}(\W|$)", text, re.I):
        text = f"{tw}, {text}" if text else tw
    with open(os.path.splitext(media_path)[0] + ".txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


def _place(src: str, dest_dir: str, index: int, ext: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, f"{index:05d}{ext}")
    shutil.copy2(src, dest)
    return dest


def prepare_dataset(cfg, work: str, videos_dir: str, images_dir: str) -> DatasetReport:
    """Download + validate; copy media and write .txt captions into videos_dir / images_dir."""
    os.makedirs(work, exist_ok=True)
    report = DatasetReport(source="url", videos_dir=videos_dir, images_dir=images_dir)
    archive = os.path.join(work, "dataset.archive")
    try:
        _download(cfg.dataset_url, archive)
    except DatasetError:
        raise
    except Exception as exc:
        raise DatasetError(f"could not download dataset_url: {redact(str(exc))[:300]}") from exc
    _extract(archive, os.path.join(work, "unpacked"))
    os.remove(archive)
    videos, images = _collect(os.path.join(work, "unpacked"))
    report.videos_found = len(videos)
    report.images_found = len(images)
    if not videos and not images:
        raise DatasetError(
            "no videos or images found (expected .mp4/.webm/.mov/.mkv and/or .jpg/.jpeg/.png/.webp, "
            "captions as same-name .txt; optional videos/ and images/ folders)"
        )

    total = len(videos) + len(images)
    if total > cfg.max_media:
        # Prefer keeping a mix: trim from the end of each list proportionally.
        report.warnings.append(f"{total} media files found, using the first {cfg.max_media} (max_media)")
        if videos and images:
            keep_v = max(1, int(round(cfg.max_media * len(videos) / total)))
            keep_v = min(keep_v, cfg.max_media - 1, len(videos))
            keep_i = min(len(images), cfg.max_media - keep_v)
            videos, images = videos[:keep_v], images[:keep_i]
        elif videos:
            videos = videos[: cfg.max_media]
        else:
            images = images[: cfg.max_media]

    base = _base_caption(cfg)

    def caption_and_place(pairs, dest_dir, kind: str) -> int:
        used = 0
        for src, cap in pairs:
            ext = os.path.splitext(src)[1].lower()
            if kind == "image":
                try:
                    from PIL import Image, ImageOps
                    with Image.open(src) as im:
                        im = ImageOps.exif_transpose(im)
                        w, h = im.size
                        if min(w, h) < MIN_IMAGE_SIDE:
                            report.skipped.append(f"{os.path.basename(src)}: too small ({w}x{h})")
                            continue
                except Exception as exc:
                    report.skipped.append(f"{os.path.basename(src)}: unreadable ({str(exc)[:80]})")
                    continue
            elif os.path.getsize(src) < 1024:
                report.skipped.append(f"{os.path.basename(src)}: video file too small")
                continue
            dest = _place(src, dest_dir, used, ext)
            used += 1
            mode = cfg.caption_mode
            if mode == "trigger" or (cap is None and mode in ("auto", "existing")):
                text = base
                report.captions_trigger_only += 1
                if cap is None and mode == "existing":
                    report.warnings.append(
                        "caption_mode=existing but some files have no .txt; used trigger word"
                    )
            else:
                text = cap
                report.captions_from_files += 1
            _write_caption(dest, text or "", cfg)
        return used

    report.videos_used = caption_and_place(videos, videos_dir, "video")
    report.images_used = caption_and_place(images, images_dir, "image")
    if report.videos_used + report.images_used == 0:
        raise DatasetError("no usable videos or images (all were unreadable, too small, or skipped)")

    if cfg.is_i2v and report.videos_used == 0:
        raise DatasetError(
            f"{cfg.base_model} is an I2V task and needs at least one video in the dataset "
            "(musubi uses the first frame as the I2V control image). Add clips under videos/."
        )

    if report.videos_used + report.images_used < 3:
        report.warnings.append(
            f"only {report.videos_used} video(s) and {report.images_used} image(s); "
            "more varied clips usually give a better LoRA"
        )
    if report.captions_trigger_only and not cfg.trigger_word:
        report.warnings.append("some files have no caption and no trigger_word was set")

    for folder in (videos_dir, images_dir):
        if not os.path.isdir(folder):
            continue
        for f in sorted(os.listdir(folder)):
            if f.endswith(".txt") and len(report.example_captions) < 3:
                with open(os.path.join(folder, f), encoding="utf-8") as fh:
                    report.example_captions.append(fh.read().strip()[:300])
    return report
