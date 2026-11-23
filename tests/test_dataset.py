import os
import zipfile

from PIL import Image

from src.config import resolve_config
from src.dataset import _collect, _extract, _write_caption


def _make_zip(tmp_path, n_img=3, n_vid=1, layout="flat"):
    src = tmp_path / "src"
    (src / "videos").mkdir(parents=True)
    (src / "images").mkdir()
    for i in range(n_img):
        dest = (src / "images" / f"img{i}.jpg") if layout == "folders" else (src / f"img{i}.jpg")
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (512, 512), (i * 40, 100, 100)).save(dest)
        if i % 2 == 0:
            dest.with_suffix(".txt").write_text(f"caption {i}")
    for i in range(n_vid):
        dest = (src / "videos" / f"clip{i}.mp4") if layout == "folders" else (src / f"clip{i}.mp4")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"\x00\x00fake-mp4-placeholder" + b"\x00" * 2048)
        dest.with_suffix(".txt").write_text(f"video {i}")
    (src / ".DS_Store").write_text("x")
    zp = tmp_path / "d.zip"
    with zipfile.ZipFile(zp, "w") as z:
        for dirpath, _dirnames, files in os.walk(src):
            for f in files:
                full = os.path.join(dirpath, f)
                z.write(full, os.path.relpath(full, src))
        z.writestr("../evil.txt", "nope")
    return zp


def test_extract_zip_slip_and_pairs(tmp_path):
    zp = _make_zip(tmp_path, layout="flat")
    out = tmp_path / "out"
    _extract(str(zp), str(out))
    assert not (tmp_path / "evil.txt").exists()
    videos, images = _collect(str(out))
    assert len(images) == 3 and len(videos) == 1
    assert images[0][1] == "caption 0" and images[1][1] is None
    assert videos[0][1] == "video 0"


def test_folder_layout(tmp_path):
    zp = _make_zip(tmp_path, layout="folders")
    out = tmp_path / "out"
    _extract(str(zp), str(out))
    videos, images = _collect(str(out))
    assert len(images) == 3 and len(videos) == 1
    assert "images" in images[0][0] and "videos" in videos[0][0]


def test_trigger_prepended(tmp_path):
    cfg = resolve_config({"dataset_url": "https://x/y.zip", "trigger_word": "zxowl"}, env={})
    img = tmp_path / "a.jpg"
    _write_caption(str(img), "a figurine on a table", cfg)
    assert (tmp_path / "a.txt").read_text().strip() == "zxowl, a figurine on a table"
    _write_caption(str(img), "a photo of zxowl", cfg)
    assert (tmp_path / "a.txt").read_text().strip() == "a photo of zxowl"
