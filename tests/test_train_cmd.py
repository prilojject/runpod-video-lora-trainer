from src.config import resolve_config
from src.models import components_for
from src.train import (auto_blocks_to_swap, build_train_command, cache_latents_cmd, cache_text_cmd,
                       use_offload_inactive, write_dataset_toml)

URL = "https://example.com/data.zip"
MODELS = {"t5": "/m/t5.pth", "vae": "/m/vae.safetensors", "dit": "/m/dit.safetensors"}


def test_dataset_toml_separate_caches(tmp_path):
    cfg = resolve_config({"dataset_url": URL, "resolution": "832x480", "target_frames": [17]}, env={})
    t = tmp_path / "d.toml"
    caches = write_dataset_toml(cfg, "/vids", "/imgs", str(tmp_path / "cache"), str(t), 2, 3)
    txt = t.read_text()
    assert 'video_directory = "/vids"' in txt
    assert 'image_directory = "/imgs"' in txt
    assert "target_frames = [17]" in txt
    assert "source_fps = 16" in txt
    assert caches["videos"] != caches["images"]
    assert caches["videos"] in txt and caches["images"] in txt
    assert txt.count("[[datasets]]") == 2


def test_image_only_toml_has_no_target_frames(tmp_path):
    cfg = resolve_config({"dataset_url": URL}, env={})
    t = tmp_path / "d.toml"
    write_dataset_toml(cfg, "/vids", "/imgs", str(tmp_path / "cache"), str(t), 0, 3)
    txt = t.read_text()
    assert "image_directory" in txt and "video_directory" not in txt
    assert "target_frames" not in txt


def test_wan13_train_command():
    cfg = resolve_config({"dataset_url": URL, "trigger_word": "zxowl", "steps": 20, "rank": 8}, env={})
    cmd = build_train_command(cfg, MODELS, "/d.toml", "/o", "n", 24.0, 0)
    s = " ".join(cmd)
    assert "wan_train_network.py" in s
    assert "--task=t2v-1.3B" in s
    assert "--network_module=networks.lora_wan" in s
    assert "--max_train_steps=20" in s
    assert "--fp8_base" in s
    assert "--optimizer_type=adamw8bit" in s
    assert "--discrete_flow_shift=3.0" in s
    assert "--blocks_to_swap" not in s
    assert "--offload_inactive_dit" not in s


def test_auto_blocks_and_offload():
    cfg13 = resolve_config({"dataset_url": URL}, env={})
    assert auto_blocks_to_swap(cfg13, 24.0) == 0
    cfg14 = resolve_config({"dataset_url": URL, "base_model": "wan-t2v-14b"}, env={})
    assert auto_blocks_to_swap(cfg14, 24.0) == 24
    assert auto_blocks_to_swap(cfg14, 48.0) == 0
    cfg22 = resolve_config({"dataset_url": URL, "base_model": "wan22-t2v-a14b"}, env={})
    assert auto_blocks_to_swap(cfg22, 24.0) == 28
    assert use_offload_inactive(cfg22, 28, 24.0) is False
    assert auto_blocks_to_swap(cfg22, 48.0) == 0
    assert use_offload_inactive(cfg22, 0, 48.0) is True
    # Mutual exclusion: never both.
    assert not (auto_blocks_to_swap(cfg22, 24.0) and use_offload_inactive(cfg22, auto_blocks_to_swap(cfg22, 24.0), 24.0))


def test_wan22_both_and_single_expert():
    both = resolve_config({"dataset_url": URL, "base_model": "wan22-t2v-a14b", "wan22_mode": "both"}, env={})
    models = {**MODELS, "dit_high": "/m/dit_high.safetensors"}
    s = " ".join(build_train_command(both, models, "/d.toml", "/o", "n", 48.0, 0))
    assert "--dit_high_noise=/m/dit_high.safetensors" in s
    assert "--timestep_boundary=0.875" in s
    assert "--offload_inactive_dit" in s
    low = resolve_config({"dataset_url": URL, "base_model": "wan22-t2v-a14b", "wan22_mode": "low"}, env={})
    s = " ".join(build_train_command(low, MODELS, "/d.toml", "/o", "n", 24.0, 24))
    assert "--preserve_distribution_shape" in s
    assert "--min_timestep=0" in s and "--max_timestep=875" in s
    assert "--dit_high_noise" not in s
    assert "--offload_inactive_dit" not in s
    assert "--blocks_to_swap=24" in s


def test_cache_cmds_i2v_and_fp8_t5():
    cfg = resolve_config({"dataset_url": URL, "base_model": "wan-i2v-14b"}, env={})
    models = {**MODELS, "clip": "/m/clip.pth"}
    lat = " ".join(cache_latents_cmd(cfg, models, "/d.toml", 24.0))
    assert "--i2v" in lat and "--clip /m/clip.pth" in lat and "--skip_broken" in lat
    assert "--vae_cache_cpu" in lat
    txt = " ".join(cache_text_cmd(cfg, models, "/d.toml", 24.0))
    assert "--fp8_t5" in txt and "--t5 /m/t5.pth" in txt
    txt48 = " ".join(cache_text_cmd(cfg, models, "/d.toml", 48.0))
    assert "--fp8_t5" not in txt48


def test_components_single_expert_skips_other_dit():
    both = resolve_config({"dataset_url": URL, "base_model": "wan22-t2v-a14b", "wan22_mode": "both"}, env={})
    comps = components_for(both)
    assert "dit" in comps and "dit_high" in comps and "clip" not in comps
    low = resolve_config({"dataset_url": URL, "base_model": "wan22-t2v-a14b", "wan22_mode": "low"}, env={})
    comps = components_for(low)
    assert "dit" in comps and "dit_high" not in comps
    i2v = resolve_config({"dataset_url": URL, "base_model": "wan-i2v-14b"}, env={})
    assert "clip" in components_for(i2v)
    for name, (repo, fn, _sz) in components_for(both).items():
        assert "fp8_scaled" not in fn
