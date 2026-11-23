import pytest

from src.config import ConfigError, render_prompt, resolve_config

URL = "https://example.com/data.zip"


def test_defaults():
    c = resolve_config({"dataset_url": URL, "trigger_word": "zxowl"}, env={})
    assert c.base_model == "wan-t2v-1.3b" and c.task == "t2v-1.3B"
    assert c.steps == 2000 and c.rank == 16 and c.alpha == 16.0
    assert c.learning_rate == pytest.approx(2e-4)
    assert c.resolution_wh == (832, 480)
    assert c.target_frames == [17]
    assert c.fp8_base is True and c.num_samples == 0
    assert c.wan22_mode == "both" and c.blocks_to_swap == "auto"


def test_input_beats_env_beats_preset():
    c = resolve_config({"dataset_url": URL, "steps": 20, "resolution": 512},
                       env={"STEPS": "500", "LORA_RANK": "32", "BASE_MODEL": "wan-t2v-14b"})
    assert c.steps == 20 and c.rank == 32 and c.base_model == "wan-t2v-14b"
    assert c.resolution_wh == (512, 512)


def test_resolution_shapes():
    c = resolve_config({"dataset_url": URL, "resolution": "832x480"}, env={})
    assert c.resolution_wh == (832, 480)
    c = resolve_config({"dataset_url": URL, "resolution": [640, 480]}, env={})
    assert c.resolution_wh == (640, 480)
    with pytest.raises(ConfigError, match="multiple of 16"):
        resolve_config({"dataset_url": URL, "resolution": 500}, env={})


def test_target_frames_validation():
    c = resolve_config({"dataset_url": URL, "target_frames": "17,25"}, env={})
    assert c.target_frames == [17, 25]
    c = resolve_config({"dataset_url": URL, "frames": 17}, env={})
    assert c.target_frames == [17]
    with pytest.raises(ConfigError, match="both"):
        resolve_config({"dataset_url": URL, "frames": 17, "target_frames": [25]}, env={})
    with pytest.raises(ConfigError, match="N×4\\+1"):
        resolve_config({"dataset_url": URL, "target_frames": [16]}, env={})


def test_tokens_rejected():
    with pytest.raises(ConfigError, match="tokens"):
        resolve_config({"dataset_url": URL, "hf_token": "hf_x"}, env={})


def test_coming_soon_rejected():
    with pytest.raises(ConfigError, match="coming soon"):
        resolve_config({"dataset_url": URL, "base_model": "hunyuan-t2v"}, env={})
    with pytest.raises(ConfigError, match="coming soon"):
        resolve_config({"dataset_url": URL, "base_model": "framepack-i2v"}, env={})


def test_dataset_required():
    with pytest.raises(ConfigError, match="dataset"):
        resolve_config({}, env={})
    with pytest.raises(ConfigError, match="http"):
        resolve_config({"dataset_url": "ftp://x"}, env={})


def test_validation_errors():
    with pytest.raises(ConfigError) as e:
        resolve_config({"dataset_url": "ftp://x", "steps": 5, "resolution": 1000, "bogus": 1}, env={})
    msg = str(e.value)
    for frag in ("dataset_url", "steps", "resolution", "unknown"):
        assert frag in msg


def test_wan22_and_14b_warning():
    c = resolve_config({"dataset_url": URL, "base_model": "wan22-t2v-a14b", "wan22_mode": "low"}, env={})
    assert c.is_wan22 and c.wan22_mode == "low" and c.task == "t2v-A14B"
    assert any("48 GB" in w for w in c.warnings)


def test_hf_repo_needs_token():
    with pytest.raises(ConfigError, match="HF_TOKEN"):
        resolve_config({"dataset_url": URL, "hf_repo_id": "me/lora"}, env={})
    c = resolve_config({"dataset_url": URL, "hf_repo_id": "me/lora", "dry_run": True}, env={})
    assert any("HF_TOKEN" in w for w in c.warnings)


def test_render_prompt():
    assert render_prompt("a video of {trigger} on a table", "zxowl", "figurine") == "a video of zxowl figurine on a table"
    assert render_prompt("{trigger} style", "", "") == "the subject style"


def test_public_dict_redacts_upload_url():
    c = resolve_config({"dataset_url": URL, "upload_url": "https://s3.example/put?sig=secret"}, env={})
    assert c.public_dict()["upload_url"] == "<presigned url>"
    assert c.public_dict()["resolution"] == [832, 480]
