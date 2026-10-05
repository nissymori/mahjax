from pathlib import Path

import jax.numpy as jnp

from mahjax.red_mahjong.state import default_state
from mahjax.red_mahjong.visualization import (
    render_round_svg,
    render_svg_animation,
    save_play_history_svg,
)


def _collect_states(count: int = 10) -> list:
    state = default_state()
    return [state for _ in range(max(1, count))]


def test_render_round_svg_contains_svg_tag() -> None:
    state = _collect_states(0)[0]
    svg = render_round_svg(state, show_all_hands=True)
    assert svg.startswith("<svg")
    assert "</svg>" in svg


def test_render_round_svg_supports_bilingual_tile_style() -> None:
    state = _collect_states(0)[0]
    svg_standard = render_round_svg(state, show_all_hands=True, tile_style="standard")
    svg_bilingual = render_round_svg(state, show_all_hands=True, tile_style="bilingual")
    assert "東1局" in svg_standard
    assert "East 1" in svg_bilingual
    assert svg_standard != svg_bilingual


def test_render_round_svg_labels_sudden_death_rounds() -> None:
    # A half game that nobody finishes above the target score enters West (round 8+).
    state = default_state()
    for round_index, ja, en in ((8, "西1局", "West 1"), (11, "西4局", "West 4")):
        st = state.replace(round_state=state.round_state.replace(round=jnp.int8(round_index)))  # type: ignore[attr-defined]
        assert ja in render_round_svg(st, show_all_hands=True, tile_style="standard")
        assert en in render_round_svg(st, show_all_hands=True, tile_style="bilingual")


def test_save_play_history_svg_for_10_steps() -> None:
    states = _collect_states(10)
    out = Path("fig/red_mahjong_10steps_test.svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    save_play_history_svg(states, out, columns=5, show_all_hands=True)
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_svg_animation_contains_keyframes() -> None:
    states = _collect_states(5)
    svg = render_svg_animation(states, frame_duration_seconds=0.1, show_all_hands=True)
    assert "@keyframes" in svg


def test_save_svg_honors_scale(tmp_path: Path) -> None:
    import mahjax

    state = default_state()
    base, scaled = tmp_path / "base.svg", tmp_path / "scaled.svg"
    mahjax.save_svg(state, base)
    mahjax.save_svg(state, scaled, scale=2.0)
    root_base = base.read_text(encoding="utf-8").split(">", 1)[0]
    root_scaled = scaled.read_text(encoding="utf-8").split(">", 1)[0]
    assert 'width="700"' in root_base
    assert 'width="1400"' in root_scaled and 'height="1400"' in root_scaled
    assert 'viewBox="0 0 700 700"' in root_scaled  # drawing coordinates are unchanged

    anim = tmp_path / "anim.svg"
    mahjax.save_svg_animation([state, state], anim, scale=0.5)
    assert 'width="350"' in anim.read_text(encoding="utf-8").split(">", 1)[0]
