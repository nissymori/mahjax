"""Rule configuration for the Hong Kong Old Style environment."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Rules:
    tile_count: int
    hand_size: int
    include_flowers: bool
    minimum_faan: int
    maximum_faan: int
    allow_seven_pairs: bool
    allow_thirteen_orphans: bool
    multiple_winners: bool
    dealer_repeats_on_win: bool
    dealer_repeats_on_draw: bool
    robbing_a_kong: bool
    replacement_tile_for_kong: bool
    replacement_tile_for_flower: bool
    discard_priority: str
    self_draw_payment: str
    discard_payment: str
    special_limit_hands: bool
    flowers_score: str
    payout_table: str

    def __post_init__(self) -> None:
        if self.tile_count != 144:
            raise ValueError("Hong Kong Old Style requires a 144-tile set")
        if self.hand_size != 13:
            raise ValueError("Hong Kong Old Style requires 13 concealed tiles")
        if self.minimum_faan != 3 or self.maximum_faan != 10:
            raise ValueError("HKOS_V1 requires a 3-faan minimum and 10-faan maximum")
        required_true = {
            "include_flowers": self.include_flowers,
            "allow_thirteen_orphans": self.allow_thirteen_orphans,
            "dealer_repeats_on_win": self.dealer_repeats_on_win,
            "dealer_repeats_on_draw": self.dealer_repeats_on_draw,
            "robbing_a_kong": self.robbing_a_kong,
            "replacement_tile_for_kong": self.replacement_tile_for_kong,
            "replacement_tile_for_flower": self.replacement_tile_for_flower,
            "special_limit_hands": self.special_limit_hands,
        }
        disabled = [name for name, enabled in required_true.items() if not enabled]
        if disabled:
            raise ValueError(f"HKOS_V1 requires {', '.join(disabled)}")
        if self.allow_seven_pairs or self.multiple_winners:
            raise ValueError("HKOS_V1 disables seven pairs and multiple winners")
        if self.discard_priority != "nearest_winner":
            raise ValueError("only nearest_winner discard priority is implemented")
        if self.self_draw_payment != "all_players_pay":
            raise ValueError("only all_players_pay self draws are implemented")
        if self.discard_payment != "discarder_only":
            raise ValueError("only discarder_only discard wins are implemented")
        if self.flowers_score != "hong_kong":
            raise ValueError("only hong_kong flower scoring is implemented")
        if self.payout_table != "hk_old_style_v1":
            raise ValueError("only the hk_old_style_v1 payout table is implemented")


HKOS_V1 = Rules(
    tile_count=144,
    hand_size=13,
    include_flowers=True,
    minimum_faan=3,
    maximum_faan=10,
    allow_seven_pairs=False,
    allow_thirteen_orphans=True,
    multiple_winners=False,
    dealer_repeats_on_win=True,
    dealer_repeats_on_draw=True,
    robbing_a_kong=True,
    replacement_tile_for_kong=True,
    replacement_tile_for_flower=True,
    discard_priority="nearest_winner",
    self_draw_payment="all_players_pay",
    discard_payment="discarder_only",
    special_limit_hands=True,
    flowers_score="hong_kong",
    payout_table="hk_old_style_v1",
)
