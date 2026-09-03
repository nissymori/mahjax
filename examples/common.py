from pathlib import Path
from typing import Any, Callable, Dict


EXAMPLES_DIR = Path(__file__).resolve().parent
OFFLINE_DATA_DIR = EXAMPLES_DIR / "offline_data"
PARAMS_DIR = EXAMPLES_DIR / "params"
FIG_DIR = EXAMPLES_DIR / "fig"


def default_dataset_path(env_name: str) -> str:
    return str(OFFLINE_DATA_DIR / f"{env_name}_offline_data.pkl")


def default_bc_params_path(env_name: str) -> str:
    return str(PARAMS_DIR / f"{env_name}_bc_params.pkl")


def default_rl_params_path(env_name: str, seed: int) -> str:
    return str(PARAMS_DIR / f"{env_name}-seed={seed}.ckpt")


def get_rule_based_player(env_name: str) -> Callable[..., Any]:
    if env_name == "red_mahjong":
        from mahjax.red_mahjong.players import rule_based_player

        return rule_based_player
    if env_name == "no_red_mahjong":
        from mahjax.no_red_mahjong.players import rule_based_player

        return rule_based_player
    raise ValueError(f"Unsupported env_name: {env_name}")


def get_network_cls(env_name: str, encoder: str = "transformer"):
    """Bind the env's tokenizer to a sequence model.

    ``network.py`` is env-agnostic; the red / no-red split lives entirely in the two
    feature modules, which expose the same contract (D_MODEL, NUM_ACTIONS,
    CLS_POSITIONS). ``encoder`` selects "transformer" or "perceiver".
    """
    try:
        from .networks.network import make_acnet
    except ImportError:
        from networks.network import make_acnet

    if env_name == "red_mahjong":
        try:
            from .networks.red_feature import ObservationTokenizer
        except ImportError:
            from networks.red_feature import ObservationTokenizer
        return make_acnet(ObservationTokenizer, encoder)
    if env_name == "no_red_mahjong":
        try:
            from .networks.no_red_feature import ObservationTokenizer
        except ImportError:
            from networks.no_red_feature import ObservationTokenizer
        return make_acnet(ObservationTokenizer, encoder)
    raise ValueError(f"Unsupported env_name: {env_name}")


def attach_dataset_metadata(dataset: Dict[str, Any], env_name: str) -> Dict[str, Any]:
    dataset["env_name"] = env_name
    dataset["observe_type"] = "dict"
    return dataset
