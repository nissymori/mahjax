"""Time ``env.step`` under ``jax.vmap`` in each round mode.

    python -I benchmarks/step_speed.py [--src <tree>] [--env red_4p] [--batch 256] [--steps 256] [--max-ratio 1.2]

Each variant runs ``--batch`` envs under ``jax.vmap``, each a ``lax.scan`` of ``--steps``
steps with random legal actions, and prints the best of ``--repeat`` runs in ms per step,
its ratio to a single game, and the bytes XLA estimates the compiled rollout to access.
mahjax is imported from the source tree ``--src``, by default the one holding this script.
Run it on a GPU. With ``--max-ratio``, it exits with status 1 when an east or a half game
takes more than that many times a single game's time per step.
"""

import argparse
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VARIANTS = {
    "single": dict(round_mode="single"),
    "east": dict(round_mode="east"),
    "half": dict(round_mode="half"),
    "half, round_end=confirm": dict(round_mode="half", round_end="confirm"),
}
CHECKED = ("east", "half")  # what --max-ratio compares with single


def rollout(env, steps):
    import jax
    import jax.numpy as jnp

    def body(carry, _):
        state, rng = carry
        rng, k_act, k_step = jax.random.split(rng, 3)
        action = jax.random.categorical(k_act, jnp.where(state.legal_action_mask, 0.0, -1e9)).astype(jnp.int32)
        state = env.step(state, action, k_step)
        return (state, rng), jnp.sum(state.rewards.astype(jnp.float32))

    def one(state, rng):
        (state, rng), out = jax.lax.scan(body, (state, rng), None, length=steps)
        return state, jnp.sum(out)

    return jax.jit(jax.vmap(one))


def measure(env, batch, steps, repeat):
    """Best time per step in ms, and XLA's estimate of the bytes accessed."""
    import jax

    keys = jax.random.split(jax.random.PRNGKey(0), batch)
    states = jax.jit(jax.vmap(env.init))(keys)
    compiled = rollout(env, steps).lower(states, keys).compile()
    jax.block_until_ready(compiled(states, keys))
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        jax.block_until_ready(compiled(states, keys))
        best = min(best, time.perf_counter() - t0)
    cost = compiled.cost_analysis()
    cost = cost[0] if isinstance(cost, (list, tuple)) else cost
    return 1000 * best / steps, (cost or {}).get("bytes accessed", float("nan"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", default=REPO, help="the source tree to import mahjax from")
    p.add_argument("--env", default="red_4p")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--max-ratio", type=float, default=None)
    args = p.parse_args()
    sys.path.insert(0, os.path.abspath(args.src))
    import jax

    import mahjax

    print(f"{args.env} from {os.path.dirname(mahjax.__file__)}, {args.batch} envs x {args.steps} steps, "
          f"jax {jax.__version__}, {jax.devices()[0].device_kind}")
    print(f"{'variant':26s} {'ms per step':>11s} {'x single':>9s} {'bytes accessed':>15s}")
    times = {}
    for name, kwargs in VARIANTS.items():
        env = mahjax.make(args.env, observe_type="dict", **kwargs)
        times[name], accessed = measure(env, args.batch, args.steps, args.repeat)
        print(f"{name:26s} {times[name]:11.2f} {times[name] / times['single']:9.2f} {accessed:15.3e}", flush=True)
    if args.max_ratio is not None:
        slow = [name for name in CHECKED if times[name] > args.max_ratio * times["single"]]
        if slow:
            print(f"FAIL: {', '.join(slow)} slower than {args.max_ratio}x single")
            sys.exit(1)
        print(f"OK: east and half within {args.max_ratio}x single")


if __name__ == "__main__":
    main()
