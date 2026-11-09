"""Digest every step of random games, to check that two source trees play the same.

    python -I benchmarks/trajectory_digest.py run <out.txt> --src <tree> [--env red_4p]
        [--round-mode half] [--round-end auto] [--seeds 0,1,2] [--batch 256] [--steps 3000]
    python -I benchmarks/trajectory_digest.py compare <a.txt> <b.txt>

``run`` imports mahjax from the source tree ``--src`` (by default the one holding this
script), so the trees run in two processes; ``-I`` keeps any other mahjax off the path.
At each step, every one of the ``--batch`` games under ``jax.vmap`` plays a random legal
action, and a game that has ended is dealt again. A line of the output holds the seed,
the step, and the sha256 of every leaf of the stepped states and of their observations.
Equal states give equal actions, so two trees that play the same write the same lines.
"""

import argparse
import hashlib
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _digest(leaves, i):
    h = hashlib.sha256()
    for leaf in leaves:
        h.update(np.ascontiguousarray(leaf[i]).tobytes())
    return h.hexdigest()


def run(args):
    src = os.path.abspath(args.src)
    sys.path.insert(0, src)
    import jax
    import jax.numpy as jnp

    import mahjax

    if os.path.dirname(os.path.abspath(mahjax.__file__)) != os.path.join(src, "mahjax"):
        sys.exit(f"mahjax was imported from {mahjax.__file__}, not from {src}; run with python -I")
    if args.cache_dir:
        jax.config.update("jax_compilation_cache_dir", args.cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    if args.steps % args.chunk:
        sys.exit("--steps must be a multiple of --chunk")
    batch = args.batch
    env = mahjax.make(args.env, round_mode=args.round_mode, round_end=args.round_end, observe_type="dict")
    v_init, v_step, v_observe = jax.vmap(env.init), jax.vmap(env.step), jax.vmap(env.observe)

    def body(carry, _):
        state, key = carry
        key, k_act, k_step, k_deal = jax.random.split(key, 4)
        logits = jnp.where(state.legal_action_mask, 0.0, -1e9)
        action = jax.vmap(jax.random.categorical)(jax.random.split(k_act, batch), logits).astype(jnp.int32)
        stepped = v_step(state, action, jax.random.split(k_step, batch))
        # The games that have ended are dealt again: computed for every game, then picked.
        dealt = v_init(jax.random.split(k_deal, batch))
        ended = stepped.terminated | stepped.truncated
        state = jax.tree_util.tree_map(
            lambda new, old: jnp.where(ended.reshape((batch,) + (1,) * (old.ndim - 1)), new, old), dealt, stepped
        )
        return (state, key), (stepped, v_observe(stepped))

    chunk = jax.jit(lambda state, key: jax.lax.scan(body, (state, key), None, length=args.chunk))
    t0 = time.time()
    ended_games = 0
    with open(args.out, "w") as out:
        for seed in [int(s) for s in args.seeds.split(",")]:
            key, k_init = jax.random.split(jax.random.PRNGKey(seed))
            state = jax.jit(v_init)(jax.random.split(k_init, batch))
            for c in range(args.steps // args.chunk):
                (state, key), (stepped, obs) = chunk(state, key)
                stepped, obs = jax.device_get((stepped, obs))
                state_leaves, obs_leaves = jax.tree_util.tree_leaves(stepped), jax.tree_util.tree_leaves(obs)
                for i in range(args.chunk):
                    out.write(f"{seed} {c * args.chunk + i} {_digest(state_leaves, i)} {_digest(obs_leaves, i)}\n")
                ended_games += int(np.asarray(stepped.terminated).sum())
    print(f"{args.env} {args.round_mode} {args.round_end}: {args.steps} steps of {batch} games for seeds {args.seeds}, "
          f"{ended_games} games ended, {time.time() - t0:.0f}s")


def compare(args):
    with open(args.a) as fa, open(args.b) as fb:
        a, b = fa.read().splitlines(), fb.read().splitlines()
    differ = [(x, y) for x, y in zip(a, b) if x != y]
    if len(a) != len(b) or differ:
        print(f"DIFFERENT: {len(a)} and {len(b)} lines, {len(differ)} of the common lines differ")
        if differ:
            seed, step, state_a, obs_a = differ[0][0].split()
            _, _, state_b, obs_b = differ[0][1].split()
            parts = [name for name, x, y in (("state", state_a, state_b), ("observation", obs_a, obs_b)) if x != y]
            print(f"first at seed {seed}, step {step}: the {' and the '.join(parts)} differ")
        return 1
    print(f"IDENTICAL: {len(a)} steps")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("out")
    r.add_argument("--src", default=REPO, help="the source tree to import mahjax from")
    r.add_argument("--env", default="red_4p")
    r.add_argument("--round-mode", default="half")
    r.add_argument("--round-end", default="auto")
    r.add_argument("--seeds", default="0,1,2")
    r.add_argument("--batch", type=int, default=256)
    r.add_argument("--steps", type=int, default=3000)
    r.add_argument("--chunk", type=int, default=100, help="steps per compiled scan")
    r.add_argument("--cache-dir", default=None, help="a JAX compilation cache to share between runs")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = p.parse_args()
    if args.cmd == "run":
        run(args)
    else:
        sys.exit(compare(args))


if __name__ == "__main__":
    main()
