"""Chess adapter for Algorithm v4 — unbounded-depth speculative actions.

Runs a single-agent self-play chess game using ``run_depth_speculative``
from ``depth_algorithm.py``. Every speculative state in the tree gets
its own Actor and Speculator call; the tree grows recursively until the
``max_inflight_actors`` budget is reached.

Reuses ``AgentManager``, ``ChessActionCleaner``, ``Config``, and
``GameLogger`` from ``Speculative_Chess.py``.

Usage:
    python depth_speculative_chess.py --config config.yml --stop-after 20 \\
           --max-inflight-actors 13
"""

from __future__ import annotations

import argparse
import asyncio
import time
import uuid
from os.path import join
from typing import Any, List, Tuple

import chess

from Speculative_Chess import (
    AgentManager,
    ChessActionCleaner,
    Config,
    GameLogger,
)
from depth_algorithm import run_depth_speculative
from utils import Utils


# ── State / policy / transition ──────────────────────────────────────


def _make_state(board: chess.Board) -> Tuple[str, Tuple[str, ...]]:
    return (board.fen(), tuple(m.uci() for m in board.move_stack))


def _board_from_state(state: Tuple[str, Tuple[str, ...]]) -> chess.Board:
    fen, _history = state
    return chess.Board(fen)


def _transition(state: Tuple[str, Tuple[str, ...]],
                action: str) -> Tuple[str, Tuple[str, ...]]:
    board = _board_from_state(state)
    uci = action.strip().strip("[]").lower()
    board.push(chess.Move.from_uci(uci))
    return _make_state(board)


def _policy(state: Tuple[str, Tuple[str, ...]]) -> Tuple[str, Tuple[Any, ...]]:
    board = _board_from_state(state)
    turn = "White" if board.turn == chess.WHITE else "Black"
    valid_moves = tuple(f'[{m.uci()}]' for m in board.legal_moves)
    observation = (
        f"[GAME] You are playing as {turn} in a game of Chess. "
        f"Make your moves in UCI format enclosed in square brackets "
        f"(e.g., [e2e4]).\n"
        f"[GAME] The current board is:\n"
        f"{Utils.board_with_coords(board)}\n"
        f"[GAME] The valid moves are: {list(valid_moves)}."
    )
    return ("chess_llm", (turn, valid_moves, observation))


def _semantic_match(a_hat: str, a_truth: str, domain: str) -> bool:
    def norm(a: str) -> str:
        return a.strip().strip("[]").lower()
    return norm(a_hat) == norm(a_truth)


# ── Runner ───────────────────────────────────────────────────────────


class ChessDepthRunner:
    def __init__(self, config: Config, max_inflight_actors: int):
        self.config = config
        self.agent_manager = AgentManager(config)
        self.agent0_name = config.agent_name0
        self.agent1_name = config.agent_name1
        self.guess_model_name = config.guess_model_name
        self.num_guesses = config.num_guesses
        self.max_inflight_actors = max_inflight_actors
        self.logger: GameLogger | None = None

    def _actor_model_for_turn(self, turn: str) -> str:
        name = self.agent0_name if turn == "White" else self.agent1_name
        if name == "OpenAI":
            return self.config.openai_model_name
        if name == "OpenRouter":
            return self.config.openrouter_model_name
        raise ValueError(f"unknown agent type: {name}")

    def _actor_sync(self, turn: str, valid_moves: Tuple[str, ...],
                    observation: str) -> str:
        model = self._actor_model_for_turn(turn)
        for attempt in range(3):
            raw, *_ = self.agent_manager.call_guess_llm(
                observation, model, retries=1)
            cleaned = ChessActionCleaner.clean_action(raw) if raw else None
            if cleaned and cleaned in valid_moves:
                if self.logger:
                    self.logger.log("ACTOR", f"{turn} → {cleaned}")
                return cleaned
            observation += self.config.retry_prompt.format(
                attempt=attempt + 1, role=turn)
        raise RuntimeError("Actor failed to produce a valid move after retries")

    def _speculator_sync(self, turn: str, valid_moves: Tuple[str, ...],
                         observation: str) -> List[str]:
        prompt = observation + self.config.guess_prompt.format(
            num_guesses=self.num_guesses)
        raw, *_ = self.agent_manager.call_guess_llm(
            prompt, self.guess_model_name, retries=3)
        if not raw:
            return []
        candidates = ChessActionCleaner.clean_actions(raw)
        legal = [c for c in candidates if c in valid_moves]
        if self.logger:
            self.logger.log("SPEC", f"{turn} candidates → {legal}")
        return legal[: self.num_guesses]

    async def _actor(self, state, h, q) -> str:
        turn, valid_moves, observation = q
        return await asyncio.to_thread(
            self._actor_sync, turn, valid_moves, observation)

    async def _speculator(self, state, h, q) -> List[str]:
        turn, valid_moves, observation = q
        return await asyncio.to_thread(
            self._speculator_sync, turn, valid_moves, observation)

    async def run(self, stop_after: int, output_dir: str) -> None:
        run_id = str(uuid.uuid4())
        self.logger = GameLogger(output_dir, run_id)
        Utils.save_file("", join(output_dir, run_id, "log.txt"))

        board = chess.Board()
        s_0 = _make_state(board)

        t0 = time.perf_counter()

        def on_event(kind: str, f: dict) -> None:
            ts = time.perf_counter() - t0
            depth = f.get("depth")
            inflight = f.get("inflight")
            tag = f"[{ts:7.2f}s inflight={inflight:>2}]"
            d = f"d={depth}" if depth is not None else ""
            if kind == "ACTOR_LAUNCH":
                msg = f"→ fire Actor @{d} for â={f.get('a_hat')}"
            elif kind == "ACTOR_LAUNCH_FORCE":
                msg = f"→ fire Actor @{d} (root/force)"
            elif kind == "SPEC_LAUNCH":
                msg = f"→ fire Speculator @{d} for â={f.get('a_hat')}"
            elif kind == "ACTOR_RESOLVED":
                mark = "ROOT " if f.get("is_root") else ""
                msg = f"← {mark}Actor @{d} returned {f.get('a_true')}"
            elif kind == "SPEC_RESOLVED":
                msg = (f"← Speculator @{d} returned "
                       f"{f.get('predictions')}")
            elif kind == "ACTOR_DEFERRED":
                msg = f"~ Actor @{d} DEFERRED (budget full)"
            elif kind == "ACTOR_DEDUP":
                msg = f"~ Actor @{d} reuses sibling (dedup)"
            elif kind == "HIT":
                msg = (f"✓ HIT step={f.get('step')} "
                       f"a_true={f.get('a_true')} matched "
                       f"{f.get('matched')}")
            elif kind == "CASCADED_HIT":
                msg = (f"✓✓ CASCADED_HIT step={f.get('step')} "
                       f"a_true={f.get('a_true')}")
            elif kind == "MISS":
                msg = (f"✗ MISS step={f.get('step')} "
                       f"a_true={f.get('a_true')} "
                       f"preds={f.get('predictions')}")
            elif kind == "ABANDON":
                msg = f"⌀ abandon subtree @{d} (â={f.get('a_hat')})"
            elif kind == "PROMOTE_DEFERRED":
                msg = f"↑ promote: force-launch DEFERRED Actor @{d}"
            elif kind == "ACTOR_ERROR":
                msg = f"! Actor error @{d}: {f.get('err')}"
            elif kind == "SPEC_ERROR":
                msg = f"! Spec error @{d}: {f.get('err')}"
            else:
                msg = f"{kind} {f}"
            line = f"{tag} {msg}"
            print(line)
            if self.logger:
                self.logger.log(kind, msg)

        states, actions, stats, step_info = await run_depth_speculative(
            s_0, stop_after,
            actor=self._actor,
            speculator=self._speculator,
            transition=_transition,
            policy=_policy,
            semantic_match=_semantic_match,
            k=self.num_guesses,
            max_inflight_actors=self.max_inflight_actors,
            domain="chess",
            assert_invariants=True,
            record_steps=True,
            on_event=on_event,
        )
        wall = time.perf_counter() - t0

        # Enrich step_info with chess-specific fields (fen, player_id) so
        # downstream analysis matches the breadth-focused stepsinfo.json.
        enriched_steps = []
        for i, entry in enumerate(step_info):
            fen_before, _ = states[i]
            fen_after = states[i + 1][0] if i + 1 < len(states) else None
            board_before = chess.Board(fen_before)
            player_id = 0 if board_before.turn == chess.WHITE else 1
            enriched_steps.append({
                **entry,
                "player_id": player_id,
                "fen_before": fen_before,
                "fen_after": fen_after,
            })

        out_dir = join(output_dir, run_id)
        stats_dict = {
            "hits": stats.hits,
            "cascaded_hits": stats.cascaded_hits,
            "misses": stats.misses,
            "actor_launches": stats.actor_launches,
            "actor_dedup_reuse": stats.actor_dedup_reuse,
            "actor_deferred": stats.actor_deferred,
            "actor_promoted_launch": stats.actor_promoted_launch,
            "speculator_launches": stats.speculator_launches,
            "max_inflight_actors_observed": stats.max_inflight_actors_observed,
            "max_tree_nodes": stats.max_tree_nodes,
            "max_tree_depth": stats.max_tree_depth,
            "max_cascade_run_length": stats.max_cascade_run_length,
        }
        run_config = {
            "run_id": run_id,
            "stop_after": stop_after,
            "k_num_guesses": self.num_guesses,
            "max_inflight_actors": self.max_inflight_actors,
            "agent0": self.agent0_name,
            "agent1": self.agent1_name,
            "guess_model": self.guess_model_name,
        }
        result = {
            "run_config": run_config,
            "moves": actions,
            "final_fen": states[-1][0],
            "wall_seconds": wall,
            "stats": stats_dict,
        }

        Utils.save_json(result, join(out_dir, "result.json"))
        Utils.save_json(enriched_steps, join(out_dir, "stepsinfo.json"))
        Utils.save_json(stats_dict, join(out_dir, "stats.json"))
        Utils.save_json(run_config, join(out_dir, "run_config.json"))

        if self.logger:
            self.logger.log("DONE", Utils.dict_to_str(stats_dict))
        print(f"[done] run_id={run_id} wall={wall:.2f}s  "
              f"hits={stats.hits}+{stats.cascaded_hits}cascade "
              f"misses={stats.misses} "
              f"max_inflight={stats.max_inflight_actors_observed} "
              f"max_tree_depth={stats.max_tree_depth}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run depth-speculative chess (Algorithm v4, "
                    "unbounded-depth tree).")
    p.add_argument("--config", default="config.yml",
                   help="Path to config YAML (default: config.yml)")
    p.add_argument("--stop-after", type=int, default=None,
                   help="Stop after N moves (default: from config)")
    p.add_argument("--trajectories-dir", default=None,
                   help="Output directory (overrides config)")
    p.add_argument("--max-inflight-actors", type=int, default=None,
                   help="Max concurrent Actor API calls. Default: value "
                        "from config.game.max_inflight_actors, or 13.")
    args = p.parse_args()

    config = Config(args.config)
    if args.trajectories_dir is not None:
        config.trajectories_path = args.trajectories_dir.rstrip("/")

    stop_after = args.stop_after or config.stop_after

    if args.max_inflight_actors is not None:
        max_inflight = args.max_inflight_actors
    else:
        max_inflight = getattr(config, "max_inflight_actors", 13)

    out_base = (f"{config.trajectories_path.rstrip('/')}/"
                f"depth_{config.agent_name0}_vs_{config.agent_name1}"
                f"_guess_{config.guess_model_name}"
                f"_budget{max_inflight}")

    runner = ChessDepthRunner(config, max_inflight_actors=max_inflight)
    asyncio.run(runner.run(stop_after, out_base))


if __name__ == "__main__":
    main()
