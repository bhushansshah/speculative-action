"""Chess adapter for Algorithm 3 (v3) — depth-focused speculative actions.

Runs a single-agent self-play chess game using ``run_depth_speculative``
from ``depth_algorithm.py``. The Actor is the high-reasoning LLM chosen
per-turn based on ``board.turn``; the Speculator is the guess model.

This module is intentionally minimal — it reuses ``AgentManager``,
``ChessActionCleaner``, ``Config``, and ``GameLogger`` from
``Speculative_Chess.py`` rather than duplicating that infrastructure.

Usage:
    python depth_speculative_chess.py --config config.yml --stop-after 10
"""

from __future__ import annotations

import argparse
import asyncio
import time
import uuid
from os.path import join
from typing import Any, List, Tuple

import chess

# Reuse everything the breadth implementation already built
from Speculative_Chess import (
    AgentManager,
    ChessActionCleaner,
    Config,
    GameLogger,
)
from depth_algorithm import run_depth_speculative
from utils import Utils


# ── State / policy / transition ──────────────────────────────────────
# State is a (board_fen, move_history_tuple). We keep it hashable so
# it can be used as a cache key. The tuple of moves is redundant with
# the FEN position but lets us replay if needed.


def _make_state(board: chess.Board) -> Tuple[str, Tuple[str, ...]]:
    return (board.fen(), tuple(m.uci() for m in board.move_stack))


def _board_from_state(state: Tuple[str, Tuple[str, ...]]) -> chess.Board:
    fen, _history = state
    return chess.Board(fen)


def _transition(state: Tuple[str, Tuple[str, ...]],
                action: str) -> Tuple[str, Tuple[str, ...]]:
    """Apply a UCI move (with or without brackets) to the state."""
    board = _board_from_state(state)
    uci = action.strip().strip("[]").lower()
    board.push(chess.Move.from_uci(uci))
    return _make_state(board)


def _policy(state: Tuple[str, Tuple[str, ...]]) -> Tuple[str, Tuple[Any, ...]]:
    """Return (api_target, params) for the LLM call at this state.

    Params include everything the Actor or Speculator needs to build a
    prompt. Kept as a hashable tuple so the algorithm's per-batch dedup
    can key on it.
    """
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
    # (target="chess_llm", (turn, valid_moves, observation))
    return ("chess_llm", (turn, valid_moves, observation))


def _semantic_match(a_hat: str, a_truth: str, domain: str) -> bool:
    """UCI-level equality, ignoring brackets/whitespace/case."""
    def norm(a: str) -> str:
        return a.strip().strip("[]").lower()
    return norm(a_hat) == norm(a_truth)


# ── LLM adapters (sync OpenAI client wrapped into asyncio) ───────────


class ChessDepthRunner:
    def __init__(self, config: Config):
        self.config = config
        self.agent_manager = AgentManager(config)
        self.agent0_name = config.agent_name0  # "OpenAI" or "OpenRouter"
        self.agent1_name = config.agent_name1
        self.guess_model_name = config.guess_model_name
        self.num_guesses = config.num_guesses
        self.logger: GameLogger | None = None

    def _actor_model_for_turn(self, turn: str) -> str:
        """Pick which main model to call based on whose turn it is."""
        name = self.agent0_name if turn == "White" else self.agent1_name
        if name == "OpenAI":
            return self.config.openai_model_name
        if name == "OpenRouter":
            return self.config.openrouter_model_name
        raise ValueError(f"unknown agent type: {name}")

    def _actor_sync(self, turn: str, valid_moves: Tuple[str, ...],
                    observation: str) -> str:
        """Synchronous Actor call — returns a cleaned, validated UCI move."""
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
        raise RuntimeError(f"Actor failed to produce a valid move after retries")

    def _speculator_sync(self, turn: str, valid_moves: Tuple[str, ...],
                         observation: str) -> List[str]:
        """Synchronous Speculator call — returns k cleaned UCI move candidates."""
        prompt = observation + self.config.guess_prompt.format(
            num_guesses=self.num_guesses)
        raw, *_ = self.agent_manager.call_guess_llm(
            prompt, self.guess_model_name, retries=3)
        if not raw:
            return []
        candidates = ChessActionCleaner.clean_actions(raw)
        # Keep only legal moves, preserve order
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

        # Seed from initial chess position
        board = chess.Board()
        s_0 = _make_state(board)

        t0 = time.perf_counter()
        states, actions, stats = await run_depth_speculative(
            s_0, stop_after,
            actor=self._actor,
            speculator=self._speculator,
            transition=_transition,
            policy=_policy,
            semantic_match=_semantic_match,
            k=self.num_guesses,
            max_inflight_actors=self.num_guesses + 1,
            domain="chess",
            assert_invariants=True,
        )
        wall = time.perf_counter() - t0

        result = {
            "run_id": run_id,
            "moves": actions,
            "final_fen": states[-1][0],
            "wall_seconds": wall,
            "stats": {
                "hits": stats.hits,
                "cascaded_hits": stats.cascaded_hits,
                "misses": stats.misses,
                "actor_launches": stats.actor_launches,
                "actor_deferred": stats.actor_deferred,
                "actor_dedup_reuse": stats.actor_dedup_reuse,
                "speculator_launches": stats.speculator_launches,
                "max_inflight_actors_observed":
                    stats.max_inflight_actors_observed,
                "max_chain_depth_observed": stats.max_chain_depth_observed,
            },
        }
        Utils.save_json(result, join(output_dir, run_id, "result.json"))
        if self.logger:
            self.logger.log("DONE", Utils.dict_to_str(result["stats"]))
        print(f"[done] run_id={run_id} wall={wall:.2f}s  "
              f"hits={stats.hits}+{stats.cascaded_hits}cascade "
              f"misses={stats.misses}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run depth-speculative chess (Algorithm 3 v3).")
    p.add_argument("--config", default="config.yml",
                   help="Path to config YAML (default: config.yml)")
    p.add_argument("--stop-after", type=int, default=None,
                   help="Stop after N moves (default: from config)")
    p.add_argument("--trajectories-dir", default=None,
                   help="Output directory (overrides config)")
    args = p.parse_args()

    config = Config(args.config)
    if args.trajectories_dir is not None:
        config.trajectories_path = args.trajectories_dir.rstrip("/")

    stop_after = args.stop_after or config.stop_after
    out_base = (f"{config.trajectories_path.rstrip('/')}/"
                f"depth_{config.agent_name0}_vs_{config.agent_name1}"
                f"_guess_{config.guess_model_name}")

    runner = ChessDepthRunner(config)
    asyncio.run(runner.run(stop_after, out_base))


if __name__ == "__main__":
    main()
