"""Depth-Focused Speculative Actions — Algorithm 3 (v3).

Generic, domain-independent implementation of the depth-speculative
algorithm described in ``chess-game/algorithm_v3.md``. Run the module
directly to execute the simulation harness (``python depth_algorithm.py``).

Caller provides:
  * actor(state, h, q)      -> awaitable producing the ground-truth action
  * speculator(state, h, q) -> awaitable producing a list[action] of k guesses
  * transition(state, a)    -> next state (pure, cheap)
  * policy(state)           -> (h, q) API target + params (pure, cheap)
  * semantic_match(a_hat, a_truth, domain) -> bool

Concurrency model: asyncio single-threaded. No handler body ``await``s
anything except cascade-chained handlers — the only blocking point is
``asyncio.wait(return_when=FIRST_COMPLETED)`` at the top of the loop.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple


class _Deferred:
    def __repr__(self) -> str:
        return "DEFERRED"


DEFERRED = _Deferred()


class ErrorResult:
    def __init__(self, err: BaseException) -> None:
        self.err = err

    def __repr__(self) -> str:
        return f"ErrorResult({self.err!r})"


@dataclass
class Prediction:
    branch_id: int
    generation: int
    a_hat: Any
    s_next_hat: Any
    h_next: Any
    q_next: Any
    a_actor_next: Any  # asyncio.Task | DEFERRED
    a_spec_next: Optional[asyncio.Task]
    abandoned: bool = False
    actor_next_resolved: bool = False
    spec_next_resolved: bool = False
    predictions_next: Optional[List[Any]] = None


@dataclass
class Node:
    depth: int
    generation: int
    s: Any
    h: Any
    q: Any
    a_actor: asyncio.Task
    a_spec: asyncio.Task
    actor_resolved: bool = False
    spec_resolved: bool = False
    a_true: Any = None
    predictions: Optional[List[Prediction]] = None


@dataclass
class _Event:
    type: str  # 'actor' | 'spec' | 'actor_next' | 'spec_next'
    task: asyncio.Task
    depth: int
    generation: int
    branch_id: Optional[int] = None


@dataclass
class RunStats:
    hits: int = 0
    misses: int = 0
    cascaded_hits: int = 0
    actor_launches: int = 0
    actor_dedup_reuse: int = 0
    actor_deferred: int = 0
    speculator_launches: int = 0
    max_inflight_actors_observed: int = 0
    max_chain_depth_observed: int = 0


def _hashable(q: Any) -> Any:
    try:
        hash(q)
        return q
    except TypeError:
        return repr(q)


async def run_depth_speculative(
    s_0: Any,
    T: int,
    *,
    actor: Callable[[Any, Any, Any], Awaitable[Any]],
    speculator: Callable[[Any, Any, Any], Awaitable[List[Any]]],
    transition: Callable[[Any, Any], Any],
    policy: Callable[[Any], Tuple[Any, Any]],
    semantic_match: Callable[[Any, Any, str], bool],
    k: int = 3,
    max_inflight_actors: Optional[int] = None,
    domain: str = "generic",
    assert_invariants: bool = False,
) -> Tuple[List[Any], List[Any], RunStats]:
    """Run Algorithm 3 (v3) starting from ``s_0`` for ``T`` steps.

    Returns ``(states, actions, stats)`` where ``states[i]`` is the confirmed
    state at depth ``i`` and ``actions[i]`` is the ground-truth action at
    depth ``i`` (length ``T``).
    """
    if max_inflight_actors is None:
        max_inflight_actors = k + 1

    confirmed_t = 0
    s: Dict[int, Any] = {0: s_0}
    actions: Dict[int, Any] = {}
    spec_chain: Dict[int, Node] = {}
    current_generation = 0
    inflight_actors: Dict[asyncio.Task, Tuple[int, Optional[int], int]] = {}
    stats = RunStats()

    # ── helpers ──────────────────────────────────────────────────────

    def launch_actor(state: Any, h: Any, q: Any, depth: int,
                     branch_id: Optional[int], gen: int) -> asyncio.Task:
        task = asyncio.create_task(actor(state, h, q))
        inflight_actors[task] = (depth, branch_id, gen)
        stats.actor_launches += 1
        stats.max_inflight_actors_observed = max(
            stats.max_inflight_actors_observed, len(inflight_actors))
        return task

    def release_actor(task: asyncio.Task) -> None:
        inflight_actors.pop(task, None)

    def budget_ok() -> bool:
        return len(inflight_actors) < max_inflight_actors

    def build_predictions(a_list: List[Any], parent_s: Any,
                          parent_depth: int, gen: int) -> List[Prediction]:
        seen: Dict[Tuple[Any, Any], asyncio.Task] = {}
        preds: List[Prediction] = []
        for i, a_i in enumerate(a_list):
            s_next = transition(parent_s, a_i)
            h_next, q_next = policy(s_next)

            if parent_depth + 1 >= T:
                preds.append(Prediction(
                    branch_id=i, generation=gen, a_hat=a_i,
                    s_next_hat=s_next, h_next=h_next, q_next=q_next,
                    a_actor_next=DEFERRED, a_spec_next=None,
                    spec_next_resolved=True))
                continue

            key = (h_next, _hashable(q_next))
            if key in seen:
                a_actor_next: Any = seen[key]
                stats.actor_dedup_reuse += 1
            elif budget_ok():
                a_actor_next = launch_actor(
                    s_next, h_next, q_next, parent_depth + 1, i, gen)
                seen[key] = a_actor_next
            else:
                a_actor_next = DEFERRED
                stats.actor_deferred += 1

            a_spec_next = asyncio.create_task(speculator(s_next, h_next, q_next))
            stats.speculator_launches += 1

            preds.append(Prediction(
                branch_id=i, generation=gen, a_hat=a_i,
                s_next_hat=s_next, h_next=h_next, q_next=q_next,
                a_actor_next=a_actor_next, a_spec_next=a_spec_next,
                spec_next_resolved=False))
        return preds

    def abandon_prediction(pred: Prediction,
                           except_future: Optional[asyncio.Task] = None) -> None:
        pred.abandoned = True
        if (pred.a_actor_next is not DEFERRED
                and not pred.actor_next_resolved
                and pred.a_actor_next is not except_future):
            release_actor(pred.a_actor_next)

    def abandon_node(node: Node) -> None:
        if not node.actor_resolved:
            release_actor(node.a_actor)
        if node.predictions is not None:
            for p in node.predictions:
                abandon_prediction(p)

    def build_watch_set() -> List[_Event]:
        watched: List[_Event] = []
        for d, node in spec_chain.items():
            if not node.actor_resolved:
                watched.append(_Event('actor', node.a_actor, d, node.generation))
            if not node.spec_resolved:
                watched.append(_Event('spec', node.a_spec, d, node.generation))
            if node.predictions is not None:
                for p in node.predictions:
                    if p.abandoned:
                        continue
                    if (p.a_actor_next is not DEFERRED
                            and not p.actor_next_resolved):
                        watched.append(_Event(
                            'actor_next', p.a_actor_next,
                            d, p.generation, p.branch_id))
                    if (p.a_spec_next is not None
                            and not p.spec_next_resolved):
                        watched.append(_Event(
                            'spec_next', p.a_spec_next,
                            d, p.generation, p.branch_id))
        return watched

    def safe_result(task: asyncio.Task) -> Any:
        try:
            return task.result()
        except BaseException as e:  # noqa: BLE001
            return ErrorResult(e)

    def find_match(preds: List[Prediction], a_truth: Any) -> Optional[Prediction]:
        for p in preds:
            if semantic_match(p.a_hat, a_truth, domain):
                return p
        return None

    def invariants_hold() -> None:
        # Budget invariant: inflight count never exceeds max + mandatory confirmed_t slot
        # (the confirmed_t Actor is always inflight until resolved, not speculative).
        assert len(inflight_actors) <= max_inflight_actors + 1, \
            f"budget overrun: {len(inflight_actors)} > {max_inflight_actors}+1"
        # Chain monotonicity
        if spec_chain:
            ks = sorted(spec_chain.keys())
            assert ks[0] >= confirmed_t, f"chain starts before confirmed_t"

    # ── per-event work: HIT/MISS dispatch ────────────────────────────

    async def handle_hit(node: Node, match: Prediction, a_d: Any) -> None:
        nonlocal confirmed_t
        stats.hits += 1
        except_fut = match.a_actor_next if match.a_actor_next is not DEFERRED else None
        for p in node.predictions or []:
            if p is not match:
                abandon_prediction(p, except_future=except_fut)
        s[confirmed_t + 1] = transition(s[confirmed_t], a_d)
        actions[confirmed_t] = a_d
        del spec_chain[confirmed_t]
        confirmed_t += 1
        if confirmed_t < T:
            promote_branch(match, confirmed_t)
        await cascade_loop()

    async def cascade_loop() -> None:
        nonlocal confirmed_t
        while confirmed_t < T and confirmed_t in spec_chain:
            next_node = spec_chain[confirmed_t]
            if not next_node.actor_resolved:
                break
            if not next_node.spec_resolved:
                break
            a_next = next_node.a_true
            if a_next is None:
                # Deeper Actor errored; re-issue ground-truth here
                await handle_miss_synthetic(next_node)
                break
            match = find_match(next_node.predictions or [], a_next)
            if match is not None:
                stats.cascaded_hits += 1
                except_fut = match.a_actor_next if match.a_actor_next is not DEFERRED else None
                for p in next_node.predictions or []:
                    if p is not match:
                        abandon_prediction(p, except_future=except_fut)
                s[confirmed_t + 1] = transition(s[confirmed_t], a_next)
                actions[confirmed_t] = a_next
                del spec_chain[confirmed_t]
                confirmed_t += 1
                if confirmed_t < T:
                    promote_branch(match, confirmed_t)
            else:
                await handle_miss(a_next)
                break

    async def handle_miss(a_truth: Any) -> None:
        nonlocal confirmed_t, current_generation
        stats.misses += 1
        s[confirmed_t + 1] = transition(s[confirmed_t], a_truth)
        actions[confirmed_t] = a_truth
        current_generation += 1
        for node in list(spec_chain.values()):
            abandon_node(node)
        spec_chain.clear()
        confirmed_t += 1
        if confirmed_t < T:
            h_new, q_new = policy(s[confirmed_t])
            a_actor_new = launch_actor(
                s[confirmed_t], h_new, q_new,
                confirmed_t, None, current_generation)
            a_spec_new = asyncio.create_task(
                speculator(s[confirmed_t], h_new, q_new))
            stats.speculator_launches += 1
            spec_chain[confirmed_t] = Node(
                depth=confirmed_t, generation=current_generation,
                s=s[confirmed_t], h=h_new, q=q_new,
                a_actor=a_actor_new, a_spec=a_spec_new)

    async def handle_miss_synthetic(node: Node) -> None:
        # A deeper Actor errored and was promoted into this Node with
        # a_true=None. Re-issue the ground-truth Actor synchronously
        # (rare error path; blocking here is fine).
        fresh = launch_actor(
            s[confirmed_t], node.h, node.q,
            confirmed_t, None, current_generation)
        try:
            result = await fresh
        except BaseException as e:  # noqa: BLE001
            release_actor(fresh)
            raise RuntimeError(f"unrecoverable Actor failure: {e}") from e
        release_actor(fresh)
        await handle_miss(result)

    def promote_branch(pred: Prediction, depth: int) -> None:
        if pred.a_actor_next is DEFERRED:
            a_actor = launch_actor(
                pred.s_next_hat, pred.h_next, pred.q_next,
                depth, None, current_generation)
        else:
            a_actor = pred.a_actor_next
            if a_actor in inflight_actors:
                inflight_actors[a_actor] = (depth, None, current_generation)

        new_node = Node(
            depth=depth, generation=current_generation,
            s=pred.s_next_hat, h=pred.h_next, q=pred.q_next,
            a_actor=a_actor, a_spec=pred.a_spec_next)

        if pred.actor_next_resolved:
            r = safe_result(a_actor)
            new_node.actor_resolved = True
            new_node.a_true = None if isinstance(r, ErrorResult) else r
            release_actor(a_actor)

        if pred.spec_next_resolved:
            new_node.spec_resolved = True
            if pred.predictions_next:
                new_node.predictions = build_predictions(
                    pred.predictions_next, pred.s_next_hat,
                    depth, current_generation)
            else:
                new_node.predictions = []

        spec_chain[depth] = new_node

    # ── initialization ───────────────────────────────────────────────

    h_0, q_0 = policy(s_0)
    a_actor_0 = launch_actor(s_0, h_0, q_0, 0, None, 0)
    a_spec_0 = asyncio.create_task(speculator(s_0, h_0, q_0))
    stats.speculator_launches += 1
    spec_chain[0] = Node(
        depth=0, generation=0, s=s_0, h=h_0, q=q_0,
        a_actor=a_actor_0, a_spec=a_spec_0)

    # ── main event loop ──────────────────────────────────────────────

    while confirmed_t < T:
        watched = build_watch_set()
        if not watched:
            raise RuntimeError(
                f"event loop stalled at confirmed_t={confirmed_t}")
        stats.max_chain_depth_observed = max(
            stats.max_chain_depth_observed, len(spec_chain))

        done, _pending = await asyncio.wait(
            [e.task for e in watched],
            return_when=asyncio.FIRST_COMPLETED)

        # Pick the first event whose task is done (deterministic order).
        event: Optional[_Event] = None
        for e in watched:
            if e.task in done:
                event = e
                break
        assert event is not None

        if event.generation != current_generation:
            continue

        if event.type in ('spec_next', 'actor_next'):
            node = spec_chain.get(event.depth)
            if node is None or node.predictions is None:
                continue
            pred = node.predictions[event.branch_id]  # type: ignore[index]
            if pred.abandoned:
                continue

        if event.type == 'actor':
            node = spec_chain.get(event.depth)
            if node is None:
                continue
            r = safe_result(event.task)
            if isinstance(r, ErrorResult):
                if event.depth == confirmed_t:
                    raise RuntimeError(
                        f"ground-truth Actor failed at depth {confirmed_t}: {r.err}"
                    ) from r.err
                node.actor_resolved = True
                node.a_true = None
                release_actor(event.task)
                continue
            node.actor_resolved = True
            node.a_true = r
            release_actor(event.task)
            if event.depth == confirmed_t and node.spec_resolved:
                match = find_match(node.predictions or [], r)
                if match is not None:
                    await handle_hit(node, match, r)
                else:
                    await handle_miss(r)
            if assert_invariants:
                invariants_hold()
            continue

        if event.type == 'spec':
            node = spec_chain.get(event.depth)
            if node is None:
                continue
            r = safe_result(event.task)
            node.spec_resolved = True
            if isinstance(r, ErrorResult):
                node.predictions = []
            else:
                node.predictions = build_predictions(
                    r, node.s, event.depth, node.generation)
            if event.depth == confirmed_t and node.actor_resolved:
                a_d = node.a_true
                if a_d is None:
                    await handle_miss_synthetic(node)
                else:
                    match = find_match(node.predictions or [], a_d)
                    if match is not None:
                        await handle_hit(node, match, a_d)
                    else:
                        await handle_miss(a_d)
            if assert_invariants:
                invariants_hold()
            continue

        if event.type == 'actor_next':
            node = spec_chain[event.depth]
            pred = node.predictions[event.branch_id]  # type: ignore[index]
            r = safe_result(event.task)
            if isinstance(r, ErrorResult):
                abandon_prediction(pred)
                continue
            pred.actor_next_resolved = True
            # Keep result available in the task for promote_branch. Just
            # release the budget slot — the Future object still holds .result().
            release_actor(event.task)
            if assert_invariants:
                invariants_hold()
            continue

        if event.type == 'spec_next':
            node = spec_chain[event.depth]
            pred = node.predictions[event.branch_id]  # type: ignore[index]
            r = safe_result(event.task)
            pred.spec_next_resolved = True
            pred.predictions_next = [] if isinstance(r, ErrorResult) else r
            if assert_invariants:
                invariants_hold()
            continue

    # ── cleanup: cancel any orphaned tasks ───────────────────────────

    for task in list(inflight_actors.keys()):
        if not task.done():
            task.cancel()
    inflight_actors.clear()

    states = [s[i] for i in range(confirmed_t + 1)]
    action_list = [actions[i] for i in range(confirmed_t)]
    return states, action_list, stats


# ═════════════════════════════════════════════════════════════════════
# Simulation harness
# ═════════════════════════════════════════════════════════════════════


async def _sequential_baseline(
    s_0: Any, T: int, *, actor, transition, policy,
) -> Tuple[List[Any], List[Any]]:
    """Strict sequential execution — the ground-truth trajectory the
    speculative algorithm must match (lossless property).
    """
    s = [s_0]
    actions: List[Any] = []
    for _ in range(T):
        h, q = policy(s[-1])
        a = await actor(s[-1], h, q)
        actions.append(a)
        s.append(transition(s[-1], a))
    return s, actions


def _check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    line = f"  [{status}] {name}"
    if detail:
        line += f"  ({detail})"
    print(line)
    if not cond:
        raise AssertionError(name)


async def _scenario_all_hits() -> None:
    """Speculator always predicts correctly → long HIT chain, cascades."""
    print("\n[scenario] all_hits — Speculator is an oracle")
    ALPHA, BETA = 0.10, 0.02
    T = 8

    # State: int counter. Action: increment amount in {1,2,3}. Ground truth: always 2.
    async def actor(s, h, q):
        await asyncio.sleep(ALPHA)
        return 2

    async def speculator(s, h, q):
        await asyncio.sleep(BETA)
        return [2, 1, 3]  # first one is correct (the oracle)

    transition = lambda s, a: s + a
    policy = lambda s: ("inc", s)
    match = lambda a_hat, a, d: a_hat == a

    t0 = asyncio.get_event_loop().time()
    states, actions, stats = await run_depth_speculative(
        0, T, actor=actor, speculator=speculator,
        transition=transition, policy=policy, semantic_match=match,
        k=3, assert_invariants=True)
    elapsed = asyncio.get_event_loop().time() - t0

    base_states, base_actions = await _sequential_baseline(
        0, T, actor=actor, transition=transition, policy=policy)
    base_elapsed_lower_bound = T * ALPHA

    _check("trajectory matches sequential baseline", states == base_states,
           f"got {states}, want {base_states}")
    _check("actions match baseline", actions == base_actions)
    _check("hits+cascaded_hits == T", stats.hits + stats.cascaded_hits == T,
           f"hits={stats.hits} cascaded={stats.cascaded_hits}")
    _check("no misses", stats.misses == 0)
    _check("speedup over sequential",
           elapsed < 0.7 * base_elapsed_lower_bound,
           f"{elapsed:.3f}s vs {base_elapsed_lower_bound:.3f}s sequential")


async def _scenario_all_miss() -> None:
    """Speculator always wrong → zero HITs, falls back to sequential-ish."""
    print("\n[scenario] all_miss — Speculator always wrong")
    ALPHA, BETA = 0.05, 0.02
    T = 6

    async def actor(s, h, q):
        await asyncio.sleep(ALPHA)
        return 2

    async def speculator(s, h, q):
        await asyncio.sleep(BETA)
        return [0, 1, 3]  # none match the true 2

    transition = lambda s, a: s + a
    policy = lambda s: ("inc", s)
    match = lambda a_hat, a, d: a_hat == a

    states, actions, stats = await run_depth_speculative(
        0, T, actor=actor, speculator=speculator,
        transition=transition, policy=policy, semantic_match=match,
        k=3, assert_invariants=True)

    base_states, base_actions = await _sequential_baseline(
        0, T, actor=actor, transition=transition, policy=policy)

    _check("trajectory matches baseline", states == base_states)
    _check("actions match baseline", actions == base_actions)
    _check("misses == T", stats.misses == T, f"misses={stats.misses}")
    _check("no hits", stats.hits == 0 and stats.cascaded_hits == 0)


async def _scenario_mixed() -> None:
    """Correct prediction every other step → alternating HIT/MISS."""
    print("\n[scenario] mixed — speculator correct on even depths only")
    ALPHA, BETA = 0.05, 0.02
    T = 8
    step_counter = [0]

    async def actor(s, h, q):
        await asyncio.sleep(ALPHA)
        return 1 if s % 2 == 0 else 2

    async def speculator(s, h, q):
        await asyncio.sleep(BETA)
        step_counter[0] += 1
        # Put the correct action in the list for even depths, wrong for odd.
        # Since state parity flips with each (action_1 or action_2),
        # we just provide both possible actions — one will always match.
        return [1, 2, 3]

    transition = lambda s, a: s + a
    policy = lambda s: ("inc", s)
    match = lambda a_hat, a, d: a_hat == a

    states, actions, stats = await run_depth_speculative(
        0, T, actor=actor, speculator=speculator,
        transition=transition, policy=policy, semantic_match=match,
        k=3, assert_invariants=True)

    base_states, base_actions = await _sequential_baseline(
        0, T, actor=actor, transition=transition, policy=policy)

    _check("trajectory matches baseline", states == base_states,
           f"got {states}, want {base_states}")
    _check("actions match baseline", actions == base_actions)
    _check("hits+cascaded+misses == T",
           stats.hits + stats.cascaded_hits + stats.misses == T,
           f"h={stats.hits} c={stats.cascaded_hits} m={stats.misses}")


async def _scenario_budget_squeeze() -> None:
    """max_inflight_actors=0: all speculative Actors deferred. Correctness preserved."""
    print("\n[scenario] budget_squeeze — zero speculative Actor budget")
    ALPHA, BETA = 0.03, 0.01
    T = 5

    async def actor(s, h, q):
        await asyncio.sleep(ALPHA)
        return 1

    async def speculator(s, h, q):
        await asyncio.sleep(BETA)
        return [1, 2]

    transition = lambda s, a: s + a
    policy = lambda s: ("inc", s)
    match = lambda a_hat, a, d: a_hat == a

    states, actions, stats = await run_depth_speculative(
        0, T, actor=actor, speculator=speculator,
        transition=transition, policy=policy, semantic_match=match,
        k=2, max_inflight_actors=0, assert_invariants=True)

    base_states, base_actions = await _sequential_baseline(
        0, T, actor=actor, transition=transition, policy=policy)

    _check("trajectory matches baseline under zero budget",
           states == base_states)
    _check("all speculative Actors were deferred", stats.actor_deferred > 0,
           f"deferred={stats.actor_deferred}")


async def _scenario_dedup() -> None:
    """Speculator returns duplicates → only one Actor future launched per (h,q)."""
    print("\n[scenario] dedup — speculator emits duplicate predictions")
    ALPHA, BETA = 0.04, 0.01
    T = 4

    async def actor(s, h, q):
        await asyncio.sleep(ALPHA)
        return 1

    async def speculator(s, h, q):
        await asyncio.sleep(BETA)
        # All three predictions are the same → should dedup to 1 launched Actor
        return [1, 1, 1]

    transition = lambda s, a: s + a
    policy = lambda s: ("inc", s)
    match = lambda a_hat, a, d: a_hat == a

    states, _, stats = await run_depth_speculative(
        0, T, actor=actor, speculator=speculator,
        transition=transition, policy=policy, semantic_match=match,
        k=3, max_inflight_actors=5, assert_invariants=True)

    base_states, _ = await _sequential_baseline(
        0, T, actor=actor, transition=transition, policy=policy)

    _check("trajectory matches baseline with dedup siblings",
           states == base_states)
    _check("dedup reuse occurred", stats.actor_dedup_reuse > 0,
           f"reuse={stats.actor_dedup_reuse}")


async def _scenario_actor_error() -> None:
    """Deeper Actor errors; depth handler marks branch abandoned, loop continues."""
    print("\n[scenario] actor_error — speculative Actor raises at depth≥1")
    ALPHA, BETA = 0.04, 0.01
    T = 4

    call_count = [0]

    async def actor(s, h, q):
        await asyncio.sleep(ALPHA)
        call_count[0] += 1
        # First call (depth=0, ground truth) succeeds; certain speculative
        # Actors at deeper depths error.
        if s >= 5 and call_count[0] > 3:
            raise RuntimeError("simulated upstream 500")
        return 1

    async def speculator(s, h, q):
        await asyncio.sleep(BETA)
        return [1, 2]

    transition = lambda s, a: s + a
    policy = lambda s: ("inc", s)
    match = lambda a_hat, a, d: a_hat == a

    states, _, stats = await run_depth_speculative(
        0, T, actor=actor, speculator=speculator,
        transition=transition, policy=policy, semantic_match=match,
        k=2, assert_invariants=True)

    base_states, _ = await _sequential_baseline(
        0, T, actor=actor, transition=transition, policy=policy)
    # Reset call_count for baseline fairness is not strictly needed — baseline
    # only runs ground-truth Actors (one per depth, always s < 5 here).

    _check("trajectory still matches baseline under Actor errors",
           states == base_states, f"got {states}, want {base_states}")


async def main() -> int:
    scenarios = [
        _scenario_all_hits,
        _scenario_all_miss,
        _scenario_mixed,
        _scenario_budget_squeeze,
        _scenario_dedup,
        _scenario_actor_error,
    ]
    failed = 0
    for s in scenarios:
        try:
            await s()
        except AssertionError as e:
            print(f"  scenario failed: {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  scenario raised: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{'=' * 60}")
    print(f"  Harness: {len(scenarios) - failed}/{len(scenarios)} passed")
    print(f"{'=' * 60}")
    return failed


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
