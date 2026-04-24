"""Depth-Focused Speculative Actions — Algorithm v4 (unbounded-depth tree).

Generic, domain-independent implementation of a tree-shaped depth-speculative
algorithm described in ``chess-game/algorithm_v4.md``. Run the module
directly to execute the simulation harness (``python depth_algorithm.py``).

Semantics (v4):
  * Every speculative state is a ``SpecNode``. There is no separate
    Prediction type — a node's children ARE its predictions.
  * When a Speculator resolves at ANY node, we immediately expand that
    node's children: for each candidate action, build the next state,
    then launch BOTH an Actor and a Speculator for that child. This
    recurses: a grandchild's Speculator resolving also expands, etc.
  * The ONLY bound on tree growth is ``max_inflight_actors``. Actor
    launches beyond that limit are marked DEFERRED (not launched until
    promotion to the root frees budget). Speculators are not budget-
    limited; they run as soon as a node is created.

Caller provides:
  * actor(state, h, q)      -> awaitable producing the ground-truth action
  * speculator(state, h, q) -> awaitable producing a list[action] of k guesses
  * transition(state, a)    -> next state (pure, cheap)
  * policy(state)           -> (h, q) API target + params (pure, cheap)
  * semantic_match(a_hat, a_truth, domain) -> bool

Concurrency model: asyncio single-threaded. Only blocking point is
``asyncio.wait(return_when=FIRST_COMPLETED)`` at the top of the loop.
"""

from __future__ import annotations

import asyncio
import time
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
class SpecNode:
    """A speculative or confirmed state in the speculation tree.

    The root of the tree is the node at ``confirmed_t`` (parent=None). All
    other nodes are speculative. Every node has its own Actor and Speculator
    call (or DEFERRED / None). Children are created eagerly when the node's
    Speculator resolves.
    """
    depth: int
    s: Any
    h: Any
    q: Any
    parent: Optional['SpecNode']
    a_hat_from_parent: Any  # the speculated action that led here; None for root
    a_actor: Any = None     # asyncio.Task | DEFERRED | None (None only at horizon)
    a_spec: Optional[asyncio.Task] = None
    actor_resolved: bool = False
    spec_resolved: bool = False
    a_true: Any = None
    predictions: Optional[List[Any]] = None  # Speculator's k candidate actions
    children: Optional[List['SpecNode']] = None
    effective_next: Optional['SpecNode'] = None  # matching child after prune
    pruned: bool = False      # whether we've run match-based sibling pruning
    abandoned: bool = False


@dataclass
class _Event:
    type: str  # 'actor' | 'spec'
    task: asyncio.Task
    node: SpecNode


@dataclass
class RunStats:
    hits: int = 0
    misses: int = 0
    cascaded_hits: int = 0
    actor_launches: int = 0
    actor_dedup_reuse: int = 0
    actor_deferred: int = 0
    actor_promoted_launch: int = 0
    speculator_launches: int = 0
    max_inflight_actors_observed: int = 0
    max_tree_nodes: int = 0
    max_tree_depth: int = 0
    max_cascade_run_length: int = 0


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
    max_inflight_actors: int = 13,
    domain: str = "generic",
    assert_invariants: bool = False,
    record_steps: bool = False,
    on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> Tuple[List[Any], List[Any], RunStats, List[Dict[str, Any]]]:
    """Run Algorithm v4 starting from ``s_0`` for ``T`` steps.

    Returns ``(states, actions, stats, step_info)`` where ``states[i]`` is
    the confirmed state at depth ``i``, ``actions[i]`` is the ground-truth
    action at depth ``i`` (length ``T``), and ``step_info`` is a list of
    per-step dicts (empty unless ``record_steps=True``).
    """
    confirmed_t = 0
    states: Dict[int, Any] = {0: s_0}
    actions: Dict[int, Any] = {}
    stats = RunStats()
    inflight_actors: Dict[asyncio.Task, SpecNode] = {}
    step_info: List[Dict[str, Any]] = []
    run_start = time.perf_counter()

    # ── helpers ──────────────────────────────────────────────────────

    def emit(kind: str, **fields: Any) -> None:
        if on_event is not None:
            fields.setdefault("inflight", len(inflight_actors))
            on_event(kind, fields)

    def budget_ok() -> bool:
        return len(inflight_actors) < max_inflight_actors

    def launch_actor_task(node: SpecNode,
                          shared: Optional[asyncio.Task] = None) -> Any:
        """Launch an Actor API call for ``node`` if budget allows.

        If ``shared`` is given, reuse that Task (sibling dedup).
        Returns the Task, a shared Task, or ``DEFERRED``.
        """
        if shared is not None:
            stats.actor_dedup_reuse += 1
            emit("ACTOR_DEDUP", depth=node.depth, a_hat=node.a_hat_from_parent)
            return shared
        if not budget_ok():
            stats.actor_deferred += 1
            emit("ACTOR_DEFERRED", depth=node.depth,
                 a_hat=node.a_hat_from_parent)
            return DEFERRED
        task = asyncio.create_task(actor(node.s, node.h, node.q))
        inflight_actors[task] = node
        stats.actor_launches += 1
        stats.max_inflight_actors_observed = max(
            stats.max_inflight_actors_observed, len(inflight_actors))
        emit("ACTOR_LAUNCH", depth=node.depth,
             a_hat=node.a_hat_from_parent)
        return task

    def launch_actor_force(node: SpecNode) -> asyncio.Task:
        """Launch an Actor unconditionally (for ground-truth root). Still
        recorded in ``inflight_actors`` but may exceed the soft budget."""
        task = asyncio.create_task(actor(node.s, node.h, node.q))
        inflight_actors[task] = node
        stats.actor_launches += 1
        stats.actor_promoted_launch += 1
        stats.max_inflight_actors_observed = max(
            stats.max_inflight_actors_observed, len(inflight_actors))
        emit("ACTOR_LAUNCH_FORCE", depth=node.depth,
             a_hat=node.a_hat_from_parent)
        return task

    def launch_speculator_task(node: SpecNode) -> asyncio.Task:
        task = asyncio.create_task(speculator(node.s, node.h, node.q))
        stats.speculator_launches += 1
        emit("SPEC_LAUNCH", depth=node.depth,
             a_hat=node.a_hat_from_parent)
        return task

    def release_actor(task: asyncio.Task) -> None:
        inflight_actors.pop(task, None)

    def make_root(s: Any, depth: int) -> SpecNode:
        h, q = policy(s)
        node = SpecNode(depth=depth, s=s, h=h, q=q,
                        parent=None, a_hat_from_parent=None)
        if depth < T:
            node.a_actor = launch_actor_force(node)
            node.a_spec = launch_speculator_task(node)
        else:
            node.a_actor = None
            node.a_spec = None
            node.actor_resolved = True
            node.spec_resolved = True
        return node

    def make_child(parent: SpecNode, a_hat: Any,
                   shared_actor: Optional[asyncio.Task]) -> SpecNode:
        s_child = transition(parent.s, a_hat)
        h, q = policy(s_child)
        depth = parent.depth + 1
        node = SpecNode(depth=depth, s=s_child, h=h, q=q,
                        parent=parent, a_hat_from_parent=a_hat)
        if depth >= T:
            # At horizon: no further expansion possible.
            node.a_actor = None
            node.a_spec = None
            node.actor_resolved = True
            node.spec_resolved = True
            return node
        # Try to launch the Actor first. If budget is full and we have
        # no sibling to share with, mark DEFERRED — and crucially skip
        # the Speculator too, otherwise the Speculator tree would grow
        # exponentially even when Actors can no longer follow it. The
        # Speculator will be launched at promotion time.
        node.a_actor = launch_actor_task(node, shared=shared_actor)
        if node.a_actor is DEFERRED:
            node.a_spec = None
        else:
            node.a_spec = launch_speculator_task(node)
        return node

    def expand_children(node: SpecNode) -> None:
        """Create children for every prediction. Called once, when the
        Speculator resolves. Siblings with the same (h, q) share one
        Actor Task to avoid duplicate API calls.
        """
        if node.children is not None or node.predictions is None:
            return
        if node.abandoned:
            return
        seen: Dict[Tuple[Any, Any], asyncio.Task] = {}
        kids: List[SpecNode] = []
        for a_hat in node.predictions:
            try:
                s_child = transition(node.s, a_hat)
            except Exception:  # noqa: BLE001 - illegal speculated action
                continue
            _h_peek, _q_peek = policy(s_child)
            key = (_h_peek, _hashable(_q_peek))
            shared = seen.get(key)
            child = make_child(node, a_hat, shared)
            if (isinstance(child.a_actor, asyncio.Task)
                    and key not in seen):
                seen[key] = child.a_actor
            kids.append(child)
            stats.max_tree_depth = max(stats.max_tree_depth, child.depth)
        node.children = kids
        stats.max_tree_nodes = max(stats.max_tree_nodes, count_tree_nodes())

    def count_tree_nodes() -> int:
        if root is None:
            return 0
        n = 0
        stack: List[SpecNode] = [root]
        while stack:
            cur = stack.pop()
            if cur.abandoned:
                continue
            n += 1
            if cur.children:
                stack.extend(cur.children)
        return n

    def abandon_subtree(node: Optional[SpecNode]) -> None:
        if node is None or node.abandoned:
            return
        emit("ABANDON", depth=node.depth,
             a_hat=node.a_hat_from_parent)
        # DFS iterative to avoid recursion limits on deep trees.
        # NB: we don't cancel in-flight Tasks. Actor futures may be shared
        # with a sibling via dedup; cancelling here would kill the winner's
        # task too. We just release the budget slot and mark the node
        # abandoned — stale events are filtered by ``node.abandoned`` in
        # the main loop, and orphaned Tasks get cancelled at final cleanup.
        stack: List[SpecNode] = [node]
        while stack:
            cur = stack.pop()
            if cur.abandoned:
                continue
            cur.abandoned = True
            if (isinstance(cur.a_actor, asyncio.Task)
                    and not cur.actor_resolved):
                release_actor(cur.a_actor)
            if cur.children:
                stack.extend(cur.children)

    def build_watch_set() -> List[_Event]:
        """DFS the live tree and collect every unresolved Actor/Spec Task."""
        watched: List[_Event] = []
        seen_tasks: set = set()
        stack: List[SpecNode] = [root] if root is not None else []
        while stack:
            cur = stack.pop()
            if cur.abandoned:
                continue
            if (isinstance(cur.a_actor, asyncio.Task)
                    and not cur.actor_resolved
                    and cur.a_actor not in seen_tasks):
                watched.append(_Event('actor', cur.a_actor, cur))
                seen_tasks.add(cur.a_actor)
            if (isinstance(cur.a_spec, asyncio.Task)
                    and not cur.spec_resolved
                    and cur.a_spec not in seen_tasks):
                watched.append(_Event('spec', cur.a_spec, cur))
                seen_tasks.add(cur.a_spec)
            if cur.children:
                stack.extend(cur.children)
        return watched

    def safe_result(task: asyncio.Task) -> Any:
        try:
            return task.result()
        except BaseException as e:  # noqa: BLE001
            return ErrorResult(e)

    def find_match(node: SpecNode, a_truth: Any) -> Optional[SpecNode]:
        if a_truth is None or not node.children:
            return None
        for c in node.children:
            if c.abandoned:
                continue
            if semantic_match(c.a_hat_from_parent, a_truth, domain):
                return c
        return None

    def prune_on_resolution(node: SpecNode) -> None:
        """Once a node has both Actor and Speculator resolved, identify
        the matching child (if any) and abandon the others. Called at
        every depth — frees budget proactively."""
        if node.pruned or node.abandoned:
            return
        if not (node.actor_resolved and node.spec_resolved):
            return
        if node.children is None:
            return  # predictions were empty
        match = find_match(node, node.a_true)
        node.effective_next = match
        node.pruned = True
        if node.children:
            for c in node.children:
                if c is not match:
                    abandon_subtree(c)

    def record_step(kind: str, a_true: Any, node_at_conf: SpecNode) -> None:
        if not record_steps:
            return
        preds = node_at_conf.predictions or []
        step_info.append({
            "step": confirmed_t,
            "action": a_true,
            "kind": kind,
            "predictions": list(preds),
            "time_to_confirm_seconds": time.perf_counter() - run_start,
            "inflight_actors_at_confirm": len(inflight_actors),
        })

    # ── initialization ───────────────────────────────────────────────

    root: Optional[SpecNode] = make_root(s_0, 0)

    # ── cascade / miss handlers ──────────────────────────────────────

    def try_cascade() -> None:
        """Advance ``confirmed_t`` as far as the current root allows.

        At each step the root must have actor AND speculator resolved
        (or actor errored, which we promote into a MISS via
        ``handle_miss_synthetic`` semantics below). Cascades bubble the
        ``effective_next`` chain eagerly.
        """
        nonlocal root, confirmed_t
        cascaded_this_call = 0
        while confirmed_t < T and root is not None:
            if not root.actor_resolved or not root.spec_resolved:
                return

            # Actor errored on root; must restart ground-truth here.
            if root.a_true is None:
                handle_miss_synthetic()
                return

            prune_on_resolution(root)
            a_true = root.a_true
            match = root.effective_next

            if match is not None:
                # Record step, advance
                if cascaded_this_call == 0:
                    stats.hits += 1
                    record_step("hit", a_true, root)
                    emit("HIT", step=confirmed_t, a_true=a_true,
                         matched=match.a_hat_from_parent)
                else:
                    stats.cascaded_hits += 1
                    record_step("cascaded_hit", a_true, root)
                    emit("CASCADED_HIT", step=confirmed_t, a_true=a_true,
                         matched=match.a_hat_from_parent)
                cascaded_this_call += 1
                stats.max_cascade_run_length = max(
                    stats.max_cascade_run_length, cascaded_this_call)

                states[confirmed_t + 1] = transition(states[confirmed_t], a_true)
                actions[confirmed_t] = a_true

                # Promote match to new root. Siblings were abandoned in
                # prune_on_resolution.
                new_root = match
                new_root.parent = None
                root = new_root
                confirmed_t += 1

                # Root MUST have a real Actor call in flight (or resolved).
                # If it was DEFERRED when created, launch now — and also
                # launch its Speculator, which we skipped at creation time.
                if confirmed_t < T and root.a_actor is DEFERRED:
                    emit("PROMOTE_DEFERRED", depth=root.depth)
                    root.a_actor = launch_actor_force(root)
                    if root.a_spec is None:
                        root.a_spec = launch_speculator_task(root)
                    # Not resolved yet; loop will exit and wait.
            else:
                # MISS
                stats.misses += 1
                record_step("miss", a_true, root)
                emit("MISS", step=confirmed_t, a_true=a_true,
                     predictions=list(root.predictions or []))
                states[confirmed_t + 1] = transition(states[confirmed_t], a_true)
                actions[confirmed_t] = a_true
                abandon_subtree(root)
                confirmed_t += 1
                if confirmed_t < T:
                    root = make_root(states[confirmed_t], confirmed_t)
                else:
                    root = None
                return

    async def handle_miss_synthetic() -> None:
        """The root's Actor errored. Re-issue the ground-truth Actor
        synchronously; rare error-recovery path."""
        nonlocal root, confirmed_t
        if root is None:
            return
        fresh = launch_actor_force(root)
        try:
            result = await fresh
        except BaseException as e:  # noqa: BLE001
            release_actor(fresh)
            raise RuntimeError(f"unrecoverable Actor failure: {e}") from e
        release_actor(fresh)
        # Treat as a MISS at the current confirmed_t.
        stats.misses += 1
        record_step("miss", result, root)
        states[confirmed_t + 1] = transition(states[confirmed_t], result)
        actions[confirmed_t] = result
        abandon_subtree(root)
        confirmed_t += 1
        if confirmed_t < T:
            root = make_root(states[confirmed_t], confirmed_t)
        else:
            root = None

    def invariants_hold() -> None:
        # Soft budget check: + small slack for the force-launched root.
        assert len(inflight_actors) <= max_inflight_actors + 1, \
            f"budget overrun: {len(inflight_actors)} > {max_inflight_actors}+1"
        if root is not None:
            assert not root.abandoned, "root is abandoned"

    # ── main event loop ──────────────────────────────────────────────

    while confirmed_t < T:
        if root is None:
            break
        watched = build_watch_set()
        if not watched:
            raise RuntimeError(
                f"event loop stalled at confirmed_t={confirmed_t}")

        done, _pending = await asyncio.wait(
            [e.task for e in watched],
            return_when=asyncio.FIRST_COMPLETED)

        # Process ALL done events this cycle (FIRST_COMPLETED may return
        # multiple simultaneously-resolved tasks).
        for e in watched:
            if e.task not in done:
                continue
            node = e.node
            if node.abandoned:
                continue

            if e.type == 'actor':
                r = safe_result(e.task)
                release_actor(e.task)
                if isinstance(r, ErrorResult):
                    emit("ACTOR_ERROR", depth=node.depth, err=str(r.err))
                    if node is root:
                        raise RuntimeError(
                            f"ground-truth Actor failed at depth "
                            f"{confirmed_t}: {r.err}") from r.err
                    # Speculative Actor errored — abandon this subtree.
                    abandon_subtree(node)
                    continue
                node.actor_resolved = True
                node.a_true = r
                emit("ACTOR_RESOLVED", depth=node.depth,
                     a_true=r, is_root=(node is root))
                prune_on_resolution(node)

            elif e.type == 'spec':
                r = safe_result(e.task)
                node.spec_resolved = True
                if isinstance(r, ErrorResult):
                    node.predictions = []
                    emit("SPEC_ERROR", depth=node.depth, err=str(r.err))
                else:
                    node.predictions = list(r)
                    emit("SPEC_RESOLVED", depth=node.depth,
                         predictions=list(r))
                expand_children(node)
                prune_on_resolution(node)

            if assert_invariants:
                invariants_hold()

        # After processing events, see if the root can cascade forward.
        if root is not None:
            # try_cascade may itself call handle_miss_synthetic which is
            # async; we handle that case separately:
            if root.actor_resolved and root.a_true is None:
                await handle_miss_synthetic()
            else:
                try_cascade()

        if assert_invariants:
            invariants_hold()

    # ── cleanup ──────────────────────────────────────────────────────

    for task in list(inflight_actors.keys()):
        if not task.done():
            task.cancel()
    inflight_actors.clear()

    states_list = [states[i] for i in range(confirmed_t + 1)]
    action_list = [actions[i] for i in range(confirmed_t)]
    return states_list, action_list, stats, step_info


# ═════════════════════════════════════════════════════════════════════
# Simulation harness
# ═════════════════════════════════════════════════════════════════════


async def _sequential_baseline(
    s_0: Any, T: int, *, actor, transition, policy,
) -> Tuple[List[Any], List[Any]]:
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
    print("\n[scenario] all_hits — Speculator is an oracle")
    ALPHA, BETA = 0.10, 0.02
    T = 8

    async def actor(s, h, q):
        await asyncio.sleep(ALPHA)
        return 2

    async def speculator(s, h, q):
        await asyncio.sleep(BETA)
        return [2, 1, 3]

    transition = lambda s, a: s + a
    policy = lambda s: ("inc", s)
    match = lambda a_hat, a, d: a_hat == a

    t0 = asyncio.get_event_loop().time()
    states, actions, stats, _ = await run_depth_speculative(
        0, T, actor=actor, speculator=speculator,
        transition=transition, policy=policy, semantic_match=match,
        k=3, max_inflight_actors=13, assert_invariants=True)
    elapsed = asyncio.get_event_loop().time() - t0

    base_states, base_actions = await _sequential_baseline(
        0, T, actor=actor, transition=transition, policy=policy)
    base_lower = T * ALPHA

    _check("trajectory matches baseline", states == base_states,
           f"got {states}, want {base_states}")
    _check("actions match baseline", actions == base_actions)
    _check("hits+cascaded == T", stats.hits + stats.cascaded_hits == T,
           f"h={stats.hits} c={stats.cascaded_hits}")
    _check("no misses", stats.misses == 0)
    _check("speedup over sequential",
           elapsed < 0.7 * base_lower,
           f"{elapsed:.3f}s vs {base_lower:.3f}s sequential")
    _check("tree depth exceeded 1",
           stats.max_tree_depth >= 2,
           f"max_tree_depth={stats.max_tree_depth} — eager recursion "
           f"should expand beyond immediate children")


async def _scenario_all_miss() -> None:
    print("\n[scenario] all_miss — Speculator always wrong")
    ALPHA, BETA = 0.05, 0.02
    T = 6

    async def actor(s, h, q):
        await asyncio.sleep(ALPHA)
        return 2

    async def speculator(s, h, q):
        await asyncio.sleep(BETA)
        return [0, 1, 3]

    transition = lambda s, a: s + a
    policy = lambda s: ("inc", s)
    match = lambda a_hat, a, d: a_hat == a

    states, actions, stats, _ = await run_depth_speculative(
        0, T, actor=actor, speculator=speculator,
        transition=transition, policy=policy, semantic_match=match,
        k=3, max_inflight_actors=13, assert_invariants=True)

    base_states, base_actions = await _sequential_baseline(
        0, T, actor=actor, transition=transition, policy=policy)

    _check("trajectory matches baseline", states == base_states)
    _check("actions match baseline", actions == base_actions)
    _check("misses == T", stats.misses == T, f"misses={stats.misses}")
    _check("no hits", stats.hits == 0 and stats.cascaded_hits == 0)


async def _scenario_mixed() -> None:
    print("\n[scenario] mixed — always at least one match in predictions")
    ALPHA, BETA = 0.05, 0.02
    T = 8

    async def actor(s, h, q):
        await asyncio.sleep(ALPHA)
        return 1 if s % 2 == 0 else 2

    async def speculator(s, h, q):
        await asyncio.sleep(BETA)
        return [1, 2, 3]

    transition = lambda s, a: s + a
    policy = lambda s: ("inc", s)
    match = lambda a_hat, a, d: a_hat == a

    states, actions, stats, _ = await run_depth_speculative(
        0, T, actor=actor, speculator=speculator,
        transition=transition, policy=policy, semantic_match=match,
        k=3, max_inflight_actors=13, assert_invariants=True)

    base_states, base_actions = await _sequential_baseline(
        0, T, actor=actor, transition=transition, policy=policy)

    _check("trajectory matches baseline", states == base_states)
    _check("actions match baseline", actions == base_actions)
    _check("all T confirmed",
           stats.hits + stats.cascaded_hits + stats.misses == T,
           f"h={stats.hits} c={stats.cascaded_hits} m={stats.misses}")


async def _scenario_budget_squeeze() -> None:
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

    # max_inflight_actors=1 means only the ground-truth root Actor fits;
    # every speculative Actor is DEFERRED.
    states, actions, stats, _ = await run_depth_speculative(
        0, T, actor=actor, speculator=speculator,
        transition=transition, policy=policy, semantic_match=match,
        k=2, max_inflight_actors=1, assert_invariants=True)

    base_states, _ = await _sequential_baseline(
        0, T, actor=actor, transition=transition, policy=policy)

    _check("trajectory matches baseline under tight budget",
           states == base_states)
    _check("speculative Actors were deferred",
           stats.actor_deferred > 0,
           f"deferred={stats.actor_deferred}")


async def _scenario_dedup() -> None:
    print("\n[scenario] dedup — speculator emits duplicates")
    ALPHA, BETA = 0.04, 0.01
    T = 4

    async def actor(s, h, q):
        await asyncio.sleep(ALPHA)
        return 1

    async def speculator(s, h, q):
        await asyncio.sleep(BETA)
        return [1, 1, 1]

    transition = lambda s, a: s + a
    policy = lambda s: ("inc", s)
    match = lambda a_hat, a, d: a_hat == a

    states, _, stats, _ = await run_depth_speculative(
        0, T, actor=actor, speculator=speculator,
        transition=transition, policy=policy, semantic_match=match,
        k=3, max_inflight_actors=13, assert_invariants=True)

    base_states, _ = await _sequential_baseline(
        0, T, actor=actor, transition=transition, policy=policy)

    _check("trajectory matches baseline under dedup",
           states == base_states)
    _check("dedup reuse occurred", stats.actor_dedup_reuse > 0,
           f"reuse={stats.actor_dedup_reuse}")


async def _scenario_actor_error() -> None:
    print("\n[scenario] actor_error — speculative Actor raises at depth≥1")
    ALPHA, BETA = 0.04, 0.01
    T = 4

    call_count = [0]

    async def actor(s, h, q):
        await asyncio.sleep(ALPHA)
        call_count[0] += 1
        if s >= 5 and call_count[0] > 3:
            raise RuntimeError("simulated upstream 500")
        return 1

    async def speculator(s, h, q):
        await asyncio.sleep(BETA)
        return [1, 2]

    transition = lambda s, a: s + a
    policy = lambda s: ("inc", s)
    match = lambda a_hat, a, d: a_hat == a

    states, _, stats, _ = await run_depth_speculative(
        0, T, actor=actor, speculator=speculator,
        transition=transition, policy=policy, semantic_match=match,
        k=2, max_inflight_actors=13, assert_invariants=True)

    base_states, _ = await _sequential_baseline(
        0, T, actor=actor, transition=transition, policy=policy)

    _check("trajectory matches baseline under Actor errors",
           states == base_states, f"got {states}, want {base_states}")


async def _scenario_deep_tree() -> None:
    """Speedup from deep speculation: Actor slow, Speculator fast, oracle."""
    print("\n[scenario] deep_tree — oracle Speculator, tall tree fits in budget")
    ALPHA, BETA = 0.20, 0.02
    T = 6

    async def actor(s, h, q):
        await asyncio.sleep(ALPHA)
        return 1

    async def speculator(s, h, q):
        await asyncio.sleep(BETA)
        return [1, 2, 3]

    transition = lambda s, a: s + a
    policy = lambda s: ("inc", s)
    match = lambda a_hat, a, d: a_hat == a

    states, actions, stats, step_info = await run_depth_speculative(
        0, T, actor=actor, speculator=speculator,
        transition=transition, policy=policy, semantic_match=match,
        k=3, max_inflight_actors=13, assert_invariants=True,
        record_steps=True)

    base_states, _ = await _sequential_baseline(
        0, T, actor=actor, transition=transition, policy=policy)

    _check("trajectory matches baseline", states == base_states)
    # Note: true cascade_run_length > 1 requires a deeper Actor to finish
    # simultaneously with (or before) its parent, which doesn't happen
    # under deterministic staggered sleeps. Cascades are still exercised
    # here via per-step HITs — speedup comes from depth pipelining, not
    # from multiple HITs in a single try_cascade call.
    _check("tree grew beyond depth 1",
           stats.max_tree_depth >= 2,
           f"max_tree_depth={stats.max_tree_depth}")
    _check("step_info populated", len(step_info) == T,
           f"got {len(step_info)} steps")


async def main() -> int:
    scenarios = [
        _scenario_all_hits,
        _scenario_all_miss,
        _scenario_mixed,
        _scenario_budget_squeeze,
        _scenario_dedup,
        _scenario_actor_error,
        _scenario_deep_tree,
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
