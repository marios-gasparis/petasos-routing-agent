"""Offline evaluation harness for the routing agent.

Runs the REAL agent pipeline (router -> local_chat -> should_escalate ->
remote_chat) over the generated testset and reports:
  - per-category and overall accuracy (the accuracy-gate proxy),
  - the agent's scored Fireworks token total (the metric being minimized),
  - a threshold sweep over escalation policies so the accuracy-vs-tokens knee
    can be chosen,
  - a "false-trust" gap analysis surfacing the gate's remaining weaknesses
    (factual/logic/summarization/code-debug have no verifiable check; math is
    covered by dual-answer agreement since 2026-07-10), rather than hiding
    them.

Two token streams are tracked SEPARATELY and never mixed:
  - agent scored tokens: src/remote_model.get_total_tokens_used() (reset per
    run) -- this is what the contest scores.
  - judge offline tokens: eval/judge.get_judge_tokens_used() -- NOT scored,
    reported only for transparency.

Threshold-sweep design (why policies, not a numeric knob):
  src/confidence.should_escalate has no single numeric threshold today -- it is
  a union of boolean rules (degenerate answer, verifiable-check failure,
  hard+hedge). There is no continuous confidence score to sweep until Phase 6
  adds the classifier's predicted-local-success probability. So rather than
  invasively bolt a fake threshold onto the shipped gate, the harness SIMULATES
  a set of nested escalation policies, from all-local to all-remote, with the
  shipped gate as one point on the curve. This keeps src/confidence.py's
  shipped behavior untouched (as instructed) while still producing an
  accuracy-vs-tokens curve. When Phase 6 lands a continuous predicted-success
  score, swap these discrete policies for a numeric threshold sweep.

Runs offline without credentials: local_chat / remote_chat / judge all degrade
to a fallback ("N/A" / SKIP) instead of raising, so this prints a valid (if
low-accuracy) table with no server and no Fireworks keys.

Usage:
  python -m eval.harness                 # single run: all_local + shipped
  python -m eval.harness --sweep         # full policy sweep (remote for all)
  python -m eval.harness --limit 8       # subset for a quick smoke test
  python -m eval.harness --self-test     # offline: stub local/remote=reference
  python -m eval.harness --no-judge      # force metric-only grading
"""

import argparse
import json
import logging
import os
import sys

# Flat imports from src/ and sibling eval/ modules (project convention).
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import remote_model  # noqa: E402  (module import so we can reset its counter)
from confidence import should_escalate  # noqa: E402
from local_model import local_chat  # noqa: E402
from remote_model import pick_remote_model, remote_chat  # noqa: E402
from router import route_task  # noqa: E402

import judge as judge_mod  # noqa: E402
from metrics import grade_metric, grade_strategy  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("harness")

CATEGORIES = [
    "factual", "math", "sentiment", "summarization",
    "ner", "code-debug", "logic", "code-gen",
]
_VERIFIABLE = {"math", "ner", "sentiment", "code-gen"}
_NO_CHECK = {"factual", "summarization", "code-debug", "logic"}

DEFAULT_TESTSET = os.path.join(os.path.dirname(__file__), "testset", "testset.json")


# --- grading ------------------------------------------------------------------

def grade_answer(category, prompt, reference, answer, use_judge):
    """Grade one answer -> (passed, via, detail). `passed` is True/False, or
    None when the category is judge-only and no judge is available (counted as
    'ungradeable', excluded from accuracy but reported as reduced coverage)."""
    strategy = grade_strategy(category)
    metric_pass, _score, detail = grade_metric(category, answer, reference)

    if strategy in ("judge_pref", "judge_only") and use_judge:
        verdict, reason = judge_mod.judge(prompt, reference, answer)
        if verdict in ("PASS", "FAIL"):
            return (verdict == "PASS"), "judge", reason
        # SKIP -> fall through to the metric fallback below.

    if metric_pass is not None:
        return metric_pass, "metric", detail
    return None, "ungradeable", "no judge available for judge-only category"


# --- escalation policies (the sweep axis) -------------------------------------
# Each policy maps a task record -> bool (escalate?). The sets are nested:
# all_local subset of verifiable_only subset of shipped subset of
# shipped_plus_hard subset of all_remote, so escalating more only ever swaps in
# more remote answers (monotonic tokens).

def _p_all_local(rec):
    return False


def _p_verifiable_only(rec):
    # difficulty forced to "easy" disables the shipped gate's hard-difficulty
    # path on no-check categories, leaving the degenerate + verifiable-check +
    # hedge-language triggers. Uses the ROUTED category, because that is the
    # check the shipped agent applies. verify_answer was computed once in
    # run_pipeline so policy re-scoring stays deterministic (no live calls).
    return should_escalate(rec["routed_category"], "easy", rec["local_answer"],
                           rec.get("verify_answer"))


def _p_shipped(rec):
    return should_escalate(rec["routed_category"], rec["difficulty"],
                           rec["local_answer"], rec.get("verify_answer"))


def _p_shipped_plus_hard(rec):
    return _p_shipped(rec) or rec["difficulty"] == "hard"


def _p_all_remote(rec):
    return True


POLICIES = [
    ("all_local", _p_all_local),
    ("verifiable_only", _p_verifiable_only),
    ("shipped", _p_shipped),
    ("shipped_plus_hard", _p_shipped_plus_hard),
    ("all_remote", _p_all_remote),
]
POLICY_MAP = dict(POLICIES)


# --- running the pipeline over the testset ------------------------------------

def _load_testset(path, limit, categories=None):
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    if categories:
        items = [it for it in items if it.get("category") in categories]
    if limit:
        items = items[:limit]
    return items


def _parse_categories(spec):
    """Parse a --categories spec ("code-gen,logic") into a validated set, or
    None when no filter was given. Unknown names are dropped with a warning so
    a typo shrinks the run visibly instead of silently running nothing."""
    if not spec:
        return None
    wanted = {c.strip() for c in spec.split(",") if c.strip()}
    unknown = wanted - set(CATEGORIES)
    if unknown:
        logger.warning("ignoring unknown categories: %s (valid: %s)",
                       sorted(unknown), ", ".join(CATEGORIES))
    return (wanted & set(CATEGORIES)) or None


def run_pipeline(items, use_judge, need_remote_all, self_test=False):
    """Execute local (+ remote where needed) over every task and grade both.

    need_remote_all=True computes a remote answer for EVERY task (required for
    the full policy sweep, including all_remote). Otherwise remote is computed
    only for tasks the shipped gate escalates (the default, token-economical
    single run).

    self_test replaces the model calls with reference-echoing stubs so the
    harness plumbing / metrics / accounting can be verified fully offline with
    no server and no credentials.
    """
    records = []
    for it in items:
        true_category = it["category"]
        # The agent ROUTES the prompt itself; a misroute is a real failure mode
        # (wrong prompt/caps + wrong escalation check), so the pipeline uses the
        # routed category. Grading, however, uses the KNOWN true category, so a
        # code task misrouted to "factual" is still graded as code.
        cat_obj, routed_category, difficulty = route_task(it["prompt"])
        reference = it["reference"]

        if self_test:
            local_answer = reference
        else:
            local_answer, _ = local_chat(
                cat_obj.system_prompt, it["prompt"],
                max_tokens=cat_obj.max_tokens, stop=cat_obj.stop,
            )

        # Math dual-answer verification (mirrors src/main.py): one extra LOCAL
        # call whose final number must agree with the primary answer's.
        # Computed once here so score_policy can re-evaluate policies without
        # live calls. self_test echoes the reference (always agrees).
        verify_answer = None
        if routed_category == "math":
            if self_test:
                verify_answer = reference
            elif (local_answer or "").strip().lower() not in ("", "n/a", "na"):
                verify_answer, _ = local_chat(
                    cat_obj.system_prompt,
                    "Recompute carefully step by step, then give the final "
                    "numeric answer on its own line.\n\n" + it["prompt"],
                    max_tokens=cat_obj.max_tokens, stop=cat_obj.stop,
                )

        rec = {
            "task_id": it["task_id"],
            "true_category": true_category,
            "routed_category": routed_category,
            "routed_correct": routed_category == true_category,
            "difficulty": difficulty,
            "prompt": it["prompt"],
            "reference": reference,
            "local_answer": local_answer,
            "verify_answer": verify_answer,
            "remote_answer": None,
            "remote_tokens": 0,
        }

        lp, lvia, ldetail = grade_answer(true_category, it["prompt"], reference,
                                         local_answer, use_judge)
        rec["local_pass"], rec["local_via"], rec["local_detail"] = lp, lvia, ldetail

        rec["shipped_escalate"] = _p_shipped(rec)
        do_remote = need_remote_all or rec["shipped_escalate"]

        if do_remote:
            if self_test:
                remote_answer, tokens = reference, 0
            else:
                model = pick_remote_model(routed_category)
                remote_answer, usage = remote_chat(
                    model, cat_obj.system_prompt, it["prompt"],
                    max_tokens=cat_obj.max_tokens, stop=cat_obj.stop,
                )
                tokens = getattr(usage, "total_tokens", 0) or 0
            rec["remote_answer"] = remote_answer
            rec["remote_tokens"] = tokens
            rp, rvia, rdetail = grade_answer(true_category, it["prompt"],
                                             reference, remote_answer, use_judge)
            rec["remote_pass"], rec["remote_via"] = rp, rvia
        else:
            rec["remote_pass"], rec["remote_via"] = None, None

        records.append(rec)
        logger.info(
            "task_id=%s true=%s routed=%s diff=%s local_pass=%s escalate=%s",
            rec["task_id"], true_category, routed_category, difficulty, lp,
            rec["shipped_escalate"],
        )
    return records


# --- scoring a policy over the records ----------------------------------------

def score_policy(records, policy_fn):
    """Compute (accuracy, tokens, gradeable, total, per_category) for a policy.

    accuracy is over gradeable tasks only (judge-only tasks with no judge are
    excluded and reported as reduced coverage). tokens is the sum of remote
    tokens over the tasks this policy escalates -- the scored metric proxy."""
    passed = 0
    gradeable = 0
    tokens = 0
    escalated = 0
    per_cat = {c: {"passed": 0, "gradeable": 0, "total": 0} for c in CATEGORIES}

    for rec in records:
        cat = rec["true_category"]
        pc = per_cat.setdefault(cat, {"passed": 0, "gradeable": 0, "total": 0})
        pc["total"] += 1

        escalate = policy_fn(rec)
        if escalate:
            escalated += 1
            tokens += rec.get("remote_tokens", 0)
            verdict = rec.get("remote_pass")
        else:
            verdict = rec.get("local_pass")

        if verdict is None:
            continue  # ungradeable -> excluded from accuracy
        gradeable += 1
        pc["gradeable"] += 1
        if verdict:
            passed += 1
            pc["passed"] += 1

    accuracy = passed / gradeable if gradeable else 0.0
    return {
        "accuracy": accuracy,
        "passed": passed,
        "gradeable": gradeable,
        "total": len(records),
        "tokens": tokens,
        "escalated": escalated,
        "per_category": per_cat,
    }


def dump_labels(records, path):
    """Persist per-task pass/fail labels for Phase 6's predicted-local-success
    head (classifier/train_classifier.py --labels). Written atomically. These
    are REAL labels from this run's local model + judge -- never fabricated;
    ungradeable tasks (local_pass is None) are written through as null and the
    trainer skips them. Producing a useful labels file therefore requires ONE
    harness run with the judge available (i.e. live creds), which can be
    captured from a sweep you are already paying for rather than a second run.
    """
    labels = [
        {
            "task_id": r["task_id"],
            "category": r["true_category"],
            "routed_category": r["routed_category"],
            "difficulty": r["difficulty"],
            "prompt": r["prompt"],
            "local_answer": r["local_answer"],
            "local_pass": r["local_pass"],
            "local_via": r.get("local_via"),
            "shipped_escalate": r["shipped_escalate"],
            "remote_pass": r.get("remote_pass"),
        }
        for r in records
    ]
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    graded = sum(1 for r in records if r["local_pass"] is not None)
    logger.info("wrote %d labels (%d graded, %d ungradeable) to %s",
                len(labels), graded, len(labels) - graded, path)


def gap_analysis(records):
    """Surface the known Phase 4 weaknesses: for the SHIPPED gate, count tasks
    whose local answer shipped (was NOT escalated) yet graded FAIL -- these are
    accuracy leaks the current gate misses. Annotate why per category."""
    rows = []
    for cat in CATEGORIES:
        recs = [r for r in records if r["true_category"] == cat]
        if not recs:
            continue
        false_trust = sum(
            1 for r in recs
            if not r["shipped_escalate"] and r["local_pass"] is False
        )
        gradeable = sum(1 for r in recs if r["local_pass"] is not None)
        if cat == "math":
            note = "dual-answer agreement check (format-only when no verify answer)"
        elif cat in _NO_CHECK:
            note = "NO verifiable check (escalates on hard OR hedge)"
        else:
            note = "verifiable check present"
        rows.append((cat, false_trust, len(recs), gradeable, note))
    return rows


# --- reporting ----------------------------------------------------------------

def _fmt_pct(x):
    return f"{x * 100:5.1f}%"


def print_report(records, policies_to_report, judge_tokens):
    print("\n" + "=" * 74)
    print("EVAL HARNESS REPORT")
    print("=" * 74)

    results = {name: score_policy(records, POLICY_MAP[name])
               for name in policies_to_report}

    # Routing accuracy: how often the heuristic router put the prompt in its
    # true category. A misroute means a wrong prompt/cap and a wrong (or
    # absent) escalation check, so this is a leading indicator of accuracy loss.
    routed_correct = sum(1 for r in records if r["routed_correct"])
    routing_acc = routed_correct / len(records) if records else 0.0
    print(f"\nRouting accuracy (routed == true category): "
          f"{routed_correct}/{len(records)} = {_fmt_pct(routing_acc)}")
    miss = {}
    for r in records:
        if not r["routed_correct"]:
            key = (r["true_category"], r["routed_category"])
            miss[key] = miss.get(key, 0) + 1
    if miss:
        print("  misroutes (true -> routed: count):")
        for (t, rt), n in sorted(miss.items(), key=lambda kv: -kv[1]):
            print(f"    {t:14s} -> {rt:14s} {n}")

    # Per-category breakdown for the shipped policy (or first available).
    focus = "shipped" if "shipped" in results else policies_to_report[0]
    fr = results[focus]
    print(f"\nPer-category accuracy (policy = {focus}):")
    print(f"  {'category':16s} {'passed/gradeable':>16s}  {'coverage':>9s}")
    for cat in CATEGORIES:
        pc = fr["per_category"].get(cat)
        if not pc or pc["total"] == 0:
            continue
        acc = pc["passed"] / pc["gradeable"] if pc["gradeable"] else 0.0
        cov = pc["gradeable"] / pc["total"] if pc["total"] else 0.0
        print(f"  {cat:16s} {pc['passed']:>7d}/{pc['gradeable']:<8d} "
              f"{_fmt_pct(acc)}  {_fmt_pct(cov)}")

    # Threshold sweep table.
    print("\nThreshold sweep (escalation policy -> accuracy vs agent tokens):")
    print(f"  {'policy':18s} {'escalated':>9s} {'agent_tokens':>12s} "
          f"{'accuracy':>9s} {'coverage':>9s}")
    for name in policies_to_report:
        r = results[name]
        cov = r["gradeable"] / r["total"] if r["total"] else 0.0
        print(f"  {name:18s} {r['escalated']:>9d} {r['tokens']:>12d} "
              f"{_fmt_pct(r['accuracy'])} {_fmt_pct(cov)}")

    # Gap analysis.
    print("\nFalse-trust gap analysis (shipped gate: local answer shipped but "
          "graded FAIL):")
    print(f"  {'category':16s} {'false_trust':>11s} {'note'}")
    total_gap = 0
    for cat, ft, n, _grad, note in gap_analysis(records):
        total_gap += ft
        flag = "  <-- LEAK" if ft > 0 else ""
        print(f"  {cat:16s} {ft:>4d}/{n:<4d}   {note}{flag}")
    print(f"  total false-trust (shipped local answers graded wrong): {total_gap}")

    # Token accounting -- two clearly separated streams.
    print("\nToken accounting (streams kept separate):")
    if "shipped" in results:
        print(f"  agent scored Fireworks tokens (shipped gate): "
              f"{results['shipped']['tokens']}")
    print(f"  agent scored Fireworks tokens (this run, remote_model counter): "
          f"{remote_model.get_total_tokens_used()}")
    print(f"  judge OFFLINE tokens (NOT scored): {judge_tokens}")
    print("=" * 74 + "\n")
    return results


# --- CLI ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Offline eval harness.")
    parser.add_argument("--testset", default=DEFAULT_TESTSET)
    parser.add_argument("--limit", type=int, default=None,
                        help="only run the first N tasks (smoke test)")
    parser.add_argument("--categories", default=None, metavar="CAT1,CAT2",
                        help="only run these categories (e.g. "
                             "code-gen,code-debug,logic,math) -- targeted "
                             "re-runs that skip re-spending judge credits on "
                             "already-validated categories")
    parser.add_argument("--sweep", action="store_true",
                        help="run the full policy sweep (remote for every task)")
    parser.add_argument("--no-judge", action="store_true",
                        help="force metric-only grading (skip the LLM judge)")
    parser.add_argument("--self-test", action="store_true",
                        help="offline: stub local/remote answers = reference")
    parser.add_argument("--dump-labels", default=None, metavar="PATH",
                        help="write per-task pass/fail labels to PATH for "
                             "Phase 6's success head (train_classifier.py "
                             "--labels). Needs the judge (live creds) to be "
                             "useful; capture it from a sweep you're already "
                             "running rather than a second paid run.")
    args = parser.parse_args()

    if not os.path.exists(args.testset):
        logger.error("testset not found: %s (run gen_testset.py first)",
                     args.testset)
        sys.exit(1)

    # Reset both token streams so each run's accounting starts at zero.
    remote_model.total_tokens_used = 0
    judge_mod.reset_judge_tokens()

    use_judge = (not args.no_judge) and (not args.self_test) \
        and judge_mod.judge_available()
    if not use_judge:
        logger.warning(
            "judge disabled (no creds / --no-judge / --self-test): open-ended "
            "and judge-only categories fall back to string metrics or are "
            "reported as ungradeable")

    if not args.self_test and not remote_model.resolve_models():
        logger.warning("ALLOWED_MODELS empty: remote escalation will no-op "
                       "(offline dry run) -- accuracy reflects local answers")

    items = _load_testset(args.testset, args.limit,
                          categories=_parse_categories(args.categories))
    logger.info("loaded %d task(s) from %s", len(items), args.testset)
    if not items:
        logger.error("no tasks matched the filter; nothing to run")
        sys.exit(1)

    records = run_pipeline(items, use_judge=use_judge,
                           need_remote_all=args.sweep, self_test=args.self_test)

    if args.dump_labels:
        dump_labels(records, args.dump_labels)
        if not use_judge:
            logger.warning("--dump-labels written WITHOUT the judge: judge-only "
                           "categories (logic/code) are null and excluded by "
                           "the success-head trainer. Re-run with live creds "
                           "for full-coverage labels.")

    if args.sweep:
        policies_to_report = [name for name, _ in POLICIES]
    else:
        policies_to_report = ["all_local", "shipped"]

    print_report(records, policies_to_report, judge_mod.get_judge_tokens_used())


if __name__ == "__main__":
    main()
