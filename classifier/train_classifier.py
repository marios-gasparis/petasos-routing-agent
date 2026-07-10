"""Train the Stage-2 learned router (Phase 6) -- OFFLINE, training-only.

Produces `classifier/router_clf.joblib`, a scikit-learn artifact baked into the
scored image and loaded at runtime by `src/classifier_router.py`. Two heads:

  (a) category      -- predicts one of the 8 task categories from the prompt.
  (b) success       -- predicts P(local model answers correctly). Trained ONLY
                       if a harness pass/fail labels file is supplied via
                       --labels; skipped gracefully otherwise (see below).

Runtime-dependency decision (the primary Phase 6 risk = image budget):
  The PHASE_6 spec names all-MiniLM-L6-v2 (sentence-transformers) as the
  embedder, but that pulls in torch (+transformers): ~1-2 GB added to the image
  and, worse, a 2-5 s torch import plus a 2-3 s MiniLM load -- which blows the
  "<2 s artifact load" and eats into the "<60 s cold start" hard constraints.
  So the SHIPPED backend is a pure scikit-learn TF-IDF (word 1-2 grams +
  char_wb 3-5 grams) + logistic regression. It adds only scikit-learn/scipy to
  the runtime (~50-60 MB), loads in <0.2 s, needs no network to train, and --
  because category routing here is a strongly lexical problem -- generalizes to
  unseen paraphrases at least as well as the (already strong) heuristic. See
  docs/phase6-classifier.md for the full rationale and the measured numbers.

Data source:
  Base (prompt -> category) pairs come from eval/gen_testset.BASE (the single
  source of truth for the testset). We augment each base case with neutral
  paraphrase wrappers (answer-preserving) to mimic unseen prompt variants, then
  do a GROUPED split by base case AND by wrapper set, so validation prompts are
  genuinely unseen content in unseen phrasings -- a strict generalization test.
  The FINAL shipped model is retrained on ALL base cases x ALL wrappers.

Run:
  python classifier/train_classifier.py                 # category head only
  python classifier/train_classifier.py --labels eval/labels.json
  python classifier/train_classifier.py --out classifier/router_clf.joblib
"""

import argparse
import json
import os
import sys

# Flat imports from src/ (the shipped inference path) and eval/ (training-only
# data source), mirroring tests/ and eval/harness.py conventions.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, os.path.join(_HERE, "..", "eval"))

DEFAULT_OUT = os.path.join(_HERE, "router_clf.joblib")

# --- paraphrase augmentation --------------------------------------------------
# Train and validation wrapper sets are DISJOINT so held-out accuracy is not
# inflated by memorized phrasings. Wrappers are answer-preserving; sentiment and
# NER get bespoke neutral wrappers (a generic "I need help with this:" prefix
# would not change their label, but keeping them task-flavored matches how the
# scored prompts actually read).
_GENERIC_TRAIN = [
    "{p}",
    "Please answer the following. {p}",
    "{p} Answer concisely.",
    "Question: {p}",
    "I need help with this: {p}",
    "Can you help with this? {p}",
    "Here is a task. {p}",
    "{p} Thanks in advance.",
]
_GENERIC_VAL = [
    "Kindly respond to this. {p}",
    "Task for you: {p}",
    "{p} Please give your response.",
]
_SENTIMENT_TRAIN = [
    "{p}",
    "Classify this. {p}",
    "{p} Give the label only.",
    "Sentiment task: {p}",
    "Determine the sentiment. {p}",
]
_SENTIMENT_VAL = [
    "Label this text. {p}",
    "{p} What is the sentiment?",
]
_NER_TRAIN = [
    "{p}",
    "{p} Return a JSON list.",
    "Entity extraction task. {p}",
    "{p} List them.",
    "Find the entities. {p}",
]
_NER_VAL = [
    "Extraction task: {p}",
    "{p} Provide the entities.",
]


def _wrapper_sets(category):
    if category == "sentiment":
        return _SENTIMENT_TRAIN, _SENTIMENT_VAL
    if category == "ner":
        return _NER_TRAIN, _NER_VAL
    return _GENERIC_TRAIN, _GENERIC_VAL


def _load_base_cases():
    """Return {category: [base_prompt, ...]} from eval/gen_testset.BASE."""
    try:
        from gen_testset import BASE
    except Exception as exc:  # pragma: no cover - only if eval/ is unavailable
        raise SystemExit(
            "cannot import eval/gen_testset.BASE (needed for training data): "
            f"{exc}"
        )
    return {cat: [p for (p, _ref) in pairs] for cat, pairs in BASE.items()}


def _build_dataset(base_cases, wrapper_selector):
    """Expand base cases into (prompt, category, group) triples. `group` is the
    base-case identity so a grouped split never puts a base case in both train
    and validation. `wrapper_selector(category) -> list[str]` picks the wrappers
    to apply."""
    prompts, categories, groups = [], [], []
    for category, bases in base_cases.items():
        wrappers = wrapper_selector(category)
        for base_idx, base in enumerate(bases):
            group = f"{category}:{base_idx}"
            for wrapper in wrappers:
                prompts.append(wrapper.format(p=base))
                categories.append(category)
                groups.append(group)
    return prompts, categories, groups


# --- model construction -------------------------------------------------------

def build_feature_pipeline():
    """Word (1-2 gram) + char_wb (3-5 gram) TF-IDF union. char_wb n-grams give
    robustness to word-level variation in unseen paraphrases; word n-grams
    capture the task-signalling keywords. Kept lightweight so the artifact loads
    in well under the 2 s budget."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion

    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                 min_df=1, sublinear_tf=True, lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                 min_df=1, sublinear_tf=True, lowercase=True)),
    ])


def build_category_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    return Pipeline([
        ("features", build_feature_pipeline()),
        ("clf", LogisticRegression(max_iter=2000, C=8.0,
                                   class_weight="balanced")),
    ])


def build_success_model():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    return Pipeline([
        ("features", build_feature_pipeline()),
        ("clf", LogisticRegression(max_iter=2000, C=4.0,
                                   class_weight="balanced")),
    ])


# --- category head: train + held-out validation -------------------------------

def _heuristic_accuracy(prompts, categories):
    """Category accuracy of the Stage-1 heuristic on the same held-out set, for
    an apples-to-apples comparison against the acceptance criterion."""
    from router import heuristic_route
    correct = 0
    for prompt, true_cat in zip(prompts, categories):
        pred, _ = heuristic_route(prompt)
        correct += (pred == true_cat)
    return correct / len(prompts) if prompts else 0.0


def train_and_validate_category(base_cases, seed=42):
    """Train on a subset of base cases (train wrappers) and validate on the
    held-out base cases (val wrappers). Returns (val_report, min_confidence).

    The report compares the classifier vs the heuristic on held-out variants
    (the PHASE_6 acceptance criterion) and picks the min-confidence gate that
    maximizes the COMBINED (classifier-or-heuristic) held-out accuracy."""
    import numpy as np
    from sklearn.model_selection import GroupShuffleSplit

    train_prompts, train_cats, groups = _build_dataset(
        base_cases, lambda c: _wrapper_sets(c)[0])
    # Split base-case GROUPS: ~30% of base cases held out entirely.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    (tr_idx, _va_idx), = gss.split(train_prompts, train_cats, groups)
    tr_groups = {groups[i] for i in tr_idx}

    fit_prompts = [train_prompts[i] for i in tr_idx]
    fit_cats = [train_cats[i] for i in tr_idx]

    # Validation prompts: held-out base cases, rendered with the DISJOINT
    # validation wrappers (unseen content AND unseen phrasing).
    val_base = {
        cat: [b for j, b in enumerate(bases) if f"{cat}:{j}" not in tr_groups]
        for cat, bases in base_cases.items()
    }
    va_prompts, va_cats, _ = _build_dataset(val_base, lambda c: _wrapper_sets(c)[1])

    model = build_category_model()
    model.fit(fit_prompts, fit_cats)

    proba = model.predict_proba(va_prompts)
    classes = list(model.named_steps["clf"].classes_)
    pred_idx = proba.argmax(axis=1)
    preds = [classes[i] for i in pred_idx]
    confs = proba.max(axis=1)

    clf_correct = sum(p == t for p, t in zip(preds, va_cats))
    clf_acc = clf_correct / len(va_cats) if va_cats else 0.0
    heur_acc = _heuristic_accuracy(va_prompts, va_cats)

    # Pick the confidence gate that maximizes combined accuracy: use the
    # classifier when conf >= gate, else the heuristic.
    from router import heuristic_route
    heur_preds = [heuristic_route(p)[0] for p in va_prompts]
    best_gate, best_combined = 0.0, -1.0
    for gate in np.linspace(0.0, 0.9, 19):
        combined = 0
        for cp, cf, hp, t in zip(preds, confs, heur_preds, va_cats):
            chosen = cp if cf >= gate else hp
            combined += (chosen == t)
        acc = combined / len(va_cats) if va_cats else 0.0
        if acc > best_combined:
            best_combined, best_gate = acc, float(gate)

    report = {
        "val_size": len(va_cats),
        "train_size": len(fit_cats),
        "classifier_acc": clf_acc,
        "heuristic_acc": heur_acc,
        "combined_acc": best_combined,
        "chosen_min_confidence": round(best_gate, 3),
        "mean_confidence": float(confs.mean()) if len(confs) else 0.0,
    }
    return report, round(best_gate, 3)


def train_final_category(base_cases):
    """Train the shipped category model on ALL base cases x ALL (train+val)
    wrappers -- maximum coverage for the artifact that goes in the image."""
    def all_wrappers(category):
        tr, va = _wrapper_sets(category)
        return tr + va

    prompts, cats, _ = _build_dataset(base_cases, all_wrappers)
    model = build_category_model()
    model.fit(prompts, cats)
    return model, sorted(set(cats))


# --- success head (blocked on Phase 5 harness labels) -------------------------

def train_success_head(labels_path):
    """Train P(local success) from a harness labels file, if present.

    The labels file is produced OFFLINE by `python -m eval.harness --dump-labels
    <path>` (added in Phase 6): a JSON list of records each with at least
    {"prompt": str, "local_pass": bool|null}. Records with local_pass=null
    (ungradeable judge-only tasks with no judge available) are skipped -- we do
    NOT fabricate labels. Returns (model, report) or (None, reason)."""
    if not labels_path:
        return None, "no --labels supplied (success head skipped; needs a " \
                     "harness run with the judge -- see docs/next-steps.md)"
    if not os.path.exists(labels_path):
        return None, f"labels file not found: {labels_path} (run harness " \
                     f"--dump-labels first)"
    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except Exception as exc:
        return None, f"could not read labels file {labels_path}: {exc}"

    prompts, labels = [], []
    for rec in records:
        prompt = rec.get("prompt")
        passed = rec.get("local_pass")
        if prompt and passed is not None:
            prompts.append(prompt)
            labels.append(1 if passed else 0)

    n = len(labels)
    n_pos = sum(labels)
    if n < 20 or n_pos == 0 or n_pos == n:
        return None, (f"insufficient/degenerate success labels (n={n}, "
                      f"pass={n_pos}); need both classes and >=20 examples")

    model = build_success_model()
    model.fit(prompts, labels)
    # In-sample accuracy only (a proper CV needs more data than a single
    # harness run yields); reported as a sanity check, not a generalization
    # claim. Threshold tuning happens against the live harness sweep.
    from sklearn.metrics import accuracy_score
    train_acc = accuracy_score(labels, model.predict(prompts))
    report = {"n": n, "n_pass": n_pos, "train_acc": float(train_acc)}
    return model, report


# --- persistence --------------------------------------------------------------

def save_artifact(out_path, category_model, categories, min_confidence,
                  success_model, success_report, val_report, sklearn_version):
    import joblib

    artifact = {
        "version": 1,
        "backend": "tfidf-logreg",
        "categories": categories,
        "category_model": category_model,
        # Below the classifier's own confidence gate, routing defers to the
        # Stage-1 heuristic (src/router.py). Tuned on held-out variants.
        "min_category_confidence": min_confidence,
        "success_model": success_model,           # None until Phase 5 labels exist
        # Escalate when P(local success) < this threshold. Placeholder until the
        # live harness sweep tunes it (Phase 6 tuning step). Only consulted when
        # success_model is not None.
        "success_threshold": 0.5,
        "sklearn_version": sklearn_version,
        "metadata": {
            "category_validation": val_report,
            "success_report": success_report,
        },
    }
    tmp = out_path + ".tmp"
    joblib.dump(artifact, tmp)
    os.replace(tmp, out_path)
    return artifact


def main():
    parser = argparse.ArgumentParser(description="Train the Stage-2 router.")
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--labels", default=None,
                        help="harness pass/fail labels JSON (enables the "
                             "success head; skipped if absent)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import sklearn

    base_cases = _load_base_cases()

    print("== Category head: held-out validation ==")
    val_report, min_conf = train_and_validate_category(base_cases, seed=args.seed)
    for k, v in val_report.items():
        print(f"  {k:22s} {v}")
    verdict = ("BEATS" if val_report["classifier_acc"] > val_report["heuristic_acc"]
               else "MATCHES" if val_report["classifier_acc"] == val_report["heuristic_acc"]
               else "TRAILS")
    print(f"  -> classifier {verdict} heuristic on held-out variants "
          f"(combined router = {val_report['combined_acc']:.3f})")

    print("\n== Category head: final fit on all data ==")
    category_model, categories = train_final_category(base_cases)
    print(f"  trained on {len(categories)} categories: {categories}")

    print("\n== Success head ==")
    success_model, success_report = train_success_head(args.labels)
    if success_model is None:
        print(f"  SKIPPED: {success_report}")
        success_report = {"skipped": success_report}
    else:
        print(f"  trained: {success_report}")

    artifact = save_artifact(
        args.out, category_model, categories, min_conf,
        success_model, success_report, val_report, sklearn.__version__)
    size_kb = os.path.getsize(args.out) / 1024
    print(f"\nWrote {args.out} ({size_kb:.0f} KB, sklearn "
          f"{artifact['sklearn_version']}, min_confidence={min_conf})")


if __name__ == "__main__":
    main()
