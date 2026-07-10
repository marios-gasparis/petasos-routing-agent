"""Stage-2 learned router package (Phase 6).

Ships `router_clf.joblib` (a scikit-learn TF-IDF + logistic-regression model)
into the scored image; `src/classifier_router.py` loads and runs it at runtime.
`train_classifier.py` is training-only (offline) and is never imported by the
scored path."""
