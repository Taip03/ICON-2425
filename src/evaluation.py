import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

def cross_validate(models, train_df, train_mod, train_avz):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows_cv = []
    for name, mdl in models.items():
        for mode, Xtr in [("Raw", train_df), ("Mod", train_mod), ("Avz", train_avz)]:
            X = Xtr.drop(columns=["target"]); y = Xtr["target"]
            f1_scores = cross_val_score(mdl, X, y, cv=cv, scoring="f1_macro")
            auc_scores = cross_val_score(mdl, X, y, cv=cv, scoring="roc_auc")
            rows_cv.append({
                "Model": name, "Mode": mode,
                "CV F1 mean": float(f1_scores.mean()), "CV F1 std": float(f1_scores.std()),
                "CV AUC mean": float(auc_scores.mean()), "CV AUC std": float(auc_scores.std())
            })
    return pd.DataFrame(rows_cv).sort_values(["Model","Mode"]).reset_index(drop=True)

def eval_on_test(model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)[:,1] if hasattr(model, "predict_proba") else pred
    return f1_score(yte, pred, average="macro"), roc_auc_score(yte, proba)

def evaluate_models(models, train_df, test_df, train_mod, test_mod, train_avz, test_avz):
    rows = []
    for name, mdl in models.items():
        f1_raw, auc_raw = eval_on_test(mdl, train_df.drop(columns=["target"]), train_df["target"],
                                             test_df.drop(columns=["target"]),  test_df["target"])
        f1_mod, auc_mod = eval_on_test(mdl, train_mod.drop(columns=["target"]), train_mod["target"],
                                             test_mod.drop(columns=["target"]),  test_mod["target"])
        f1_avz, auc_avz = eval_on_test(mdl, train_avz.drop(columns=["target"]), train_avz["target"],
                                             test_avz.drop(columns=["target"]),  test_avz["target"])
        rows.append({
            "Model": name,
            "Raw F1": f1_raw, "Raw AUC": auc_raw,
            "Moderato F1": f1_mod, "Moderato AUC": auc_mod,
            "Avanzato F1": f1_avz, "Avanzato AUC": auc_avz
        })
    return pd.DataFrame(rows).sort_values("Model").reset_index(drop=True)

