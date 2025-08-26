def enrich_with_ontology(df, mapping, mode="moderato"):
    df = df.copy()
    for category in sorted(set(mapping.values())):
        if category not in df.columns:
            df[category] = 0
    for feat, category in mapping.items():
        if feat in df.columns:
            if df[feat].dropna().isin([0,1]).all():
                df[category] = (df[category] | (df[feat] == 1)).astype(int)
            else:
                df[category] = (df[category] | (df[feat] > df[feat].median())).astype(int)

    if "chol" in df.columns:
        df["high_chol"] = (df["chol"] > 240).astype(int)
        df["very_high_chol"] = (df["chol"] > 280).astype(int)

    if "trestbps" in df.columns:
        df["high_bp"] = (df["trestbps"] >= 140).astype(int)

    if mode == "avanzato":
        if "thalach" in df.columns:
            df["tachycardia"] = (df["thalach"] > 100).astype(int)
            df["bradycardia"] = (df["thalach"] < 60).astype(int)
        if "fbs" in df.columns:
            df["fbs_high"] = (df["fbs"] > 1).astype(int)

    return df
