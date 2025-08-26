from utils import normalize_label, search_concept

manual_mapping = {
    "trestbps": "blood pressure measurement",
    "chol": "blood cholesterol measurement",
    "thalach": "heart rate measurement",
    "oldpeak": "st segment depression measurement",
    "restecg": "electrocardiogram",
    "fbs": "blood glucose measurement",
    "ca": "cardiac catheterization finding",
    "cp": "chest pain symptom",
    "thal": "thalassemia"
}

CATEGORY_LABELS = {
    "blood_measurement": "blood measurement",
    "blood_pressure_measurement": "blood pressure measurement",
    "cardiac_function_measurement": "heart rate measurement",
    "electrical_activity": "electrocardiogram",
    "general_symptom": "general symptom",
    "cardiovascular_symptom": "cardiovascular system symptom",
    "metabolic_disorder": "metabolic disease"
}

def build_symptom_mapping(symptom_cols, ontologies, ALL_LABELS, ALL_CLASS_BY_LABEL):
    symptom_to_category = {}
    for col in symptom_cols:
        concept = search_concept(col, ALL_LABELS, ALL_CLASS_BY_LABEL)
        if concept:
            symptom_to_category[col] = normalize_label(getattr(concept, "label", [col])[0])
        elif col in manual_mapping:
            symptom_to_category[col] = normalize_label(manual_mapping[col])

    mapped_cols = sorted(symptom_to_category.keys())
    unmapped_cols = sorted(set(symptom_cols) - set(mapped_cols))

    print(f"[Mapping] Feature mappate: {len(mapped_cols)} / {len(symptom_cols)}")
    print(f"[Mapping] Esempi mappati: {mapped_cols[:10]}")
    print(f"[Mapping] Esempi NON mappati: {unmapped_cols[:10]}")
    return symptom_to_category, mapped_cols, unmapped_cols

def build_category_classes(ontologies):
    CATEGORY_CLASSES = {}
    for feat, lab in CATEGORY_LABELS.items():
        for onto in ontologies.values():
            cls = onto.search_one(label=lab)
            if cls:
                CATEGORY_CLASSES[feat] = cls
                break
    CATEGORY_CLASSES = {k: v for k, v in CATEGORY_CLASSES.items() if v is not None}
    print("[Categorie ontologiche usate]:", list(CATEGORY_CLASSES.keys()))
    return CATEGORY_CLASSES

def get_ancestors(cls, max_depth=6):
    out, frontier, seen = [], [cls], {cls}
    depth = 0
    while frontier and depth < max_depth:
        new_frontier = []
        for c in frontier:
            for sup in getattr(c, "is_a", []):
                if sup not in seen:
                    seen.add(sup)
                    out.append(sup)
                    new_frontier.append(sup)
        frontier = new_frontier
        depth += 1
    return out

def build_feature_to_category(symptom_to_category, ontologies, CATEGORY_CLASSES):
    feature_to_category = {}
    for feat, concept_label in symptom_to_category.items():
        for onto in ontologies.values():
            concept = onto.search_one(label=concept_label)
            if concept:
                ancestors = get_ancestors(concept, max_depth=6)
                for cat, cat_cls in CATEGORY_CLASSES.items():
                    if cat_cls in ancestors or concept == cat_cls:
                        feature_to_category[feat] = cat
                        break
    print(f"[Mapping a categorie] Copertura: {len(feature_to_category)} feature")
    return feature_to_category
