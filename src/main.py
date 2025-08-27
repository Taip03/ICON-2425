from data_loader import load_data
from ontology_loader import load_ontologies
from utils import build_label_dict
from mapping import build_symptom_mapping, build_category_classes, build_feature_to_category
from enrichment import enrich_with_ontology
from models import models
from evaluation import cross_validate, evaluate_models
from export_results import export_results
from sklearn.model_selection import StratifiedKFold

# 1) DATA
df, train_df, test_df, symptom_cols = load_data()

# 2) ONTOLOGIE
ontologies = load_ontologies()
ALL_LABELS, ALL_CLASS_BY_LABEL = build_label_dict(ontologies)

# 3) MAPPING
symptom_to_category, mapped_cols, unmapped_cols = build_symptom_mapping(symptom_cols, ontologies, ALL_LABELS, ALL_CLASS_BY_LABEL)
CATEGORY_CLASSES = build_category_classes(ontologies)
feature_to_category = build_feature_to_category(symptom_to_category, ontologies, CATEGORY_CLASSES)

# 4) ARRICCHIMENTO
train_mod = enrich_with_ontology(train_df, feature_to_category, mode="moderato")
test_mod  = enrich_with_ontology(test_df, feature_to_category, mode="moderato")
train_avz = enrich_with_ontology(train_df, feature_to_category, mode="avanzato")
test_avz  = enrich_with_ontology(test_df, feature_to_category, mode="avanzato")

print("\n=== COLONNE DATASET ===")
print(train_df.columns)
print(train_mod.columns)
print(train_avz.columns)

# 5) CV
cv_df = cross_validate(models, train_df, train_mod, train_avz)
print("\n=== 5-fold CV (TRAIN) Raw vs Moderato vs Avanzato ===")
print(cv_df.to_string(index=False))

# 6) TEST
results_df = evaluate_models(models, train_df, test_df, train_mod, test_mod, train_avz, test_avz)
print("\n=== RISULTATI SU TEST (Raw vs Moderato vs Avanzato) ===")
print(results_df.to_string(index=False))

# 8) EXPORT
export_results(results_df, cv_df)

# 9) LOG
print("\n=== LOG PER RELAZIONE ===")
print(f"- Colonne dataset: {len(symptom_cols)} cliniche")
print(f"- Ontologie usate: {list(ontologies.keys())}")
print(f"- Feature mappate: {len(mapped_cols)} -> categorie: {len(feature_to_category)}")
print(f"- Macro-categorie (moderato): {sorted(set(train_mod.columns) - set(train_df.columns))}")
print(f"- Macro-categorie (avanzato): {sorted(set(train_avz.columns) - set(train_df.columns))}")
print(f"- Esempi feature non mappate: {unmapped_cols[:10]}")
