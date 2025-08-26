import re
import difflib

def normalize_label(s: str) -> str:
    x = s.lower().strip().replace("_", " ")
    x = re.sub(r"\s+", " ", x).strip()
    return x

def build_label_dict(ontologies):
    ALL_LABELS = []
    ALL_CLASS_BY_LABEL = {}
    for onto in ontologies.values():
        for cls in onto.classes():
            for lab in getattr(cls, "label", []):
                l = lab.lower()
                ALL_LABELS.append(l)
                ALL_CLASS_BY_LABEL.setdefault(l, set()).add(cls)
            for syn_prop in ("hasExactSynonym", "hasRelatedSynonym", "hasBroadSynonym", "hasNarrowSynonym"):
                for syn in getattr(cls, syn_prop, []):
                    s = syn.lower()
                    ALL_LABELS.append(s)
                    ALL_CLASS_BY_LABEL.setdefault(s, set()).add(cls)
    ALL_LABELS = sorted(set(ALL_LABELS))
    ALL_CLASS_BY_LABEL = {k: list(v) for k, v in ALL_CLASS_BY_LABEL.items()}
    return ALL_LABELS, ALL_CLASS_BY_LABEL

def search_concept(term: str, ALL_LABELS, ALL_CLASS_BY_LABEL):
    t = normalize_label(term)
    if t in ALL_CLASS_BY_LABEL:
        return ALL_CLASS_BY_LABEL[t][0]
    cand = difflib.get_close_matches(t, ALL_LABELS, n=1, cutoff=0.92)
    if cand:
        return ALL_CLASS_BY_LABEL[cand[0]][0]
    return None
