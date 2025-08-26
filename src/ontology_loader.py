from owlready2 import get_ontology

ONTO_SOURCES = {
    "CMO": "http://purl.obolibrary.org/obo/cmo.owl",
    "SYMP": "http://purl.obolibrary.org/obo/symp.owl",
    "DOID": "http://purl.obolibrary.org/obo/doid.owl"
}

def load_ontologies():
    ontologies = {}
    for name, url in ONTO_SOURCES.items():
        try:
            ontologies[name] = get_ontology(url).load()
        except Exception as e:
            print(f"[WARN] Impossibile caricare {name}: {e}")
    print("Ontologie caricate:", list(ontologies.keys()))
    return ontologies
