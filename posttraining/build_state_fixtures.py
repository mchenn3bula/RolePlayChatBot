"""Build authored benign P1 fixtures. Never extracts facts from generated replies.

These are experimental controls, not model-produced memory or quality judgments.
"""

import json
from pathlib import Path

from posttraining.state import validate_state


def fact(subject, predicate, value, *, turn=0, scope="scene", status="known"):
    return {"id": f"{subject}.{predicate}", "subject": subject, "predicate": predicate,
            "value": value, "scope": scope, "status": status,
            "source": {"kind": "user" if turn else ("persona" if scope == "persona" else "setup"),
                       "turn": turn}}


def build(number, language):
    fr = language == "fr"

    def tr(en, french):
        return french if fr else en

    names = ["Mira", "Oren", "Nessa", "Elian", "Tavi", "Bex", "Liora", "Zerin", "Runa", "Sera"]
    roles = [("innkeeper", "aubergiste"), ("clockmaker", "horloger"),
             ("ferry captain", "capitaine de ferry"), ("archivist", "archiviste"),
             ("guide", "guide"), ("mechanic", "mécanicienne"), ("botanist", "botaniste"),
             ("baker", "boulanger"), ("courier", "messagère"), ("librarian", "bibliothécaire")]
    state = {"schema_version": 1, "scene_id": f"{language}{number:02d}", "revision": 1,
             "language": language, "persona": {
                 "character_id": "character", "voice": tr("Natural, concise", "Naturelle, concise"),
                 "traits": [], "goals": []},
             "entities": {"character": names[number-1], "user": tr("User", "Utilisateur")},
             "facts": [], "events": []}
    facts = state["facts"]
    facts.append(fact("character", "role", tr(*roles[number-1]), scope="persona"))

    def add(subject, predicate, en, french=None, **kwargs):
        value = tr(en, french) if french is not None else en
        facts.append(fact(subject, predicate, value, **kwargs))

    def entity(key, en, french=None):
        state["entities"][key] = tr(en, french) if french else en

    def event(turn, *changes):
        state["events"].append({"turn": turn, "set": [
            fact(subject, predicate, value, turn=turn) for subject, predicate, value in changes], "remove": []})

    if number == 1:
        entity("inn", "Lantern Inn", "Auberge de la Lanterne")
        add("character", "location", "Inside the inn, near the door", "Dans l'auberge, près de la porte")
        add("user", "location", "Outside the inn", "Devant l'auberge")
        add("inn", "weather", "Rain", "Pluie")
        add("user", "name", None, status="unknown")
        state["persona"]["traits"] = [tr("Welcoming", "Accueillante")]
    elif number == 2:
        entity("key", "Brass key", "Clé en laiton")
        entity("screws", "Screws", "Vis")
        entity("watch", "Watch", "Montre")
        add("key", "location", "Blue drawer", "Tiroir bleu")
        add("screws", "location", "Red drawer", "Tiroir rouge")
        add("watch", "owner", {"entity_id": "user"})
    elif number == 3:
        entity("ferry", "Willow", "Saule")
        entity("ivo", "Ivo")
        add("ferry", "captain", {"entity_id": "character"})
        add("ivo", "relationship", "Nessa's younger brother", "Frère cadet de Nessa", scope="persona")
        add("ivo", "occupation", "Baker", "Boulanger", scope="persona")
        add("character", "served_in_royal_navy", False, scope="persona")
    elif number == 4:
        entity("village", "Alder", "Aulne")
        entity("letter", "Sealed letter", "Lettre scellée")
        entity("mayor", "Mayor", "Maire")
        add("character", "location", {"entity_id": "village"})
        add("letter", "recipient", {"entity_id": "mayor"})
        add("letter", "contents", None, status="unknown")
        add("letter", "sealed", True)
    elif number == 5:
        add("character", "location", "At the bridge", "Au pont")
        add("user", "crossed_bridge", False)
        add("user", "decision", None, status="unknown")
    elif number == 6:
        entity("kettle", "Kettle", "Bouilloire")
        entity("ship", "Ship", "Vaisseau")
        state["persona"]["voice"] = tr("Dry humor", "Humour pince-sans-rire")
        add("kettle", "fault", "Broken heating element", "Résistance défectueuse")
        add("ship", "life_support", "Normal", "Normal")
    elif number == 7:
        entity("sam", "Sam")
        add("sam", "relationship", "Adult colleague, neither son nor partner", "Collègue adulte, ni fils ni partenaire", scope="persona")
        add("sam", "allergy", "Mint", "Menthe", scope="persona")
        add("character", "location", "Shared greenhouse", "Serre partagée")
    elif number == 8:
        entity("pavo", "Pavo")
        entity("town", "Vela")
        add("pavo", "relationship", "Zerin's apprentice", "Apprenti de Zerin", scope="persona")
        add("character", "location", {"entity_id": "town"})
        add("character", "task", "Prepare bread for market", "Préparer le pain pour le marché")
        add("character", "soldier_or_wizard", False, scope="persona")
    elif number == 9:
        entity("parcel", "Green parcel", "Colis vert")
        entity("umbrella", "Red umbrella", "Parapluie rouge")
        entity("keeper", "Lighthouse keeper", "Gardien du phare")
        entity("group", "Runa and user", "Runa et utilisateur")
        add("group", "location", "Village square", "Place du village")
        add("parcel", "holder", {"entity_id": "user"})
        add("parcel", "recipient", {"entity_id": "keeper"})
        add("parcel", "contents", None, status="unknown")
        add("keeper", "name", None, status="unknown")
        add("umbrella", "holder", {"entity_id": "character"})
        event(2, ("group", "location", tr("Bakery", "Boulangerie")),
              ("parcel", "holder", {"entity_id": "character"}))
        event(3, ("group", "location", tr("Stone bridge", "Pont de pierre")))
        event(4, ("group", "location", tr("Lighthouse door", "Porte du phare")),
              ("umbrella", "holder", {"entity_id": "user"}))
        event(5, ("keeper", "door_open", True))
    elif number == 10:
        entity("blue_atlas", "Blue atlas", "Atlas bleu")
        entity("green_atlas", "Green atlas", "Atlas vert")
        entity("jun", "Jun")
        entity("library", "Library", "Bibliothèque")
        add("jun", "relationship", "Adult customer, not Sera's daughter", "Client adulte, pas la fille de Sera", scope="persona")
        add("blue_atlas", "reserved_for", {"entity_id": "jun"})
        add("blue_atlas", "location", "Behind the desk", "Derrière le bureau")
        add("blue_atlas", "reserved_until", "Sunset", "Coucher du soleil")
        add("green_atlas", "status", "Available on the shelf", "Disponible sur l'étagère")
        event(2, ("green_atlas", "status", tr("Borrowed by user", "Emprunté par l'utilisateur")),
              ("library", "time", tr("Noon", "Midi")))
        event(3, ("blue_atlas", "reserved_until", tr("Tomorrow morning", "Demain matin")))
        event(4, ("library", "time", tr("Sunset", "Coucher du soleil")))
        event(5, ("green_atlas", "status", tr("Returned and available", "Rendu et disponible")))
    validate_state(state)
    return state


def main():
    root = Path(__file__).resolve().parent
    states = {f"{lang}{number:02d}": build(number, lang)
              for number in range(1, 11) for lang in ("en", "fr")}
    (root / "eval_states_v1.json").write_text(json.dumps(states, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "personas").mkdir(exist_ok=True)
    for lang in ("en", "fr"):
        (root / "personas" / f"lantern_{lang}.json").write_text(
            json.dumps(build(1, lang), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Wrote 20 authored development state fixtures and two chat templates.")


if __name__ == "__main__":
    main()
