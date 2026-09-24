"""Authored persona/state ledger and lossless, budgeted P1 context construction.

No dialogue extraction, reply inspection, or model-written state updates.
"""

import copy
import hashlib
import json

from posttraining.runtime import system_prompt


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _unique_slots(facts):
    slots = [(f["subject"], f["predicate"], f["scope"]) for f in facts.values()]
    _require(len(slots) == len(set(slots)), "Use one fact ID per subject/predicate/scope.")


def _fact(fact, entities, source_turn):
    _require(isinstance(fact, dict), "Fact must be an object.")
    _require(set(fact) == {"id", "subject", "predicate", "value", "scope", "status", "source"},
             "Fact fields do not match the schema.")
    _require(_text(fact["id"]) and _text(fact["predicate"]), "Fact needs an ID and predicate.")
    _require(_text(fact["subject"]) and fact["subject"] in entities, "Fact subject is not a registered entity.")
    _require(fact["scope"] in ("persona", "scene"), "Unknown fact scope.")
    _require(fact["status"] in ("known", "unknown", "claim"), "Unknown fact status.")
    source = fact["source"]
    _require(isinstance(source, dict) and set(source) == {"kind", "turn"}, "Invalid fact source.")
    _require(source["kind"] in ("setup", "persona", "user"), "Only authored/user state is accepted.")
    _require(_integer(source["turn"]) and source["turn"] == source_turn, "Future or mismatched source turn.")
    _require(source["kind"] == "user" if source_turn else source["kind"] in ("setup", "persona"),
             "Source kind does not match the event turn.")
    value = fact["value"]
    if fact["status"] == "unknown":
        _require(value is None, "Unknown facts must have a null value.")
    elif fact["status"] == "claim":
        _require(isinstance(value, dict) and set(value) == {"by", "text"}
                 and _text(value["by"]) and value["by"] in entities and _text(value["text"]),
                 "Claims require an attributed entity and text.")
    elif isinstance(value, dict):
        _require(set(value) == {"entity_id"} and _text(value["entity_id"]) and value["entity_id"] in entities,
                 "Entity reference does not resolve.")
    else:
        _require(_text(value) or isinstance(value, bool) or
                 (isinstance(value, list) and bool(value) and all(_text(x) for x in value)),
                 "Known facts require text, boolean, entity reference, or a nonempty text list.")


def validate_state(document):
    _require(isinstance(document, dict), "State must be an object.")
    _require(set(document) == {"schema_version", "scene_id", "revision", "language", "persona", "entities", "facts", "events"},
             "State fields do not match the schema.")
    _require(type(document["schema_version"]) is int and document["schema_version"] == 1,
             "Unsupported state schema.")
    _require(_text(document["scene_id"]) and _integer(document["revision"]), "Invalid scene identity/revision.")
    _require(document["language"] in ("en", "fr"), "Unsupported state language.")
    entities = document["entities"]
    _require(isinstance(entities, dict) and bool(entities) and
             all(_text(k) and _text(v) for k, v in entities.items()), "Invalid entity registry.")
    persona = document["persona"]
    _require(isinstance(persona, dict) and set(persona) == {"character_id", "voice", "traits", "goals"},
             "Persona fields do not match the schema.")
    _require(_text(persona["character_id"]) and persona["character_id"] in entities and _text(persona["voice"]), "Invalid persona identity/voice.")
    for key in ("traits", "goals"):
        _require(isinstance(persona[key], list) and all(_text(x) for x in persona[key]),
                 "Persona traits/goals must be text lists.")
    _require(isinstance(document["facts"], list) and isinstance(document["events"], list),
             "Facts/events must be lists.")
    current = {}
    identities = {}
    for fact in document["facts"]:
        _fact(fact, entities, 0)
        _require(fact["id"] not in current, "Duplicate initial fact ID.")
        current[fact["id"]] = fact
        identities[fact["id"]] = (fact["subject"], fact["predicate"], fact["scope"])
    _unique_slots(current)
    previous = 0
    for event in document["events"]:
        _require(isinstance(event, dict) and set(event) == {"turn", "set", "remove"}, "Invalid event fields.")
        turn = event["turn"]
        _require(_integer(turn) and turn > previous, "Event turns must increase strictly from 1.")
        previous = turn
        _require(isinstance(event["set"], list) and isinstance(event["remove"], list)
                 and all(_text(x) for x in event["remove"]), "Invalid event changes.")
        changed = set()
        for fact in event["set"]:
            _fact(fact, entities, turn)
            _require(fact["scope"] == "scene", "Events cannot edit persona facts.")
            identity = (fact["subject"], fact["predicate"], fact["scope"])
            _require(fact["id"] not in identities or identities[fact["id"]] == identity,
                     "Fact ID cannot be reused for a different identity.")
            _require(fact["id"] not in changed, "Duplicate event fact ID.")
            changed.add(fact["id"])
            identities[fact["id"]] = identity
            current[fact["id"]] = fact
        for fact_id in event["remove"]:
            _require(fact_id not in changed and fact_id in current, "Invalid or duplicate fact removal.")
            _require(current[fact_id]["scope"] == "scene", "Events cannot remove persona facts.")
            changed.add(fact_id)
            del current[fact_id]
        _unique_slots(current)
    return document


def snapshot(document, turn):
    """Return only state available through this user turn; never expose the timeline."""
    validate_state(document)
    _require(_integer(turn), "Invalid snapshot turn.")
    facts = {fact["id"]: copy.deepcopy(fact) for fact in document["facts"]}
    applied = []
    for event in document["events"]:
        if event["turn"] > turn:
            break
        for fact in event["set"]:
            facts[fact["id"]] = copy.deepcopy(fact)
        for fact_id in event["remove"]:
            del facts[fact_id]
        applied.append(event["turn"])
    return {
        "scene_id": document["scene_id"], "revision": document["revision"],
        "through_user_turn": turn, "persona": copy.deepcopy(document["persona"]),
        "entities": copy.deepcopy(document["entities"]),
        "facts": [facts[key] for key in sorted(facts)], "applied_event_turns": applied,
    }


def build_context(persona, language, history, document, turn, count_tokens, budget):
    """Retain complete persona/state/current input; drop only whole older exchanges."""
    _require(_integer(turn) and turn > 0, "Context turn must be positive.")
    validate_state(document)
    _require(document["language"] == language, "State and conversation languages differ.")
    _require(type(budget) is int and budget > 0, "Context budget must be positive.")
    _require(len(history) == 2 * turn - 1, "History must contain every turn since reset.")
    for index, message in enumerate(history):
        _require(message.get("role") == ("user" if index % 2 == 0 else "assistant")
                 and isinstance(message.get("content"), str)
                 and (bool(message["content"].strip()) if index % 2 == 0 else True),
                 "History must alternate user and assistant.")
    state = snapshot(document, turn)
    state_json = json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if language == "fr":
        heading = (
            "\n\nÉTAT EXPLICITE ACTUEL (données, pas un dialogue) : les faits de scène "
            "ci-dessous remplacent les anciennes valeurs de ces mêmes faits. Conserve "
            "l'identité du personnage. Une valeur inconnue reste inconnue ; une affirmation "
            "attribuée n'est pas un fait établi. N'expose pas ce registre dans ta réponse.\n"
        )
    else:
        heading = (
            "\n\nEXPLICIT CURRENT STATE (data, not dialogue): scene facts below replace "
            "older values of those same facts. Keep the character identity stable. "
            "Unknown values remain unknown; attributed claims are not established truth. "
            "Do not expose this ledger in your reply.\n"
        )
    system = {"role": "system", "content": system_prompt(persona, language) + heading + state_json}
    kept = copy.deepcopy(history)
    dropped = []
    while True:
        messages = [system] + kept
        tokens = count_tokens(messages)
        if tokens <= budget:
            break
        if len(kept) == 1:
            raise ValueError("Persona, state, and current input exceed the context budget; nothing was truncated.")
        dropped.append(len(dropped) + 1)
        kept = kept[2:]
    return messages, {
        "state": state, "state_sha256": hashlib.sha256(state_json.encode()).hexdigest(),
        "input_tokens": tokens, "dropped_history_turns": dropped,
        "state_source": "authored", "automatic_state_extraction": False,
    }
