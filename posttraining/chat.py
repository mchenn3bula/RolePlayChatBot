"""Local file-output chat with authored persona and scene state."""

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from posttraining.artifacts import ReplyWriter
from posttraining.runtime import Runtime, read_config, system_prompt
from posttraining.state import build_context, validate_state


def load_state(path, language):
    state = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    validate_state(state)
    if state["language"] != language:
        raise ValueError("State language differs from the selected chat language.")
    return state


def reload_state(previous, candidate):
    validate_state(candidate)
    if (candidate["scene_id"], candidate["language"]) != (previous["scene_id"], previous["language"]):
        raise ValueError("Reload must keep scene identity and language; start a new chat to change these.")
    if candidate != previous and candidate["revision"] <= previous["revision"]:
        raise ValueError("Increase revision when changing state.")
    return candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ministral_p1.json"))
    parser.add_argument("--language", choices=("en", "fr"), default="en")
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--persona", default=None)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--adapter", type=Path)
    args = parser.parse_args()
    config = read_config(args.config)
    mode = config.get("context_mode", "p0")
    if mode not in ("p0", "p1"):
        raise ValueError("Unknown context mode.")
    if mode == "p0" and args.state_file:
        raise ValueError("A state file requires the P1 profile.")
    directory = args.output_dir or Path("reports") / (
        "chat-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)
    state_path = None
    state = None
    if mode == "p1":
        state_path = args.state_file
        if state_path is None:
            # Personal edits stay in ignored reports, outside tracked example files.
            state_path = directory / "state.json"
            state_path.write_bytes(Path(f"posttraining/personas/lantern_{args.language}.json").read_bytes())
        state_path = state_path.resolve()
        state = load_state(state_path, args.language)
    persona = args.persona or ("Use the explicit persona and scene below." if mode == "p1" else (
        "Tu es Mira, l'aubergiste accueillante de l'auberge de la Lanterne. Il pleut. "
        "Tu es à l'intérieur, près de la porte. Tu ne connais pas encore le voyageur."
        if args.language == "fr" else
        "You are Mira, the welcoming keeper of the Lantern Inn. It is raining. "
        "You are inside near the door and do not yet know the traveler."
    ))
    runtime = Runtime(config, adapter_path=args.adapter)
    preflight = runtime.preflight()
    (directory / "manifest.json").write_text(json.dumps({
        "config": config, "preflight": preflight, "adapter": runtime.adapter,
        "state_file": str(state_path) if state_path else None,
        "review_policy": "user-only", "automatic_state_extraction": False,
    }, indent=2), encoding="utf-8")
    history, turn, session = [], 0, 1
    print(f"Ready. Replies: {(directory / 'replies.html').resolve()}", flush=True)
    print("Refresh that file after each reply. /reload loads edited state; /state shows its path; "
          "/reset clears history and resets the event clock; /quit exits.", flush=True)
    if state_path:
        print(f"State file: {state_path}", flush=True)
    with ReplyWriter(directory) as writer:
        while True:
            try:
                user = input(f"You [next turn {turn+1}]> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if user == "/quit":
                break
            if user == "/reset":
                history, turn, session = [], 0, session + 1
                print("History cleared; authored event clock reset to turn 1.", flush=True)
                continue
            if user == "/state":
                print(f"State file: {state_path}" if state_path else "P0 has no state file.", flush=True)
                continue
            if user == "/reload":
                if state_path is None:
                    print("P0 has no state file.", flush=True)
                    continue
                try:
                    state = reload_state(state, load_state(state_path, args.language))
                except (ValueError, OSError, TypeError, KeyError):
                    print("State reload rejected. Check schema, language, scene ID, and increased revision.", flush=True)
                    continue
                print(f"State revision {state['revision']} loaded for next user turn {turn+1}.", flush=True)
                continue
            if not user:
                continue
            trial = history + [{"role": "user", "content": user}]
            try:
                if state is not None:
                    messages, audit = build_context(persona, args.language, trial, state, turn + 1,
                                                    runtime.count_tokens, config["max_context_tokens"])
                else:
                    messages = [{"role": "system", "content": system_prompt(persona, args.language)}] + trial
                    audit = {"state_source": None, "dropped_history_turns": []}
                result = runtime.generate(messages, 42 + 1000003 * turn)
            except ValueError:
                print("Input/state rejected or context budget exceeded. Simplify state/input or /reset.", flush=True)
                continue
            except Exception as error:
                print(f"Inference stopped ({type(error).__name__}); prior replies remain saved.", flush=True)
                return 1
            writer.append({"id": f"session{session}-turn{turn+1}", "language": args.language,
                           "turn": turn + 1, "session": session, "messages": messages,
                           "context": audit, **result})
            history = trial + [{"role": "assistant", "content": result["text"]}]
            turn += 1
            print(f"Reply {turn} saved | {result['tokens_per_second']:.1f} tok/s | "
                  f"history exchanges dropped: {len(audit['dropped_history_turns'])} | "
                  f"{directory / 'replies.html'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
