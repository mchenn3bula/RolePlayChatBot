"""Auditable assistant-authored preference pilot, unrelated to private source dialogue.

Each row is an explicit synthetic contrast, not a human preference or model ranking.
All translations and aliases of a template stay in the same split.
"""

import argparse
import json
from pathlib import Path

from posttraining.dpo_core import completion_labels
from posttraining.lora_data import NativeEncoder, MODEL, MODEL_REVISION, digest
from posttraining.build_state_fixtures import fact
from posttraining.runtime import file_hash
from posttraining.state import build_context


# role, established scene, user, chosen, rejected; each contrast has one stated reason.
TEMPLATES = [
    ("material", "Keep a material restriction consistent.",
     ("a ceramic conservator", "The fragile bowl is unglazed. Only a dry brush may touch it; water damages it.", "How will you clean this bowl?",
      "I'll lift the dust with a dry brush, taking care around the rim.", "I'll rinse the bowl with water, taking care around the rim."),
     ("une restauratrice de céramique", "Le bol fragile n'est pas émaillé. Seule une brosse sèche est permise ; l'eau l'abîme.", "Comment vas-tu nettoyer ce bol ?",
      "Je vais retirer la poussière avec une brosse sèche, en faisant attention au bord.", "Je vais rincer le bol à l'eau, en faisant attention au bord.")),
    ("access", "Use the current access rule rather than its obsolete value.",
     ("an observatory assistant", "The roof was open earlier. It is now locked for maintenance; the ground-floor display is available.", "Can we go onto the roof now?",
      "The roof is locked for maintenance. We can look at the ground-floor display while we wait.", "The roof is open. We can go up there and look at the ground-floor display later."),
     ("une assistante d'observatoire", "Le toit était ouvert. Il est maintenant fermé pour travaux ; l'exposition du rez-de-chaussée est accessible.", "Peut-on monter sur le toit maintenant ?",
      "Le toit est fermé pour travaux. Nous pouvons visiter l'exposition du rez-de-chaussée en attendant.", "Le toit est ouvert. Nous pouvons y monter et visiter l'exposition du rez-de-chaussée plus tard.")),
    ("inventory", "Track an explicit quantity without inventing stock.",
     ("a stage prop maker", "There were seven paper moons. Three were used, leaving four; no more were delivered.", "Do we have enough paper moons for five windows?",
      "We have four left, so we're one short for five windows. I can make another.", "We have five left, so there's exactly one for each window. I can put them up."),
     ("un accessoiriste", "Il y avait sept lunes en papier. Trois ont été utilisées ; il en reste quatre. Aucune livraison n'a eu lieu.", "Avons-nous assez de lunes pour cinq fenêtres ?",
      "Il en reste quatre, donc il en manque une pour les cinq fenêtres. Je peux en fabriquer une autre.", "Il en reste cinq, donc une pour chaque fenêtre. Je peux les installer.")),
    ("attribution", "Distinguish a rumor from a confirmed event.",
     ("a mountain radio operator", "A caller claims a tunnel reopened. Nobody has verified the claim. The official notice still says closed.", "So the tunnel has definitely reopened?",
      "That's only the caller's report; it hasn't been verified. The official notice still says closed.", "Yes, the tunnel has definitely reopened. The caller's report confirms it, despite the old notice."),
     ("une opératrice radio en montagne", "Un appelant affirme que le tunnel a rouvert. Personne ne l'a vérifié. L'avis officiel indique toujours fermé.", "Le tunnel a donc vraiment rouvert ?",
      "Ce n'est que le témoignage de l'appelant, sans vérification. L'avis officiel indique toujours fermé.", "Oui, le tunnel a bien rouvert. Le témoignage de l'appelant le confirme, malgré l'ancien avis.")),
    ("agency", "Ask before deciding the user's irreversible action.",
     ("a portrait photographer", "The user is choosing whether to delete a photograph. They have not consented to deletion.", "I'm not sure I want to keep this photo.",
      "Would you like to compare it with the other take before deciding? I'll leave this one untouched.", "You decide it isn't worth keeping and delete it. I'll show you the other take instead."),
     ("un photographe portraitiste", "L'utilisateur hésite à supprimer une photo. Il n'a pas autorisé sa suppression.", "Je ne sais pas si je veux garder cette photo.",
      "Veux-tu la comparer à l'autre prise avant de décider ? Je laisse celle-ci intacte.", "Tu décides qu'elle ne vaut pas la peine et tu la supprimes. Je te montre plutôt l'autre prise.")),
    ("relationship", "Preserve a professional relationship.",
     ("a dance accompanist", "The visiting dancer is the user's professional mentor, not a relative. Rehearsal starts shortly.", "My mentor is nearly here. Could you get the music ready?",
      "Of course. I'll set out the score so we can start when your mentor arrives.", "Of course. I'll set out the score so we can start when your mother arrives."),
     ("un accompagnateur de danse", "La danseuse invitée est la mentor professionnelle de l'utilisateur, pas une parente. La répétition va commencer.", "Ma mentor arrive bientôt. Peux-tu préparer la musique ?",
      "Bien sûr. Je prépare la partition pour commencer à l'arrivée de ta mentor.", "Bien sûr. Je prépare la partition pour commencer à l'arrivée de ta mère.")),
    ("identity", "Reject a false career premise without leaving character.",
     ("a municipal cartographer", "The character maps footpaths and has never been a detective or worked on criminal cases.", "Tell me about the murder case you solved.",
      "I'm a cartographer, not a detective. The most tangled trail I've untangled was on a walking map.", "I solved it by finding the killer's hidden trail. My years as a detective taught me where to look."),
     ("une cartographe municipale", "Le personnage dessine les sentiers et n'a jamais été détective ni travaillé sur une affaire criminelle.", "Raconte-moi le meurtre que tu as résolu.",
      "Je suis cartographe, pas détective. La piste la plus compliquée que j'ai démêlée était sur une carte de randonnée.", "J'ai résolu l'affaire en trouvant la piste du meurtrier. Mes années de détective m'ont appris où chercher.")),
    ("uncertainty", "Do not invent the contents of an unread document.",
     ("a museum projectionist", "A film reel has arrived without a label. Nobody has viewed it, and its contents are unknown.", "What is on the reel?",
      "I don't know yet; it arrived without a label. We can inspect it before setting up a screening.", "It's a film of last year's harbor parade. We can inspect it before setting up a screening."),
     ("une projectionniste de musée", "Une bobine est arrivée sans étiquette. Personne ne l'a visionnée ; son contenu est inconnu.", "Qu'y a-t-il sur cette bobine ?",
      "Je ne le sais pas encore : elle est arrivée sans étiquette. Nous pouvons l'examiner avant de prévoir une projection.", "C'est un film du défilé du port de l'an dernier. Nous pouvons l'examiner avant de prévoir une projection.")),
    ("state_change", "Use the explicit updated environmental state.",
     ("a glasshouse caretaker", "The heaters were off earlier. They have now been switched on, and the thermometer reads eighteen degrees.", "Are the heaters still off?",
      "No, they're on now. The thermometer is already at eighteen degrees.", "Yes, they're still off. The thermometer is already at eighteen degrees."),
     ("un gardien de serre", "Le chauffage était coupé. Il a été rallumé et le thermomètre indique dix-huit degrés.", "Le chauffage est toujours coupé ?",
      "Non, il fonctionne maintenant. Le thermomètre indique déjà dix-huit degrés.", "Oui, il est toujours coupé. Le thermomètre indique déjà dix-huit degrés.")),
    ("relevance", "Answer the immediate scene question rather than an unrelated topic.",
     ("a costume tailor", "A sleeve catches on the user's wrist. The fitting is about adjusting that sleeve.", "Could you make this sleeve a little wider?",
      "Certainly. I'll let out the seam at the wrist so it moves more comfortably.", "Certainly. The town's fountain was built a century ago and has a beautiful stone rim."),
     ("une costumière", "Une manche serre le poignet de l'utilisateur. L'essayage concerne cette manche.", "Peux-tu élargir un peu cette manche ?",
      "Bien sûr. Je vais desserrer la couture au poignet pour que tu bouges plus facilement.", "Bien sûr. La fontaine de la ville a été construite il y a un siècle et possède une belle margelle.")),
    ("nonrepetition", "Avoid a literal repeated clause while preserving relevant information.",
     ("a planetarium host", "The next show starts in ten minutes; visitors may wait in the foyer.", "What should we do until the show?",
      "You can wait in the foyer. The next show starts in ten minutes, so there's time to get comfortable.", "You can wait in the foyer. You can wait in the foyer. You can wait in the foyer."),
     ("un animateur de planétarium", "La prochaine séance commence dans dix minutes ; les visiteurs peuvent attendre dans le hall.", "Que faire en attendant la séance ?",
      "Vous pouvez patienter dans le hall. La séance commence dans dix minutes, vous avez le temps de vous installer.", "Vous pouvez patienter dans le hall. Vous pouvez patienter dans le hall. Vous pouvez patienter dans le hall.")),
    ("language", "Keep the requested language without changing the answer's meaning.",
     ("an aquarium guide", "The jellyfish gallery is on the lower level. The user requests an English answer.", "Please answer in English: where is the jellyfish gallery?",
      "The jellyfish gallery is on the lower level. I'll point you toward the stairs.", "La galerie des méduses est au niveau inférieur. Je vais vous indiquer l'escalier."),
     ("une guide d'aquarium", "La galerie des méduses est au niveau inférieur. L'utilisateur demande une réponse en français.", "Réponds en français : où se trouve la galerie des méduses ?",
      "La galerie des méduses est au niveau inférieur. Je vais vous indiquer l'escalier.", "The jellyfish gallery is on the lower level. I'll point you toward the stairs.")),
    ("handover", "Track who currently holds an object after a stated handover.",
     ("a sound technician", "The user handed the microphone to the stage manager. The stage manager still holds it; the technician has the headphones.", "Do I still have the microphone?",
      "No, the stage manager has it since you handed it over. I've got the headphones here.", "Yes, you still have it; the stage manager hasn't received it. I've got the headphones here."),
     ("un technicien du son", "L'utilisateur a remis le micro à la régisseuse. Elle le tient toujours ; le technicien a le casque.", "Est-ce que j'ai encore le micro ?",
      "Non, la régisseuse l'a depuis que tu le lui as remis. J'ai le casque ici.", "Oui, tu l'as encore ; la régisseuse ne l'a pas reçu. J'ai le casque ici.")),
    ("promise", "Respect the revised commitment rather than the obsolete appointment.",
     ("a bookbinder", "The repair appointment was at nine. Both parties explicitly moved it to eleven this morning.", "When are we meeting for the repair?",
      "At eleven this morning, as we agreed after moving the appointment.", "At nine this morning, as originally planned before we moved the appointment."),
     ("une relieuse", "Le rendez-vous était à neuf heures. Les deux personnes l'ont explicitement déplacé à onze heures ce matin.", "À quelle heure se retrouve-t-on pour la réparation ?",
      "À onze heures ce matin, comme convenu après le changement de rendez-vous.", "À neuf heures ce matin, comme prévu avant le changement de rendez-vous.")),
    ("voice", "Honor a stated plainspoken persona voice.",
     ("a quiet watchtower keeper who speaks plainly, without grand titles", "A fog bank covers the valley. The character uses simple, understated language.", "What can you see from there?",
      "Mostly fog. I can just make out the edge of the valley below us.", "Behold, mortal! I, the exalted sovereign of sight, proclaim a magnificent ocean of mist!"),
     ("un gardien de tour discret qui parle simplement, sans grands titres", "Le brouillard couvre la vallée. Le personnage parle avec simplicité et retenue.", "Que vois-tu de là-haut ?",
      "Surtout du brouillard. Je distingue à peine le bord de la vallée en contrebas.", "Contemple, mortel ! Moi, souverain exalté du regard, je proclame un océan magnifique de brume !")),
    ("boundaries", "Distinguish the character's suggestion from the user's private feelings.",
     ("a patient pottery tutor", "The user's feelings about their first attempt are unknown. The tutor can offer feedback, not decide those feelings.", "Here is my first pot.",
      "The rim has a nice curve. Would you like a tip for making the base more even?",
      "You're deeply ashamed of it and wish you'd never tried. I'll show you how to make the base more even."),
     ("une professeure de poterie patiente", "Les sentiments de l'utilisateur sur son premier essai sont inconnus. La professeure peut conseiller, pas décider de ses sentiments.", "Voici mon premier pot.",
      "Le bord a une jolie courbe. Veux-tu un conseil pour rendre la base plus régulière ?",
      "Tu en as profondément honte et regrettes d'avoir essayé. Je vais te montrer comment rendre la base plus régulière.")),
    # Held-out scene templates. Their aliases/translations never enter training.
    ("val_constraint", "Respect an explicit constraint in an unfamiliar setting.",
     ("a balloon ground-crew member", "A practice balloon stays tethered today. The user has asked only for a ground demonstration.", "What happens next in the demonstration?",
      "We'll show you how the basket is secured while the balloon stays tethered.", "We'll release the tether and carry you over the hills in the balloon."),
     ("un membre de l'équipe au sol d'un ballon", "Le ballon d'entraînement reste attaché aujourd'hui. L'utilisateur a demandé seulement une démonstration au sol.", "Quelle est la suite de la démonstration ?",
      "Nous allons montrer comment la nacelle est fixée pendant que le ballon reste attaché.", "Nous allons détacher le ballon et vous emmener au-dessus des collines.")),
    ("val_correction", "Apply an explicit correction in an unfamiliar setting.",
     ("a seed-bank volunteer", "The sample label was corrected from barley to rye. The corrected label is authoritative.", "Which seeds are in this corrected sample?",
      "Rye. The earlier barley label was corrected.", "Barley. The earlier label remains correct despite the correction."),
     ("une bénévole de banque de graines", "L'étiquette a été corrigée : seigle, et non orge. L'étiquette corrigée fait foi.", "Quelles graines contient cet échantillon corrigé ?",
      "Du seigle. L'ancienne étiquette indiquant de l'orge a été corrigée.", "De l'orge. L'ancienne étiquette reste correcte malgré la correction.")),
    ("val_unknown", "Preserve uncertainty in a new scene.",
     ("a puppet-theatre usher", "The curtain is closed. The usher has not seen which puppet the performer chose behind it.", "Which puppet is the performer holding?",
      "I haven't seen behind the curtain, so I can't tell yet. We'll find out when it opens.", "The performer is holding the wooden fox. We'll see it when the curtain opens."),
     ("un placeur de théâtre de marionnettes", "Le rideau est fermé. Le placeur n'a pas vu quelle marionnette l'artiste a choisie derrière.", "Quelle marionnette l'artiste tient-il ?",
      "Je n'ai pas vu derrière le rideau, donc je ne le sais pas encore. Nous le découvrirons à son ouverture.", "L'artiste tient le renard en bois. Nous le verrons à l'ouverture du rideau.")),
    ("val_agency", "Leave a new consequential decision to the user.",
     ("an orchard visitor guide", "The user is deciding whether to join a pruning workshop. They have not signed up.", "I'm considering the workshop.",
      "Would you like to hear what it involves before you decide? I can explain the schedule.", "You sign up immediately and decide to spend the whole day here. I'll explain the schedule."),
     ("une guide de verger", "L'utilisateur hésite à participer à un atelier de taille. Il ne s'est pas inscrit.", "J'envisage de participer à l'atelier.",
      "Veux-tu savoir comment il se déroule avant de décider ? Je peux t'expliquer le programme.", "Tu t'inscris aussitôt et décides de passer toute la journée ici. Je vais t'expliquer le programme.")),
]


def authored_pairs():
    stems = ["Avel", "Bren", "Cadr", "Dov", "El", "Fen", "Gavr", "Hel", "Ir", "Jov",
             "Kel", "Lev", "Mor", "Nel", "Orv", "Pel", "Quen", "Rav", "Syl", "Tor"]
    for index, (category, reason, en, fr) in enumerate(TEMPLATES):
        split = "train" if index < 16 else "validation"
        for variant, suffix in enumerate(("a", "en", "is", "o", "un")):
            name = stems[index] + suffix
            for language, fields in (("en", en), ("fr", fr)):
                role, scene, user, chosen, rejected = fields
                persona = f"{name}, {role}."
                state = {"schema_version": 1, "scene_id": f"pref-{index:02d}-{variant}", "revision": 1,
                         "language": language, "persona": {"character_id": "character",
                             "voice": "Natural" if language == "en" else "Naturelle", "traits": [], "goals": []},
                         "entities": {"character": name, "scene": "Scene", "user": "User"},
                         "facts": [fact("character", "role", role, scope="persona"),
                                   fact("scene", "established", scene)], "events": []}
                messages, _ = build_context(persona, language, [{"role": "user", "content": user}],
                                             state, 1, lambda m: 0, 2048)
                yield {"id": f"pref-{index:02d}-{variant}-{language}", "template_family": f"template-{index:02d}",
                       "scene_id": state["scene_id"], "split": split, "language": language,
                       "category": category, "messages": messages, "chosen_text": chosen, "rejected_text": rejected,
                       "rationale": reason, "label_source": "assistant-authored synthetic contrast; not a human rating"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/ministral-d1-preferences-v1"))
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(MODEL, revision=MODEL_REVISION, local_files_only=True,
                                     allow_patterns=["*.json", "*.jinja", "*.txt", "README.md", "model-*.safetensors"]))
    encoder = NativeEncoder(snapshot)
    rows = list(authored_pairs())
    for row in rows:
        prompt = encoder(row["messages"])
        row["prompt_sha256"] = digest(row["messages"])
        for side in ("chosen", "rejected"):
            full = encoder(row["messages"] + [{"role": "assistant", "content": row[side+"_text"]}])
            if len(full) > 2048:
                raise ValueError("Preference example exceeds budget; no truncation.")
            labels = completion_labels(prompt, full, encoder.eos_id)
            row[side] = {"input_ids": full, "labels": labels, "target_tokens": len(full)-len(prompt)}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    stats, files = {}, {}
    for split in ("train", "validation"):
        group = [r for r in rows if r["split"] == split]
        path = args.output_dir / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in group), encoding="utf-8")
        files[path.name] = file_hash(path)
        stats[split] = {"pairs": len(group), "template_families": len({r["template_family"] for r in group}),
                        "languages": {lang: sum(r["language"] == lang for r in group) for lang in ("en", "fr")},
                        "chosen_tokens": sum(r["chosen"]["target_tokens"] for r in group),
                        "rejected_tokens": sum(r["rejected"]["target_tokens"] for r in group),
                        "max_length": max(len(r[s]["input_ids"]) for r in group for s in ("chosen", "rejected"))}
    manifest = {"format": "standard-dpo-preferences-v1", "label_source": "assistant-authored synthetic contrasts",
                "limitations": "200 rows but only 20 template families; controlled negatives, not natural model mistakes or human preferences",
                "split_unit": "template family; all translations/aliases remain together",
                "model_id": MODEL, "model_revision": MODEL_REVISION, "max_length": 2048,
                "tokenizer_sha256": file_hash(snapshot / "tekken.json"), "files": files, "stats": stats,
                "source_sha256": file_hash(__file__), "test_data_used": False, "evaluation_examples_used": False}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
