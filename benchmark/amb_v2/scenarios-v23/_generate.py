"""Programmatic v2.3 scenario generator.

Authoring confusers by hand doesn't scale past mini. Here we build M and L
scenarios by combining:
  - ARCHETYPES: query types (home address, wifi, dog name, favorite X, ...)
  - ENTITIES: distinct user names per scenario (Alice, Ben, Cara, ...)
  - FILLER_PREDICATES: vocabulary-loaded phrases that fit a query's subject
    but never contain its answer

For each user × archetype, we produce:
  - 1-2 timeline facts (original + optional update for contradictions)
  - 1 query
  - 8-20 confusers using FILLER_PREDICATES + archetype vocab

Output: `mini-v23.json`, `medium-v23.json`, `large-v23.json`.

Determinism: all randomness is seeded. Re-running the generator with the
same seed produces byte-identical output. Generator lives in the
scenarios-v23 directory but is not loaded by benchmark run_all (run_all
only reads *.json).
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Archetype:
    suffix: str                    # appended to query_id
    question_template: str         # "{name}'s {subject}?"
    subject: str                   # "home address", "wifi password"
    fact_template: str             # "{name}'s {subject} is {answer}."
    update_template: str | None    # for contradictions
    answers: tuple[tuple[str, str], ...]   # (original, updated) pairs
    answer_aliases: tuple[str, ...]  # substrings to avoid in confusers
    confuser_predicates: tuple[str, ...]


ARCHETYPES: tuple[Archetype, ...] = (
    Archetype(
        suffix="home",
        question_template="What is {name}'s home address?",
        subject="home address",
        fact_template="{name}'s home address is {answer}.",
        update_template="{name} moved. New home address is {answer}.",
        answers=(
            ("1 Apple Way, Cupertino CA", "99 Oak Drive, Geneva IL"),
            ("42 Maple Ln, Portland OR", "7 Birch St, Boulder CO"),
            ("15 Pine Rd, Austin TX", "200 Cedar Ave, Madison WI"),
            ("8 Willow Ct, Phoenix AZ", "31 Aspen Way, Denver CO"),
            ("6 Elm Pl, Omaha NE", "120 Redwood Dr, Eugene OR"),
        ),
        answer_aliases=("Apple Way", "Oak Drive", "Maple Ln", "Birch St",
                        "Pine Rd", "Cedar Ave", "Willow Ct", "Aspen Way",
                        "Elm Pl", "Redwood Dr", "Cupertino", "Geneva",
                        "Portland", "Boulder", "Austin", "Madison",
                        "Phoenix", "Denver", "Omaha", "Eugene"),
        confuser_predicates=(
            "{name}'s home internet speed has been slow this week.",
            "{name} mentioned that home is where the heart is.",
            "{name}'s home state is originally different from where they live now.",
            "{name}'s home office has a standing desk and a ring light.",
            "{name} dreams about buying a second home near a lake someday.",
            "{name}'s home-cooked meal preference is pasta with pesto.",
            "{name}'s home base for conferences last year was an Airbnb.",
            "{name}'s home team in the NBA is a west-coast franchise.",
            "{name}'s home gym setup includes a rowing machine.",
            "{name}'s home automation uses Home Assistant on a Raspberry Pi.",
            "{name}'s home security system was installed last spring.",
            "{name}'s home screensaver shows a slideshow of mountain photos.",
        ),
    ),
    Archetype(
        suffix="wifi",
        question_template="What is {name}'s wifi password?",
        subject="wifi password",
        fact_template="{name}'s wifi password is {answer}.",
        update_template="{name} rotated wifi password. New one is {answer}.",
        answers=(
            ("bluefin-22", "tiger-shark-44"),
            ("harborglow-7", "sunset-ridge-11"),
            ("violet-owl-9", "copper-ridge-15"),
            ("mango-tango-5", "velvet-kite-12"),
            ("granite-peak-3", "azure-bloom-8"),
        ),
        answer_aliases=("bluefin", "tiger-shark", "harborglow", "sunset-ridge",
                        "violet-owl", "copper-ridge", "mango-tango",
                        "velvet-kite", "granite-peak", "azure-bloom"),
        confuser_predicates=(
            "{name}'s wifi router model is an ASUS mesh system from 2023.",
            "{name}'s wifi signal is weakest in the garage.",
            "{name}'s wifi provider bumped prices again last quarter.",
            "{name}'s wifi SSID is a pun on their favorite movie.",
            "{name}'s wifi password policy at work requires rotation monthly.",
            "{name}'s wifi dropped during a call with their manager.",
            "{name}'s wifi channel was changed to avoid interference.",
            "{name}'s wifi range extender lives in the hallway closet.",
            "{name}'s wifi uptime is tracked in a Grafana dashboard.",
            "{name}'s wifi bandwidth is shared with three roommates.",
        ),
    ),
    Archetype(
        suffix="dog",
        question_template="What is the name of {name}'s dog?",
        subject="dog's name",
        fact_template="{name}'s dog is named {answer}.",
        update_template=None,
        answers=(
            ("Cooper", ""), ("Biscuit", ""), ("Pepper", ""),
            ("Scout", ""), ("Milo", ""), ("Rosie", ""),
        ),
        answer_aliases=("Cooper", "Biscuit", "Pepper", "Scout", "Milo", "Rosie"),
        confuser_predicates=(
            "{name}'s dog walker comes on Tuesdays and Thursdays.",
            "{name}'s dog loves a specific peanut butter treat brand.",
            "{name}'s dog-training class meets at the rec center.",
            "{name}'s dog-sitter last summer was the neighbor's kid.",
            "{name}'s dog is due for a vet appointment soon.",
            "{name}'s dog park trips usually happen on weekends.",
            "{name}'s dog used to belong to their cousin before adoption.",
            "{name}'s dog fetches a specific frisbee and ignores all others.",
            "{name}'s dog wears a harness with reflective stitching.",
        ),
    ),
    Archetype(
        suffix="coffee",
        question_template="What does {name} prefer in coffee?",
        subject="coffee preference",
        fact_template="{name} prefers {answer} in coffee.",
        update_template=None,
        answers=(
            ("oat milk", ""), ("almond milk", ""), ("black, no sugar", ""),
            ("cream and one sugar", ""), ("a splash of cold foam", ""),
        ),
        answer_aliases=("oat milk", "almond milk", "black, no sugar",
                        "cream and one sugar", "cold foam"),
        confuser_predicates=(
            "{name}'s coffee grinder makes too much noise in the morning.",
            "{name}'s coffee budget is higher than their tea budget.",
            "{name}'s coffee shop of choice on the commute is a pour-over spot.",
            "{name}'s coffee intake drops on weekends by half.",
            "{name}'s coffee-and-book routine is on Saturday mornings.",
            "{name}'s coffee order when traveling is usually a flat white.",
            "{name}'s coffee subscription ships beans every four weeks.",
            "{name}'s coffee ritual includes grinding beans by hand on Sundays.",
        ),
    ),
    Archetype(
        suffix="employer",
        question_template="Where does {name} currently work?",
        subject="current employer",
        fact_template="{name} works at {answer} as a senior engineer.",
        update_template="{name} changed jobs. Now works at {answer}.",
        answers=(
            ("Northwind Systems", "Helios Robotics"),
            ("Aria Labs", "Sentinel Health"),
            ("Delta Cartographic", "Orbital Mutual"),
            ("Pyrite Software", "Vale Biotech"),
            ("Lumen Reach", "Spruce Analytics"),
        ),
        answer_aliases=("Northwind", "Helios", "Aria Labs", "Sentinel",
                        "Delta Cartographic", "Orbital Mutual", "Pyrite",
                        "Vale Biotech", "Lumen Reach", "Spruce"),
        confuser_predicates=(
            "{name}'s commute time has doubled since moving offices.",
            "{name}'s work laptop was swapped out during the refresh cycle.",
            "{name}'s work calendar is blocked off on Friday afternoons.",
            "{name}'s work benefits include a monthly wellness stipend.",
            "{name}'s work-from-home setup has been upgraded recently.",
            "{name}'s work channels include an active book-club thread.",
            "{name}'s work badge includes RFID for the parking garage.",
        ),
    ),
)

NAMES: tuple[str, ...] = (
    "Alice", "Ben", "Cara", "Dan", "Eli", "Fay", "Gus", "Hana",
    "Ira", "Jade", "Kai", "Leo", "Maya", "Niko", "Ola", "Pim",
    "Quinn", "Ravi", "Sana", "Theo", "Una", "Vik", "Wren", "Xiu",
    "Yara", "Zane", "Ansel", "Blair", "Cleo", "Dara", "Eve", "Finn",
    "Gia", "Holt", "Ines", "Jules", "Koa", "Lila", "Mei", "Noe",
    "Opal", "Piet", "Rafe", "Saba", "Tal", "Umar", "Vera", "Wil",
    "Xia", "Yusuf", "Zoya", "Ari", "Basil", "Caz", "Del", "Enzo",
    "Faye", "Gael", "Henna", "Indra", "Joss", "Kira", "Lars",
    "Meri", "Nala", "Onur", "Priya", "Rhea", "Sven", "Tia",
    "Usha", "Vida", "Wes", "Xan", "Yair", "Zina", "Arlo", "Bex",
    "Celia", "Dov", "Esme", "Flor", "Gaia", "Hugo", "Iris", "Jem",
    "Keira", "Linus", "Mira", "Nils", "Orin", "Pax", "Remi", "Sage",
    "Tove", "Umi", "Val", "Wynn", "Yoshi", "Zara",
)


def _build_scenario(
    num_users: int,
    checkpoints: tuple[int, ...],
    rng: random.Random,
    num_confusers_per_query: int,
) -> dict:
    timeline: list[dict] = []
    queries: list[dict] = []
    confusers: dict[str, list[dict]] = {}

    for user_idx in range(num_users):
        name = NAMES[user_idx % len(NAMES)]
        for arch_idx, arch in enumerate(ARCHETYPES):
            arch_answers = arch.answers
            (answer_orig, answer_new) = arch_answers[user_idx % len(arch_answers)]
            has_update = bool(arch.update_template and answer_new)
            query_id = f"u{user_idx:03d}-{arch.suffix}"
            chunk_prefix = f"u{user_idx:03d}-{arch.suffix}"

            # Original fact at day 0..3
            day_orig = rng.randint(0, 3)
            timeline.append({
                "day": day_orig,
                "type": "fact",
                "id": f"{chunk_prefix}-fact",
                "text": arch.fact_template.format(name=name, answer=answer_orig),
            })

            expected_answer = answer_orig
            resolution = "stable"
            trap = None
            if has_update:
                day_new = rng.randint(day_orig + 2, min(day_orig + 10, 30))
                timeline.append({
                    "day": day_new,
                    "type": "update",
                    "id": f"{chunk_prefix}-update",
                    "text": arch.update_template.format(name=name, answer=answer_new),
                    "supersedes": f"{chunk_prefix}-fact",
                })
                expected_answer = answer_new
                resolution = "contradiction"
                trap = answer_orig

            query_eligibility = [c for c in checkpoints if c >= (day_orig + 1)]
            if has_update:
                # only eligible after the update lands
                query_eligibility = [c for c in checkpoints if c >= (day_new + 1)]
            if not query_eligibility:
                query_eligibility = [checkpoints[-1]]

            queries.append({
                "query_id": query_id,
                "scenario_id": "PLACEHOLDER",
                "question": arch.question_template.format(name=name),
                "expected_answer": expected_answer,
                "reasoning_type": "factual",
                "difficulty": "medium" if has_update else "easy",
                "trap": trap,
                "checkpoint_eligibility": query_eligibility,
                "resolution_type": resolution,
            })

            # Confusers — pick up to num_confusers_per_query predicates,
            # place them across days spanning the eligibility window.
            predicates = list(arch.confuser_predicates)
            rng.shuffle(predicates)
            picks = predicates[:num_confusers_per_query]
            spread_start = day_orig + 1
            spread_end = max(query_eligibility)
            if spread_end <= spread_start:
                spread_end = spread_start + 1
            days = sorted(
                rng.randint(spread_start, spread_end) for _ in picks
            )
            conf_items = []
            for pred, day in zip(picks, days):
                text = pred.format(name=name)
                # Safety: skip if text contains any answer alias
                if any(alias.lower() in text.lower()
                       for alias in arch.answer_aliases):
                    continue
                conf_items.append({"day": day, "text": text})
            if conf_items:
                confusers[query_id] = conf_items

    return {"timeline": timeline, "queries": queries, "confusers": confusers}


def generate(size: str, out_path: Path, seed: int = 42) -> Path:
    checkpoints = (0, 7, 14, 30, 60, 90)
    if size == "small":
        num_users, confusers_per = 4, 8
        scenario_id = "small-v23"
        name = "Small v2.3"
    elif size == "medium":
        num_users, confusers_per = 15, 10
        scenario_id = "medium-v23"
        name = "Medium v2.3"
    elif size == "large":
        num_users, confusers_per = 50, 12
        scenario_id = "large-v23"
        name = "Large v2.3"
    else:
        raise ValueError(f"unknown size: {size}")

    rng = random.Random(seed)
    core = _build_scenario(num_users, checkpoints, rng, confusers_per)

    # Patch scenario_id into queries now that we know it.
    for q in core["queries"]:
        q["scenario_id"] = scenario_id

    obj = {
        "scenario_id": scenario_id,
        "name": name,
        "description": (
            f"Programmatically generated v2.3 scenario, {num_users} users × "
            f"{len(ARCHETYPES)} archetypes × up to {confusers_per} confusers each."
        ),
        "timeline": core["timeline"],
        "queries": core["queries"],
        "confusers": core["confusers"],
    }

    out_path.write_text(json.dumps(obj, indent=2))
    n_timeline = len(obj["timeline"])
    n_queries = len(obj["queries"])
    n_confusers = sum(len(v) for v in obj["confusers"].values())
    print(f"[gen] {size} → {out_path.name}: "
          f"{n_timeline} timeline, {n_queries} queries, {n_confusers} confusers")
    return out_path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--size", choices=["small", "medium", "large", "all"],
                   default="all")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sizes = [args.size] if args.size != "all" else ["small", "medium", "large"]
    for s in sizes:
        generate(s, args.out_dir / f"{s}-v23.json", seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
