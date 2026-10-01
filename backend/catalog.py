"""
Player catalog - the factual layer of the application.

Two groups live here:

1. ``PLAYERS``  - real Indian cricketers. Only public career information is
   stored (roles, debut years, well-documented records).  **This file is metadata
   about people; it deliberately contains no photographs.** You must add your own
   legally usable images to ``dataset/<folder>/`` before the model can recognise
   anyone.

2. ``DEMO_PERSONAS`` - artistically generated, entirely fictional people used by
   the built-in demo dataset. They are labelled as fictional everywhere in the
   UI; they are not real players and must not be presented as such.

Fields are intentionally conservative: anything that could not be verified from
public career records is left as ``None`` and the UI renders it as "—".
"""
from __future__ import annotations

from typing import Dict, List, Optional

SOURCE_NOTE = ("Factual fields are drawn from public career records "
               "(BCCI / ESPNcricinfo-level information) and are provided for the "
               "profile cards only. Always verify before publishing.")

PLAYERS: Dict[str, dict] = {
    "Virat_Kohli": {
        "name": "Virat Kohli",
        "role": "Right-handed top-order batter · former captain (all formats)",
        "batting_style": "Right-handed",
        "bowling_style": "Right-arm medium pace (rarely used)",
        "jersey_number": 18,
        "born": "5 November 1988 · Delhi, India",
        "bio": ("Delhi-born top-order batter who captained India across all three formats "
                "from 2014 to 2022. ODI debut 2008, T20I debut 2010 and Test debut 2011. "
                "Holds the record for the most centuries in ODI cricket and is among the "
                "highest run-scorers in the history of international cricket."),
        "career_highlights": [
            "Most centuries in men's ODI cricket (50)",
            "Padma Shri (2017); Major Dhyan Chand Khel Ratna (2018); Arjuna Award (2013)",
            "Member of India's 2011 ICC Cricket World Cup and 2013 ICC Champions Trophy winning squads",
            "Led India to the 2021 World Test Championship and 2023 ICC Cricket World Cup finals",
        ],
    },
    "Rohit_Sharma": {
        "name": "Rohit Sharma",
        "role": "Right-handed opening batter · captain (ODI)",
        "batting_style": "Right-handed",
        "bowling_style": "Right-arm off-break (occasional)",
        "jersey_number": 45,
        "born": "30 April 1987 · Nagpur, Maharashtra, India",
        "bio": ("Opening batter who captained India to the 2024 ICC Men's T20 World Cup and "
                "the 2025 ICC Champions Trophy titles. ODI and T20I debuts in 2007, Test "
                "debut in 2013. Holds the highest individual score in ODI cricket (264)."),
        "career_highlights": [
            "Highest individual score in ODI cricket: 264 vs Sri Lanka (2014)",
            "Three ODI double centuries, the most by any batter",
            "Captained India to the 2024 T20 World Cup and 2025 Champions Trophy",
            "Major Dhyan Chand Khel Ratna (2020); Arjuna Award (2015)",
        ],
    },
    "Jasprit_Bumrah": {
        "name": "Jasprit Bumrah",
        "role": "Right-arm fast bowler",
        "batting_style": "Right-handed (lower order)",
        "bowling_style": "Right-arm fast",
        "jersey_number": 93,
        "born": "6 December 1993 · Ahmedabad, Gujarat, India",
        "bio": ("Right-arm fast bowler with an unorthodox action, known for yorkers, "
                "deceptive pace and a very low economy at the death. ODI and T20I debuts "
                "in 2016, Test debut in 2018. Widely rated the leading fast bowler of his "
                "generation."),
        "career_highlights": [
            "ICC Men's Cricketer of the Year (Sir Garfield Sobers Trophy) for 2024",
            "ICC Men's Test Cricketer of the Year for 2024",
            "First Indian pace bowler to take a hat-trick in Test cricket (2021 vs West Indies)",
            "Member of India's 2024 T20 World Cup and 2025 Champions Trophy winning squads",
        ],
    },
    "Ravindra_Jadeja": {
        "name": "Ravindra Jadeja",
        "role": "Left-arm spin-bowling all-rounder",
        "batting_style": "Left-handed",
        "bowling_style": "Slow left-arm orthodox",
        "jersey_number": 8,
        "born": "6 December 1988 · Jamnagar, Gujarat, India",
        "bio": ("Left-arm spinner and lower-order batter, and one of the finest fielders of "
                "his era. ODI and T20I debuts in 2009, Test debut in 2012. The only Indian "
                "with three first-class triple centuries."),
        "career_highlights": [
            "Leading wicket-taker of the 2013 ICC Champions Trophy",
            "Member of India's 2011 World Cup, 2013 Champions Trophy, 2024 T20 World Cup and 2025 Champions Trophy winning squads",
            "Three first-class triple centuries",
            "Ranked the No. 1 Test all-rounder during his peak years",
        ],
    },
    "KL_Rahul": {
        "name": "K. L. Rahul",
        "role": "Right-handed top-order batter · wicket-keeper",
        "batting_style": "Right-handed",
        "bowling_style": None,
        "jersey_number": 1,
        "born": "18 April 1992 · Karnataka, India",
        "bio": ("Right-handed top-order batter who also keeps wicket. Test debut in 2014, "
                "ODI debut in 2016 and T20I debut in 2016. Known for textbook timing and "
                "for batting anywhere in the top four."),
        "career_highlights": [
            "First Indian to score a century on ODI debut (100* vs Zimbabwe, 2016)",
            "Century off 79 balls against England (2017), among the fastest Test hundreds by an Indian",
            "Member of India's 2024 T20 World Cup and 2025 Champions Trophy winning squads",
        ],
    },
    "Hardik_Pandya": {
        "name": "Hardik Pandya",
        "role": "Right-handed fast-bowling all-rounder",
        "batting_style": "Right-handed",
        "bowling_style": "Right-arm medium-fast",
        "jersey_number": 33,
        "born": "11 October 1993 · Surat, Gujarat, India",
        "bio": ("Batting all-rounder who bowls in the middle overs and at the death. ODI and "
                "T20I debuts in 2016, Test debut in 2017. Vice-captain of India's 2024 ICC "
                "Men's T20 World Cup winning side."),
        "career_highlights": [
            "Vice-captain of India's 2024 ICC Men's T20 World Cup winning squad",
            "Bowled the final over of the 2024 T20 World Cup final",
            "Captained Gujarat Titans to the IPL title in their debut season (2022)",
        ],
    },
    "Shubman_Gill": {
        "name": "Shubman Gill",
        "role": "Right-handed top-order batter · captain (Test)",
        "batting_style": "Right-handed",
        "bowling_style": "Right-arm off-break (rarely used)",
        "jersey_number": 77,
        "born": "8 September 1999 · Fazilka, Punjab, India",
        "bio": ("Right-handed opening batter who made his Test debut at the MCG in December "
                "2020 and became India's Test captain in 2025. ODI debut in 2019 and T20I "
                "debut in 2023."),
        "career_highlights": [
            "India's Test captain since 2025",
            "Top-scored with 91 at the Gabba in the 2020-21 Border-Gavaskar Trophy",
            "Member of India's 2025 ICC Champions Trophy and 2024 T20 World Cup winning squads",
        ],
    },
    "Rishabh_Pant": {
        "name": "Rishabh Pant",
        "role": "Left-handed wicket-keeper batter",
        "batting_style": "Left-handed",
        "bowling_style": None,
        "jersey_number": 17,
        "born": "4 October 1997 · Roorkee, Uttarakhand, India",
        "bio": ("Wicket-keeper batter whose counter-attacking Test centuries include maiden "
                "hundreds in England and Australia. T20I debut in 2017; Test and ODI debuts "
                "in 2018."),
        "career_highlights": [
            "Test centuries in England, Australia and South Africa",
            "Made his Test debut in 2018 and scored his maiden Test century in England the same year",
            "Member of India's 2024 T20 World Cup and 2025 Champions Trophy winning squads",
        ],
    },
    "Shreyas_Iyer": {
        "name": "Shreyas Iyer",
        "role": "Right-handed middle-order batter",
        "batting_style": "Right-handed",
        "bowling_style": "Right-arm leg-break (occasional)",
        "jersey_number": 41,
        "born": "6 December 1994 · Mumbai, Maharashtra, India",
        "bio": ("Middle-order batter known for playing spin and pace equally well in the "
                "middle overs. ODI and T20I debuts in 2017, Test debut in 2021. Captained "
                "Kolkata Knight Riders to the IPL title in 2024."),
        "career_highlights": [
            "Captained Kolkata Knight Riders to the IPL 2024 title",
            "Member of India's 2023 ICC Cricket World Cup squad",
            "Member of India's 2025 ICC Champions Trophy winning squad",
        ],
    },
    "Mohammed_Siraj": {
        "name": "Mohammed Siraj",
        "role": "Right-arm fast bowler",
        "batting_style": "Right-handed (lower order)",
        "bowling_style": "Right-arm fast-medium",
        "jersey_number": None,
        "born": "13 March 1994 · Hyderabad, Telangana, India",
        "bio": ("Right-arm pace bowler from Hyderabad known for his outswinger and his "
                "aggressive new-ball spells. T20I debut 2017, ODI debut 2019 and Test debut "
                "at the MCG in December 2020."),
        "career_highlights": [
            "Took 5/77 at the Gabba (2021) in his second Test",
            "Member of India's 2024 T20 World Cup and 2025 Champions Trophy winning squads",
            "Has led India's new-ball attack in all three formats",
        ],
    },
    "Axar_Patel": {
        "name": "Axar Patel",
        "role": "Left-arm spin-bowling all-rounder",
        "batting_style": "Left-handed",
        "bowling_style": "Slow left-arm orthodox",
        "jersey_number": None,
        "born": "20 January 1994 · Anand, Gujarat, India",
        "bio": ("Left-arm orthodox spinner and lower-order batter who often bowls the tough "
                "middle overs. ODI debut 2014, T20I debut 2015 and Test debut 2021, where he "
                "took a five-wicket haul in his first Test innings."),
        "career_highlights": [
            "Took 5/60 on Test debut vs England (2021)",
            "Member of India's 2024 T20 World Cup and 2025 Champions Trophy winning squads",
        ],
    },
    "Yuzvendra_Chahal": {
        "name": "Yuzvendra Chahal",
        "role": "Right-arm leg-break bowler",
        "batting_style": "Right-handed (lower order)",
        "bowling_style": "Right-arm leg-break / googly",
        "jersey_number": None,
        "born": "23 July 1990 · Jind, Haryana, India",
        "bio": ("Leg-spinner and India's leading wicket-taker in T20 internationals. T20I "
                "debut 2016 and ODI debut 2017. Known for flight, drift and a well-disguised "
                "googly."),
        "career_highlights": [
            "India's leading wicket-taker in T20 internationals",
            "First Indian to take a six-wicket haul in T20Is (6/25 vs England, 2017)",
        ],
    },
}


# --------------------------------------------------------------------------
# Fictional demo personas (AI-generated faces shipped in dataset_demo/)
# --------------------------------------------------------------------------
DEMO_PERSONAS: Dict[str, dict] = {
    "Demo_Persona_A": {
        "name": "Aadhya Rao (Demo Persona A)",
        "fictional": True,
        "role": "Demo persona · fictional, AI-generated",
        "batting_style": "Fictional record — not a real person",
        "bowling_style": None,
        "jersey_number": None,
        "born": None,
        "bio": ("This person does not exist. The portrait was generated by an AI image "
                "model so the demo dataset contains no real likeness. Use it to explore "
                "the app, then replace dataset_demo/ with your own dataset/ images."),
        "career_highlights": [
            "Fictional persona used only to demonstrate training and identification",
            "No real-world identity, no real career data",
        ],
    },
    "Demo_Persona_B": {
        "name": "Arjun Nair (Demo Persona B)",
        "fictional": True,
        "role": "Demo persona · fictional, AI-generated",
        "batting_style": "Fictional record — not a real person",
        "bowling_style": None,
        "jersey_number": None,
        "born": None,
        "bio": ("This person does not exist. The portrait was generated by an AI image "
                "model so the demo dataset contains no real likeness."),
        "career_highlights": ["Fictional persona used only to demonstrate the pipeline"],
    },
    "Demo_Persona_C": {
        "name": "Vikram Iyer (Demo Persona C)",
        "fictional": True,
        "role": "Demo persona · fictional, AI-generated",
        "batting_style": "Fictional record — not a real person",
        "bowling_style": None,
        "jersey_number": None,
        "born": None,
        "bio": ("This person does not exist. The portrait was generated by an AI image "
                "model so the demo dataset contains no real likeness."),
        "career_highlights": ["Fictional persona used only to demonstrate the pipeline"],
    },
}


def slugify(name: str) -> str:
    return name.strip().lower().replace(" ", "-")


def _entry(folder: str, data: dict, demo: bool = False) -> dict:
    merged = {
        "id": slugify(folder),
        "folder": folder,
        "fictional": bool(data.get("fictional", demo)),
        "source_note": SOURCE_NOTE,
        "dataset_hint": ("demo data — AI generated"
                         if demo else
                         "add your own legally usable images to this folder"),
    }
    merged.update(data)
    for key in ("bowling_style", "jersey_number", "born"):
        merged.setdefault(key, None)
    return merged


ALL_PROFILES: Dict[str, dict] = {
    **{f: _entry(f, d) for f, d in PLAYERS.items()},
    **{f: _entry(f, d, demo=True) for f, d in DEMO_PERSONAS.items()},
}

BY_ID: Dict[str, dict] = {entry["id"]: entry for entry in ALL_PROFILES.values()}


def all_profiles() -> List[dict]:
    """Real players first, then fictional demo personas."""
    real = [_entry(f, d) for f, d in PLAYERS.items()]
    demo = [_entry(f, d, demo=True) for f, d in DEMO_PERSONAS.items()]
    return real + demo


def get_profile(player_id: str) -> Optional[dict]:
    """Look up by slug id ('virat-kohli') or folder name ('Virat_Kohli')."""
    if player_id in ALL_PROFILES:
        return ALL_PROFILES[player_id]
    return BY_ID.get(player_id)


def profile_for_folder(folder: str) -> Optional[dict]:
    return ALL_PROFILES.get(folder)


def catalog_keyed_by_slug() -> Dict[str, dict]:
    """Used by the ML predictor for optional profile enrichment."""
    return dict(BY_ID)


def folders(demo: bool = False) -> List[str]:
    if demo:
        return sorted(DEMO_PERSONAS)
    return sorted(PLAYERS)


def demo_folders() -> List[str]:
    return sorted(DEMO_PERSONAS)
