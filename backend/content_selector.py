"""
content_selector.py — Rule-based contextual content selection
Replaces contextual bandit temporarily for demo stability.
"""

CONTENT_RULES = {
    "Happy":    {"mode": "visual",  "difficulty": "Medium", "title": "Visual Flashcards",   "desc": "Bright image cards with shape & colour matching. Reward sounds on correct answers."},
    "Neutral":  {"mode": "visual",  "difficulty": "Medium", "title": "Reading & Matching",  "desc": "Letter and word matching tasks. Standard calm pacing."},
    "Surprise": {"mode": "visual",  "difficulty": "Medium", "title": "Interactive Story",   "desc": "Attention re-engaged! Interactive story with comprehension question."},
    "Sad":      {"mode": "calm",    "difficulty": "Easy",   "title": "Calm Mode — Colours", "desc": "Soothing colour identification. Gentle audio and encouragement messages."},
    "Fear":     {"mode": "calm",    "difficulty": "Easy",   "title": "Calm Mode — Shapes",  "desc": "Low-stimulation content. Soft colours, slow pace, no sudden sounds."},
    "Angry":    {"mode": "break",   "difficulty": "Easy",   "title": "Sensory Break",        "desc": "2-minute breathing break. Soft animation guide to calm the child."},
    "Disgust":  {"mode": "calm",    "difficulty": "Easy",   "title": "Calm Audio Prompt",    "desc": "Gentle audio narration with simple tap tasks. Reduced visual load."},
}


def select_content(emotion: str) -> dict:
    """Return content recommendation dict for a given emotion string."""
    return CONTENT_RULES.get(emotion, CONTENT_RULES["Neutral"])


def get_attention_score(emotion: str) -> int:
    """Map emotion to attention score 0-100."""
    ATTENTION_MAP = {
        "Happy": 90, "Neutral": 78, "Surprise": 72,
        "Sad": 40,   "Fear": 30,    "Angry": 25, "Disgust": 20,
    }
    return ATTENTION_MAP.get(emotion, 50)


if __name__ == "__main__":
    for emo in CONTENT_RULES:
        content = select_content(emo)
        attn    = get_attention_score(emo)
        print(f"{emo:10s} → {content['mode']:8s} | difficulty={content['difficulty']:6s} | attn={attn}%")
