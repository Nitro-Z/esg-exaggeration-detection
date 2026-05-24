import os
import re
import csv
from tqdm import tqdm

# =============================================
# Lexicon — from pipeline_classification.py
# =============================================
exaggeration_dict_tier_1 = [
    "number one", "world-class", "valuable asset", "strive to create", "highest quality",
    "most suitable", "unprecedented", "global leader", "most advanced", "most comprehensive",
    "industry leader", "groundbreaking", "leading company", "non-gmo", "additive-free",
    "leading enterprise", "miracle", "world leader", "well-equipped", "leading provider",
    "pollution-free", "record-breaking", "soar", "sacred", "one and only", "most innovative",
    "most sustainable", "best in class", "exaggerated", "market leader", "unparalleled",
    "constant attention", "globally renowned", "first of its kind", "foolproof", "uncompromising",
    "gold standard", "zero harm", "legendary", "explosive growth", "impeccable", "world-famous",
    "exaggeration", "beyond expectations", "unrivaled", "flawless", "exponential growth",
    "unmatched", "sustainability leader", "most responsible", "leading the world", "fail-safe",
    "error-free", "spotless", "world no.1", "top performer", "omnipotent", "all-time high",
    "far exceeded", "unequivocally", "immortal", "second to none", "world record",
    "one hundred percent", "exaggerate", "zero additives", "top-ranked", "zero impact",
    "peerless", "invincible", "fully circular", "heavenly", "leading corporation", "highest ever",
    "epoch-making", "dominate the market", "unquestionable", "best performer",
    "overwhelming advantage", "godlike", "supernatural", "sweeping the globe", "skyrocket",
    "best ever", "climate champion", "unquestionably", "absolute advantage", "unsurpassed",
    "mythical", "world-changing", "completely circular",
]

exaggeration_dict_tier_2 = [
    "dedicated to", "positive impact", "sustainability vision", "superior", "exceptional",
    "exclusive", "premium", "extremely", "ultimate", "green innovation", "greatly", "super",
    "perfect", "best practices", "vigorous", "landmark", "firmly believe", "huge", "remarkable",
    "thriving", "zero tolerance", "highest standards", "extraordinary", "top-level",
    "industry-leading", "pioneering", "attach great importance", "surge", "unconventional",
    "record high", "transformative", "endless", "clean technology", "remain committed",
    "infinite", "we commit", "exquisite", "historic", "fully committed", "exciting", "giant",
    "we aspire", "supreme", "firmly committed", "we endeavor", "tremendous", "top-tier",
    "working towards", "zero deforestation", "world-leading", "explosive", "proud of",
    "at the forefront", "deeply committed", "incredible", "profound impact", "significant progress",
    "countless", "we pledge", "top-notch", "wholeheartedly", "striving for excellence",
    "state-of-the-art", "take the lead", "actively pursuing", "boundless", "pursuing excellence",
    "unforgettable", "passionate about", "comprehensive approach", "glorious", "ubiquitous",
    "best-in-class", "enormous", "forward-thinking", "important responsibility", "eye-catching",
    "golden age", "sustainability journey", "stirring", "disruptive", "exclusive to", "spike",
    "making progress", "overwhelming", "to the fullest", "undoubtedly", "to the extreme",
    "significant contribution", "integrated approach", "everlasting", "strongly believe",
    "superb", "splendid", "sweep", "magnificent", "mysterious", "spectacular",
    "innovative approach", "unstoppable", "sustainability excellence", "luxurious", "captivating",
    "relentless pursuit", "ingenious", "lasting impact", "top ranking", "meaningful contribution",
    "deeply believe", "steadfastly committed", "dazzling", "great achievements", "fastest growing",
    "making strides", "first-rate", "setting the standard", "real impact", "lead the market",
    "epic", "important mission", "unique charm", "robust framework", "lead the trend",
    "climate-friendly", "worldwide attention", "fearless", "nature-friendly", "full of vitality",
    "leading sustainability", "making a difference", "commit to continuous", "known to all",
    "myriad", "undisputed", "green solution", "ultimate experience", "proud to announce",
    "ahead of the curve", "to the utmost", "far more than", "extremely difficult",
    "unwavering dedication", "transcendent", "substantial improvement", "tremendous impact",
    "thrilling", "irresistible", "immeasurable", "beyond normal", "sustainable solution",
    "driving change", "real difference", "premium grade", "first-mover advantage",
    "transformational impact", "golden era", "recognized leader", "masterful", "unimaginable",
    "undisputed leader", "playing a leading role", "nurturing the future", "sensational",
    "tirelessly working", "gaining momentum", "red-hot", "trailblazing", "game-changing",
    "making more contributions", "indescribable", "truly exceptional", "incomparable",
    "exceedingly", "going above and beyond", "rock-solid", "raising the bar", "highly creative",
    "holistic strategy",
]

scope_amplifiers = [
    "worldwide", "internationally", "globally", "in asia", "in the world", "in the region",
    "across the globe", "across the industry", "in history", "industry-wide",
    "universally", "across all sectors",
]

# =============================================
# Count frequency in corpus_train
# =============================================
CORPUS_DIR = r'C:\Users\Administrator\Desktop\ESG_Project\Dataset\corpus_train'

def tokenize(text):
    return re.findall(r"[0-9]+(?:\.[0-9]+)?%?|[a-zA-Z]+(?:'[a-zA-Z]+)?", text.lower())

def count_term_frequency(term, corpus_dir):
    pattern = r'(?<![a-z])' + re.escape(term) + r'(?![a-z])'
    count = 0
    for fn in os.listdir(corpus_dir):
        if not fn.lower().endswith('.txt'):
            continue
        fpath = os.path.join(corpus_dir, fn)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                text = f.read().lower()
            count += len(re.findall(pattern, text))
        except Exception:
            continue
    return count

# Build rows
rows = []
all_terms = (
    [(t, 'Tier 1') for t in exaggeration_dict_tier_1] +
    [(t, 'Tier 2') for t in exaggeration_dict_tier_2] +
    [(t, 'Scope Amplifier') for t in scope_amplifiers]
)

print(f"Counting frequency for {len(all_terms)} terms in corpus_train...")
for term, tier in tqdm(all_terms):
    freq = count_term_frequency(term, CORPUS_DIR)
    rows.append({'term': term, 'tier': tier, 'frequency_corpus_train': freq})

# Sort by tier then frequency
tier_order = {'Tier 1': 1, 'Tier 2': 2, 'Scope Amplifier': 3}
rows.sort(key=lambda x: (tier_order[x['tier']], -x['frequency_corpus_train']))

# Export CSV
OUT = r'C:\Users\Administrator\Desktop\ESG_Project\lexicon_full.csv'
with open(OUT, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=['term', 'tier', 'frequency_corpus_train'])
    writer.writeheader()
    writer.writerows(rows)

print(f"\n✅ Exported {len(rows)} terms → {OUT}")
print(f"   Tier 1: {len(exaggeration_dict_tier_1)} terms")
print(f"   Tier 2: {len(exaggeration_dict_tier_2)} terms")
print(f"   Scope Amplifiers: {len(scope_amplifiers)} terms")
