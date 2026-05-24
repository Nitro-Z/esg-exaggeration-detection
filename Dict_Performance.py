import pandas as pd
import re
from sklearn.metrics import precision_recall_fscore_support, classification_report

ANNOTATION_FILE = r'C:\Users\Administrator\Desktop\ESG_Project\Dataset\annotation_task_nam.xlsx'

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

ALL_DICT = list(set(exaggeration_dict_tier_1 + exaggeration_dict_tier_2))

def dict_predict(sentence):
    s = str(sentence).lower()
    for term in ALL_DICT:
        pattern = r'(?<![a-z])' + re.escape(term) + r'(?![a-z])'
        if re.search(pattern, s):
            return 1
    return 0

def get_matched_terms(sentence):
    s = str(sentence).lower()
    matched = []
    for term in ALL_DICT:
        pattern = r'(?<![a-z])' + re.escape(term) + r'(?![a-z])'
        if re.search(pattern, s):
            matched.append(term)
    return matched

# Đọc annotation
df = pd.read_excel(ANNOTATION_FILE)
df = df[df['label'].isin([0, 1])].copy()
df['dict_pred']     = df['sentence'].apply(dict_predict)
df['matched_terms'] = df['sentence'].apply(get_matched_terms)

y_true = df['label'].astype(int).values
y_pred = df['dict_pred'].astype(int).values

p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)

print(f"{'='*50}")
print(f"DICT PERFORMANCE vs HUMAN LABELS")
print(f"{'='*50}")
print(f"Precision : {p:.3f}")
print(f"Recall    : {r:.3f}")
print(f"F1        : {f1:.3f}")
print(f"\n{classification_report(y_true, y_pred, target_names=['Normal','Exaggeration'], zero_division=0)}")

# Chi tiết
tp = df[(y_true==1) & (y_pred==1)]
fn = df[(y_true==1) & (y_pred==0)]
fp = df[(y_true==0) & (y_pred==1)]

print(f"TP={len(tp)} | FN={len(fn)} | FP={len(fp)}")
print(f"\nFALSE NEGATIVE — dict bỏ sót:")
for _, row in fn.iterrows():
    print(f"  [{row['id']}] {str(row['sentence'])[:80]}...")

print(f"\nFALSE POSITIVE — dict báo nhầm:")
for _, row in fp.iterrows():
    print(f"  [{row['id']}] {str(row['sentence'])[:80]}... | terms: {row['matched_terms']}")