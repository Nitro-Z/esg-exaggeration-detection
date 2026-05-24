import os
import re
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

nltk.download('punkt')
nltk.download('punkt_tab')

# =============================================
# 1. ĐỌC CORPUS
# =============================================
path_to_corpus = r'C:\Users\Administrator\Desktop\ESG_Project\Dataset\corpus_train'

exaggeration_dict_tier_1 = ["number one", "world-class", "valuable asset", "strive to create", "highest quality", "most suitable", "unprecedented", "global leader", "most advanced", "most comprehensive", "industry leader", "groundbreaking", "leading company", "non-gmo", "additive-free", "leading enterprise", "miracle", "world leader", "well-equipped", "leading provider", "pollution-free", "record-breaking", "soar", "sacred", "one and only", "most innovative", "market leader", "exaggerated", "best in class", "most sustainable", "constant attention", "unparalleled", "globally renowned", "first of its kind", "foolproof", "uncompromising", "gold standard", "zero harm", "legendary", "explosive growth", "impeccable", "exaggeration", "world-famous", "beyond expectations", "unrivaled", "flawless", "exponential growth", "sustainability leader", "unmatched", "most responsible", "leading the world", "error-free", "fail-safe", "spotless", "world no.1", "top performer", "omnipotent", "all-time high", "immortal", "far exceeded", "second to none", "unequivocally", "world record", "fully circular", "heavenly", "one hundred percent", "exaggerate", "zero additives", "top-ranked", "zero impact", "peerless", "invincible", "epoch-making", "dominate the market", "highest ever", "leading corporation", "world-changing", "overwhelming advantage", "godlike", "mythical", "completely circular", "unquestionable", "best performer", "supernatural", "sweeping the globe", "skyrocket", "best ever", "climate champion", "unquestionably", "unsurpassed", "absolute advantage"]

exaggeration_dict_tier_2 = ["dedicated to", "positive impact", "sustainability vision", "superior", "exceptional", "exclusive", "premium", "extremely", "ultimate", "green innovation", "greatly", "super", "perfect", "best practices", "vigorous", "landmark", "firmly believe", "huge", "remarkable", "thriving", "zero tolerance", "highest standards", "extraordinary", "top-level", "industry-leading", "pioneering", "attach great importance", "surge", "unconventional", "record high", "transformative", "endless", "clean technology", "remain committed", "infinite", "we commit", "fully committed", "exquisite", "historic", "exciting", "giant", "we aspire", "supreme", "firmly committed", "we endeavor", "tremendous", "working towards", "top-tier", "zero deforestation", "world-leading", "explosive", "proud of", "at the forefront", "deeply committed", "incredible", "profound impact", "significant progress", "countless", "we pledge", "wholeheartedly", "top-notch", "striving for excellence", "actively pursuing", "state-of-the-art", "take the lead", "boundless", "pursuing excellence", "passionate about", "comprehensive approach", "unforgettable", "glorious", "ubiquitous", "enormous", "best-in-class", "forward-thinking", "important responsibility", "eye-catching", "golden age", "stirring", "sustainability journey", "disruptive", "exclusive to", "spike", "overwhelming", "making progress", "to the fullest", "undoubtedly", "to the extreme", "significant contribution", "integrated approach", "everlasting", "strongly believe", "splendid", "superb", "sweep", "spectacular", "mysterious", "magnificent", "innovative approach", "unstoppable", "sustainability excellence", "relentless pursuit", "captivating", "luxurious", "lasting impact", "ingenious", "top ranking", "steadfastly committed", "dazzling", "deeply believe", "great achievements", "meaningful contribution", "fastest growing", "making strides", "lead the market", "setting the standard", "real impact", "first-rate", "robust framework", "unique charm", "important mission", "epic", "nature-friendly", "fearless", "worldwide attention", "lead the trend", "climate-friendly", "leading sustainability", "making a difference", "full of vitality", "myriad", "commit to continuous", "known to all", "unwavering dedication", "extremely difficult", "far more than", "to the utmost", "ahead of the curve", "proud to announce", "ultimate experience", "green solution", "undisputed", "transcendent", "substantial improvement", "tremendous impact", "thrilling", "irresistible", "immeasurable", "beyond normal", "sustainable solution", "driving change", "real difference", "premium grade", "first-mover advantage", "transformational impact", "golden era", "red-hot", "trailblazing", "sensational", "nurturing the future", "playing a leading role", "undisputed leader", "unimaginable", "masterful", "tirelessly working", "recognized leader", "gaining momentum", "indescribable", "exceedingly", "incomparable", "truly exceptional", "making more contributions", "holistic strategy", "highly creative", "raising the bar", "going above and beyond", "rock-solid", "game-changing"]

scope_amplifiers = ["worldwide", "internationally", "globally", "in asia", "in the world", "in the region", "across the globe", "across the industry", "in history", "industry-wide", "universally", "across all sectors"]

# =============================================
# 2. ĐỌC TOÀN BỘ TEXT
# =============================================
print("Đang đọc corpus...")
all_text = ""
file_list = [fn for fn in os.listdir(path_to_corpus) if fn.lower().endswith('.txt')]
for filename in tqdm(file_list, desc="Reading files"):
    with open(os.path.join(path_to_corpus, filename), 'r', encoding='utf-8') as f:
        all_text += " " + f.read().lower()

print(f"Tổng ký tự: {len(all_text):,}")

# =============================================
# 3. WHOLE-WORD MATCHING
# =============================================
def count_whole_word(text, term):
    pattern = r'(?<![a-z])' + re.escape(term) + r'(?![a-z])'
    return len(re.findall(pattern, text))

# =============================================
# 4. ĐẾM TẦN SUẤT
# =============================================
results = []
all_terms = [
    ("TIER 1", exaggeration_dict_tier_1),
    ("TIER 2", exaggeration_dict_tier_2),
    ("SCOPE AMPLIFIER", scope_amplifiers),
]

for tier, words in all_terms:
    for w in tqdm(words, desc=f"Counting {tier}"):
        count = count_whole_word(all_text, w)
        results.append({'term': w, 'tier': tier, 'count': count})

freq_df = pd.DataFrame(results).sort_values('count', ascending=False)

# =============================================
# 5. BÁO CÁO
# =============================================
t1_match = freq_df[(freq_df['tier'] == 'TIER 1') & (freq_df['count'] > 0)]
t2_match = freq_df[(freq_df['tier'] == 'TIER 2') & (freq_df['count'] > 0)]
t1_zero  = freq_df[(freq_df['tier'] == 'TIER 1') & (freq_df['count'] == 0)]
t2_zero  = freq_df[(freq_df['tier'] == 'TIER 2') & (freq_df['count'] == 0)]

print("\n📊 TỔNG QUAN (whole-word matching):")
print(f"  Tier 1 — có match: {len(t1_match)}/{len(exaggeration_dict_tier_1)}")
print(f"  Tier 2 — có match: {len(t2_match)}/{len(exaggeration_dict_tier_2)}")

print(f"\n🔴 TIER 1 — KHÔNG MATCH LẦN NÀO ({len(t1_zero)} terms):")
print(t1_zero['term'].tolist())

print(f"\n🟡 TIER 2 — KHÔNG MATCH LẦN NÀO ({len(t2_zero)} terms):")
print(t2_zero['term'].tolist())

print("\n🏆 TOP 30 TERMS HAY MATCH NHẤT:")
print(freq_df.head(30).to_string(index=False))


# =============================================
# TÍNH LABEL=1 RATIO
# =============================================
print("\nĐang tính label=1 ratio (cần chia câu)...")
from nltk.tokenize import sent_tokenize

all_dict = exaggeration_dict_tier_1 + exaggeration_dict_tier_2

sentences = []
for filename in tqdm(file_list, desc="Tokenizing"):
    fpath = os.path.join(path_to_corpus, filename)
    with open(fpath, 'r', encoding='utf-8') as f:
        text = f.read().lower()
        for sent in sent_tokenize(text):
            sent = sent.strip()
            if len(sent) < 20: continue
            if "...." in sent: continue
            if len(sent.split()) < 5: continue
            sentences.append(sent)

total = len(sentences)
matched = sum(1 for s in sentences if any(
    re.search(r'(?<![a-z])' + re.escape(w) + r'(?![a-z])', s)
    for w in all_dict
))

ratio = matched / total if total > 0 else 0
print(f"\n📈 Tổng câu: {total:,}")
print(f"📈 Câu có label=1: {matched:,}")
print(f"📈 Label=1 ratio: {ratio:.1%}")
print(f"   → Lý tưởng: 15–30%")
print(f"   → Nếu >50%: dict còn noise")
print(f"   → Nếu <10%: dict quá strict, cần thêm từ")

# =============================================
# 6. XUẤT CSV
# =============================================

# File 1: Toàn bộ kết quả (có cả count=0)
freq_df.to_csv('frequency_check_full.csv', index=False, encoding='utf-8-sig')
print(f"\n✅ Đã xuất: frequency_check_full.csv (toàn bộ {len(freq_df)} terms)")

# File 2: Chỉ những từ CÓ XUẤT HIỆN (count > 0) — dùng làm dict sạch
matched_df = freq_df[freq_df['count'] > 0].sort_values(['tier', 'count'], ascending=[True, False])
matched_df.to_csv('frequency_check_matched.csv', index=False, encoding='utf-8-sig')
print(f"✅ Đã xuất: frequency_check_matched.csv ({len(matched_df)} terms có match)")

# File 3: Chỉ những từ KHÔNG XUẤT HIỆN (count = 0) — danh sách cần xem xét bỏ
zero_df = freq_df[freq_df['count'] == 0].sort_values('tier')
zero_df.to_csv('frequency_check_zero.csv', index=False, encoding='utf-8-sig')
print(f"✅ Đã xuất: frequency_check_zero.csv ({len(zero_df)} terms count=0)")

# In tóm tắt
print(f"\n📋 TÓM TẮT 3 FILE XUẤT:")
print(f"  frequency_check_full.csv    → toàn bộ dict + count")
print(f"  frequency_check_matched.csv → chỉ từ có match → dùng làm dict final")
print(f"  frequency_check_zero.csv    → từ không match  → xem xét bỏ")

