import os
import sys
import re
import numpy as np
import pandas as pd
import nltk
import joblib
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, precision_recall_fscore_support,
                              accuracy_score, roc_auc_score, confusion_matrix)
from sklearn.base import clone

nltk.download('punkt')

# =============================================
# 1. PATHS
# =============================================
BASE_DIR    = r'C:\Users\Administrator\Desktop\ESG_Project\Dataset'
TRAIN_DIR   = os.path.join(BASE_DIR, 'corpus_train')
VAL_DIR     = os.path.join(BASE_DIR, 'corpus_val')
TEST_DIR    = os.path.join(BASE_DIR, 'corpus_test')
OUTPUT_DIR  = os.path.join(BASE_DIR, 'Output')
os.makedirs(OUTPUT_DIR, exist_ok=True)  # tự tạo nếu chưa có


# Annotation gold standard (300 câu đã label thủ công)
ANNOTATION_FILE = r'C:\Users\Administrator\Desktop\ESG_Project\Dataset\annotation_task_nam.xlsx'
# =============================================
# 2. DICT V6 FINAL
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

ALL_DICT = list(set(exaggeration_dict_tier_1 + exaggeration_dict_tier_2))

# =============================================
# 3. HELPER FUNCTIONS
# =============================================
def clean_sent(text):
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_valid_sent(sent):
    words = sent.split()
    if len(words) < 5:       return False
    if len(sent) < 20:       return False
    if '....' in sent:       return False
    if sent.isupper():       return False
    return True

def dict_label(sent):
    s = sent.lower()
    for term in ALL_DICT:
        pattern = r'(?<![a-z])' + re.escape(term) + r'(?![a-z])'
        if re.search(pattern, s):
            return 1
    return 0

def tokenize(text):
    txt = str(text).lower()
    return re.findall(r"[0-9]+(?:\.[0-9]+)?%?|[a-zA-Z]+(?:'[a-zA-Z]+)?", txt)

# =============================================
# 4. ĐỌC CORPUS VÀ GÁN NHÃN
# =============================================
def read_corpus(folder, label_source='dict'):
    """Đọc corpus từ folder, trả về DataFrame với sentence và label."""
    records = []
    file_list = [f for f in os.listdir(folder) if f.lower().endswith('.txt')]
    for fn in tqdm(file_list, desc=f"Reading {os.path.basename(folder)}"):
        fpath = os.path.join(folder, fn)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                text = f.read()
        except:
            continue
        for sent in sent_tokenize(text):
            sent = clean_sent(sent.strip())
            if not is_valid_sent(sent):
                continue
            lbl = dict_label(sent)
            records.append({'file': fn, 'sentence': sent, 'label': lbl})
    return pd.DataFrame(records)

print("=" * 60)
print("BƯỚC 1: ĐỌC CORPUS VÀ GÁN NHÃN")
print("=" * 60)

df_train = read_corpus(TRAIN_DIR)
df_val   = read_corpus(VAL_DIR)
df_test  = read_corpus(TEST_DIR)

# Kiểm tra dataset không rỗng và có đủ 2 classes
for name, df in [('TRAIN', df_train), ('VAL', df_val), ('TEST', df_test)]:
    if len(df) == 0:
        print(f"[ERROR] {name} corpus rỗng — kiểm tra lại folder path")
        sys.exit(1)
    if df['label'].nunique() < 2:
        print(f"[WARNING] {name} chỉ có 1 class — model có thể không train được")

for name, df in [('TRAIN', df_train), ('VAL', df_val), ('TEST', df_test)]:
    ratio = df['label'].mean()
    print(f"  {name}: {len(df):,} câu | label=1: {df['label'].sum():,} ({ratio:.1%})")

# Lưu processed data
df_train.to_csv(os.path.join(OUTPUT_DIR, 'processed_train.csv'), index=False, encoding='utf-8-sig')
df_val.to_csv(os.path.join(OUTPUT_DIR, 'processed_val.csv'), index=False, encoding='utf-8-sig')
df_test.to_csv(os.path.join(OUTPUT_DIR, 'processed_test.csv'), index=False, encoding='utf-8-sig')
print("✅ Đã lưu processed CSV")

# =============================================
# 5. TRAIN WORD2VEC TRÊN TOÀN BỘ TRAIN CORPUS
# =============================================
print("\n" + "=" * 60)
print("BƯỚC 2: TRAIN WORD2VEC")
print("=" * 60)

class EpochLogger(CallbackAny2Vec):
    def __init__(self): self.epoch = 0
    def on_epoch_end(self, model):
        print(f"  Epoch {self.epoch} done")
        self.epoch += 1

W2V_SIZE = 100
tokens_train = [tokenize(s) for s in tqdm(df_train['sentence'], desc="Tokenizing for W2V")]

w2v_model = Word2Vec(vector_size=W2V_SIZE, window=5, min_count=2, workers=4, seed=42)
w2v_model.build_vocab(tokens_train)
w2v_model.train(tokens_train, total_examples=w2v_model.corpus_count,
                epochs=5, callbacks=[EpochLogger()])
w2v_model.save(os.path.join(OUTPUT_DIR, 'word2vec_esg.model'))
print("✅ Word2Vec saved")

def get_sent_vec(sent, size=W2V_SIZE):
    tokens = tokenize(sent)
    vecs = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(size)

# =============================================
# 6. FEATURE EXTRACTION
# =============================================
print("\n" + "=" * 60)
print("BƯỚC 3: FEATURE EXTRACTION")
print("=" * 60)

# TF-IDF
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    tokenizer=tokenize,  # ← thêm
    token_pattern=None,  # ← thêm
)
X_train_tfidf = tfidf.fit_transform(df_train['sentence'])
X_val_tfidf   = tfidf.transform(df_val['sentence'])
X_test_tfidf  = tfidf.transform(df_test['sentence'])
print(f"  TF-IDF shape: train={X_train_tfidf.shape}, val={X_val_tfidf.shape}, test={X_test_tfidf.shape}")

# Word2Vec
print("  Building W2V vectors...")
X_train_w2v = np.vstack([get_sent_vec(s) for s in tqdm(df_train['sentence'], desc="Train W2V")])
X_val_w2v   = np.vstack([get_sent_vec(s) for s in tqdm(df_val['sentence'],   desc="Val W2V")])
X_test_w2v  = np.vstack([get_sent_vec(s) for s in tqdm(df_test['sentence'],  desc="Test W2V")])

scaler = StandardScaler()
X_train_w2v_sc = scaler.fit_transform(X_train_w2v)
X_val_w2v_sc   = scaler.transform(X_val_w2v)
X_test_w2v_sc  = scaler.transform(X_test_w2v)
print("  ✅ Features ready")

y_train = df_train['label'].values
y_val   = df_val['label'].values
y_test  = df_test['label'].values

# =============================================
# 7. TRAIN VÀ EVALUATE — 6 MODELS
# =============================================
print("\n" + "=" * 60)
print("BƯỚC 4: TRAIN & EVALUATE 6 MODELS")
print("=" * 60)

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42,
                                              class_weight='balanced'),
    'SVM':                LinearSVC(max_iter=5000, random_state=42,
                                    class_weight='balanced'),
    'RandomForest':       RandomForestClassifier(n_estimators=100, n_jobs=-1,
                                                  random_state=42,
                                                  class_weight='balanced'),
}

features = {
    'TF-IDF':    (X_train_tfidf,    X_val_tfidf,    X_test_tfidf),
    'Word2Vec':  (X_train_w2v_sc,   X_val_w2v_sc,   X_test_w2v_sc),
}

metrics_rows = []
best_f1_val  = 0
best_model_info = None

for feat_name, (Xtr, Xvl, Xte) in features.items():
    for model_name, model in models.items():
        combo = f"{model_name} + {feat_name}"
        print(f"\n▶ {combo}")

        # Clone a fresh model instance for this feature set to avoid cross-run state
        clf = clone(model)
        try:
            clf.fit(Xtr, y_train)
        except Exception as e:
            print(f"[ERROR] training {combo} failed: {e}")
            continue

        # Val
        y_val_pred = clf.predict(Xvl)
        p_val, r_val, f1_val, _ = precision_recall_fscore_support(
            y_val, y_val_pred, average='binary', zero_division=0)
        acc_val = accuracy_score(y_val, y_val_pred)
        print(f"  VAL  — P={p_val:.3f} R={r_val:.3f} F1={f1_val:.3f} Acc={acc_val:.3f}")

        # Test
        y_test_pred = clf.predict(Xte)
        p_te, r_te, f1_te, _ = precision_recall_fscore_support(
            y_test, y_test_pred, average='binary', zero_division=0)
        acc_te = accuracy_score(y_test, y_test_pred)

        # AUC — thử predict_proba trước, fallback decision_function
        try:
            if hasattr(clf, 'predict_proba'):
                scores = clf.predict_proba(Xte)[:, 1]
            elif hasattr(clf, 'decision_function'):
                scores = clf.decision_function(Xte)
            else:
                scores = None
            auc = roc_auc_score(y_test, scores) if scores is not None else None
        except Exception:
            auc = None

        print(f"  TEST — P={p_te:.3f} R={r_te:.3f} F1={f1_te:.3f} Acc={acc_te:.3f}"
              + (f" AUC={auc:.3f}" if auc else ""))
        print(classification_report(y_test, y_test_pred, target_names=['Normal','Exaggeration'],
                                    zero_division=0))

        metrics_rows.append({
            'model': model_name, 'feature': feat_name,
            'val_precision': p_val, 'val_recall': r_val, 'val_f1': f1_val, 'val_acc': acc_val,
            'test_precision': p_te, 'test_recall': r_te, 'test_f1': f1_te, 'test_acc': acc_te,
            'test_auc': auc,
        })

        # Lưu best model theo val F1
        if f1_val > best_f1_val:
            best_f1_val = f1_val
            # store the fitted clone (clf) as the best model
            best_model_info = (combo, clf, feat_name, Xte, y_test)

# =============================================
# 8. BEST MODEL — EVALUATE TRÊN ANNOTATION (300 câu)
# =============================================
print("\n" + "=" * 60)
print("BƯỚC 5: EVALUATE BEST MODEL TRÊN ANNOTATION GOLD STANDARD")
print("=" * 60)

# Guard: kiểm tra có best model không
if best_model_info is None:
    print("[ERROR] Không có model nào train được — kiểm tra lại dataset và labels")
    sys.exit(1)

best_combo, best_model, best_feat, _, _ = best_model_info
print(f"Best model: {best_combo} (Val F1={best_f1_val:.3f})")

if os.path.exists(ANNOTATION_FILE):
    df_ann = pd.read_excel(ANNOTATION_FILE)
    df_ann = df_ann[df_ann['label'].isin([0, 1])].copy()

    if best_feat == 'TF-IDF':
        X_ann = tfidf.transform(df_ann['sentence'])
    else:
        X_ann_w2v = np.vstack([get_sent_vec(s) for s in df_ann['sentence']])
        X_ann = scaler.transform(X_ann_w2v)

    y_ann_true = df_ann['label'].astype(int).values
    y_ann_pred = best_model.predict(X_ann)

    p, r, f1, _ = precision_recall_fscore_support(y_ann_true, y_ann_pred,
                                                    average='binary', zero_division=0)
    DICT_BASELINE_F1 = 0.692
    print(f"\nAnnotation Gold Standard (300 câu):")
    print(f"  Precision={p:.3f} | Recall={r:.3f} | F1={f1:.3f}")
    print(classification_report(y_ann_true, y_ann_pred,
                                  target_names=['Normal','Exaggeration'], zero_division=0))
    print(f"  → Dict baseline F1: {DICT_BASELINE_F1}")
    print(f"  → Best model F1:    {f1:.3f}")
    if f1 > DICT_BASELINE_F1:
        print(f"  → Model outperforms dictionary baseline ✅")
    else:
        print(f"  → Model does not outperform dictionary baseline ⚠️")

    # Lưu kết quả — thêm cột pred vào file Excel
    df_ann['pred'] = y_ann_pred          # ← thêm vào đây
    output_ann = os.path.join(OUTPUT_DIR, 'annotation_evaluated.xlsx')
    df_ann.to_excel(output_ann, index=False)
    print(f"✅ Đã lưu: {output_ann}")

else:
    print(f"  ⚠️ Annotation file không tìm thấy: {ANNOTATION_FILE}")




# =============================================
# 9. SAVE OUTPUTS
# =============================================
print("\n" + "=" * 60)
print("BƯỚC 6: LƯU KẾT QUẢ")
print("=" * 60)

# Metrics
metrics_df = pd.DataFrame(metrics_rows)
metrics_path = os.path.join(OUTPUT_DIR, 'metrics_results.csv')
metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
print(f"✅ Metrics: {metrics_path}")

# Best model
model_path = os.path.join(OUTPUT_DIR, 'best_model.pkl')
joblib.dump(best_model, model_path)
print(f"✅ Best model: {model_path}")

# TF-IDF vectorizer và scaler (cần để predict sau này)
tfidf_path = os.path.join(OUTPUT_DIR, 'tfidf_vectorizer.pkl')
scaler_path = os.path.join(OUTPUT_DIR, 'scaler.pkl')
joblib.dump(tfidf, tfidf_path)
joblib.dump(scaler, scaler_path)
print(f"✅ TF-IDF vectorizer: {tfidf_path}")
print(f"✅ Scaler: {scaler_path}")

# Summary
print("\n" + "=" * 60)
print("TỔNG KẾT 6 MODELS (Test F1):")
print("=" * 60)
metrics_df_sorted = metrics_df.sort_values('test_f1', ascending=False)
for _, row in metrics_df_sorted.iterrows():
    print(f"  {row['model']:25} + {row['feature']:10} "
          f"| Val F1={row['val_f1']:.3f} | Test F1={row['test_f1']:.3f} "
          f"| P={row['test_precision']:.3f} R={row['test_recall']:.3f}")

print(f"\n🏆 Best model: {best_combo} (Val F1={best_f1_val:.3f})")
print("\n✅ PIPELINE HOÀN TẤT")
