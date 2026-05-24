import os
import random
import shutil

# =============================================
# CẤU HÌNH
# =============================================
source_dir = r'C:\Users\Administrator\Desktop\ESG_Project\pdf_texts_2'
output_dir = r'C:\Users\Administrator\Desktop\ESG_Project'

train_ratio = 0.70
val_ratio   = 0.15
test_ratio  = 0.15  # phần còn lại

SEED = 42  # giữ nguyên seed để reproduce được kết quả

# =============================================
# TẠO FOLDER
# =============================================
train_dir = os.path.join(output_dir, 'corpus_train')
val_dir   = os.path.join(output_dir, 'corpus_val')
test_dir  = os.path.join(output_dir, 'corpus_test')

for d in [train_dir, val_dir, test_dir]:
    os.makedirs(d, exist_ok=True)

# =============================================
# LẤY DANH SÁCH FILE VÀ SHUFFLE
# =============================================
all_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.txt')]
print(f"Tổng số file: {len(all_files)}")

random.seed(SEED)
random.shuffle(all_files)  # shuffle IN-PLACE, hoàn toàn random

# =============================================
# TÍNH SỐ LƯỢNG
# =============================================
n_total = len(all_files)
n_train = int(n_total * train_ratio)
n_val   = int(n_total * val_ratio)
n_test  = n_total - n_train - n_val  # phần còn lại tránh lệch do làm tròn

train_files = all_files[:n_train]
val_files   = all_files[n_train:n_train + n_val]
test_files  = all_files[n_train + n_val:]

print(f"\nSau khi chia:")
print(f"  Train: {len(train_files)} files ({len(train_files)/n_total:.1%})")
print(f"  Val:   {len(val_files)} files ({len(val_files)/n_total:.1%})")
print(f"  Test:  {len(test_files)} files ({len(test_files)/n_total:.1%})")

# =============================================
# COPY FILE
# =============================================
def copy_files(file_list, src, dst, label):
    print(f"\nĐang copy {label}...")
    for fn in file_list:
        shutil.copy2(os.path.join(src, fn), os.path.join(dst, fn))
    print(f"  ✅ Xong {len(file_list)} files → {dst}")

copy_files(train_files, source_dir, train_dir, "TRAIN")
copy_files(val_files,   source_dir, val_dir,   "VAL")
copy_files(test_files,  source_dir, test_dir,  "TEST")

# =============================================
# LƯU MANIFEST — biết file nào thuộc tập nào
# =============================================
import csv
manifest_path = os.path.join(output_dir, 'split_manifest.csv')
with open(manifest_path, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'split'])
    for fn in train_files: writer.writerow([fn, 'train'])
    for fn in val_files:   writer.writerow([fn, 'val'])
    for fn in test_files:  writer.writerow([fn, 'test'])

print(f"\n📋 Manifest lưu tại: {manifest_path}")
print(f"   (Dùng file này để cite trong thesis: 'split was performed with seed=42')")
print(f"\n✅ HOÀN TẤT!")
print(f"   Train: {train_dir}")
print(f"   Val:   {val_dir}")
print(f"   Test:  {test_dir}")