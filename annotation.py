import os, random, shutil

val_dir        = r'C:\Users\Administrator\Desktop\ESG_Project\Dataset\corpus_val'
annotation_dir = r'C:\Users\Administrator\Desktop\ESG_Project\Dataset\annotation_sample'

os.makedirs(annotation_dir, exist_ok=True)

all_val_files = [f for f in os.listdir(val_dir) if f.endswith('.txt')]
random.seed(42)
random.shuffle(all_val_files)

move_files = all_val_files[:50]

for fn in move_files:
    shutil.move(
        os.path.join(val_dir, fn),
        os.path.join(annotation_dir, fn)
    )

print(f"✅ Đã move {len(move_files)} files → annotation_sample/")
print(f"   corpus_val còn lại: {len(all_val_files) - len(move_files)} files")