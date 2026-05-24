import pandas as pd

# Đọc file
df = pd.read_csv(
    'vader_lexicon.txt',
    sep='\t',
    header=None,
    names=['word', 'valence', 'std', 'raw']
)

# Chỉ lấy cột word và valence
df = df[['word', 'valence']].copy()
df['valence'] = pd.to_numeric(df['valence'], errors='coerce')

# Filter theo ngưỡng
# Valence >= 2.5 → từ tích cực mạnh (overclaim positive)
# Valence <= -2.5 → từ tiêu cực mạnh (có thể bỏ)
positive_strong = df[df['valence'] >= 2.5].sort_values('valence', ascending=False)
negative_strong = df[df['valence'] <= -2.5].sort_values('valence')

print(f"Từ tích cực mạnh (>=2.5): {len(positive_strong)}")
print(f"Từ tiêu cực mạnh (<=-2.5): {len(negative_strong)}")
print(positive_strong.head(20))

# Xuất Excel để review thủ công
with pd.ExcelWriter('vader_filtered.xlsx') as writer:
    positive_strong.to_excel(writer, sheet_name='Positive_Strong', index=False)
    negative_strong.to_excel(writer, sheet_name='Negative_Strong', index=False)
    df.to_excel(writer, sheet_name='All', index=False)

print("✅ Xuất xong: vader_filtered.xlsx")