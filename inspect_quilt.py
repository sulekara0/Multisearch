import pandas as pd, zipfile, glob, os

print("="*60)
print("1a) CSV YAPISI")
print("="*60)
csv_path = r"C:\Users\Çağrı\Downloads\quilt_1M_lookup.csv"
df = pd.read_csv(csv_path, nrows=5)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 80)
pd.set_option('display.width', 200)
print("Sutunlar:", list(df.columns))
print()
print(df.to_string())
print()
print(df.dtypes)

print()
print("="*60)
print("1b) ZIP ICERIGI (images_part_1.zip - ilk 20)")
print("="*60)
zip_path = r"C:\Users\Çağrı\Downloads\images_part_1.zip"
with zipfile.ZipFile(zip_path, 'r') as z:
    names = z.namelist()
    print(f"Bu ZIP'teki toplam dosya: {len(names)}")
    for n in names[:20]:
        print(" ", n)
    exts = set(n.rsplit('.',1)[-1].lower() for n in names if '.' in n)
    print(f"Uzantilar: {exts}")

print()
print("="*60)
print("1c) TUM ZIP DOSYALARI")
print("="*60)
zips = sorted(glob.glob(r"C:\Users\Çağrı\Downloads\images_part_*.zip"))
print(f"Toplam ZIP: {len(zips)}")
for z in zips:
    print(f"  {os.path.basename(z)}  -  {os.path.getsize(z)/1024/1024:.0f} MB")
