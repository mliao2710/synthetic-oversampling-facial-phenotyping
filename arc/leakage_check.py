from datasets import load_dataset
from pathlib import Path
from PIL import Image
import imagehash, hashlib, io, csv
from collections import Counter

REAL_ROOT = Path("/mnt/isilon/wang_lab/liaom/projects/data/gmdb_crops")
OUT = Path("leakage_analysis")
OUT.mkdir(exist_ok=True)

print("Loading datasets...")
gmdb = load_dataset("aaronwzl/gmdb_ten_disease_subset")
pdidb = load_dataset("GestaltGANwglabshare/PDIDB")

ev = gmdb["test"]
syn = pdidb["train"]

print(f"GMDB evaluation: {len(ev)}")
print(f"PDIDB synthetic: {len(syn)}")

assert len(ev) == 558
assert len(syn) == 1727

def real_path(row):
    # crop_path is e.g. gmdb_crops/2567_aligned.jpg
    p = Path(str(row["crop_path"]))
    candidates = [
        REAL_ROOT / p.name,
        REAL_ROOT.parent / p,
    ]
    for x in candidates:
        if x.exists():
            return x
    raise FileNotFoundError(f"Cannot find {row['image_id']}: {candidates}")

def canonical_sha(img):
    # Pixel-level hash after standardized RGB conversion.
    # Avoids missing identical pixels solely because of JPEG/PNG encoding.
    img = img.convert("RGB")
    return hashlib.sha256(
        img.size[0].to_bytes(4,"big") +
        img.size[1].to_bytes(4,"big") +
        img.tobytes()
    ).hexdigest()

print("Computing GMDB hashes...")
real_records = []
for i, r in enumerate(ev):
    p = real_path(r)
    with Image.open(p) as im:
        im = im.convert("RGB")
        real_records.append({
            "index": i,
            "id": str(r["image_id"]),
            "disease": str(r["disease"]),
            "sha": canonical_sha(im),
            "phash": imagehash.phash(im)
        })

print("Computing PDIDB hashes...")
syn_records = []
for i, r in enumerate(syn):
    im = r["image"].convert("RGB")
    syn_records.append({
        "index": i,
        "id": str(r["image_id"]),
        "disease": str(r["disease"]),
        "sha": canonical_sha(im),
        "phash": imagehash.phash(im)
    })

# Exact pixel duplicates
real_sha = {}
for r in real_records:
    real_sha.setdefault(r["sha"], []).append(r)

exact = []
for s in syn_records:
    for r in real_sha.get(s["sha"], []):
        exact.append((s,r))

print("\n" + "="*70)
print("EXACT PIXEL DUPLICATES")
print("="*70)
print("Count:", len(exact))
for s,r in exact[:20]:
    print(s["id"], "<->", r["id"], "|", s["disease"], "|", r["disease"])

# Exhaustive pHash comparison: 1727 x 558 is tiny
print("\nComputing all perceptual-hash distances...")
pairs = []
for s in syn_records:
    for r in real_records:
        d = s["phash"] - r["phash"]
        pairs.append((
            d,
            s["id"], r["id"],
            s["disease"], r["disease"],
            s["index"], r["index"]
        ))

pairs.sort(key=lambda x:x[0])

print("\n" + "="*70)
print("TOP 30 CLOSEST SYNTHETIC <-> EVALUATION PAIRS BY pHash")
print("="*70)

for x in pairs[:30]:
    print(
        f"d={x[0]:2d} | SYN {x[1]:20s} | EVAL {x[2]:8s} | "
        f"{x[3]} -> {x[4]}"
    )

for threshold in [0,2,4,6,8,10]:
    n = sum(x[0] <= threshold for x in pairs)
    print(f"Pairs with pHash distance <= {threshold}: {n}")

with open(OUT/"phash_top_pairs.csv","w",newline="") as f:
    w=csv.writer(f)
    w.writerow([
        "phash_distance","synthetic_id","eval_id",
        "synthetic_disease","eval_disease",
        "synthetic_index","eval_index"
    ])
    w.writerows(pairs[:200])

print("\nSaved:", OUT/"phash_top_pairs.csv")
