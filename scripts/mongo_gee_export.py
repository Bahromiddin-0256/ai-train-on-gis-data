"""
export_to_gee.py
MongoDB bugdoy (wheat) ma'lumotlarini GEE Asset uchun GeoJSON ga export qilish
Andijan viloyati, crop_year=2025
"""

from pymongo import MongoClient
import json
from shapely import wkb, wkt
from shapely.geometry import mapping, shape
import binascii

MONGO_URI  = "mongodb://localhost:27019"
DB_NAME    = "gis-census"
COLLECTION = "uzcosmos_flats"

client = MongoClient(MONGO_URI)
db     = client[DB_NAME]
col    = db[COLLECTION]

# ─────────────────────────────────────────────────────────────
# DEBUG: bitta document ko'rish — geom_2 qanday format ekanligini aniqlash
# ─────────────────────────────────────────────────────────────
sample = col.find_one({"viloyat": "Andijon viloyati", "crop_type": "bugdoy"})
if sample:
    g = sample.get("geom_2", "")
    print("=== geom_2 DEBUG ===")
    print(f"  type(geom_2) : {type(g)}")
    print(f"  value[:120]  : {str(g)[:120]}")
    print(f"  len          : {len(str(g))}")
    print("====================\n")

# ─────────────────────────────────────────────────────────────
# Universal geometry parser — WKB hex / WKT / GeoJSON dict hammasini qabul qiladi
# ─────────────────────────────────────────────────────────────
def parse_geom(g):
    if g is None:
        return None

    # 1. Agar allaqachon dict (GeoJSON) bo'lsa
    if isinstance(g, dict):
        try:
            return mapping(shape(g))
        except Exception as e:
            print(f"  GeoJSON parse xatosi: {e}")
            return None

    g_str = str(g).strip()

    # 2. WKT format: "MULTIPOLYGON (((...)))" yoki "POLYGON ((...))"
    if g_str.upper().startswith(("MULTIPOLYGON", "POLYGON", "POINT", "LINESTRING")):
        try:
            return mapping(wkt.loads(g_str))
        except Exception as e:
            print(f"  WKT parse xatosi: {e}")
            return None

    # 3. EWKT format: "SRID=4326;MULTIPOLYGON (((...)))"
    if ";" in g_str and g_str.upper().startswith("SRID"):
        try:
            geom_part = g_str.split(";", 1)[1]
            return mapping(wkt.loads(geom_part))
        except Exception as e:
            print(f"  EWKT parse xatosi: {e}")
            return None

    # 4. WKB hex string (PostGIS binary)
    hex_clean = g_str.replace(" ", "")
    # SRID prefix bo'lsa olib tashlash (birinchi 10 char = "0106000020" + SRID)
    try:
        raw = binascii.unhexlify(hex_clean)
        geom = wkb.loads(raw, hex=False)
        return mapping(geom)
    except Exception:
        pass

    # 5. WKB with SRID stripped (skip first 4 bytes = type flag + SRID)
    try:
        # Try loading directly as hex string (shapely can do this)
        geom = wkb.loads(hex_clean, hex=True)
        return mapping(geom)
    except Exception as e:
        print(f"  WKB hex parse xatosi: {e} | sample: {g_str[:40]}")
        return None

# ─────────────────────────────────────────────────────────────
# Query va export
# ─────────────────────────────────────────────────────────────
def export_collection(query, label, limit=None):
    cursor = col.find(query, {
        "_id": 0, "id": 1, "fid": 1, "crop_type": 1, "crop_year": 1,
        "tuman": 1, "tuman_code": 1, "viloyat": 1, "maydon": 1,
        "gis_area": 1, "geom_2": 1
    })
    if limit:
        cursor = cursor.limit(limit)

    features, skipped = [], 0
    for doc in cursor:
        geom = parse_geom(doc.get("geom_2"))
        if geom is None:
            skipped += 1
            continue
        features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": {
                "label":      label,
                "crop_type":  doc.get("crop_type", ""),
                "crop_year":  int(doc.get("crop_year") or 0),
                "tuman":      doc.get("tuman", ""),
                "tuman_code": int(doc.get("tuman_code") or 0),
                "viloyat":    doc.get("viloyat", ""),
                "gis_area":   float(doc.get("gis_area") or 0),
                "field_id":   str(doc.get("id", "")),
            }
        })
    print(f"  label={label} → {len(features)} OK, {skipped} o'tkazildi")
    return features

print("Bugdoy dalalari yuklanmoqda...")
wheat_features = export_collection(
    {"viloyat": "Andijon viloyati", "crop_type": "bugdoy", "crop_year": 2025},
    label=1,
    limit=1000
)

print("Non-wheat dalalari yuklanmoqda...")
nonwheat_features = export_collection(
    {"viloyat": "Andijon viloyati", "crop_year": 2025, "crop_type": {"$ne": "bugdoy"}},
    label=0,
    limit=500
)

# ─────────────────────────────────────────────────────────────
# Natija nol bo'lsa — geom field nomini tekshirish
# ─────────────────────────────────────────────────────────────
if len(wheat_features) == 0:
    print("\n⚠️  Hech narsa export bo'lmadi!")
    print("   Barcha field nomlarini ko'rish:")
    s = col.find_one({"viloyat": "Andijon viloyati"})
    if s:
        for k, v in s.items():
            print(f"   {k}: {str(v)[:80]}")

# ─────────────────────────────────────────────────────────────
# GeoJSON fayllarni saqlash
# ─────────────────────────────────────────────────────────────
all_features = wheat_features + nonwheat_features

for fname, feats in [
    ("andijan_wheat_2025.geojson",            wheat_features),
    ("andijan_nonwheat_2025.geojson",          nonwheat_features),
    ("andijan_training_combined_2025.geojson", all_features),
]:
    with open(fname, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats},
                  f, ensure_ascii=False, indent=2)
    print(f"✅ {fname} → {len(feats)} feature")

print(f"\nJami: Wheat={len(wheat_features)}, Non-wheat={len(nonwheat_features)}")

client.close()

import geemap
import ee

try:
    ee.Initialize(project="my-agro-research")
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project="my-agro-research")

# Convert GeoJSON to EE FeatureCollection
ee_object = geemap.geojson_to_ee('andijan_wheat_2025.geojson')

# Optional: Export it to your GEE Assets
exportTask = ee.batch.Export.table.toAsset(
    collection=ee_object,
    description='upload_geojson_asset',
    assetId='users/my-agro-research/andijan_wheat_2025'
)
exportTask.start()