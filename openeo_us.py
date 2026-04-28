"""
WorldCereal Crop Map — Andijan viloyati, Asaka tumani, O'zbekiston
=================================================================
CDSE (Copernicus Data Space Ecosystem) + openEO orqali:
  - Cropland extent (doimiy ekin yer xaritasi)
  - Winter cereals (bug'doy/arpa/javdar) xaritasi

Talablar:
    pip install worldcereal openeo openeo-gfmap geopandas shapely rasterio matplotlib

CDSE bepul akkaunt kerak: https://dataspace.copernicus.eu/
  → Ro'yxatdan o'tgach 10,000 bepul processing credits beriladi (har oy yangilanadi)
"""

import os
import json
from pathlib import Path
from datetime import datetime

import geopandas as gpd
import openeo
from dotenv import load_dotenv
from openeo_gfmap import BoundingBoxExtent, TemporalContext
from shapely.geometry import box

load_dotenv(Path(__file__).resolve().parent / ".env")

# ─────────────────────────────────────────────────────────────────────────────
# 1. AOI — Andijan viloyati, Asaka tumani
# ─────────────────────────────────────────────────────────────────────────────
# Asaka tumani taxminiy chegarasi (EPSG:4326)
# To'g'riroq chegara uchun o'zingizning shapefile'ingizni ishlatishingiz mumkin.
ASAKA_BBOX = (69.55, 40.55, 70.00, 40.90)   # (minx, miny, maxx, maxy)

# Butun Andijan viloyati uchun kattaroq bbox:
ANDIJAN_BBOX = (69.40, 40.40, 70.50, 41.00)

# Qaysi AOI ishlatishni tanlang:
USE_BBOX = ASAKA_BBOX          # yoki ANDIJAN_BBOX

OUTPUT_DIR = Path("./worldcereal_output_uzbekistan")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. Vaqt oralig'i
#    WorldCereal cropland = to'liq bir yil (masalan, 2023-11-01 → 2024-10-31)
#    Winter cereals = bug'doy/arpa yig'im-terimi bo'lgan mavsum
# ─────────────────────────────────────────────────────────────────────────────
# O'zbekistonda bug'doy — bahor yig'im (aprel–iyun mavsumidir).
# WorldCereal "winter cereals" produkti uchun standart temporal_extent:
TEMPORAL_EXTENT = TemporalContext(
    start_date="2024-11-01",  # 2025 bug'doy mavsumi boshlanishi (kuzgi ekish)
    end_date="2025-10-31",    # 2025 mavsumining tugashi
)

# ─────────────────────────────────────────────────────────────────────────────
# Mongo (uzcosmos) — bitta bug'doy polygon
# ─────────────────────────────────────────────────────────────────────────────
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27019")
MONGO_DB = os.environ.get("MONGO_DB", "gis-census")
MONGO_COLLECTION = os.environ.get("MONGO_COLLECTION", "uzcosmos_flats")

# ─────────────────────────────────────────────────────────────────────────────
# 3. GeoDataFrame — AOI
# ─────────────────────────────────────────────────────────────────────────────
def build_aoi_gdf(bbox: tuple) -> gpd.GeoDataFrame:
    """bbox (minx, miny, maxx, maxy) dan GeoDataFrame yasaydi."""
    minx, miny, maxx, maxy = bbox
    geom = box(minx, miny, maxx, maxy)
    gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
    gdf["id"] = "asaka_andijan_uz"
    return gdf


def load_one_bugdoy_from_mongo() -> gpd.GeoDataFrame:
    """uzcosmos_flats dan crop_type=bugdoy bo'lgan bitta poligon oladi."""
    from pymongo import MongoClient
    from shapely import wkt

    print(f"🗄️  Mongo: {MONGO_URI} / {MONGO_DB}.{MONGO_COLLECTION}")
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5_000)
    col = client[MONGO_DB][MONGO_COLLECTION]
    doc = col.find_one(
        {"crop_type": {"$regex": "^bugdoy", "$options": "i"}},
        {"_id": 1, "crop_type": 1, "geom_2": 1, "tuman": 1, "viloyat": 1},
    )
    if doc is None:
        raise RuntimeError("uzcosmos_flats da bugdoy poligoni topilmadi.")

    geom = wkt.loads(doc["geom_2"])
    gdf = gpd.GeoDataFrame(
        [{
            "id": str(doc["_id"]),
            "crop_type": doc.get("crop_type"),
            "tuman": doc.get("tuman"),
            "viloyat": doc.get("viloyat"),
        }],
        geometry=[geom],
        crs="EPSG:4326",
    )
    print(f"   _id: {doc['_id']}  crop_type: {doc.get('crop_type')}")
    print(f"   joy: {doc.get('viloyat')} / {doc.get('tuman')}")
    print(f"   bbox: {tuple(round(x, 5) for x in gdf.total_bounds)}")
    return gdf

# Ixtiyoriy: o'z shapefile'ingizni yuklash
def load_custom_shapefile(path: str) -> gpd.GeoDataFrame:
    """Mahalliy shapefile yoki GeoJSON yuklaydi."""
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    gdf["id"] = gdf.index.astype(str)
    return gdf

# ─────────────────────────────────────────────────────────────────────────────
# 4. CDSE ulanish va autentifikatsiya
# ─────────────────────────────────────────────────────────────────────────────
def connect_cdse() -> openeo.Connection:
    """CDSE openEO backendiga ulanadi va autentifikatsiya qiladi."""
    print("🔌 CDSE openEO backendiga ulanilmoqda...")
    conn = openeo.connect("https://openeo.dataspace.copernicus.eu")

    # Autentifikatsiya usuli:
    # 1) OIDC (brauzer orqali) — interaktiv muhit uchun
    # 2) Basic (username/password) — skript/server uchun
    auth_method = os.environ.get("CDSE_AUTH_METHOD", "oidc")

    if auth_method == "basic":
        username = os.environ.get("SENTINEL_USER") or os.environ.get("CDSE_USERNAME")
        password = os.environ.get("SENTINEL_PASSWORD") or os.environ.get("CDSE_PASSWORD")
        if not username or not password:
            raise RuntimeError(
                "CDSE_AUTH_METHOD=basic uchun .env faylida SENTINEL_USER va "
                "SENTINEL_PASSWORD ko'rsatilishi kerak."
            )
        conn.authenticate_basic(username, password)
    else:
        # OIDC device code flow — brauzer ochilmasa ham ishlaydi
        conn.authenticate_oidc()

    account = conn.describe_account() or {}
    print(f"✅ Ulandi: {account.get('name', account.get('user_id', 'unknown'))}")
    return conn

# ─────────────────────────────────────────────────────────────────────────────
# 5a. worldcereal Python paketi orqali (tavsiya etiladi)
# ─────────────────────────────────────────────────────────────────────────────
def run_worldcereal_package(
    aoi_gdf: gpd.GeoDataFrame,
    conn: openeo.Connection,
    pad_m: float = 320.0,
):
    """
    worldcereal 2.6.1 API orqali bitta poligon ustida cropland + croptype
    inferens jobini ishga tushiradi (CDSE).

    pad_m — poligon atrofiga qo'shiladigan bufer (metr). WorldCereal tile_size
    minimal 128 piksel (1.28 km). Kichik dalalar uchun bbox kengaytiriladi.
    """
    from worldcereal.job import (
        DEFAULT_INFERENCE_JOB_OPTIONS,
        WorldCerealProductType,
        create_inference_process_graph,
    )
    from worldcereal.openeo.workflow_config import build_config_from_params

    print("\n🌾 WorldCereal paketi orqali ish boshlanmoqda (CDSE inference)...")

    # UTM ga o'tkazib bbox ni padding qilamiz, keyin EPSG:4326 ga qaytarmaymiz
    # (BoundingBoxExtent UTM EPSG bilan ishlay oladi)
    utm_gdf = aoi_gdf.to_crs(aoi_gdf.estimate_utm_crs())
    minx, miny, maxx, maxy = utm_gdf.total_bounds
    minx -= pad_m; miny -= pad_m; maxx += pad_m; maxy += pad_m
    epsg_utm = int(utm_gdf.crs.to_epsg())

    spatial_extent = BoundingBoxExtent(
        west=float(minx), south=float(miny),
        east=float(maxx), north=float(maxy),
        epsg=epsg_utm,
    )
    print(f"   AOI (EPSG:{epsg_utm}): "
          f"{maxx - minx:.0f} m × {maxy - miny:.0f} m")

    aoi_id = str(aoi_gdf.iloc[0].get("id", "aoi"))

    # phase_ii_multitask preset season_ids: None — yillik oynani 2 ga bo'lib
    # (tc-s1 = qish/bahor, tc-s2 = yoz/kuz) global modelga yetkazib beramiz.
    season_windows = {
        "tc-s1": (TEMPORAL_EXTENT.start_date, "2025-04-30"),
        "tc-s2": ("2025-05-01", TEMPORAL_EXTENT.end_date),
    }
    workflow_config = build_config_from_params(
        season_ids=["tc-s1", "tc-s2"],
        season_windows=season_windows,
    )

    for label, product in [
        ("cropland", WorldCerealProductType.CROPLAND),
        ("croptype", WorldCerealProductType.CROPTYPE),
    ]:
        print(f"\n📍 {label} → process graph quriladi...")
        result = create_inference_process_graph(
            spatial_extent=spatial_extent,
            temporal_extent=TEMPORAL_EXTENT,
            product_type=product,
            connection=conn,
            target_epsg=epsg_utm,
            workflow_config=workflow_config,
        )

        title = f"WorldCereal_{product.value}_{aoi_id}_{datetime.now():%Y%m%d_%H%M%S}"
        job = conn.create_job(
            result,
            title=title,
            description=f"WorldCereal {product.value} (uzcosmos bugdoy {aoi_id})",
            additional=dict(DEFAULT_INFERENCE_JOB_OPTIONS),
        )
        print(f"   🚀 Job: {job.job_id}  ({title})")
        job.start_and_wait()

        out_subdir = OUTPUT_DIR / label / aoi_id
        out_subdir.mkdir(parents=True, exist_ok=True)
        job.get_results().download_files(str(out_subdir))
        print(f"   ✅ {label} → {out_subdir}")

# ─────────────────────────────────────────────────────────────────────────────
# 5b. To'g'ridan-to'g'ri openEO orqali (WorldCereal 2021 global mahsulot)
#     worldcereal paketi o'rnatilmagan bo'lsa ishlatiladi
# ─────────────────────────────────────────────────────────────────────────────
def run_openeo_direct(conn: openeo.Connection, aoi_gdf: gpd.GeoDataFrame):
    """
    openEO orqali ESA_WORLDCEREAL_10m to'plamdan ma'lumot yuklab oladi.
    Bu yo'l global 2021 mahsulotiga kirish imkonini beradi.
    """
    print("\n🔭 openEO to'g'ridan-to'g'ri ulanish orqali yuklanmoqda...")

    minx, miny, maxx, maxy = aoi_gdf.total_bounds
    bbox_extent = {
        "west": minx, "east": maxx,
        "south": miny, "north": maxy,
        "crs": "EPSG:4326"
    }
    time_extent = ["2021-01-01", "2021-12-31"]  # global 2021 mahsulot

    # ── Winter Cereals collection ──────────────────────────────────────────
    print("   Mavjud WorldCereal to'plamlari tekshirilmoqda...")
    collections = conn.list_collections()
    wc_collections = [c["id"] for c in collections if "WORLDCEREAL" in c["id"].upper()]
    print(f"   Topilgan to'plamlar: {wc_collections}")

    if not wc_collections:
        print("   ⚠️  WorldCereal to'plamlari topilmadi. openEO Algorithm Plaza ishlatilsin.")
        return

    for col_id in wc_collections:
        print(f"\n   📥 Yuklanmoqda: {col_id}")
        try:
            dc = conn.load_collection(
                col_id,
                spatial_extent=bbox_extent,
                temporal_extent=time_extent,
                bands=["classification", "confidence"],
            )
            job_title = f"WorldCereal_{col_id}_Andijan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Batch job sifatida ishga tushirish
            job = dc.create_job(
                title=job_title,
                out_format="GTiff",
                job_options={"tile_grid": "wgs84-1degree"}
            )
            print(f"   🚀 Job yaratildi: {job.job_id}")
            job.start_and_wait()
            results = job.get_results()
            results.download_files(str(OUTPUT_DIR / col_id.lower()))
            print(f"   ✅ Yuklandi → {OUTPUT_DIR / col_id.lower()}")
        except Exception as e:
            print(f"   ❌ Xato ({col_id}): {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Natijani vizualizatsiya qilish
# ─────────────────────────────────────────────────────────────────────────────
def visualize_results(tif_path: str, title: str = "WorldCereal — Asaka, Andijan"):
    """GeoTIFF faylni matplotlib orqali ko'rsatadi."""
    try:
        import rasterio
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np

        with rasterio.open(tif_path) as src:
            data = src.read(1)
            extent = (
                src.bounds.left, src.bounds.right,
                src.bounds.bottom, src.bounds.top,
            )

        fig, ax = plt.subplots(figsize=(12, 8))

        # WorldCereal classification: 0 = ekin emas, 100 = ekin
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "wc", ["#e8f5e9", "#2e7d32"], N=101
        )
        im = ax.imshow(data, cmap=cmap, vmin=0, vmax=100,
                       extent=extent, aspect="auto")
        plt.colorbar(im, ax=ax, label="Ekin ehtimolligi (0–100)")
        ax.set_title(f"{title}\n{tif_path}", fontsize=11)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        out_png = Path(tif_path).with_suffix(".png")
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"   🖼️  Vizualizatsiya saqlandi: {out_png}")
        plt.show()
    except ImportError:
        print("   ⚠️  rasterio yoki matplotlib o'rnatilmagan.")
    except Exception as e:
        print(f"   ❌ Vizualizatsiya xatosi: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Statistika — ekin maydonini hisoblash
# ─────────────────────────────────────────────────────────────────────────────
def compute_crop_area(tif_path: str, threshold: int = 50) -> dict:
    """
    GeoTIFF dan ekin maydoni statistikasini hisoblaydi.
    threshold: classification > threshold bo'lgan piksellar ekin deb hisoblanadi.
    """
    try:
        import rasterio
        import numpy as np
        from rasterio.crs import CRS
        from pyproj import Transformer

        with rasterio.open(tif_path) as src:
            data = src.read(1)
            transform = src.transform
            crs = src.crs

            # Piksel maydoni (kv.m)
            if crs.is_geographic:
                # Taxminiy piksel maydoni geografik koordinatada
                # Markaziy kenglik bo'yicha hisoblash
                center_lat = (src.bounds.top + src.bounds.bottom) / 2
                import math
                lat_m = 111320  # 1 gradus kenglik ≈ 111.32 km
                lon_m = 111320 * math.cos(math.radians(center_lat))
                pixel_area_m2 = abs(transform.a) * lon_m * abs(transform.e) * lat_m
            else:
                pixel_area_m2 = abs(transform.a * transform.e)

        crop_pixels = int(np.sum(data >= threshold))
        total_pixels = int(data.size)
        crop_area_ha = crop_pixels * pixel_area_m2 / 10_000
        crop_pct = crop_pixels / total_pixels * 100 if total_pixels else 0.0

        stats = {
            "fayl": tif_path,
            "ekin_piksellar": int(crop_pixels),
            "jami_piksellar": int(total_pixels),
            "ekin_maydoni_ga": round(crop_area_ha, 2),
            "ekin_ulushi_%": round(crop_pct, 2),
            "piksel_maydoni_m2": round(pixel_area_m2, 2),
        }

        print("\n📊 Ekin maydoni statistikasi:")
        for k, v in stats.items():
            print(f"   {k:30s}: {v}")

        # JSON faylga saqlash
        tif_p = Path(tif_path)
        stats_path = tif_p.with_name(f"{tif_p.stem}_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"   💾 Statistika saqlandi: {stats_path}")
        return stats

    except ImportError:
        print("   ⚠️  rasterio o'rnatilmagan.")
        return {}
    except Exception as e:
        print(f"   ❌ Statistika xatosi: {e}")
        return {}

# ─────────────────────────────────────────────────────────────────────────────
# 8. Asosiy funksiya
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  WorldCereal — Andijan viloyati, Asaka tumani, O'zbekiston")
    print("=" * 65)

    # AOI — uzcosmos_flats dan bitta bug'doy poligoni
    aoi_gdf = load_one_bugdoy_from_mongo()
    aoi_id = str(aoi_gdf.iloc[0]["id"])
    print(f"\n📌 AOI: bugdoy poligon (id={aoi_id})")
    utm_crs = aoi_gdf.estimate_utm_crs()
    area_ha = aoi_gdf.to_crs(utm_crs).area.sum() / 10_000
    print(f"   Maydon: ~{area_ha:.2f} ga  (UTM EPSG:{utm_crs.to_epsg()})")

    # AOI GeoJSON sifatida saqlash (tekshirish uchun)
    aoi_path = OUTPUT_DIR / f"aoi_bugdoy_{aoi_id}.geojson"
    aoi_gdf.to_file(aoi_path, driver="GeoJSON")
    print(f"   AOI saqlandi: {aoi_path}")

    # CDSE ulanish
    try:
        conn = connect_cdse()
    except Exception as e:
        print(f"❌ CDSE ulanish xatosi: {e}")
        print("   CDSE akkaunt kerak: https://dataspace.copernicus.eu/")
        return

    # Usul tanlash
    print("\n⚙️  Ishga tushirish usuli tanlanmoqda...")
    try:
        import worldcereal  # noqa: F401
        print("   worldcereal paketi topildi → 5a usul ishlatiladi")
        run_worldcereal_package(aoi_gdf, conn)
    except ImportError:
        print("   worldcereal paketi topilmadi → openEO to'g'ridan-to'g'ri usul")
        run_openeo_direct(conn, aoi_gdf)

    # Natijalarni ko'rsatish va statistika
    print("\n" + "=" * 65)
    print("  Natijalar tekshirilmoqda...")
    for tif_file in OUTPUT_DIR.rglob("*.tif"):
        print(f"\n🗂️  {tif_file.name}")
        compute_crop_area(str(tif_file))
        visualize_results(str(tif_file),
                          title=f"WorldCereal — {tif_file.stem}\nAsaka, Andijan, O'zbekiston")

    print("\n" + "=" * 65)
    print("✅ Barcha ishlar yakunlandi!")
    print(f"   Natijalar: {OUTPUT_DIR.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()