"""
run_test.py — One-click cat recognition accuracy & performance test.

Usage:
    python run_test.py

    # Custom paths:
    python run_test.py --gd_config path/to/config.py --gd_checkpoint path/to/gd.pth \
                       --sam_checkpoint path/to/sam.pth

    # Force rebuild gallery:
    python run_test.py --rebuild

    # Adjust threshold:
    python run_test.py --threshold 0.40
"""
import os
import sys
import time
import argparse
import numpy as np
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from core_pipeline import CatIdentificationPipeline


# ==================== Database ====================

def init_database(conn_params):
    """Create database, enable pgvector, create table."""
    base_params = {k: v for k, v in conn_params.items() if k != "dbname"}
    conn = psycopg2.connect(**base_params)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s",
        (conn_params["dbname"],),
    )
    if not cur.fetchone():
        cur.execute(f"CREATE DATABASE {conn_params['dbname']}")
        print(f"  Created database '{conn_params['dbname']}'")
    cur.close()
    conn.close()

    conn = psycopg2.connect(**conn_params)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS cat_gallery (
            id SERIAL PRIMARY KEY,
            cat_name TEXT,
            image_path TEXT,
            embedding vector(768)
        )
    """)
    cur.close()
    conn.close()
    print("  Database ready.")


def clear_gallery(conn_params):
    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()
    cur.execute("DELETE FROM cat_gallery")
    conn.commit()
    cur.close()
    conn.close()


def gallery_count(conn_params):
    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM cat_gallery")
    n = cur.fetchone()[0]
    cur.close()
    conn.close()
    return n


# ==================== Gallery Build ====================

def build_gallery(pipeline, conn_params, gallery_dir):
    """Detect + segment + extract features for all gallery images, store in DB."""
    n = gallery_count(conn_params)
    if n > 0:
        print(f"  Gallery already has {n} entries — skipping. Use --rebuild to force.")
        return

    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()
    total = 0
    failed = 0

    for cat_dir in sorted(os.listdir(gallery_dir)):
        cat_path = os.path.join(gallery_dir, cat_dir)
        if not os.path.isdir(cat_path) or cat_dir == "videos":
            continue

        for img_file in sorted(os.listdir(cat_path)):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(cat_path, img_file)
            result = pipeline.process_image(img_path)

            if result is None:
                failed += 1
                print(f"    SKIP {cat_dir}/{img_file} (no cat detected)")
                continue

            vector, _ = result
            cur.execute(
                "INSERT INTO cat_gallery (cat_name, image_path, embedding) VALUES (%s, %s, %s)",
                (cat_dir, img_path, vector),
            )
            conn.commit()
            total += 1

    cur.close()
    conn.close()
    print(f"  Gallery built: {total} embeddings stored, {failed} images skipped.")


# ==================== Search ====================

def search_similar(conn_params, query_vector, top_k=5):
    """
    Search gallery by cosine distance, aggregated per cat (min distance).
    Returns: [(cat_name, distance), ...]
    """
    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()
    vec_str = "[" + ",".join(map(str, query_vector)) + "]"
    cur.execute(
        f"""
        SELECT cat_name, MIN(embedding <=> %s::vector) AS distance
        FROM cat_gallery
        GROUP BY cat_name
        ORDER BY distance
        LIMIT %s
        """,
        (vec_str, top_k),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows  # [(cat_name, distance), ...]


# ==================== Test Runner ====================

UNREGISTERED_CAT = "cat_xiaomai"


def run_test(pipeline, conn_params, query_dir, threshold):
    """Run recognition test on all query images. Returns structured results."""
    results = {}

    for cat_dir in sorted(os.listdir(query_dir)):
        cat_path = os.path.join(query_dir, cat_dir)
        if not os.path.isdir(cat_path):
            continue

        is_registered = cat_dir != UNREGISTERED_CAT
        results[cat_dir] = []

        for img_file in sorted(os.listdir(cat_path)):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(cat_path, img_file)
            entry = {"file": img_file}

            # --- Stage 1: Detect ---
            t0 = time.time()
            image_source, box = pipeline.cat_pipeline.detect(img_path)
            entry["t_detect"] = time.time() - t0

            if box is None:
                entry.update(detected=False, top1=False, top3=False,
                             top1_name=None, distance=None,
                             t_segment=0, t_extract=0, t_search=0)
                entry["t_total"] = entry["t_detect"]
                results[cat_dir].append(entry)
                continue

            # --- Stage 2: Segment ---
            t1 = time.time()
            masked_image, _ = pipeline.cat_pipeline.segment(image_source, box)
            entry["t_segment"] = time.time() - t1

            # --- Stage 3: Feature extraction ---
            t2 = time.time()
            query_vector = pipeline.extract_feature(masked_image)
            entry["t_extract"] = time.time() - t2

            # --- Stage 4: Search ---
            t3 = time.time()
            matches = search_similar(conn_params, query_vector, top_k=5)
            entry["t_search"] = time.time() - t3

            entry["t_total"] = time.time() - t0
            entry["detected"] = True

            if not matches:
                entry.update(top1=False, top3=False, top1_name=None, distance=None)
            else:
                top1_name, top1_dist = matches[0]
                top3_names = [m[0] for m in matches[:3]]
                entry["top1_name"] = top1_name
                entry["distance"] = top1_dist

                if is_registered:
                    entry["top1"] = top1_name == cat_dir
                    entry["top3"] = cat_dir in top3_names
                else:
                    # Unregistered: correct = rejected (distance >= threshold)
                    entry["top1"] = top1_dist >= threshold
                    entry["top3"] = entry["top1"]

            results[cat_dir].append(entry)

    return results


# ==================== Report ====================

def print_report(results, threshold):
    registered = [k for k in results if k != UNREGISTERED_CAT]

    # ---------- Accuracy ----------
    print("\n" + "=" * 70)
    print("  ACCURACY RESULTS")
    print("=" * 70)
    print(f"  {'Cat':<22} {'N':>4}   {'Top-1':>8}   {'Top-3':>8}")
    print("  " + "-" * 50)

    sum_n = sum_t1 = sum_t3 = 0
    for cat in registered:
        entries = results[cat]
        n = len(entries)
        t1 = sum(1 for e in entries if e["top1"])
        t3 = sum(1 for e in entries if e["top3"])
        print(f"  {cat:<22} {n:>4}   {t1/n*100:>7.1f}%   {t3/n*100:>7.1f}%")
        sum_n += n
        sum_t1 += t1
        sum_t3 += t3

    print("  " + "-" * 50)
    print(f"  {'TOTAL':<22} {sum_n:>4}   {sum_t1/sum_n*100:>7.1f}%   {sum_t3/sum_n*100:>7.1f}%")

    if UNREGISTERED_CAT in results:
        entries = results[UNREGISTERED_CAT]
        n = len(entries)
        rej = sum(1 for e in entries if e["top1"])
        print(f"\n  Unregistered ({UNREGISTERED_CAT}):  {n} images, "
              f"rejection rate = {rej/n*100:.1f}%")

    # ---------- Timing ----------
    print("\n" + "=" * 70)
    print("  TIMING (seconds, excluding first cold-start request)")
    print("=" * 70)

    all_entries = []
    for cat in results:
        all_entries.extend([e for e in results[cat] if e["detected"]])

    # Skip first entry (cold start)
    warm = all_entries[1:] if len(all_entries) > 1 else all_entries

    if warm:
        print(f"  {'Stage':<12} {'Mean':>8}  {'Median':>8}  {'P95':>8}")
        print("  " + "-" * 42)
        for key, label in [("t_detect", "Detect"), ("t_segment", "Segment"),
                           ("t_extract", "Extract"), ("t_search", "Search"),
                           ("t_total", "Total")]:
            vals = [e[key] for e in warm]
            print(f"  {label:<12} {np.mean(vals):>7.3f}s  {np.median(vals):>7.3f}s  "
                  f"{np.percentile(vals, 95):>7.3f}s")

        cold = all_entries[0]
        print(f"\n  Cold start (first request): {cold['t_total']:.2f}s")

    # ---------- Detail ----------
    print("\n" + "=" * 70)
    print("  DETAILED RESULTS")
    print("=" * 70)

    for cat in results:
        is_reg = cat != UNREGISTERED_CAT
        print(f"\n  --- {cat} {'(registered)' if is_reg else '(UNREGISTERED)'} ---")
        for e in results[cat]:
            mark = "OK" if e["top1"] else "MISS"
            if not e["detected"]:
                print(f"    [{mark}] {e['file']}: no cat detected")
            else:
                dist_str = f"{e['distance']:.4f}" if e['distance'] is not None else "N/A"
                print(f"    [{mark}] {e['file']}: "
                      f"top1={e['top1_name']} dist={dist_str} "
                      f"time={e['t_total']:.2f}s")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(
        description="One-click cat recognition test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gd_config", default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--gd_checkpoint", default="weights/groundingdino_swint_ogc.pth")
    parser.add_argument("--sam_checkpoint", default="weights/sam_vit_h_4b8939.pth")
    parser.add_argument("--sam_type", default="vit_h")
    parser.add_argument("--gallery_dir", default="dataset/gallery")
    parser.add_argument("--query_dir", default="dataset/query")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--db_host", default="localhost")
    parser.add_argument("--db_user", default="postgres")
    parser.add_argument("--db_password", default="123456")
    parser.add_argument("--db_name", default="cat_recognition")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild gallery embeddings")
    args = parser.parse_args()

    conn_params = dict(dbname=args.db_name, user=args.db_user,
                       password=args.db_password, host=args.db_host)

    print("=" * 70)
    print("  Cat Recognition Pipeline — Accuracy & Performance Test")
    print("=" * 70)

    # Step 1: Database
    print("\n[1/4] Initializing database...")
    init_database(conn_params)
    if args.rebuild:
        clear_gallery(conn_params)
        print("  Gallery cleared.")

    # Step 2: Load models
    print("\n[2/4] Loading models (this may take a while on first run)...")
    pipeline = CatIdentificationPipeline(
        gd_config_path=args.gd_config,
        gd_checkpoint_path=args.gd_checkpoint,
        sam_checkpoint_path=args.sam_checkpoint,
        sam_type=args.sam_type,
    )

    # Step 3: Build gallery
    print("\n[3/4] Building gallery embeddings...")
    build_gallery(pipeline, conn_params, args.gallery_dir)

    # Step 4: Run test
    print("\n[4/4] Running recognition test on query images...")
    results = run_test(pipeline, conn_params, args.query_dir, args.threshold)

    # Report
    print_report(results, args.threshold)
    print()


if __name__ == "__main__":
    main()
