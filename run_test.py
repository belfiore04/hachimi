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

    # Compare multiple thresholds without rerunning model inference:
    python run_test.py --thresholds 0.30,0.35,0.40,0.45,0.50

    # Save report to a custom path:
    python run_test.py --report_path results/test_report.txt
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


def list_registered_cats(gallery_dir):
    """Return cat folder names that are present in the gallery."""
    cats = set()
    for name in os.listdir(gallery_dir):
        path = os.path.join(gallery_dir, name)
        if os.path.isdir(path) and name != "videos":
            cats.add(name)
    return cats


def count_images_by_cat(root_dir):
    """Count image files in each direct child folder."""
    counts = {}
    for name in sorted(os.listdir(root_dir)):
        path = os.path.join(root_dir, name)
        if not os.path.isdir(path) or name == "videos":
            continue
        counts[name] = sum(
            1 for file_name in os.listdir(path)
            if file_name.lower().endswith((".jpg", ".jpeg", ".png"))
        )
    return counts


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

def run_test(pipeline, conn_params, query_dir, threshold, registered_cats):
    """Run recognition test on all query images. Returns structured results."""
    results = {}

    for cat_dir in sorted(os.listdir(query_dir)):
        cat_path = os.path.join(query_dir, cat_dir)
        if not os.path.isdir(cat_path):
            continue

        is_registered = cat_dir in registered_cats
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
                             decision_correct=False, decision="no_cat_detected",
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
                entry.update(top1=False, top3=False, decision_correct=False,
                             decision="unknown", top1_name=None, distance=None)
            else:
                top1_name, top1_dist = matches[0]
                top3_names = [m[0] for m in matches[:3]]
                entry["top1_name"] = top1_name
                entry["distance"] = top1_dist
                accepted = top1_dist < threshold
                entry["decision"] = top1_name if accepted else "unknown"

                if is_registered:
                    entry["top1"] = top1_name == cat_dir
                    entry["top3"] = cat_dir in top3_names
                    entry["decision_correct"] = accepted and top1_name == cat_dir
                else:
                    # Unregistered: correct = rejected (distance >= threshold)
                    entry["top1"] = top1_dist >= threshold
                    entry["top3"] = entry["top1"]
                    entry["decision_correct"] = not accepted

            results[cat_dir].append(entry)

    return results


# ==================== Report ====================

def parse_thresholds(raw_thresholds):
    """Parse comma-separated thresholds into a sorted unique list."""
    if not raw_thresholds:
        return []

    thresholds = []
    for value in raw_thresholds.split(","):
        value = value.strip()
        if not value:
            continue
        thresholds.append(float(value))

    return sorted(set(thresholds))


def summarize_threshold(results, threshold, registered_cats):
    """Calculate decision metrics for one threshold from stored distances."""
    known_n = known_top1 = known_top3 = known_final = 0
    unknown_n = unknown_reject = 0

    for cat, entries in results.items():
        is_registered = cat in registered_cats

        if is_registered:
            known_n += len(entries)
            known_top1 += sum(1 for e in entries if e["top1"])
            known_top3 += sum(1 for e in entries if e["top3"])
            known_final += sum(
                1 for e in entries
                if e["distance"] is not None
                and e["distance"] < threshold
                and e["top1_name"] == cat
            )
        else:
            unknown_n += len(entries)
            unknown_reject += sum(
                1 for e in entries
                if e["distance"] is not None and e["distance"] >= threshold
            )

    return {
        "threshold": threshold,
        "known_n": known_n,
        "known_top1": known_top1,
        "known_top3": known_top3,
        "known_final": known_final,
        "unknown_n": unknown_n,
        "unknown_reject": unknown_reject,
        "known_top1_rate": known_top1 / known_n * 100 if known_n else None,
        "known_top3_rate": known_top3 / known_n * 100 if known_n else None,
        "known_final_rate": known_final / known_n * 100 if known_n else None,
        "unknown_reject_rate": unknown_reject / unknown_n * 100 if unknown_n else None,
    }


def print_report(results, threshold, registered_cats, gallery_counts=None,
                 query_counts=None, threshold_values=None, report_path=None):
    lines = []

    def emit(line=""):
        print(line)
        lines.append(line)

    registered = [k for k in results if k in registered_cats]
    unregistered = [k for k in results if k not in registered_cats]

    # ---------- Dataset summary ----------
    if gallery_counts is not None and query_counts is not None:
        emit("\n" + "=" * 70)
        emit("  DATASET SUMMARY")
        emit("=" * 70)
        emit(f"  {'Cat':<22} {'Type':<14} {'Gallery':>8} {'Query':>8}")
        emit("  " + "-" * 56)
        for cat in sorted(set(gallery_counts) | set(query_counts)):
            cat_type = "registered" if cat in registered_cats else "unregistered"
            emit(f"  {cat:<22} {cat_type:<14} {gallery_counts.get(cat, 0):>8} {query_counts.get(cat, 0):>8}")
        emit("  " + "-" * 56)
        emit(f"  {'TOTAL':<22} {'':<14} {sum(gallery_counts.values()):>8} {sum(query_counts.values()):>8}")

    # ---------- Accuracy ----------
    emit("\n" + "=" * 70)
    emit("  ACCURACY RESULTS")
    emit("=" * 70)
    emit(f"  {'Cat':<22} {'N':>4}   {'Top-1':>8}   {'Top-3':>8}")
    emit("  " + "-" * 50)

    sum_n = sum_t1 = sum_t3 = sum_decision = 0
    for cat in registered:
        entries = results[cat]
        n = len(entries)
        t1 = sum(1 for e in entries if e["top1"])
        t3 = sum(1 for e in entries if e["top3"])
        if n:
            emit(f"  {cat:<22} {n:>4}   {t1/n*100:>7.1f}%   {t3/n*100:>7.1f}%")
        else:
            emit(f"  {cat:<22} {n:>4}   {'N/A':>8}   {'N/A':>8}")
        sum_n += n
        sum_t1 += t1
        sum_t3 += t3
        sum_decision += sum(1 for e in entries if e["decision_correct"])

    emit("  " + "-" * 50)
    if sum_n:
        emit(f"  {'TOTAL':<22} {sum_n:>4}   {sum_t1/sum_n*100:>7.1f}%   {sum_t3/sum_n*100:>7.1f}%")
        emit(f"\n  Known final decision accuracy: {sum_decision}/{sum_n} = {sum_decision/sum_n*100:.1f}%")
    else:
        emit(f"  {'TOTAL':<22} {sum_n:>4}   {'N/A':>8}   {'N/A':>8}")

    if unregistered:
        emit("\n  Unregistered cats:")
        total_unknown_n = 0
        total_unknown_rej = 0
        for cat in unregistered:
            entries = results[cat]
            n = len(entries)
            rej = sum(1 for e in entries if e["decision_correct"])
            total_unknown_n += n
            total_unknown_rej += rej
            rate = rej / n * 100 if n else 0
            emit(f"    {cat:<20} {n:>4} images, rejection rate = {rate:.1f}%")
        total_rate = total_unknown_rej / total_unknown_n * 100 if total_unknown_n else 0
        emit(f"    {'TOTAL':<20} {total_unknown_n:>4} images, rejection rate = {total_rate:.1f}%")

    # ---------- Threshold comparison ----------
    if threshold_values:
        emit("\n" + "=" * 70)
        emit("  THRESHOLD COMPARISON")
        emit("=" * 70)
        emit(f"  {'Threshold':>9} {'Top-1':>8} {'Top-3':>8} {'Known Final':>14} {'Unknown Reject':>16}")
        emit("  " + "-" * 64)
        for item in [summarize_threshold(results, t, registered_cats) for t in threshold_values]:
            top1 = f"{item['known_top1_rate']:.1f}%" if item["known_top1_rate"] is not None else "N/A"
            top3 = f"{item['known_top3_rate']:.1f}%" if item["known_top3_rate"] is not None else "N/A"
            known_final = f"{item['known_final_rate']:.1f}%" if item["known_final_rate"] is not None else "N/A"
            unknown_reject = f"{item['unknown_reject_rate']:.1f}%" if item["unknown_reject_rate"] is not None else "N/A"
            emit(f"  {item['threshold']:>9.2f} {top1:>8} {top3:>8} {known_final:>14} {unknown_reject:>16}")

        emit("\n  Note: Top-1 and Top-3 are ranking metrics and do not change with threshold.")
        emit("  Known Final and Unknown Reject are threshold-dependent decision metrics.")

    # ---------- Timing ----------
    emit("\n" + "=" * 70)
    emit("  TIMING (seconds, excluding first cold-start request)")
    emit("=" * 70)

    all_entries = []
    for cat in results:
        all_entries.extend([e for e in results[cat] if e["detected"]])

    # Skip first entry (cold start)
    warm = all_entries[1:] if len(all_entries) > 1 else all_entries

    if warm:
        emit(f"  {'Stage':<12} {'Mean':>8}  {'Median':>8}  {'P95':>8}")
        emit("  " + "-" * 42)
        for key, label in [("t_detect", "Detect"), ("t_segment", "Segment"),
                           ("t_extract", "Extract"), ("t_search", "Search"),
                           ("t_total", "Total")]:
            vals = [e[key] for e in warm]
            emit(f"  {label:<12} {np.mean(vals):>7.3f}s  {np.median(vals):>7.3f}s  "
                 f"{np.percentile(vals, 95):>7.3f}s")

        cold = all_entries[0]
        emit(f"\n  Cold start (first request): {cold['t_total']:.2f}s")

    # ---------- Detail ----------
    emit("\n" + "=" * 70)
    emit("  DETAILED RESULTS")
    emit("=" * 70)

    for cat in results:
        is_reg = cat in registered_cats
        emit(f"\n  --- {cat} {'(registered)' if is_reg else '(UNREGISTERED)'} ---")
        for e in results[cat]:
            mark = "OK" if e["top1"] else "MISS"
            if not e["detected"]:
                emit(f"    [{mark}] {e['file']}: no cat detected")
            else:
                dist_str = f"{e['distance']:.4f}" if e['distance'] is not None else "N/A"
                emit(f"    [{mark}] {e['file']}: "
                     f"top1={e['top1_name']} decision={e['decision']} dist={dist_str} "
                     f"time={e['t_total']:.2f}s")

    if report_path:
        report_dir = os.path.dirname(report_path)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            f.write("\n")
        print(f"\n  Report saved to: {report_path}")


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
    parser.add_argument("--thresholds", default="",
                        help="Comma-separated thresholds for comparison, e.g. 0.30,0.35,0.40,0.45,0.50")
    parser.add_argument("--db_host", default="localhost")
    parser.add_argument("--db_user", default="postgres")
    parser.add_argument("--db_password", default="123456")
    parser.add_argument("--db_name", default="cat_recognition")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild gallery embeddings")
    parser.add_argument("--report_path", default="test_report.txt",
                        help="Path to save the text report. Use an empty string to disable.")
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
    registered_cats = list_registered_cats(args.gallery_dir)
    gallery_counts = count_images_by_cat(args.gallery_dir)
    query_counts = count_images_by_cat(args.query_dir)
    print(f"  Registered cats: {', '.join(sorted(registered_cats))}")

    # Step 4: Run test
    print("\n[4/4] Running recognition test on query images...")
    results = run_test(pipeline, conn_params, args.query_dir, args.threshold, registered_cats)

    # Report
    threshold_values = parse_thresholds(args.thresholds)
    report_path = args.report_path or None
    print_report(results, args.threshold, registered_cats,
                 gallery_counts, query_counts, threshold_values, report_path)
    print()


if __name__ == "__main__":
    main()
