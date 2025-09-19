#!/usr/bin/env python3
"""
Usage:
  python download_videos.py --csv datatrain.csv --out videos --limit 0 \
    --jobs 4 --format 'mp4[height<=480]/mp4/best' \
    --cookies-from-browser firefox
"""
import argparse, csv, os, sys, subprocess, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

def is_drive(url: str) -> bool:
    host = urllib.parse.urlparse(url).netloc.lower()
    return "drive.google.com" in host or "docs.google.com" in host

def is_generic(url: str) -> bool:
    # anything not Drive → generic/yt-dlp decides; we still keep a last-resort HTTP downloader
    return True

def ensure_dropbox_direct(url: str) -> str:
    # Turn www.dropbox.com/s/<id>/<name>?dl=0 → dl=1
    u = urllib.parse.urlparse(url)
    if "dropbox.com" in u.netloc and (u.query == "" or "dl=" in u.query):
        qs = urllib.parse.parse_qs(u.query)
        qs["dl"] = ["1"]
        return urllib.parse.urlunparse(u._replace(query=urllib.parse.urlencode(qs, doseq=True)))
    return url

def run(cmd: list) -> int:
    try:
        return subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr).returncode
    except FileNotFoundError:
        return 127

def download_with_ytdlp(url: str, outpath: str, ytdlp_format: str, cookies_file: str, cookies_browser: str, extra: str) -> int:
    cmd = [sys.executable, "-m", "yt_dlp",
           "-o", outpath,
           "-f", ytdlp_format,
           "--no-warnings",
           "--retry-sleep", "1",
           "--retries", "20",
           "--fragment-retries", "20"]
    if cookies_file:
        cmd += ["--cookies", cookies_file]
    if cookies_browser:
        cmd += ["--cookies-from-browser", cookies_browser]
    if extra:
        cmd += extra.split()
    cmd.append(url)
    print("YT-DLP:", " ".join(cmd))
    return run(cmd)

def download_with_gdown(url: str, outpath: str) -> int:
    # --fuzzy lets gdown accept many Drive URL shapes
    cmd = [sys.executable, "-m", "gdown", "--fuzzy", url, "-O", outpath, "--quiet"]
    print("GDOWN :", " ".join(cmd))
    return run(cmd)

def download_http(url: str, outpath: str, timeout: int = 60) -> int:
    # Lightweight fallback for direct links
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r, open(outpath, "wb") as f:
            chunk = r.read(1024 * 1024)
            while chunk:
                f.write(chunk)
                chunk = r.read(1024 * 1024)
        return 0
    except Exception as e:
        print(f"[http-err] {url} -> {e}", file=sys.stderr)
        return 1

def safe_name(name: str) -> str:
    # prevent weird chars in filenames
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)

def worker(row: dict, args) -> Tuple[str, str]:
    url = row[args.url_col].strip()
    vid = safe_name(row[args.id_col].strip())
    outpath = os.path.join(args.out, f"{vid}.mp4")

    if os.path.exists(outpath) and os.path.getsize(outpath) > 0:
        return ("skip", vid)

    # choose strategy
    if is_drive(url):
        rc = download_with_gdown(url, outpath)
        if rc != 0:
            # try yt-dlp as a second chance (some shared Drives also work)
            rc = download_with_ytdlp(url, outpath, args.format, args.cookies, args.cookies_from_browser, args.extra)
    else:
        # normalize a few providers
        url2 = ensure_dropbox_direct(url)
        # primary: yt-dlp (covers Instagram/TikTok/etc.)
        rc = download_with_ytdlp(url2, outpath, args.format, args.cookies, args.cookies_from_browser, args.extra)
        if rc != 0 and url2.startswith(("http://", "https://")):
            # last resort plain HTTP (for direct files)
            rc = download_http(url2, outpath)

    return ("ok", vid) if rc == 0 else ("fail", vid)

def main():
    p = argparse.ArgumentParser()
    # p.add_argument("--csv",        default="datatrain.csv")
    p.add_argument("--csv", default="satdat-bdc-2025/datatest_revisi.csv")
    p.add_argument("--url_col",    default="video")
    p.add_argument("--id_col",     default="id")
    p.add_argument("--out",        default="videos")
    p.add_argument("--limit",      type=int, default=0, help="limit number of downloads (0 = all)")
    p.add_argument("--jobs",       type=int, default=4, help="parallel downloads")
    p.add_argument("--format",     default='mp4[height<=480]/mp4/best', help="yt-dlp format selector")
    p.add_argument("--cookies",    default="", help="path to cookies.txt for yt-dlp")
    p.add_argument("--cookies-from-browser", dest="cookies_from_browser", default="", help="e.g. chrome / firefox")
    p.add_argument("--extra",      default="", help="extra flags passed to yt-dlp")
    args = p.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        missing = [c for c in (args.url_col, args.id_col) if c not in rdr.fieldnames]
        if missing:
            print(f"Missing columns in CSV: {missing}", file=sys.stderr)
            sys.exit(1)
        for r in rdr:
            rows.append(r)

    if args.limit > 0:
        rows = rows[:args.limit]

    ok, skip, fail = 0, 0, 0
    failed_ids = []

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = [ex.submit(worker, r, args) for r in rows]
        for fut in as_completed(futs):
            status, vid = fut.result()
            if status == "ok":
                ok += 1
                print(f"[ok]   {vid}")
            elif status == "skip":
                skip += 1
                print(f"[skip] {vid}")
            else:
                fail += 1
                failed_ids.append(vid)
                print(f"[fail] {vid}")

    # summary
    total = ok + skip + fail
    print("\n=== Download Summary ===")
    print(f"Total rows : {total}")
    print(f"Downloaded : {ok}")
    print(f"Skipped    : {skip}")
    print(f"Failed     : {fail}")

    if failed_ids:
        fail_log = os.path.join(args.out, "download_failures.txt")
        with open(fail_log, "w", encoding="utf-8") as g:
            g.write("\n".join(failed_ids))
        print(f"Failed IDs saved to: {fail_log}")

if __name__ == "__main__":
    main()
