"""
Usage:
  python download_videos.py --csv datatrain.csv --out videos --limit 0
"""

import argparse, csv, os, subprocess, sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="datatrain.csv")
    p.add_argument("--url_col", default="video")
    p.add_argument("--id_col", default="id")
    p.add_argument("--out", default="videos")
    p.add_argument("--limit", type=int, default=0, help="limit number of downloads (0 = all)")
    p.add_argument("--extra", default="", help="extra flags passed to yt-dlp, e.g. --cookies cookies.txt")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    rows = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            if args.url_col not in r or args.id_col not in r:
                print(f"Columns {args.url_col}/{args.id_col} not found in CSV", file=sys.stderr)
                sys.exit(1)
            rows.append(r)

    if args.limit > 0:
        rows = rows[:args.limit]

    for r in rows:
        url = r[args.url_col]
        vid = r[args.id_col]
        outpath = os.path.join(args.out, f"{vid}.mp4")
        if os.path.exists(outpath):
            print(f"[skip] {outpath}")
            continue
        cmd = f'yt-dlp -o "{outpath}" -f "mp4[height<=480]/mp4/best" {args.extra} "{url}"'
        print(cmd)
        rc = subprocess.call(cmd, shell=True)
        if rc != 0:
            print(f"[warn] failed: {url}")

if __name__ == "__main__":
    main()
