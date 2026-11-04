"""Download the historical datasets used for backtesting (both MIT licensed).

  * NBA results + open/close lines 2011-12..2021-22: github.com/flancast90/sportsbookreview-scraper
  * NBA player box scores 2010-11..2023-24:         github.com/NocturneBear/NBA-Data-2010-2024
"""
from pathlib import Path
import urllib.request

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
FILES = {
    "nba_archive_10Y.json":
        "https://raw.githubusercontent.com/flancast90/sportsbookreview-scraper/main/data/nba_archive_10Y.json",
    **{f"regular_season_box_scores_2010_2024_part_{i}.csv":
       f"https://raw.githubusercontent.com/NocturneBear/NBA-Data-2010-2024/main/regular_season_box_scores_2010_2024_part_{i}.csv"
       for i in (1, 2, 3)},
}

if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        dest = RAW / name
        if dest.exists():
            print(f"have {name}")
            continue
        print(f"downloading {name} ...")
        urllib.request.urlretrieve(url, dest)
    print("done")
