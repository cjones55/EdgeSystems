"""Download the historical datasets used for backtesting and demo mode.

  NBA/NFL/NHL results + lines 2011-2022   github.com/flancast90/sportsbookreview-scraper (MIT)
  NBA player box scores 2010-2024         github.com/NocturneBear/NBA-Data-2010-2024 (MIT)
  College football 2015-2026 + lines      github.com/matthewrichey3317/cfb-data (MIT; data from CollegeFootballData.com)
  College basketball results 2015-2026    Kaggle "March Machine Learning Mania" files, mirrored at
                                          github.com/lingmingzi/NCAA2026 (Kaggle competition data)
  College basketball closing odds 2021-23 github.com/JDSB123/ncaam-historical-data (no license file;
                                          personal research use)

Usage:  python scripts/fetch_data.py [nba nfl nhl ncaaf ncaab]   (default: all)
Re-running skips files that are already present. Pass --refresh-current to re-download the
current college football season (it is updated weekly upstream).
"""
from __future__ import annotations

import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
GH = "https://raw.githubusercontent.com"
SBR = f"{GH}/flancast90/sportsbookreview-scraper/main/data"
CFB = f"{GH}/matthewrichey3317/cfb-data/main"
KAG = f"{GH}/lingmingzi/NCAA2026/main"
NCAAM = f"{GH}/JDSB123/ncaam-historical-data/main"


def files(sport: str) -> dict[str, str]:
    if sport == "nba":
        return {"nba_archive_10Y.json": f"{SBR}/nba_archive_10Y.json",
                **{f"regular_season_box_scores_2010_2024_part_{i}.csv":
                   f"{GH}/NocturneBear/NBA-Data-2010-2024/main/regular_season_box_scores_2010_2024_part_{i}.csv"
                   for i in (1, 2, 3)}}
    if sport in ("nfl", "nhl"):
        return {f"{sport}_archive_10Y.json": f"{SBR}/{sport}_archive_10Y.json"}
    if sport == "ncaaf":
        with urllib.request.urlopen(f"{CFB}/export/index.json", timeout=60) as r:
            index = json.load(r)
        names = [p["file"] for p in index["files"]]
        out = {f"ncaaf/{Path(n).name}": f"{CFB}/export/{Path(n).name}" for n in names}
        for y in range(2015, date.today().year + 1):
            out[f"ncaaf/lines_{y}.csv"] = f"{CFB}/data/lines_{y}.csv"
        return out
    if sport == "ncaab":
        out = {f"ncaab/{f}.csv": f"{KAG}/{f}.csv" for f in (
            "MRegularSeasonCompactResults", "MNCAATourneyCompactResults",
            "MSecondaryTourneyCompactResults", "MTeamSpellings", "MSeasons", "MTeams")}
        out["ncaab/odds_consolidated_canonical.csv"] = f"{NCAAM}/odds/normalized/odds_consolidated_canonical.csv"
        return out
    raise ValueError(sport)


def main(argv: list[str]) -> None:
    refresh = "--refresh-current" in argv
    sports = [a for a in argv if not a.startswith("--")] or ["nba", "nfl", "nhl", "ncaaf", "ncaab"]
    for sport in sports:
        for name, url in files(sport).items():
            dest = RAW / name
            current = str(date.today().year) in name and sport == "ncaaf"
            if dest.exists() and not (refresh and current):
                print(f"have {name}")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            print(f"downloading {name} ...")
            try:
                urllib.request.urlretrieve(url, dest)
            except Exception as e:  # e.g. a season file that does not exist yet
                print(f"  skipped ({e})")
    print("done")


if __name__ == "__main__":
    main(sys.argv[1:])
