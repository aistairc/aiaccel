# Copyright (C) 2026 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from datetime import date
from pathlib import Path
import sys

COPYRIGHT_PREFIX = "Copyright (C) "


def main() -> int:
    path = Path(__file__).resolve().parents[2] / "LICENSE_HEADER"
    current_year = str(date.today().year)

    # newline="" により、LF / CRLF を読み書き時に変換しない
    with path.open(encoding="utf-8", newline="") as file:
        lines = file.readlines()

    matching_indices = [index for index, line in enumerate(lines) if line.startswith(COPYRIGHT_PREFIX)]

    if len(matching_indices) != 1:
        print(
            f"Expected one Copyright line in {path}, found {len(matching_indices)}",
            file=sys.stderr,
        )
        return 1

    index = matching_indices[0]
    year_start = len(COPYRIGHT_PREFIX)
    year_end = year_start + 4
    year = lines[index][year_start:year_end]

    if (
        len(year) != 4
        or not all("0" <= character <= "9" for character in year)
        or lines[index][year_end : year_end + 1] not in (" ", "\t")
    ):
        print(f"Expected a four-digit year after Copyright prefix in {path}", file=sys.stderr)
        return 1

    if year == current_year:
        return 0

    lines[index] = lines[index][:year_start] + current_year + lines[index][year_end:]

    with path.open("w", encoding="utf-8", newline="") as file:
        file.writelines(lines)

    print(f"Updated {path} from {year} to {current_year}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
