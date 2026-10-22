# Copyright (C) 2026 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from datetime import date
from pathlib import Path
import subprocess
import sys
import tempfile


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    template = (repo_root / ".license-header.txt").read_text(encoding="utf-8")

    if template.count("@YEAR@") != 1:
        print(f"{repo_root / '.license-header.txt'} must contain exactly one @YEAR@")
        return 1

    header = template.replace("@YEAR@", str(date.today().year))

    with tempfile.TemporaryDirectory() as tmp_dir:
        license_path = Path(tmp_dir) / ".license-header.txt"
        license_path.write_text(header, encoding="utf-8")

        result = subprocess.run(
            [
                "insert_license",
                f"--license-filepath={license_path}",
                "--detect-license-in-X-top-lines=16",
                "--allow-past-years",
                *sys.argv[1:],
            ],
            check=False,
        )
        return result.returncode


if __name__ == "__main__":
    sys.exit(main())
