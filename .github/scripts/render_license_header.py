# Copyright (C) 2026 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from datetime import date
from pathlib import Path
import sys

from jinja2 import Environment, StrictUndefined, TemplateError, nodes


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    template_path = repo_root / "templete.txt"
    output_path = repo_root / ".license-header.txt"

    try:
        template_text = template_path.read_text(encoding="utf-8")
        environment = Environment(undefined=StrictUndefined, keep_trailing_newline=True)
        parsed_template = environment.parse(template_text)
        year_references = sum(node.name == "current_year" for node in parsed_template.find_all(nodes.Name))
        if year_references != 1:
            raise ValueError("templete.txt must reference {{ current_year }} exactly once")

        header = environment.from_string(template_text).render(current_year=date.today().year)
        output_path.write_text(header, encoding="utf-8")
    except (OSError, TemplateError, ValueError) as error:
        print(f"Failed to generate {output_path}: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
