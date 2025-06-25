#!/usr/bin/env python

"""Path cross-references into generated notebook markdown files."""

import sys

import sax


def _process_content(content: str) -> str:
    blocks = content.split("```")

    for i, block in enumerate(blocks):
        if i % 2:
            if (admonition := parse_admonition(block)) is not None:
                blocks[i] = admonition
            else:
                blocks[i] = f"```{block}```"
            continue
        lines = block.split("\n")
        for j, line in enumerate(lines):
            parts = line.split("`")
            for k, part in enumerate(parts):
                if k % 2 == 0:
                    continue
                if hasattr(sax, part):
                    parts[k] = f"[`{part}`][sax.{part}]"
                elif hasattr(sax.models, part):
                    parts[k] = f"[`{part}`][sax.models.{part}]"
                else:
                    parts[k] = f"`{part}`"
            lines[j] = "".join(parts)
        blocks[i] = "\n".join(lines)

    content = "".join(blocks)
    return content


def parse_admonition(content: str) -> str | None:
    """Format content as an admonition."""
    lines = content.strip().split("\n")
    first = lines[0].strip()
    rest = lines[1:]
    if not (first.startswith("{") and first.endswith("}")):
        return None
    admonition_type = first[1:-1].strip()
    ret = f"!!! {admonition_type}\n\n"
    for line in rest:
        ret += f"    {line.strip()}\n"
    return ret


if __name__ == "__main__":
    paths = [0] if len(sys.argv) < 2 else sys.argv[1:]

    for path in paths:
        with open(path) as file:  # noqa: PTH123
            content = file.read()

        content = _process_content(content)

        if path == 0:
            sys.stdout.write(content)
        else:
            with open(path, "w") as file:  # noqa: PTH123
                file.write(content)
