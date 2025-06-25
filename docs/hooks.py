"""MkDocs hooks for SAX-specific preprocessing."""

from typing import Any

try:
    import sax
except ImportError:
    sax = None


def on_page_markdown(markdown: str, page: Any, config: Any, files: Any) -> str:
    """Process markdown content before it's converted to HTML."""
    if sax is None:
        return markdown

    return _process_content(markdown)


def _process_content(content: str) -> str:
    """Process content by handling code blocks and admonitions."""
    blocks = content.split("```")

    for i, block in enumerate(blocks):
        if i % 2:
            # This is a code block
            if (admonition := _parse_admonition(block)) is not None:
                blocks[i] = admonition
            else:
                blocks[i] = f"```{block}```"
            continue

        # This is regular markdown content
        lines = block.split("\n")
        for j, line in enumerate(lines):
            parts = line.split("`")
            for k, part in enumerate(parts):
                if k % 2 == 0:
                    continue
                # This is inline code
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


def _parse_admonition(content: str) -> str | None:
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
