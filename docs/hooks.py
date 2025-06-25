"""MkDocs hooks for SAX-specific preprocessing."""

import hashlib
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from textwrap import dedent
from typing import Any

import matplotlib.pyplot as plt

import sax

plt.rcParams.update(
    {
        "figure.figsize": (6, 2.5),
        "axes.grid": True,
    }
)


def on_startup(command: str, dirty: bool, **kwargs: Any) -> None:
    """Called once when the MkDocs build starts."""


def on_shutdown(**kwargs: Any) -> None:
    """Called once when the MkDocs build ends."""


def on_serve(server: Any, config: Any, builder: Any, **kwargs: Any) -> None:
    """Called when the MkDocs development server starts."""


def on_config(config: Any, **kwargs: Any) -> Any:
    """Called after config file is loaded but before validation."""
    return config


def on_pre_build(config: Any, **kwargs: Any) -> None:
    """Called before the build starts."""


def on_files(files: Any, config: Any, **kwargs: Any) -> Any:
    """Called after files are gathered but before processing."""
    return files


def on_nav(nav: Any, config: Any, files: Any, **kwargs: Any) -> Any:
    """Called after navigation is built."""
    return nav


def on_env(env: Any, config: Any, files: Any, **kwargs: Any) -> Any:
    """Called after Jinja2 environment is created."""
    return env


def on_post_build(config: Any, **kwargs: Any) -> None:
    """Called after the build is complete."""


def on_pre_template(
    template: Any, template_name: str, config: Any, **kwargs: Any
) -> Any:
    """Called before a template is rendered."""
    return template


def on_template_context(
    context: Any, template_name: str, config: Any, **kwargs: Any
) -> Any:
    """Called after template context is created."""
    return context


def on_post_template(
    output: str, template_name: str, config: Any, **kwargs: Any
) -> str:
    """Called after template is rendered."""
    return output


def on_pre_page(page: Any, config: Any, files: Any, **kwargs: Any) -> Any:
    """Called before a page is processed."""
    return page


def on_page_read_source(page: Any, config: Any, **kwargs: Any) -> str | None:
    """Called to read the raw source of a page."""
    return None


def on_page_markdown(
    markdown: str, page: Any, config: Any, files: Any, **kwargs: Any
) -> str:
    """Process markdown content before it's converted to HTML."""
    blocks = markdown.split("```")

    for i, block in enumerate(blocks):
        if i % 2:
            # This is a code block
            if (special := _parse_special(block)) is not None:
                blocks[i] = special
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


def on_page_content(  # noqa: C901
    html: str, page: Any, config: Any, files: Any, **kwargs: Any
) -> str:
    """Called after markdown is converted to HTML."""
    if "```{svgbob}" not in html:
        return html

    source_parts = []
    for part in html.split("```"):
        if not part.startswith("{svgbob}"):
            continue
        lines = part.split("\n")[1:]
        for i, line in enumerate(lines):
            lines[i] = "".join(re.split(r"[<>]", line)[::2])
        part = dedent("\n".join(lines))
        source_parts.append(lines)

    rendered_parts = []
    for i, part in enumerate(html.split('<div class="language-text highlight">')):
        if i > 0:
            part = f'<div class="language-text highlight">{part}'
        first, *rest = part.split("</span></code></pre></div>")
        rest = "</span></code></pre></div>".join(rest)
        first = f"{first}</span></code></pre></div>"
        rendered_parts.append(first)
        rendered_parts.append(rest)

    for i, part in enumerate(rendered_parts):
        if not part.startswith('<div class="language-text highlight">'):
            continue
        lines = part.split("\n")
        if (svgbob_source := _svgbob_source(lines, source_parts)) is None:
            continue
        if (svg := _svgbob_svg(svgbob_source)) is None:
            continue
        rendered_parts[i] = svg

    return "".join(rendered_parts)


def on_page_context(
    context: Any, page: Any, config: Any, nav: Any, **kwargs: Any
) -> Any:
    """Called after page context is created."""
    return context


def on_post_page(output: str, page: Any, config: Any, **kwargs: Any) -> str:
    """Called after page is fully processed."""
    return output


def _parse_special(content: str) -> str | None:
    """Format contents of a special code block differently."""
    lines = content.strip().split("\n")
    first = lines[0].strip()
    rest = lines[1:]
    if not (first.startswith("{") and first.endswith("}")):
        return None
    code_block_type = first[1:-1].strip()
    return _format_admonition(code_block_type, rest)


def _format_admonition(admonition_type: str, lines: list[str]) -> str:
    """Format lines as an admonition."""
    ret = f"!!! {admonition_type}\n\n"
    for line in lines:
        ret += f"    {line.strip()}\n"
    return ret


def _svgbob_svg(source: str) -> str | None:
    source = source.replace("&lt;", "<").replace("&gt;", ">")
    svgbob = shutil.which("svgbob_cli")
    if not svgbob:
        print("Warning: svgbob_cli is not installed or not found in PATH.")  # noqa: T201
        return None
    content_hash = hashlib.md5(source.encode()).hexdigest()
    txt_filename = f"svgbob_{content_hash}.txt"
    temp_path = Path(tempfile.gettempdir()).resolve() / "svgbob" / txt_filename
    temp_path.parent.mkdir(exist_ok=True)
    try:
        temp_path.write_text(source)
        return subprocess.check_output([svgbob, str(temp_path)]).decode()  # noqa: S603
    except Exception as e:  # noqa: BLE001
        print(f"Warning: Error generating SVG with svgbob: {e}")  # noqa: T201
        return None
    finally:
        temp_path.unlink()


def _svgbob_source(
    rendered_lines: list[str], source_parts: list[list[str]]
) -> str | None:
    for source_lines in source_parts:
        if all(
            sl.strip() in rl
            for sl, rl in zip(source_lines, rendered_lines, strict=False)
        ):
            return "\n".join(source_lines)
    return None
