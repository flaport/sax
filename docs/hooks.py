"""MkDocs hooks for SAX-specific preprocessing."""

import hashlib
import shutil
import subprocess
from pathlib import Path
from typing import Any

import sax


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


def on_page_content(
    html: str, page: Any, config: Any, files: Any, **kwargs: Any
) -> str:
    """Called after markdown is converted to HTML."""
    return html


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
    if code_block_type == "svgbob":
        return _format_svgbob(rest)
    return _format_admonition(code_block_type, rest)


def _format_admonition(admonition_type: str, lines: list[str]) -> str:
    """Format lines as an admonition."""
    print("ADMONITION!")
    print(admonition_type)
    print("\n".join(lines))
    ret = f"!!! {admonition_type}\n\n"
    for line in lines:
        ret += f"    {line.strip()}\n"
    return ret


def _format_svgbob(lines: list[str]) -> str | None:
    """Format lines as a svgbob code block."""
    print("SVGBOB!")
    print("\n".join(lines))
    if not lines:
        return None

    # Join the lines to create the ASCII art content
    content = "\n".join(lines)

    # Create a hash of the content to generate a unique filename
    content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
    svg_filename = f"svgbob_{content_hash}.svg"
    txt_filename = f"svgbob_{content_hash}.txt"

    # Define the path where the SVG will be saved
    docs_dir = Path(__file__).resolve().parent
    assets_dir = docs_dir / "assets" / "svgbob"
    svg_path = assets_dir / svg_filename
    txt_path = assets_dir / txt_filename

    # Create assets directory if it doesn't exist
    assets_dir.mkdir(exist_ok=True)

    # Check if SVG already exists (cache)
    if svg_path.exists():
        return f"![svgbob diagram](assets/svgbob/{svg_filename})\n"

    txt_path.write_text(content)
    svgbob = shutil.which("svgbob_cli")
    if not svgbob:
        print("Warning: svgbob_cli is not installed or not found in PATH.")  # noqa: T201
        return None
    try:
        subprocess.check_call(  # noqa: S603
            [svgbob, str(txt_path), "-o", str(svg_path)],
        )
    except Exception as e:  # noqa: BLE001
        print(f"Warning: Error generating SVG with svgbob: {e}")  # noqa: T201
        return None

    return f"![svgbob diagram](assets/svgbob/{svg_filename})\n"
