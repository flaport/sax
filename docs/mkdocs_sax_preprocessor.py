"""MkDocs extension for SAX-specific preprocessing."""

from typing import Any

from mkdocs.config import config_options
from mkdocs.plugins import BasePlugin
from mkdocs.structure.pages import Page

try:
    import sax
except ImportError:
    sax = None


class SaxPreprocessorConfig:
    """Configuration for SaxPreprocessor plugin."""

    enabled = config_options.Type(bool, default=True)


class SaxPreprocessorPlugin(BasePlugin[SaxPreprocessorConfig]):
    """MkDocs plugin for SAX-specific preprocessing."""

    def on_page_markdown(
        self, markdown: str, page: Page, config: Any, files: Any
    ) -> str:
        """Process markdown content before it's converted to HTML."""
        if not self.config.enabled or sax is None:
            return markdown

        return self._process_content(markdown)

    def _process_content(self, content: str) -> str:
        """Process content by handling code blocks and admonitions."""
        blocks = content.split("```")

        for i, block in enumerate(blocks):
            if i % 2:
                # This is a code block
                if (admonition := self._parse_admonition(block)) is not None:
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

    def _parse_admonition(self, content: str) -> str | None:
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
