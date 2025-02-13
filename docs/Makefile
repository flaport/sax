SPHINXOPTS    ?=
SOURCEDIR     = source
BUILDDIR      = build

help:
	uv run sphinx-build -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

%: Makefile
	uv run sphinx-build -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html:
	uv run sphinx-build -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	find "$(BUILDDIR)" -name "*.html" | xargs sed -i 's|docs/examples|examples|g'
	find "$(BUILDDIR)" -name "*.html" | xargs sed -i 's|docs/internals|internals|g'
	find "$(BUILDDIR)" -name "*.html" | xargs sed -i 's|docs/source/_|_|g'

.PHONY: help Makefile html
