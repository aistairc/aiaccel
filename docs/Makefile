# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: livehtml html apidoc

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
html: apidoc
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

livehtml: apidoc
	sphinx-autobuild -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

apidoc:
	sphinx-apidoc --maxdepth 2 -f -o $(SOURCEDIR)/api_reference/ ../aiaccel/
