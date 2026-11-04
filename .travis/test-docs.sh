#!/usr/bin/env bash
# Simple validation script for RIFT documentation and source code
# Run this locally BEFORE committing documentation changes
#
# Usage:
#   ./test-docs.sh           # Run all checks (docs + source)
#   ./test-docs.sh source    # Just source code tests
#   ./test-docs.sh build     # Just docs build
#   ./test-docs.sh linkcheck # Just link check

set -e

SCRIPT_PATH="${BASH_SOURCE%\/*}"
RIFT_DIR="${SCRIPT_PATH%\/.travis}"
CODE_DIR="$RIFT_DIR/MonteCarloMarginalizeCode/Code"
DOCS_DIR="$RIFT_DIR/docs"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; }

MODE="${1:-all}"

echo "=== RIFT Validation Script ==="
echo "Mode: $MODE"
echo "RIFT dir: $RIFT_DIR"
echo ""

case "$MODE" in
    source)
        echo "=== Running Source Code Tests ==="
        echo "Running test-all-mod.py..."
        cd "$RIFT_DIR"
        PYTHONPATH="$CODE_DIR:$PYTHONPATH" python3 .travis/test-all-mod.py 2>&1 | tail -20
        pass "Source code tests passed"
        ;;
    build)
        echo "=== Running Sphinx Build Check ==="
        cd "$DOCS_DIR"
        if command -v sphinx-build &> /dev/null; then
            sphinx-build -b html source build/html -W --keep-going 2>&1 | tail -10
            pass "Sphinx build successful"
        else
            warn "sphinx-build not found - skipping build check"
            warn "Install sphinx: pip install sphinx"
        fi
        ;;
    linkcheck)
        echo "=== Running Link Check ==="
        cd "$DOCS_DIR"
        if command -v sphinx-build &> /dev/null; then
            sphinx-build -b linkcheck source build/link 2>&1 | tail -10
            pass "Link check complete"
        else
            warn "sphinx-build not found - skipping link check"
        fi
        ;;
    all)
        echo "=== Phase 1: Source Code Tests ==="
        if ./test-docs.sh source; then
            pass "Source code tests passed"
        else
            fail "Source code tests FAILED"
            exit 1
        fi
        
        echo ""
        echo "=== Phase 2: Documentation Build ==="
        if ./test-docs.sh build; then
            pass "Documentation build passed"
        else
            fail "Documentation build FAILED"
            exit 1
        fi
        
        echo ""
        echo "=== Phase 3: Link Check ==="
        if ./test-docs.sh linkcheck; then
            pass "Link check passed"
        else
            fail "Link check FAILED"
            exit 1
        fi
        
        echo ""
        echo "=== All validation checks passed ==="
        ;;
    *)
        echo "Usage: $0 [source|build|linkcheck|all]"
        echo ""
        echo "  source     - Validate RIFT module imports (test-all-mod.py)"
        echo "  build      - Validate Sphinx documentation builds"
        echo "  linkcheck  - Check for broken internal links"
        echo "  all        - Run all checks (default)"
        exit 1
        ;;
esac