#!/usr/bin/env bash
# Build the pure-Python wheel of shapekit that the Lab's "Run it yourself"
# panel installs in Pyodide (docs/lab/py/shapekit-0.1.0-py3-none-any.whl).
#
#   bash scripts/build_lab_wheel.sh            # -> docs/lab/py/<wheel>, prints its name
#
# Uses `python -m build --wheel` (PyPA build) and falls back to `pip wheel .`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/docs/lab/py"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cd "$ROOT"
HAD_BUILD_DIR=$([ -d build ] && echo 1 || echo 0)
HAD_EGG_INFO=$([ -d shapekit.egg-info ] && echo 1 || echo 0)
if python -c "import build" 2>/dev/null; then
    python -m build --wheel --outdir "$TMP" >"$TMP/build.log" 2>&1 || { cat "$TMP/build.log"; exit 1; }
else
    pip wheel . --no-deps --no-build-isolation -w "$TMP" >"$TMP/build.log" 2>&1 || { cat "$TMP/build.log"; exit 1; }
fi

# setuptools leaves build/ and shapekit.egg-info/ in the project; remove them unless they were there before
[ "$HAD_BUILD_DIR" = 1 ] || rm -rf "$ROOT/build"
[ "$HAD_EGG_INFO" = 1 ] || rm -rf "$ROOT/shapekit.egg-info"

WHEEL="$(ls "$TMP"/shapekit-*-py3-none-any.whl | head -n 1)"
[ -f "$WHEEL" ] || { echo "no pure-Python wheel was built:"; ls "$TMP"; exit 1; }

mkdir -p "$OUT"
rm -f "$OUT"/shapekit-*.whl
cp "$WHEEL" "$OUT/"
NAME="$(basename "$WHEEL")"
SIZE="$(wc -c <"$OUT/$NAME")"
echo "$NAME"
echo "written to docs/lab/py/$NAME ($SIZE bytes)" >&2
