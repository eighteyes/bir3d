#!/usr/bin/env bash
# deploy-pages.sh
# Build the Vite site and publish dist/ to the gh-pages branch of origin.
# Responsibilities:
#   - Build the production bundle under the Pages base path (skips tsc for speed)
#   - Promote the glider page (index-bird.html) to the site root index.html
#   - Preserve the original vector-system demo at vector.html
#   - Disable Jekyll processing (.nojekyll) so Vite's _-prefixed assets serve
#   - Force-push the built output to the gh-pages branch of origin
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

REMOTE_URL="$(git remote get-url origin)"
# Derive the Pages base path from the repo name: git@.../bir3d.git -> /bir3d/
REPO_NAME="$(basename "$REMOTE_URL" .git)"
export VITE_BASE="/${REPO_NAME}/"

echo "Building with base ${VITE_BASE} ..."
node_modules/.bin/vite build

cd dist
cp index.html vector.html      # keep the vector-system demo reachable
cp index-bird.html index.html  # glider is the front door
touch .nojekyll

git init -q
git checkout -q -B gh-pages   # -B: create or reset, so re-runs over a leftover dist/.git are idempotent
git add -A
git -c user.email="deploy@bir3d" -c user.name="bir3d-deploy" \
  commit -qm "deploy site"
git push -f "$REMOTE_URL" gh-pages

echo "Published to gh-pages on ${REMOTE_URL}"
