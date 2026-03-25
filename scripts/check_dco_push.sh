#!/usr/bin/env bash
# DCO check for GitHub push events (before / after SHAs).
# Usage: scripts/check_dco_push.sh <before_sha> <after_sha>
# - Normal push: verifies commits in before..after (non-merge).
# - before = 40×0 (new ref): verifies all non-merge commits reachable from after.
# - before == after: empty update, success.
# Merge commits are skipped (--no-merges).
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <before_sha> <after_sha>" >&2
  exit 2
fi

BEFORE="$1"
AFTER="$2"
ZERO="0000000000000000000000000000000000000000"

if [[ "$BEFORE" == "$AFTER" ]]; then
  echo "DCO push: empty range, OK"
  exit 0
fi

if ! git cat-file -e "${AFTER}^{commit}" 2>/dev/null; then
  echo "check_dco_push: unknown after commit: $AFTER" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$BEFORE" != "$ZERO" ]] && git cat-file -e "${BEFORE}^{commit}" 2>/dev/null; then
  exec bash "$SCRIPT_DIR/check_dco.sh" "$BEFORE" "$AFTER"
fi

# New ref or missing before: check full non-merge history reachable from AFTER
missing=()
while IFS= read -r commit; do
  [[ -z "$commit" ]] && continue
  msg=$(git show -s --format=%B "$commit")
  if ! printf '%s\n' "$msg" | grep -qE '^Signed-off-by:'; then
    missing+=("$commit")
  fi
done < <(git rev-list --no-merges "$AFTER")

if [[ ${#missing[@]} -gt 0 ]]; then
  echo "DCO: the following commit(s) lack a Signed-off-by line in the message:" >&2
  for c in "${missing[@]}"; do
    echo "  - $c $(git show -s --format=%s "$c")" >&2
  done
  echo >&2
  echo "Fix with: git rebase -i ... and git commit -s, or amend with git commit -s --amend" >&2
  exit 1
fi

echo "DCO push: OK (full scan to ${AFTER:0:7})"
