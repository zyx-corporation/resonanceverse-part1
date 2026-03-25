#!/usr/bin/env bash
# Verify each commit in BASE..HEAD contains a Signed-off-by line (DCO).
# Usage: scripts/check_dco.sh <base_sha> <head_sha>
# Merge commits are skipped (--no-merges).
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <base_sha> <head_sha>" >&2
  exit 2
fi

BASE="$1"
HEAD="$2"

if ! git cat-file -e "${BASE}^{commit}" 2>/dev/null; then
  echo "check_dco: unknown base commit: $BASE" >&2
  exit 1
fi
if ! git cat-file -e "${HEAD}^{commit}" 2>/dev/null; then
  echo "check_dco: unknown head commit: $HEAD" >&2
  exit 1
fi

missing=()
while IFS= read -r commit; do
  [[ -z "$commit" ]] && continue
  msg=$(git show -s --format=%B "$commit")
  if ! printf '%s\n' "$msg" | grep -qE '^Signed-off-by:'; then
    missing+=("$commit")
  fi
done < <(git rev-list --no-merges "${BASE}..${HEAD}")

if [[ ${#missing[@]} -gt 0 ]]; then
  echo "DCO: the following commit(s) lack a Signed-off-by line in the message:" >&2
  for c in "${missing[@]}"; do
    echo "  - $c $(git show -s --format=%s "$c")" >&2
  done
  echo >&2
  echo "Fix with: git rebase -i ... and git commit -s, or amend with git commit -s --amend" >&2
  exit 1
fi

echo "DCO: OK (${BASE:0:7}..${HEAD:0:7})"
