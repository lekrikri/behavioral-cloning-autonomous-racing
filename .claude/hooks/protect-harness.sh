#!/usr/bin/env bash
# PreToolUse(Edit|Write): editing the project harness (CLAUDE.md + .claude/**) is allowed
# but NEVER silent — it always requires an explicit confirmation. The team rules must only
# change as a deliberate, justified act (ideally a PR), not as incidental agent drift.
# Pre-consent for a batch of deliberate edits: run the session with ALLOW_HARNESS_EDIT=1.
set -euo pipefail

input=$(cat)

if command -v jq >/dev/null 2>&1; then
  path=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')
else
  path=$(printf '%s' "$input" | grep -o '"file_path"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | sed 's/.*:[[:space:]]*"//; s/"$//')
fi

[ -z "${path:-}" ] && exit 0
[ "${ALLOW_HARNESS_EDIT:-0}" = "1" ] && exit 0

# Personal, gitignored overrides are never gated.
case "$path" in
  *settings.local.json|*CLAUDE.local.md) exit 0 ;;
esac

json_escape() {
  if command -v jq >/dev/null 2>&1; then
    printf '%s' "$1" | jq -Rs .
  else
    printf '"%s"' "$(printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g' | tr '\n' ' ')"
  fi
}

case "$path" in
  CLAUDE.md|*/CLAUDE.md|.claude/*|*/.claude/*)
    reason="'$path' fait partie du harness projet (regles d equipe). Modifier ces regles doit etre un choix explicite et justifie, idealement via PR. Confirme uniquement si c est intentionnel — voir .claude/rules/50-discipline.md."
    printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"ask","permissionDecisionReason":%s}}\n' "$(json_escape "$reason")"
    exit 0
    ;;
esac

exit 0
