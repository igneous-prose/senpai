# Senpai agent plugin

This directory is the runtime-owned integration point for Senpai workflow
capabilities. Claude Code and OpenHands both accept the existing
`.claude-plugin/plugin.json` manifest, so the plugin intentionally has one
manifest rather than provider-specific copies.

OpenHands receives this directory through `PluginSource` before the first user
message. It natively loads:

- `skills/` for GitHub routing and experiment workflow helpers;
- `.mcp.json` for the Exa research tool; and
- optional future `agents/` and `hooks/hooks.json` resources.

Keep Senpai-owned workflow skills here instead of relying on a provider's user
skill directory. Add `${ENV_VAR}` placeholders for secrets in `.mcp.json`; the
OpenHands conversation expands them at startup, and secret values must not be
committed. Provider-specific compatibility code should only adapt names and
paths—the plugin remains the source of truth.
