# Senpai agent plugin

This directory is the OpenHands-native integration point for Senpai workflow
capabilities. Its manifest lives at `.plugin/plugin.json`.

OpenHands receives this directory through `PluginSource` before the first user
message. It natively loads:

- `skills/` for GitHub routing and experiment workflow helpers;
- `.mcp.json` for the Exa research tool; and
- optional future `agents/` and `hooks/hooks.json` resources.

Keep Senpai-owned workflow skills here instead of relying on a provider's user
skill directory. Add `${ENV_VAR}` placeholders for secrets in `.mcp.json`; the
OpenHands conversation expands them at startup, and secret values must not be
committed. The plugin remains the source of truth.
