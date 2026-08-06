# Senpai agent plugin

This directory is the OpenHands-native integration point for Senpai workflow
capabilities. Its manifest lives at `.plugin/plugin.json`.

OpenHands receives this directory through `PluginSource` before the first user
message. It natively loads:

- `skills/` as a progressively disclosed workflow catalog; and
- `hooks/hooks.json` for early command-policy and lifecycle feedback.

GitHub mutations and training supervision are native typed Senpai tools, not
skill shell commands. Exa is also a skill/script integration rather than an MCP
server; launch preflight makes one `instant` publication search with one result
to validate the key.

Keep Senpai-owned workflow skills here rather than relying on a provider's user
skill directory. Never commit secret values. The plugin remains the source of
truth for reusable workflow guidance, while Python remains the source of truth
for state transitions.
