# GlitchScribe Project Ear

Private, local workspace metadata for GlitchScribe.

## Data boundary

Project Ear sends only:

- workspace display names;
- bounded workspace-relative file paths;
- a random per-editor-window client ID.

It does **not** send file contents, selections, diagnostics, terminal output, commands, environment variables, source-control data, or absolute paths. The GlitchScribe server keeps the latest inventory in memory only.

## Transport

The extension reads `%LOCALAPPDATA%\DrewLocalVoice\project-ear.json`, validates an explicit `127.0.0.1` endpoint, and sends one authenticated JSON `PUT`. It does not follow redirects, use proxies, add telemetry, or fall back to a remote service.

Project Ear is disabled in untrusted workspaces.

## Commands

- `GlitchScribe: Sync Project Ear Metadata`
- `GlitchScribe: Show Project Ear Status`

Automatic sync is debounced after workspace-folder, file-create, and file-delete events. Configure the path limit and exclude glob under **GlitchScribe Project Ear** settings.
