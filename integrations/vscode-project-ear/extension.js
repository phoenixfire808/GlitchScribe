"use strict";

const crypto = require("node:crypto");
const fs = require("node:fs/promises");
const os = require("node:os");
const path = require("node:path");
const {
  buildInventoryPayload,
  buildMetadataPayload,
  sendInventory,
  validateDescriptor,
} = require("./protocol");

function workspaceRelativePath(vscode, uri) {
  if (!uri || uri.scheme !== "file" || !vscode.workspace.getWorkspaceFolder(uri)) {
    return undefined;
  }
  return vscode.workspace.asRelativePath(uri, true);
}

function diagnosticCode(value) {
  if (typeof value === "string" || typeof value === "number") {
    return String(value);
  }
  if (value && (typeof value.value === "string" || typeof value.value === "number")) {
    return String(value.value);
  }
  return "unknown";
}

function collectEditorMetadata(vscode, options = {}) {
  const approveActiveFile = options.approveActiveFile === true;
  const approveDiagnostics = options.approveDiagnostics === true;
  const maximum = Math.max(1, Math.min(Number(options.maxDiagnostics) || 100, 200));
  let activeFile;
  if (approveActiveFile) {
    const document = vscode.window.activeTextEditor?.document;
    const relativePath = document && workspaceRelativePath(vscode, document.uri);
    if (relativePath && document.languageId) {
      activeFile = {
        relative_path: relativePath,
        language_id: document.languageId,
      };
    }
  }
  const diagnostics = [];
  if (approveDiagnostics) {
    for (const [uri, values] of vscode.languages.getDiagnostics()) {
      const relativePath = workspaceRelativePath(vscode, uri);
      if (!relativePath || !Array.isArray(values)) continue;
      for (const value of values) {
        if (!value?.range?.start || !value?.range?.end) continue;
        diagnostics.push({
          relative_path: relativePath,
          severity: value.severity,
          code: diagnosticCode(value.code),
          source: value.source || "unknown",
          start_line: value.range.start.line,
          start_character: value.range.start.character,
          end_line: value.range.end.line,
          end_character: value.range.end.character,
        });
        if (diagnostics.length >= maximum) break;
      }
      if (diagnostics.length >= maximum) break;
    }
  }
  return { activeFile, diagnostics };
}

function createProjectEar(vscode, dependencies = {}) {
  const fileSystem = dependencies.fs || fs;
  const protocol = dependencies.protocol || {
    buildInventoryPayload,
    buildMetadataPayload,
    sendInventory,
    validateDescriptor,
  };
  const clientId = dependencies.clientId || `editor-${crypto.randomUUID()}`;
  const localAppData = dependencies.localAppData || process.env.LOCALAPPDATA ||
    path.join(os.homedir(), "AppData", "Local");
  const rendezvousPath = path.join(localAppData, "DrewLocalVoice", "project-ear.json");
  let debounceTimer;
  let statusItem;
  let lastStatus = "Waiting for GlitchScribe";

  function setStatus(text, kind = "waiting") {
    lastStatus = text;
    if (!statusItem) return;
    statusItem.text = kind === "ready" ? `$(radio-tower) Ear: ${text}` : `$(circle-slash) Ear: ${text}`;
    statusItem.tooltip = (
      "GlitchScribe Project Ear sends workspace names and relative paths. " +
      "Active-file/language and diagnostic metadata require separate settings approvals; source bodies are never sent."
    );
    statusItem.command = "glitchscribeProjectEar.showStatus";
    statusItem.show();
  }

  async function readDescriptor() {
    const raw = JSON.parse(await fileSystem.readFile(rendezvousPath, "utf8"));
    return protocol.validateDescriptor(raw);
  }

  async function synchronize({ notify = false } = {}) {
    try {
      const folders = vscode.workspace.workspaceFolders || [];
      if (!folders.length) {
        setStatus("no workspace");
        return { accepted: false, reason: "no_workspace" };
      }
      const configuration = vscode.workspace.getConfiguration("glitchscribeProjectEar");
      const maxFiles = Math.max(100, Math.min(configuration.get("maxFiles", 5000), 5000));
      const exclude = configuration.get(
        "exclude",
        "**/{.git,node_modules,.venv,venv,__pycache__,dist,build,out}/**"
      );
      const uris = await vscode.workspace.findFiles("**/*", exclude, maxFiles);
      const relativePaths = uris.map((uri) => vscode.workspace.asRelativePath(uri, true));
      const approveActiveFile = configuration.get("shareActiveFileMetadata", false) === true;
      const approveDiagnostics = configuration.get("shareDiagnosticMetadata", false) === true;
      const maxDiagnostics = Math.max(
        1,
        Math.min(configuration.get("maxDiagnostics", 100), 200)
      );
      const metadata = collectEditorMetadata(vscode, {
        approveActiveFile,
        approveDiagnostics,
        maxDiagnostics,
      });
      const payload = protocol.buildMetadataPayload(
        clientId,
        folders.map((folder) => folder.name),
        relativePaths,
        maxFiles,
        {
          approveActiveFile,
          approveDiagnostics,
          activeFile: metadata.activeFile,
          diagnostics: metadata.diagnostics,
        }
      );
      const descriptor = await readDescriptor();
      const accepted = await protocol.sendInventory(descriptor, payload, 5000);
      const statusParts = [`${accepted.file_count} files`];
      if (approveActiveFile && metadata.activeFile) statusParts.push("active metadata on");
      if (approveDiagnostics) statusParts.push(`${metadata.diagnostics.length} diagnostics`);
      setStatus(statusParts.join(" • "), "ready");
      if (notify) {
        vscode.window.showInformationMessage(
          `GlitchScribe Project Ear synchronized ${accepted.file_count} relative paths.`
        );
      }
      return accepted;
    } catch (error) {
      const message = error && error.code === "ENOENT"
        ? "GlitchScribe offline"
        : "sync unavailable";
      setStatus(message);
      if (notify) {
        vscode.window.showWarningMessage(`GlitchScribe Project Ear: ${message}.`);
      }
      return { accepted: false, reason: message };
    }
  }

  function scheduleSync() {
    const configuration = vscode.workspace.getConfiguration("glitchscribeProjectEar");
    if (!configuration.get("autoSync", true)) return;
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => synchronize(), 750);
  }

  function activate(context) {
    statusItem = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Right, 90);
    context.subscriptions.push(
      statusItem,
      vscode.commands.registerCommand("glitchscribeProjectEar.sync", () => synchronize({ notify: true })),
      vscode.commands.registerCommand("glitchscribeProjectEar.showStatus", () =>
        vscode.window.showInformationMessage(`GlitchScribe Project Ear: ${lastStatus}.`)
      ),
      vscode.workspace.onDidChangeWorkspaceFolders(scheduleSync),
      vscode.workspace.onDidChangeConfiguration((event) => {
        if (event.affectsConfiguration("glitchscribeProjectEar")) scheduleSync();
      }),
      vscode.window.onDidChangeActiveTextEditor(scheduleSync),
      vscode.languages.onDidChangeDiagnostics(scheduleSync)
    );
    const watcher = vscode.workspace.createFileSystemWatcher("**/*");
    context.subscriptions.push(
      watcher,
      watcher.onDidCreate(scheduleSync),
      watcher.onDidDelete(scheduleSync),
      { dispose }
    );
    setStatus("starting");
    scheduleSync();
    return { synchronize };
  }

  function dispose() {
    clearTimeout(debounceTimer);
  }

  return { activate, dispose, synchronize };
}

function activate(context) {
  const vscode = require("vscode");
  return createProjectEar(vscode).activate(context);
}

function deactivate() {}

module.exports = { activate, collectEditorMetadata, createProjectEar, deactivate };
