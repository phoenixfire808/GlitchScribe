"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");
const test = require("node:test");

const protocolPath = path.join(
  __dirname,
  "..",
  "integrations",
  "vscode-project-ear",
  "protocol.js"
);

test("Project Ear protocol is strict loopback and metadata bounded", () => {
  assert.equal(fs.existsSync(protocolPath), true, "protocol.js must exist");
  const { buildInventoryPayload, validateDescriptor } = require(protocolPath);

  const descriptor = validateDescriptor({
    protocol: 1,
    host: "127.0.0.1",
    port: 49152,
    token: "1234567890abcdef",
  });
  assert.equal(descriptor.port, 49152);
  for (const host of ["localhost", "192.168.1.2", "api.example.com"]) {
    assert.throws(
      () => validateDescriptor({ ...descriptor, host }),
      /loopback/i
    );
  }

  const payload = buildInventoryPayload(
    "editor-window-1",
    ["GlitchScribe"],
    ["local_voice.py", "tests\\test_core.py", "local_voice.py", "extra.py"],
    3
  );
  assert.deepEqual(payload, {
    protocol: 1,
    client_id: "editor-window-1",
    workspace_names: ["GlitchScribe"],
    relative_paths: ["local_voice.py", "tests/test_core.py", "extra.py"],
  });
  assert.equal(JSON.stringify(payload).includes("file://"), false);
});

test("Project Ear v2 emits only separately approved editor metadata", () => {
  const protocol = require(protocolPath);
  assert.equal(typeof protocol.buildMetadataPayload, "function");
  const payload = protocol.buildMetadataPayload(
    "editor-window-1",
    ["GlitchScribe"],
    ["src/main.py"],
    5000,
    {
      approveActiveFile: true,
      approveDiagnostics: true,
      activeFile: { relative_path: "src/main.py", language_id: "python" },
      diagnostics: [{
        relative_path: "src/main.py",
        severity: 1,
        code: "E0602",
        source: "pylint",
        start_line: 7,
        start_character: 3,
        end_line: 7,
        end_character: 12,
        message: "private source-derived detail",
      }],
    }
  );

  assert.equal(payload.protocol, 2);
  assert.deepEqual(payload.metadata_approvals, {
    active_file: true,
    diagnostics: true,
  });
  assert.deepEqual(payload.active_file, {
    relative_path: "src/main.py",
    language_id: "python",
  });
  assert.equal(payload.diagnostics[0].code, "E0602");
  assert.equal(Object.hasOwn(payload.diagnostics[0], "message"), false);
  assert.equal(JSON.stringify(payload).includes("private source-derived detail"), false);

  const disabled = protocol.buildMetadataPayload(
    "editor-window-1",
    ["GlitchScribe"],
    ["src/main.py"],
    5000,
    {
      approveActiveFile: false,
      approveDiagnostics: false,
      activeFile: { relative_path: "src/main.py", language_id: "python" },
      diagnostics: payload.diagnostics,
    }
  );
  assert.equal(Object.hasOwn(disabled, "active_file"), false);
  assert.deepEqual(disabled.diagnostics, []);
});

test("Project Ear transport authenticates one loopback inventory PUT", async () => {
  const { sendInventory } = require(protocolPath);
  assert.equal(typeof sendInventory, "function");
  const received = [];
  const server = http.createServer((request, response) => {
    let body = "";
    request.setEncoding("utf8");
    request.on("data", (chunk) => { body += chunk; });
    request.on("end", () => {
      received.push({
        method: request.method,
        url: request.url,
        authorization: request.headers.authorization,
        origin: request.headers.origin,
        body: JSON.parse(body),
      });
      response.writeHead(202, { "Content-Type": "application/json" });
      response.end('{"accepted":true,"file_count":1}');
    });
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    const result = await sendInventory(
      {
        protocol: 1,
        host: "127.0.0.1",
        port: server.address().port,
        token: "1234567890abcdef",
      },
      {
        protocol: 1,
        client_id: "editor-window-1",
        workspace_names: ["Project"],
        relative_paths: ["src/main.js"],
      },
      2000
    );
    assert.equal(result.file_count, 1);
    assert.deepEqual(received, [{
      method: "PUT",
      url: "/v1/workspace-inventory",
      authorization: "Bearer 1234567890abcdef",
      origin: undefined,
      body: {
        protocol: 1,
        client_id: "editor-window-1",
        workspace_names: ["Project"],
        relative_paths: ["src/main.js"],
      },
    }]);
  } finally {
    await new Promise((resolve) => server.close(resolve));
  }
});

test("Project Ear extension manifest is local, bounded, and dependency free", () => {
  const extensionRoot = path.dirname(protocolPath);
  const manifestPath = path.join(extensionRoot, "package.json");
  const extensionPath = path.join(extensionRoot, "extension.js");
  assert.equal(fs.existsSync(manifestPath), true, "package.json must exist");
  assert.equal(fs.existsSync(extensionPath), true, "extension.js must exist");
  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  assert.deepEqual(manifest.extensionKind, ["ui"]);
  assert.equal(manifest.capabilities.untrustedWorkspaces.supported, false);
  assert.equal(manifest.main, "./extension.js");
  assert.deepEqual(manifest.dependencies || {}, {});
  const commands = manifest.contributes.commands.map((item) => item.command);
  assert.deepEqual(commands.sort(), [
    "glitchscribeProjectEar.showStatus",
    "glitchscribeProjectEar.sync",
  ]);
  const implementation = require(extensionPath);
  assert.equal(typeof implementation.activate, "function");
  assert.equal(typeof implementation.createProjectEar, "function");
  assert.equal(typeof implementation.collectEditorMetadata, "function");
});

test("Project Ear collects only approved workspace-contained editor metadata", () => {
  const extensionRoot = path.dirname(protocolPath);
  const implementation = require(path.join(extensionRoot, "extension.js"));
  assert.equal(typeof implementation.collectEditorMetadata, "function");
  const uri = { scheme: "file", fsPath: "C:/work/src/main.py" };
  const outside = { scheme: "file", fsPath: "C:/private/secret.py" };
  const vscode = {
    window: {
      activeTextEditor: {
        document: {
          uri,
          languageId: "python",
          getText() { throw new Error("source body must never be read"); },
        },
      },
    },
    workspace: {
      getWorkspaceFolder(candidate) {
        return candidate === uri ? { name: "Project" } : undefined;
      },
      asRelativePath(candidate) {
        if (candidate === uri) return "src/main.py";
        return "C:/private/secret.py";
      },
    },
    languages: {
      getDiagnostics() {
        return [
          [uri, [{
            severity: 1,
            code: { value: "E0602", target: "https://example.invalid" },
            source: "pylint",
            message: "private source-derived detail",
            range: {
              start: { line: 7, character: 3 },
              end: { line: 7, character: 12 },
            },
          }]],
          [outside, [{
            severity: 0,
            code: "SECRET",
            source: "outside",
            message: "must not cross the workspace boundary",
            range: {
              start: { line: 0, character: 0 },
              end: { line: 0, character: 1 },
            },
          }]],
        ];
      },
    },
  };

  const metadata = implementation.collectEditorMetadata(vscode, {
    approveActiveFile: true,
    approveDiagnostics: true,
    maxDiagnostics: 25,
  });
  assert.deepEqual(metadata.activeFile, {
    relative_path: "src/main.py",
    language_id: "python",
  });
  assert.equal(metadata.diagnostics.length, 1);
  assert.equal(metadata.diagnostics[0].code, "E0602");
  assert.equal(Object.hasOwn(metadata.diagnostics[0], "message"), false);
  assert.equal(JSON.stringify(metadata).includes("private source-derived detail"), false);

  const disabled = implementation.collectEditorMetadata(vscode, {
    approveActiveFile: false,
    approveDiagnostics: false,
    maxDiagnostics: 25,
  });
  assert.deepEqual(disabled, { activeFile: undefined, diagnostics: [] });
});
