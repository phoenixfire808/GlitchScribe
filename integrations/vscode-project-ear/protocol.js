"use strict";

const http = require("node:http");

function boundedText(value, label, maximum) {
  if (typeof value !== "string") {
    throw new Error(`${label} must be text`);
  }
  const rendered = value.trim();
  if (!rendered || rendered.length > maximum || rendered.includes("\0")) {
    throw new Error(`${label} exceeds its bound`);
  }
  return rendered;
}

function validateDescriptor(value) {
  if (!value || value.protocol !== 1) {
    throw new Error("Unsupported Project Ear protocol");
  }
  if (value.host !== "127.0.0.1") {
    throw new Error("Project Ear endpoint must use explicit loopback");
  }
  if (!Number.isInteger(value.port) || value.port < 1 || value.port > 65535) {
    throw new Error("Project Ear port is invalid");
  }
  const token = boundedText(value.token, "Project Ear token", 256);
  if (token.length < 16) {
    throw new Error("Project Ear token is too short");
  }
  return { protocol: 1, host: value.host, port: value.port, token };
}

function safeRelativePath(value) {
  const rendered = boundedText(value, "relative path", 512).replaceAll("\\", "/");
  if (rendered.startsWith("/") || /^[A-Za-z]:\//.test(rendered)) {
    throw new Error("Absolute paths are not allowed");
  }
  if (rendered.split("/").some((part) => !part || part === "." || part === "..")) {
    throw new Error("Path traversal is not allowed");
  }
  return rendered;
}

function buildInventoryPayload(clientId, workspaceNames, relativePaths, maxFiles = 5000) {
  const boundedMaximum = Math.max(1, Math.min(Number(maxFiles) || 5000, 5000));
  if (!Array.isArray(workspaceNames) || workspaceNames.length > 32) {
    throw new Error("workspace names must be a bounded array");
  }
  if (!Array.isArray(relativePaths)) {
    throw new Error("relative paths must be an array");
  }
  const unique = [];
  const seen = new Set();
  for (const candidate of relativePaths) {
    const rendered = safeRelativePath(candidate);
    if (!seen.has(rendered)) {
      seen.add(rendered);
      unique.push(rendered);
    }
    if (unique.length >= boundedMaximum) {
      break;
    }
  }
  return {
    protocol: 1,
    client_id: boundedText(clientId, "client ID", 128),
    workspace_names: workspaceNames.map((name) => boundedText(name, "workspace name", 128)),
    relative_paths: unique,
  };
}

function boundedInteger(value, label, maximum) {
  if (!Number.isInteger(value) || value < 0 || value > maximum) {
    throw new Error(`${label} exceeds its bound`);
  }
  return value;
}

function buildMetadataPayload(
  clientId,
  workspaceNames,
  relativePaths,
  maxFiles = 5000,
  options = {}
) {
  const base = buildInventoryPayload(clientId, workspaceNames, relativePaths, maxFiles);
  const approveActiveFile = options.approveActiveFile === true;
  const approveDiagnostics = options.approveDiagnostics === true;
  const payload = {
    ...base,
    protocol: 2,
    metadata_approvals: {
      active_file: approveActiveFile,
      diagnostics: approveDiagnostics,
    },
    diagnostics: [],
  };
  if (approveActiveFile && options.activeFile) {
    payload.active_file = {
      relative_path: safeRelativePath(options.activeFile.relative_path),
      language_id: boundedText(options.activeFile.language_id, "language ID", 64),
    };
  }
  if (approveDiagnostics) {
    const values = Array.isArray(options.diagnostics) ? options.diagnostics : [];
    payload.diagnostics = values.slice(0, 200).map((value) => ({
      relative_path: safeRelativePath(value.relative_path),
      severity: boundedInteger(value.severity, "diagnostic severity", 3),
      code: boundedText(String(value.code), "diagnostic code", 128),
      source: boundedText(value.source || "unknown", "diagnostic source", 64),
      start_line: boundedInteger(value.start_line, "diagnostic start line", 10000000),
      start_character: boundedInteger(
        value.start_character,
        "diagnostic start character",
        1000000
      ),
      end_line: boundedInteger(value.end_line, "diagnostic end line", 10000000),
      end_character: boundedInteger(
        value.end_character,
        "diagnostic end character",
        1000000
      ),
    }));
  }
  return payload;
}

function sendInventory(rawDescriptor, payload, timeoutMs = 5000) {
  const descriptor = validateDescriptor(rawDescriptor);
  const body = Buffer.from(JSON.stringify(payload), "utf8");
  if (body.length < 1 || body.length > 262144) {
    return Promise.reject(new Error("Project Ear payload exceeds transport bound"));
  }
  return new Promise((resolve, reject) => {
    const request = http.request({
      hostname: descriptor.host,
      port: descriptor.port,
      path: "/v1/workspace-inventory",
      method: "PUT",
      headers: {
        Authorization: `Bearer ${descriptor.token}`,
        "Content-Type": "application/json",
        "Content-Length": body.length,
      },
      timeout: Math.max(500, Math.min(Number(timeoutMs) || 5000, 15000)),
    }, (response) => {
      const chunks = [];
      let size = 0;
      response.on("data", (chunk) => {
        size += chunk.length;
        if (size > 65536) {
          request.destroy(new Error("Project Ear response exceeds transport bound"));
          return;
        }
        chunks.push(chunk);
      });
      response.on("end", () => {
        if (response.statusCode !== 202) {
          reject(new Error(`Project Ear rejected inventory (${response.statusCode})`));
          return;
        }
        try {
          resolve(JSON.parse(Buffer.concat(chunks).toString("utf8")));
        } catch (_error) {
          reject(new Error("Project Ear returned invalid JSON"));
        }
      });
    });
    request.on("timeout", () => request.destroy(new Error("Project Ear timed out")));
    request.on("error", reject);
    request.end(body);
  });
}

module.exports = {
  buildInventoryPayload,
  buildMetadataPayload,
  sendInventory,
  validateDescriptor,
};
