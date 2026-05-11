const { app, BrowserWindow, ipcMain } = require("electron");
const path = require("path");
const fs = require("fs");
const { spawn } = require("child_process");

const CONFIG_DIR = path.join(
  app ? app.getPath("home") : process.env.HOME,
  "Library",
  "Application Support",
  "dub-lite"
);
const CONFIG_PATH = path.join(CONFIG_DIR, "config.json");

const DEFAULT_CONFIG = {
  api_base: "http://159.89.182.232",
  default_target_lang: "zh",
};

function loadConfig() {
  try {
    if (!fs.existsSync(CONFIG_PATH)) return { ...DEFAULT_CONFIG };
    const raw = fs.readFileSync(CONFIG_PATH, "utf8");
    return { ...DEFAULT_CONFIG, ...JSON.parse(raw) };
  } catch (err) {
    return { ...DEFAULT_CONFIG, _loadError: err.message };
  }
}

function saveConfig(cfg) {
  fs.mkdirSync(CONFIG_DIR, { recursive: true });
  const sanitized = { ...DEFAULT_CONFIG, ...cfg };
  delete sanitized._loadError;
  fs.writeFileSync(CONFIG_PATH, JSON.stringify(sanitized, null, 2), {
    mode: 0o600,
  });
  return sanitized;
}

const SIDECAR_PYTHON = "/Users/alishali/.pyenv/versions/3.12.0/bin/python3";
const SIDECAR_SCRIPT = path.resolve(
  __dirname,
  "..",
  "chromeExtension",
  "native-host",
  "host.py"
);

let mainWindow = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 480,
    height: 360,
    title: "Dub Lite",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  mainWindow.loadFile("index.html");
}

// Chrome native messaging protocol: 4-byte LE length prefix + JSON body.
// host.py is one-shot per invocation, so we spawn a fresh process per request.
function callSidecar(message) {
  return new Promise((resolve, reject) => {
    const proc = spawn(SIDECAR_PYTHON, [SIDECAR_SCRIPT], {
      stdio: ["pipe", "pipe", "pipe"],
    });

    const chunks = [];
    let stderr = "";

    proc.stdout.on("data", (chunk) => chunks.push(chunk));
    proc.stderr.on("data", (chunk) => (stderr += chunk.toString()));

    proc.on("error", reject);

    proc.on("close", (code) => {
      if (code !== 0 && chunks.length === 0) {
        return reject(new Error(`sidecar exit ${code}: ${stderr}`));
      }
      try {
        const buf = Buffer.concat(chunks);
        if (buf.length < 4) {
          return reject(new Error(`sidecar short response: ${buf.length} bytes`));
        }
        const len = buf.readUInt32LE(0);
        const body = buf.subarray(4, 4 + len).toString("utf8");
        resolve(JSON.parse(body));
      } catch (err) {
        reject(new Error(`sidecar parse: ${err.message} | stderr: ${stderr}`));
      }
    });

    const payload = Buffer.from(JSON.stringify(message), "utf8");
    const header = Buffer.alloc(4);
    header.writeUInt32LE(payload.length, 0);
    proc.stdin.write(header);
    proc.stdin.write(payload);
    proc.stdin.end();
  });
}

ipcMain.handle("sidecar:ping", async () => {
  const start = Date.now();
  const response = await callSidecar({ type: "ping" });
  return { response, latencyMs: Date.now() - start };
});

ipcMain.handle("sidecar:dub", async (_event, { url, targetLang }) =>
  callSidecar({ type: "dub-url", url, target_lang: targetLang })
);

ipcMain.handle("sidecar:poll", async (_event, { jobId }) =>
  callSidecar({ type: "poll-job", job_id: jobId })
);

ipcMain.handle("settings:load", async () => loadConfig());

ipcMain.handle("settings:save", async (_event, cfg) => {
  const saved = saveConfig(cfg);
  return { saved };
});

app.whenReady().then(createWindow);

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
