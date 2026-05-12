const POLL_INTERVAL_MS = 2000;

const urlInput = document.getElementById("url");
const langSelect = document.getElementById("lang");
const dubBtn = document.getElementById("dub-btn");
const progressSection = document.getElementById("progress-section");
const stageText = document.getElementById("stage-text");
const stagePct = document.getElementById("stage-pct");
const barFill = document.getElementById("bar-fill");
const resultBox = document.getElementById("result");
const errMsg = document.getElementById("err-msg");

const pingBtn = document.getElementById("ping-btn");
const dot = document.getElementById("dot");
const statusText = document.getElementById("status-text");

const settingsBtn = document.getElementById("settings-btn");
const mainPanel = document.getElementById("main-panel");
const settingsPanel = document.getElementById("settings-panel");
const saveSettingsBtn = document.getElementById("save-settings-btn");
const cancelSettingsBtn = document.getElementById("cancel-settings-btn");
const settingsStatus = document.getElementById("settings-status");

const apiBaseInput = document.getElementById("api-base");
const defaultLangSelect = document.getElementById("default-lang");

let pollTimer = null;

function showProgress(stage, pct, { done = false, err = false } = {}) {
  progressSection.classList.add("visible");
  stageText.textContent = stage;
  stagePct.textContent = `${Math.round(pct)}%`;
  barFill.style.width = `${pct}%`;
  barFill.classList.toggle("done", done);
  barFill.classList.toggle("err", err);
}

function showResult(outputUrl) {
  resultBox.style.display = "block";
  resultBox.innerHTML = `<strong>Done.</strong><br/><a href="${outputUrl}" target="_blank">${outputUrl}</a>`;
}

function showError(msg) {
  const firstLine = String(msg).split("\n")[0];
  errMsg.textContent = firstLine.length < String(msg).length
    ? `${firstLine}\n\n--- full ---\n${msg}`
    : firstLine;
}

function resetUI() {
  progressSection.classList.remove("visible");
  resultBox.style.display = "none";
  resultBox.innerHTML = "";
  errMsg.textContent = "";
  barFill.classList.remove("done", "err");
  barFill.style.width = "0%";
}

async function startDub() {
  const url = urlInput.value.trim();
  if (!url) {
    showError("Enter a URL first.");
    return;
  }
  resetUI();
  dubBtn.disabled = true;
  dubBtn.textContent = "Working…";
  showProgress("Downloading audio…", 0);

  try {
    const dubResp = await window.dub.dub(url, langSelect.value);
    if (dubResp.error) throw new Error(dubResp.error);
    if (!dubResp.job_id) throw new Error("no job_id in response");
    pollJob(dubResp.job_id);
  } catch (err) {
    showProgress("Failed", 0, { err: true });
    showError(err.message || String(err));
    dubBtn.disabled = false;
    dubBtn.textContent = "Dub video";
  }
}

function pollJob(jobId) {
  const tick = async () => {
    try {
      const resp = await window.dub.poll(jobId);
      if (resp.error) {
        showProgress("Failed", 0, { err: true });
        showError(resp.error);
        cleanup();
        return;
      }
      const pct = resp.progress || 0;
      const stage = resp.stage || resp.status || "Processing…";
      const status = (resp.status || "").toLowerCase();
      const isDone = status === "completed";
      const isFail = status === "failed";

      if (isDone) {
        showProgress("Done", 100, { done: true });
        if (resp.output_url) showResult(resp.output_url);
        cleanup();
        return;
      }
      if (isFail) {
        showProgress("Failed", pct, { err: true });
        showError(resp.error || "pipeline failed");
        cleanup();
        return;
      }
      showProgress(stage, pct);
    } catch (err) {
      showError(`poll error: ${err.message || err}`);
      cleanup();
    }
  };
  tick();
  pollTimer = setInterval(tick, POLL_INTERVAL_MS);
}

function cleanup() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
  dubBtn.disabled = false;
  dubBtn.textContent = "Dub video";
}

dubBtn.addEventListener("click", startDub);

pingBtn.addEventListener("click", async () => {
  pingBtn.disabled = true;
  statusText.textContent = "Sidecar: pinging…";
  dot.className = "dot";
  try {
    const { response, latencyMs } = await window.dub.ping();
    if (response && response.pong === true) {
      dot.className = "dot ok";
      statusText.textContent = `Sidecar: connected (${latencyMs}ms)`;
    } else {
      dot.className = "dot err";
      statusText.textContent = `Sidecar: unexpected: ${JSON.stringify(response)}`;
    }
  } catch (err) {
    dot.className = "dot err";
    statusText.textContent = `Sidecar: error: ${err.message || err}`;
  } finally {
    pingBtn.disabled = false;
  }
});

// --- Settings ---

function showPanel(which) {
  if (which === "settings") {
    settingsBtn.classList.add("active");
    mainPanel.classList.remove("visible");
    settingsPanel.classList.add("visible");
  } else {
    settingsBtn.classList.remove("active");
    settingsPanel.classList.remove("visible");
    mainPanel.classList.add("visible");
  }
}

async function loadSettings() {
  const cfg = await window.dub.loadSettings();
  apiBaseInput.value = cfg.api_base || "";
  defaultLangSelect.value = cfg.default_target_lang || "zh";
  langSelect.value = cfg.default_target_lang || "zh";
}

settingsBtn.addEventListener("click", () => {
  const showing = settingsPanel.classList.contains("visible");
  showPanel(showing ? "main" : "settings");
});

cancelSettingsBtn.addEventListener("click", () => {
  loadSettings();
  settingsStatus.textContent = "";
  settingsStatus.className = "settings-status";
  showPanel("main");
});

saveSettingsBtn.addEventListener("click", async () => {
  saveSettingsBtn.disabled = true;
  settingsStatus.textContent = "Saving…";
  settingsStatus.className = "settings-status";

  const cfg = {
    api_base: apiBaseInput.value.trim() || "https://dub-lite.alishali.info",
    default_target_lang: defaultLangSelect.value,
  };

  try {
    await window.dub.saveSettings(cfg);
    settingsStatus.textContent = "Saved.";
    settingsStatus.className = "settings-status ok";
    langSelect.value = cfg.default_target_lang;
  } catch (err) {
    settingsStatus.textContent = `Error: ${err.message || err}`;
    settingsStatus.className = "settings-status err";
  } finally {
    saveSettingsBtn.disabled = false;
  }
});

loadSettings().catch((err) => console.error("loadSettings failed:", err));
