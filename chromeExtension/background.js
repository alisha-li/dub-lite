console.log("[dub-lite] background service worker started");

var HOST_NAME = "com.dub_lite.host";

function sendNative(payload, sendResponse) {
  chrome.runtime.sendNativeMessage(HOST_NAME, payload, function(response) {
    if (chrome.runtime.lastError) {
      console.error("[dub-lite] native error:", chrome.runtime.lastError.message);
      sendResponse({ error: chrome.runtime.lastError.message });
      return;
    }
    sendResponse(response);
  });
}

chrome.runtime.onMessage.addListener(function(message, sender, sendResponse) {

  // NEW: Mac app calls Modal directly. Pipeline yt-dlp → Spaces upload → Modal spawn.
  if (message.type === "dub-url") {
    console.log("[dub-lite] starting dub for:", message.url);
    sendNative({
      type: "dub-url",
      url: message.url,
      target_lang: message.target_lang || "zh",
    }, sendResponse);
    return true;
  }

  // NEW: poll Modal job status via native host (no API server).
  if (message.type === "poll-job") {
    sendNative({ type: "poll-job", job_id: message.job_id }, sendResponse);
    return true;
  }

  if (message.type === "ping") {
    sendNative({ type: "ping" }, sendResponse);
    return true;
  }

  if (message.type === "transcribe-url") {
    sendNative({
      type: "transcribe-url",
      url: message.url,
      language: message.language,
    }, sendResponse);
    return true;
  }

  // LEGACY: API server path. Kept temporarily for fallback.
  if (message.type === "download-audio") {
    sendNative({
      type: "download-audio",
      url: message.url,
      target_lang: message.target_lang || "zh",
      api_base: message.api_base || "http://localhost:8000",
    }, sendResponse);
    return true;
  }

  if (message.type === "poll-job-api") {
    fetch(message.api_base + "/api/jobs/" + message.job_id)
      .then(function(r) { return r.json(); })
      .then(function(job) { sendResponse({ job: job }); })
      .catch(function(err) { sendResponse({ error: err.message }); });
    return true;
  }
});
