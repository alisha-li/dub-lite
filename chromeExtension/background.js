console.log("[dub-lite] background service worker started");

var HOST_NAME = "com.dub_lite.host";

// Handle messages from the content script
chrome.runtime.onMessage.addListener(function(message, sender, sendResponse) {

  if (message.type === "download-audio") {
    // Send the YouTube URL to the native host, which runs yt-dlp
    chrome.runtime.sendNativeMessage(HOST_NAME, {
      type: "download-audio",
      url: message.url,
      target_lang: message.target_lang || "zh",
      api_base: message.api_base || "http://localhost:8000",
    }, function(response) {
      if (chrome.runtime.lastError) {
        console.error("[dub-lite] native messaging error:", chrome.runtime.lastError.message);
        sendResponse({ error: chrome.runtime.lastError.message });
        return;
      }
      console.log("[dub-lite] got response from native host, size:", response.size);
      sendResponse(response);
    });

    // Return true to indicate we'll call sendResponse asynchronously
    return true;
  }

  if (message.type === "poll-job") {
    // Content scripts can't fetch localhost from youtube.com (Private Network Access blocks it).
    // The background service worker can, because of host_permissions in manifest.json.
    fetch(message.api_base + "/api/jobs/" + message.job_id)
      .then(function(r) { return r.json(); })
      .then(function(job) { sendResponse({ job: job }); })
      .catch(function(err) { sendResponse({ error: err.message }); });
    return true;
  }

  if (message.type === "ping") {
    console.log("[dub-lite] sending ping to native host...");
    chrome.runtime.sendNativeMessage(HOST_NAME, { type: "ping" }, function(response) {
      if (chrome.runtime.lastError) {
        console.error("[dub-lite] native error:", chrome.runtime.lastError.message);
        sendResponse({ error: chrome.runtime.lastError.message });
        return;
      }
      console.log("[dub-lite] native response:", response);
      sendResponse(response);
    });
    return true;
  }
});
