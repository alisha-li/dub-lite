console.log("[dub-lite] content script loaded");

var API_BASE = "https://dub-lite.alishali.info";

// Press "q" to start dubbing the current video
document.addEventListener("keydown", function(e) {
  if (e.key !== "q" || e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

  var url = window.location.href;
  console.log("[dub-lite] starting dub for:", url);

  // Step 1: Send URL to background → native host → yt-dlp → API
  chrome.runtime.sendMessage({
    type: "download-audio",
    url: url,
    target_lang: "zh",
    api_base: API_BASE,
  }, function(response) {
    if (chrome.runtime.lastError) {
      console.error("[dub-lite] error:", chrome.runtime.lastError.message);
      return;
    }
    if (response.error) {
      console.error("[dub-lite] error:", response.error);
      return;
    }

    console.log("[dub-lite] job created!", response.job_id, "audio size:", response.size);

    // Step 2: Poll for progress
    pollJob(response.job_id);
  });
});

// --- Dub audio playback + cleanup ---
// YouTube is a SPA: navigating to another video doesn't reload content.js, so
// the old <audio> element + listeners stick around. We track them and tear
// them down when the user navigates, and also when a fresh dub starts.
var dubAudio = null;
var attachedVideo = null;
var videoHandlers = null;
var activePollInterval = null;

function cleanupDub() {
  if (activePollInterval !== null) {
    clearInterval(activePollInterval);
    activePollInterval = null;
  }
  if (dubAudio) {
    try { dubAudio.pause(); } catch (e) {}
    dubAudio.src = "";
    dubAudio = null;
  }
  if (attachedVideo && videoHandlers) {
    attachedVideo.removeEventListener("play", videoHandlers.play);
    attachedVideo.removeEventListener("pause", videoHandlers.pause);
    attachedVideo.removeEventListener("seeked", videoHandlers.seeked);
    attachedVideo.removeEventListener("ratechange", videoHandlers.ratechange);
    try { attachedVideo.muted = false; } catch (e) {}
  }
  attachedVideo = null;
  videoHandlers = null;
}

// YouTube dispatches this on SPA navigation (back/forward, sidebar click, etc.)
window.addEventListener("yt-navigate-start", cleanupDub);
window.addEventListener("yt-navigate-finish", cleanupDub);
// Fallback: catch manual history changes just in case
window.addEventListener("popstate", cleanupDub);

function playDub(audioUrl) {
  var video = document.querySelector("video");
  if (!video) {
    console.error("[dub-lite] no <video> element found");
    return;
  }

  // Tear down any existing dub state before attaching a new one
  cleanupDub();

  dubAudio = new Audio(audioUrl);
  dubAudio.preload = "auto";
  video.muted = true;

  function syncTime() {
    if (dubAudio && Math.abs(dubAudio.currentTime - video.currentTime) > 0.3) {
      dubAudio.currentTime = video.currentTime;
    }
  }

  videoHandlers = {
    play: function() {
      if (!dubAudio) return;
      syncTime();
      dubAudio.play().catch(function(e) { console.error("[dub-lite] play failed:", e); });
    },
    pause: function() { if (dubAudio) dubAudio.pause(); },
    seeked: syncTime,
    ratechange: function() { if (dubAudio) dubAudio.playbackRate = video.playbackRate; },
  };

  video.addEventListener("play", videoHandlers.play);
  video.addEventListener("pause", videoHandlers.pause);
  video.addEventListener("seeked", videoHandlers.seeked);
  video.addEventListener("ratechange", videoHandlers.ratechange);
  attachedVideo = video;

  if (!video.paused) {
    syncTime();
    dubAudio.play().catch(function(e) { console.error("[dub-lite] play failed:", e); });
  }

  console.log("[dub-lite] dub audio attached. YouTube muted, dub playing.");
}

function pollJob(jobId) {
  // Cancel any in-flight poll from a previous dub
  if (activePollInterval !== null) {
    clearInterval(activePollInterval);
  }
  activePollInterval = setInterval(function() {
    chrome.runtime.sendMessage({
      type: "poll-job",
      job_id: jobId,
      api_base: API_BASE,
    }, function(response) {
      if (!response || response.error) {
        console.error("[dub-lite] poll error:", response && response.error);
        return;
      }
      var job = response.job;
      console.log("[dub-lite] status:", job.status, "progress:", job.progress + "%", job.stage || "");

      if (job.status === "completed") {
        clearInterval(activePollInterval);
        activePollInterval = null;
        console.log("[dub-lite] DONE! download URL:", job.output_url);
        playDub(job.output_url);
      } else if (job.status === "failed") {
        clearInterval(activePollInterval);
        activePollInterval = null;
        console.error("[dub-lite] job failed:", job.error);
      }
    });
  }, 3000);
}
