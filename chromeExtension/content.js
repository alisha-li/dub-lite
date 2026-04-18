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

// Mute YouTube and play the dubbed audio synced to the video's currentTime.
var dubAudio = null;

function playDub(audioUrl) {
  var video = document.querySelector("video");
  if (!video) {
    console.error("[dub-lite] no <video> element found");
    return;
  }

  // Clean up any previous dub
  if (dubAudio) {
    dubAudio.pause();
    dubAudio.src = "";
  }

  dubAudio = new Audio(audioUrl);
  dubAudio.preload = "auto";

  video.muted = true;

  // Seek audio to match video on every sync event
  function syncTime() {
    if (Math.abs(dubAudio.currentTime - video.currentTime) > 0.3) {
      dubAudio.currentTime = video.currentTime;
    }
  }

  video.addEventListener("play", function() {
    syncTime();
    dubAudio.play().catch(function(e) { console.error("[dub-lite] play failed:", e); });
  });
  video.addEventListener("pause", function() { dubAudio.pause(); });
  video.addEventListener("seeked", syncTime);
  video.addEventListener("ratechange", function() { dubAudio.playbackRate = video.playbackRate; });

  // Start immediately if video is already playing
  if (!video.paused) {
    syncTime();
    dubAudio.play().catch(function(e) { console.error("[dub-lite] play failed:", e); });
  }

  console.log("[dub-lite] dub audio attached. YouTube muted, dub playing.");
}

function pollJob(jobId) {
  var interval = setInterval(function() {
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
        clearInterval(interval);
        console.log("[dub-lite] DONE! download URL:", job.output_url);
        playDub(job.output_url);
      } else if (job.status === "failed") {
        clearInterval(interval);
        console.error("[dub-lite] job failed:", job.error);
      }
    });
  }, 3000); // poll every 3 seconds
}
