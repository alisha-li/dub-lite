console.log("[dub-lite] content script loaded");

// Press "q" to start dubbing the current video
document.addEventListener("keydown", function(e) {
  if (e.key !== "q" || e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

  var url = window.location.href;
  console.log("[dub-lite] starting dub for:", url);

  // Mac app calls Modal directly. No API server in the loop.
  chrome.runtime.sendMessage({
    type: "dub-url",
    url: url,
    target_lang: "zh",
  }, function(response) {
    if (chrome.runtime.lastError) {
      console.error("[dub-lite] error:", chrome.runtime.lastError.message);
      return;
    }
    if (!response || response.error) {
      console.error("[dub-lite] error:", response && response.error, response && response.traceback);
      return;
    }
    console.log("[dub-lite] job created!", response.job_id);
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
    try { attachedVideo.volume = 1; } catch (e) {}
  }
  attachedVideo = null;
  videoHandlers = null;
}

window.addEventListener("yt-navigate-start", cleanupDub);
window.addEventListener("yt-navigate-finish", cleanupDub);
window.addEventListener("popstate", cleanupDub);

function playDub(audioUrl) {
  var video = document.querySelector("video");
  if (!video) {
    console.error("[dub-lite] no <video> element found");
    return;
  }

  cleanupDub();

  dubAudio = new Audio(audioUrl);
  dubAudio.preload = "auto";

  // YouTube uses MSE + custom player wiring. video.muted alone sometimes leaks
  // through. Belt + suspenders: mute the element AND zero its volume AND click
  // YouTube's own mute button if available.
  function muteOriginal() {
    try { video.muted = true; } catch (e) {}
    try { video.volume = 0; } catch (e) {}
  }
  muteOriginal();
  // Some YT updates reset .muted on play/seek events. Re-mute every animation
  // frame for the first second to defeat that.
  var muteGuard = setInterval(muteOriginal, 100);
  setTimeout(function() { clearInterval(muteGuard); }, 2000);

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
  if (activePollInterval !== null) {
    clearInterval(activePollInterval);
  }
  activePollInterval = setInterval(function() {
    chrome.runtime.sendMessage({
      type: "poll-job",
      job_id: jobId,
    }, function(response) {
      if (!response || response.error) {
        console.error("[dub-lite] poll error:", response && response.error);
        return;
      }
      // New shape: poll-job returns {status, progress, stage, output_url?} directly
      var status = response.status;
      console.log("[dub-lite] status:", status, "progress:", response.progress + "%", response.stage || "");

      if (status === "completed") {
        clearInterval(activePollInterval);
        activePollInterval = null;
        console.log("[dub-lite] DONE! download URL:", response.output_url);
        playDub(response.output_url);
      } else if (status === "failed") {
        clearInterval(activePollInterval);
        activePollInterval = null;
        console.error("[dub-lite] job failed:", response.error);
      }
    });
  }, 3000);
}
