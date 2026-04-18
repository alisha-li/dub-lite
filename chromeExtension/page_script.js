var player = document.querySelector("#movie_player");
var result = { found: false };

if (player) {
  // Log all methods on the player that might give us video/audio URLs
  var methods = [];
  for (var key in player) {
    if (typeof player[key] === "function" && key.toLowerCase().includes("url")) {
      methods.push(key);
    }
  }
  result.urlMethods = methods;

  // Try getVideoUrl
  if (player.getVideoUrl) result.videoUrl = player.getVideoUrl();

  // Try getVideoData — might contain more info
  if (player.getVideoData) result.videoData = player.getVideoData();

  // Check for any method with "stream" or "manifest" in the name
  var streamMethods = [];
  for (var key in player) {
    if (typeof player[key] === "function" && (key.toLowerCase().includes("stream") || key.toLowerCase().includes("manifest"))) {
      streamMethods.push(key);
    }
  }
  result.streamMethods = streamMethods;

  result.found = true;
}

window.postMessage({ type: "dub-lite-player-data", result: result }, "*");
