const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("dub", {
  ping: () => ipcRenderer.invoke("sidecar:ping"),
  dub: (url, targetLang) => ipcRenderer.invoke("sidecar:dub", { url, targetLang }),
  poll: (jobId) => ipcRenderer.invoke("sidecar:poll", { jobId }),
  loadSettings: () => ipcRenderer.invoke("settings:load"),
  saveSettings: (cfg) => ipcRenderer.invoke("settings:save", cfg),
});
