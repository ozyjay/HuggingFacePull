const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("hfpullDesktop", {
  selectCacheDirectory: () => ipcRenderer.invoke("select-hf-cache-directory"),
  restartForCacheDirectory: () => ipcRenderer.invoke("restart-for-cache-directory"),
});
