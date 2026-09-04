const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('aaaLite', {
  checkComfy: (endpoint) => ipcRenderer.invoke('comfy:check', endpoint),
  scanComfyCatalog: (endpoint) => ipcRenderer.invoke('comfy:catalog', endpoint),
  loadLocalComfyWorkflow: (endpoint, workflowPath) => ipcRenderer.invoke('comfy:load-local-workflow', endpoint, workflowPath),
  importComfyWorkflow: () => ipcRenderer.invoke('comfy:import-workflow'),
  inspectComfyWorkflow: (endpoint, workflow) => ipcRenderer.invoke('comfy:inspect-workflow', endpoint, workflow),
  uploadComfyImage: (endpoint) => ipcRenderer.invoke('comfy:upload-image', endpoint),
  uploadComfyVideo: (endpoint, reference) => ipcRenderer.invoke('comfy:upload-video', endpoint, reference),
  uploadComfyReference: (endpoint, reference) => ipcRenderer.invoke('comfy:upload-reference', endpoint, reference),
  runComfyWorkflow: (endpoint, workflow) => ipcRenderer.invoke('comfy:run-workflow', endpoint, workflow),
  getRecentComfyActivity: (endpoint, limit) => ipcRenderer.invoke('comfy:recent-activity', endpoint, limit),
  interruptComfy: (endpoint, promptId) => ipcRenderer.invoke('comfy:interrupt', endpoint, promptId),
  cancelComfyTask: (endpoint, promptId) => ipcRenderer.invoke('comfy:cancel-task', endpoint, promptId),
  cancelAllComfyTasks: (endpoint) => ipcRenderer.invoke('comfy:cancel-all', endpoint),
  onComfyEvent: (callback) => {
    const listener = (_event, payload) => callback(payload)
    ipcRenderer.on('comfy:event', listener)
    return () => ipcRenderer.removeListener('comfy:event', listener)
  },
  getSystemStats: () => ipcRenderer.invoke('system:stats'),
  releaseResources: (endpoint, type) => ipcRenderer.invoke('system:release', endpoint, type),
  stopAllAndRelease: (endpoint) => ipcRenderer.invoke('comfy:stop-all-and-release', endpoint),
  openComfyWeb: (endpoint) => ipcRenderer.invoke('comfy:open-web', endpoint),
  quitApp: () => ipcRenderer.invoke('app:quit'),
  getDoubaoConfig: () => ipcRenderer.invoke('doubao:get-config'),
  openDoubaoWeb: () => ipcRenderer.invoke('doubao:open-web'),
  saveDoubaoConfig: (config) => ipcRenderer.invoke('doubao:save-config', config),
  activateSeedanceConnection: (connectionId) => ipcRenderer.invoke('seedance:activate-connection', connectionId),
  chatWithDoubao: (messages) => ipcRenderer.invoke('doubao:chat', messages),
  createApiImage: (input) => ipcRenderer.invoke('api:image-generate', input),
  copyNodeContent: (input) => ipcRenderer.invoke('clipboard:copy-node-content', input),
  createSeedanceTask: (input) => ipcRenderer.invoke('seedance:create-task', input),
  getSeedanceTask: (taskId, connectionId) => ipcRenderer.invoke('seedance:get-task', taskId, connectionId),
  cancelSeedanceTask: (taskId, connectionId) => ipcRenderer.invoke('seedance:cancel-task', taskId, connectionId),
  saveCanvas: (canvasId, canvas) => ipcRenderer.invoke('canvas:save', canvasId, canvas),
  loadCanvas: (canvasId) => ipcRenderer.invoke('canvas:load', canvasId),
  storeAsset: (input) => ipcRenderer.invoke('asset:store', input),
  onResourcesReleased: (callback) => ipcRenderer.on('system:released', (_event, type) => callback(type)),
  platform: process.platform
})
