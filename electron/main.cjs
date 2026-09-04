const { app, BrowserWindow, clipboard, dialog, ipcMain, nativeImage, session, shell, Tray, Menu, safeStorage, protocol, net } = require('electron')
const path = require('path')
const fsSync = require('fs')
const os = require('os')
const fs = require('fs/promises')
const crypto = require('crypto')
const { pathToFileURL } = require('url')
const { Readable } = require('stream')
const { pipeline } = require('stream/promises')
const { execFile, spawn } = require('child_process')
const { promisify } = require('util')

const execFileAsync = promisify(execFile)

protocol.registerSchemesAsPrivileged([
  { scheme: 'comfy-media', privileges: { standard: true, secure: true, supportFetchAPI: true, stream: true } },
  { scheme: 'freedom-asset', privileges: { standard: true, secure: true, supportFetchAPI: true, stream: true } }
])

const isDev = !app.isPackaged
let previousCpu = null
let gpuCache = { at: 0, data: null }
const activeComfyRuns = new Map()
const comfyMediaCache = new Map()
let mainWindow = null
let tray = null
let isQuitting = false

function canvasStorageDirectory() {
  return path.join(app.getPath('userData'), 'canvas-data')
}

function assetStorageDirectory() {
  return path.join(app.getPath('userData'), 'canvas-assets')
}

function safeStorageId(value, fallback = 'item') {
  const cleaned = String(value || '').replace(/[^a-zA-Z0-9._-]/g, '-').replace(/^-+|-+$/g, '').slice(0, 180)
  return cleaned || fallback
}

function assetUrl(filename) {
  return `freedom-asset://local/${encodeURIComponent(filename)}`
}

async function writeJsonAtomic(target, value) {
  await fs.mkdir(path.dirname(target), { recursive: true })
  const temporary = `${target}.${process.pid}.${Date.now()}.tmp`
  await fs.writeFile(temporary, JSON.stringify(value), 'utf8')
  await fs.rename(temporary, target)
}

async function isComfyUiRunning() {
  try {
    const response = await fetch('http://127.0.0.1:8188/system_stats', { signal: AbortSignal.timeout(1800) })
    return response.ok
  } catch {
    return false
  }
}

function findComfyUiRuntime() {
  const roots = [
    process.env.COMFYUI_ROOT,
    'E:\\ComfyUI',
    path.join(os.homedir(), 'ComfyUI'),
    'C:\\ComfyUI',
    'D:\\ComfyUI',
    'F:\\ComfyUI'
  ].filter(Boolean)
  for (const root of roots) {
    const python = path.join(root, 'python_embeded', 'python.exe')
    const main = path.join(root, 'ComfyUI', 'main.py')
    if (fsSync.existsSync(python) && fsSync.existsSync(main)) return { root, python, main }
  }
  return null
}

function ensureHiddenComfyUiLauncher() {
  const helperDirectory = path.join(app.getPath('userData'), 'helpers')
  const helperPath = path.join(helperDirectory, 'start-comfyui-hidden.vbs')
  const helperSource = [
    'Set launcher = CreateObject("WScript.Shell")',
    'launcher.CurrentDirectory = WScript.Arguments(0)',
    'q = Chr(34)',
    'command = "cmd.exe /d /s /c " & q & q & WScript.Arguments(1) & q & " -s " & q & WScript.Arguments(2) & q & " --windows-standalone-build --disable-auto-launch --preview-method auto --fast fp16_accumulation --cuda-malloc" & q',
    'launcher.Run command, 0, False'
  ].join('\r\n')
  fsSync.mkdirSync(helperDirectory, { recursive: true })
  if (!fsSync.existsSync(helperPath) || fsSync.readFileSync(helperPath, 'utf8') !== helperSource) {
    fsSync.writeFileSync(helperPath, helperSource, 'utf8')
  }
  return helperPath
}

async function startComfyUiWithApp() {
  if (await isComfyUiRunning()) return
  const runtime = findComfyUiRuntime()
  if (!runtime) {
    console.warn('ComfyUI runtime was not found; automatic startup was skipped.')
    return
  }
  try {
    const helperPath = ensureHiddenComfyUiLauncher()
    const torchCache = path.join(runtime.root, 'cache', 'torchinductor')
    const tritonCache = path.join(runtime.root, 'cache', 'triton')
    fsSync.mkdirSync(torchCache, { recursive: true })
    fsSync.mkdirSync(tritonCache, { recursive: true })
    const child = spawn('wscript.exe', [helperPath, runtime.root, runtime.python, runtime.main], {
      cwd: runtime.root,
      detached: true,
      windowsHide: true,
      shell: false,
      stdio: 'ignore',
      env: {
        ...process.env,
        HF_ENDPOINT: process.env.HF_ENDPOINT || 'https://hf-mirror.com',
        PYTHONUTF8: '1',
        PYTHONIOENCODING: 'utf-8',
        COMFYUI_SOL_ATTN_SKIP_WARMUP: '1',
        TORCHINDUCTOR_CACHE_DIR: torchCache,
        TRITON_CACHE_DIR: tritonCache
      }
    })
    child.unref()
  } catch (error) {
    console.warn(`Unable to start ComfyUI automatically: ${error.message}`)
  }
}

// Keep using the former data directory so existing local canvases survive the rename.
// A separate development-only directory lets UI checks run without touching the user's live session.
const developmentUserData = isDev ? process.env.AAA_LITE_DEV_USER_DATA : null
app.setPath('userData', developmentUserData || path.join(app.getPath('appData'), 'AAA Lite'))
app.setName('自由者')
app.setAppUserModelId('local.ziyouzhe.desktop')
const hasSingleInstanceLock = app.requestSingleInstanceLock()
if (!hasSingleInstanceLock) app.quit()
else {
  // Start ComfyUI as early as possible and in parallel with the Electron UI.
  void startComfyUiWithApp()
  app.on('second-instance', () => { if (app.isReady()) showMainWindow() })
}

function cpuSnapshot() {
  return os.cpus().reduce((sum, cpu) => {
    const total = Object.values(cpu.times).reduce((value, item) => value + item, 0)
    return { idle: sum.idle + cpu.times.idle, total: sum.total + total }
  }, { idle: 0, total: 0 })
}

function readCpuPercent() {
  const current = cpuSnapshot()
  if (!previousCpu) { previousCpu = current; return 0 }
  const idle = current.idle - previousCpu.idle
  const total = current.total - previousCpu.total
  previousCpu = current
  return total > 0 ? Math.max(0, Math.min(100, Math.round((1 - idle / total) * 100))) : 0
}

async function readGpuStats() {
  if (Date.now() - gpuCache.at < 700 && gpuCache.data) return gpuCache.data
  try {
    const { stdout } = await execFileAsync('nvidia-smi.exe', [
      '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
      '--format=csv,noheader,nounits'
    ], { timeout: 1800, windowsHide: true })
    const rows = stdout.trim().split(/\r?\n/).map((line) => line.split(/\s*,\s*/).map(Number)).filter((row) => row.length >= 4 && row.every(Number.isFinite))
    const [utilization, used, total, temperature] = rows.sort((a, b) => b[1] - a[1] || b[0] - a[0])[0] || []
    gpuCache = { at: Date.now(), data: { utilization: utilization ?? null, used: used ?? null, total: total ?? null, temperature: temperature ?? null } }
  } catch {
    gpuCache = { at: Date.now(), data: { utilization: null, used: null, total: null, temperature: null } }
  }
  return gpuCache.data
}

function isTrustedEmbeddedChatUrl(rawUrl) {
  try {
    const url = new URL(rawUrl)
    if (url.protocol !== 'https:') return false
    const hostname = url.hostname.toLowerCase()
    return hostname === 'chatgpt.com' || hostname.endsWith('.chatgpt.com') ||
      hostname === 'openai.com' || hostname.endsWith('.openai.com') ||
      hostname === 'accounts.google.com' || hostname === 'oauth2.googleapis.com' ||
      hostname === 'login.microsoftonline.com' || hostname === 'login.live.com' ||
      hostname === 'appleid.apple.com' ||
      hostname === 'qianwen.com' || hostname.endsWith('.qianwen.com') ||
      hostname === 'qwen.ai' || hostname.endsWith('.qwen.ai') ||
      hostname === 'aliyun.com' || hostname.endsWith('.aliyun.com') ||
      hostname === 'alibaba.com' || hostname.endsWith('.alibaba.com') ||
      hostname === 'taobao.com' || hostname.endsWith('.taobao.com') ||
      hostname === 'doubao.com' || hostname.endsWith('.doubao.com')
  } catch { return false }
}

function createWindow() {
  const window = new BrowserWindow({
    width: 1380,
    height: 880,
    minWidth: 1050,
    minHeight: 680,
    backgroundColor: '#090b10',
    title: '自由者',
    icon: path.join(__dirname, '..', 'build', 'icon.ico'),
    autoHideMenuBar: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      webviewTag: true
    }
  })

  const chatGptSession = session.fromPartition('persist:chatgpt-web')
  chatGptSession.setPermissionRequestHandler((_webContents, _permission, callback) => callback(false))
  const qwenSession = session.fromPartition('persist:qwen-web')
  qwenSession.setPermissionRequestHandler((_webContents, _permission, callback) => callback(false))
  const doubaoWebSession = session.fromPartition('persist:doubao-web')
  doubaoWebSession.setPermissionRequestHandler((_webContents, _permission, callback) => callback(false))
  window.webContents.on('will-attach-webview', (event, webPreferences, params) => {
    if (!isTrustedEmbeddedChatUrl(params.src)) { event.preventDefault(); return }
    delete webPreferences.preload
    webPreferences.nodeIntegration = false
    webPreferences.contextIsolation = true
    webPreferences.sandbox = true
    webPreferences.webSecurity = true
  })
  window.webContents.on('did-attach-webview', (_event, contents) => {
    contents.setWindowOpenHandler(({ url }) => {
      if (isTrustedEmbeddedChatUrl(url)) void contents.loadURL(url)
      return { action: 'deny' }
    })
    contents.on('will-navigate', (event, url) => { if (!isTrustedEmbeddedChatUrl(url)) event.preventDefault() })
  })

  if (isDev) window.loadURL('http://127.0.0.1:5173')
  else window.loadFile(path.join(__dirname, '..', 'dist', 'index.html'))
  window.on('close', (event) => {
    if (isQuitting) return
    event.preventDefault()
    window.hide()
    if (tray && !tray.__closeTipShown) {
      tray.__closeTipShown = true
      tray.displayBalloon({ title: '自由者仍在后台运行', content: '右键任务栏托盘图标可重新打开或彻底退出。', iconType: 'info' })
    }
  })
  window.on('closed', () => { if (mainWindow === window) mainWindow = null })
  mainWindow = window
  return window
}

function showMainWindow() {
  if (!mainWindow || mainWindow.isDestroyed()) createWindow()
  mainWindow.show()
  if (mainWindow.isMinimized()) mainWindow.restore()
  mainWindow.focus()
}

function createTray() {
  if (tray) return
  tray = new Tray(path.join(__dirname, '..', 'build', 'icon.ico'))
  tray.setToolTip('自由者 · 后台运行中')
  tray.setContextMenu(Menu.buildFromTemplate([
    { label: '打开自由者', click: showMainWindow },
    { type: 'separator' },
    { label: '退出自由者', click: () => { isQuitting = true; app.quit() } }
  ]))
  tray.on('click', showMainWindow)
}

function normalizeEndpoint(raw) {
  const url = new URL(raw)
  if (!['http:', 'https:'].includes(url.protocol)) throw new Error('仅支持 HTTP 或 HTTPS')
  return url.origin
}

async function readDoubaoConfig() {
  try { return JSON.parse(await fs.readFile(path.join(app.getPath('userData'), 'doubao-config.json'), 'utf8')) } catch { return {} }
}

const DEFAULT_SEEDANCE_MODEL = 'doubao-seedance-2-0-mini-260615'
const DEFAULT_SEEDANCE_BASE_URL = 'https://ark.cn-beijing.volces.com/api/v3'

function normalizeApiBaseUrl(raw, fallback = DEFAULT_SEEDANCE_BASE_URL) {
  const value = String(raw || fallback).trim()
  const url = new URL(value)
  if (!['http:', 'https:'].includes(url.protocol)) throw new Error('接口地址仅支持 HTTP 或 HTTPS')
  if (url.username || url.password || url.search || url.hash) throw new Error('接口地址不能包含账号、查询参数或锚点')
  const pathname = url.pathname.replace(/\/+$/, '').replace(/\/(?:contents\/generations\/tasks|images\/generations)(?:\/.*)?$/i, '')
  return `${url.origin}${pathname}`
}

function seedanceConnections(config) {
  const saved = Array.isArray(config.videoConnections) ? config.videoConnections.filter((item) => item?.id && item?.encryptedKey) : []
  if (saved.length) return saved
  if (!config.encryptedKey) return []
  return [{ id: 'legacy-default', name: '默认连接', videoBaseUrl: config.videoBaseUrl || DEFAULT_SEEDANCE_BASE_URL, videoModel: config.videoModel || DEFAULT_SEEDANCE_MODEL, videoModels: [config.videoModel || DEFAULT_SEEDANCE_MODEL], videoModelLabels: {}, encryptedKey: config.encryptedKey }]
}

function activeSeedanceConnection(config, connectionId = '') {
  const connections = seedanceConnections(config)
  return connections.find((item) => item.id === connectionId)
    || connections.find((item) => item.id === config.activeVideoConnectionId)
    || connections[0]
    || null
}

function publicSeedanceConnections(config) {
  return seedanceConnections(config).map((item) => {
    const videoModels = Array.isArray(item.videoModels) && item.videoModels.length ? item.videoModels : [item.videoModel || DEFAULT_SEEDANCE_MODEL]
    const modelMediaKinds = item.modelMediaKinds && typeof item.modelMediaKinds === 'object' ? item.modelMediaKinds : {}
    return { id: item.id, name: item.name || '未命名连接', videoBaseUrl: normalizeApiBaseUrl(item.videoBaseUrl), videoModel: item.videoModel || videoModels[0], videoModels, videoModelLabels: item.videoModelLabels && typeof item.videoModelLabels === 'object' ? item.videoModelLabels : {}, modelMediaKinds: Object.fromEntries(videoModels.map((model) => [model, modelMediaKinds[model] === 'image' ? 'image' : 'video'])), hasKey: Boolean(item.encryptedKey) }
  })
}

function seedanceApiBase(config, connectionId = '') {
  const connection = activeSeedanceConnection(config, connectionId)
  return normalizeApiBaseUrl(connection?.videoBaseUrl || config.videoBaseUrl || DEFAULT_SEEDANCE_BASE_URL)
}

function arkApiKey(config, connectionId = '') {
  const encryptedKey = activeSeedanceConnection(config, connectionId)?.encryptedKey || config.encryptedKey
  if (!encryptedKey) throw new Error('请先在设置中配置火山方舟 API Key')
  if (!safeStorage.isEncryptionAvailable()) throw new Error('Windows 安全存储不可用')
  return safeStorage.decryptString(Buffer.from(encryptedKey, 'base64'))
}

ipcMain.handle('doubao:get-config', async () => {
  const config = await readDoubaoConfig()
  const model = config.chatModel || config.model || ''
  const connection = activeSeedanceConnection(config)
  const videoModel = connection?.videoModel || config.videoModel || DEFAULT_SEEDANCE_MODEL
  const videoBaseUrl = seedanceApiBase(config)
  return { ok: true, configured: Boolean(config.encryptedKey && model), videoConfigured: Boolean(connection?.encryptedKey || config.encryptedKey) && Boolean(videoModel), model, videoModel, videoModels: Array.isArray(connection?.videoModels) && connection.videoModels.length ? connection.videoModels : [videoModel], videoModelLabels: connection?.videoModelLabels || {}, modelMediaKinds: connection?.modelMediaKinds || {}, videoBaseUrl, activeVideoConnectionId: connection?.id || '', videoConnectionName: connection?.name || '', videoConnections: publicSeedanceConnections(config) }
})

ipcMain.handle('doubao:open-web', async () => {
  try {
    await shell.openExternal('https://www.doubao.com/chat/')
    return { ok: true, message: '已在浏览器打开豆包官方网页版，可使用抖音扫码登录' }
  } catch (error) {
    return { ok: false, message: `豆包网页版打开失败：${error.message}` }
  }
})

ipcMain.handle('comfy:open-web', async (_event, endpoint) => {
  let base = 'http://127.0.0.1:8188'
  try {
    base = normalizeEndpoint(endpoint || base)
    const focusScript = [
      "$windows = Get-Process chrome,msedge,firefox -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -match '(?i)ComfyUI' }",
      '$window = $windows | Select-Object -First 1',
      "if (-not $window) { Write-Output 'NOT_FOUND'; exit 3 }",
      "Add-Type -TypeDefinition 'using System; using System.Runtime.InteropServices; public static class ComfyWindowFocus { [DllImport(\"user32.dll\")] public static extern bool ShowWindowAsync(IntPtr hWnd, int nCmdShow); [DllImport(\"user32.dll\")] public static extern bool SetForegroundWindow(IntPtr hWnd); }'",
      '[ComfyWindowFocus]::ShowWindowAsync($window.MainWindowHandle, 9) | Out-Null',
      'Start-Sleep -Milliseconds 80',
      '$shell = New-Object -ComObject WScript.Shell',
      '$activated = $shell.AppActivate($window.Id)',
      '$foreground = [ComfyWindowFocus]::SetForegroundWindow($window.MainWindowHandle)',
      "if ($activated -or $foreground) { Write-Output 'FOCUSED'; exit 0 }",
      "Write-Output 'FOCUS_FAILED'; exit 4"
    ].join('; ')
    const { stdout } = await execFileAsync('powershell.exe', ['-NoProfile', '-NonInteractive', '-WindowStyle', 'Hidden', '-Command', focusScript], { timeout: 4000, windowsHide: true })
    if (!String(stdout).includes('FOCUSED')) throw new Error('未能激活现有网页')
    return { ok: true, message: '现有 ComfyUI 网页已恢复并显示到桌面前方' }
  } catch (error) {
    const output = `${error?.stdout || ''} ${error?.message || ''}`
    if (error?.code === 3 || output.includes('NOT_FOUND') || /exit code 3/i.test(output)) return { ok: false, message: '未找到已经打开的 ComfyUI 网页；不会新建页面' }
    return { ok: false, message: `切换到 ComfyUI 网页失败：${error.message}` }
  }
})

ipcMain.handle('doubao:save-config', async (_event, input) => {
  try {
    const existing = await readDoubaoConfig()
    const model = String(input?.model || '').trim()
    const createConnection = Boolean(input?.createConnection)
    const currentConnection = createConnection ? null : activeSeedanceConnection(existing, String(input?.connectionId || ''))
    const videoModel = String(input?.videoModel || currentConnection?.videoModel || existing.videoModel || DEFAULT_SEEDANCE_MODEL).trim()
    const videoBaseUrl = normalizeApiBaseUrl(input?.videoBaseUrl || currentConnection?.videoBaseUrl || existing.videoBaseUrl || DEFAULT_SEEDANCE_BASE_URL)
    const connectionName = String(input?.connectionName || currentConnection?.name || '默认连接').trim().slice(0, 80)
    if (!connectionName) throw new Error('请填写连接名称')
    if (model && (model.length > 200 || !/^[\w.:/-]+$/.test(model))) throw new Error('模型 ID 或 Endpoint ID 格式无效')
    if (!videoModel || videoModel.length > 200 || !/^[\w.:/-]+$/.test(videoModel)) throw new Error('Seedance 模型 ID 格式无效')
    let encryptedKey = currentConnection?.encryptedKey || (!createConnection ? existing.encryptedKey : '') || ''
    const apiKey = String(input?.apiKey || '').trim()
    if (apiKey) {
      if (apiKey.length < 12 || apiKey.length > 500) throw new Error('API Key 格式无效')
      if (!safeStorage.isEncryptionAvailable()) throw new Error('Windows 安全存储暂不可用')
      encryptedKey = safeStorage.encryptString(apiKey).toString('base64')
    }
    if (!encryptedKey) throw new Error('请填写火山方舟 API Key')
    const videoModels = [...new Set((Array.isArray(input?.videoModels) ? input.videoModels : [videoModel]).map((item) => String(item || '').trim()).filter((item) => /^[\w.:/-]{2,200}$/.test(item)))]
    if (!videoModels.includes(videoModel)) videoModels.unshift(videoModel)
    const inputLabels = input?.videoModelLabels && typeof input.videoModelLabels === 'object' ? input.videoModelLabels : {}
    const videoModelLabels = Object.fromEntries(videoModels.map((item) => [item, String(inputLabels[item] || item).trim().slice(0, 80)]))
    const inputKinds = input?.modelMediaKinds && typeof input.modelMediaKinds === 'object' ? input.modelMediaKinds : {}
    const modelMediaKinds = Object.fromEntries(videoModels.map((item) => [item, inputKinds[item] === 'image' ? 'image' : 'video']))
    const connections = seedanceConnections(existing)
    const connectionId = createConnection || !currentConnection ? `video-${Date.now()}-${Math.random().toString(36).slice(2, 8)}` : currentConnection.id
    const savedConnection = { id: connectionId, name: connectionName, videoModel, videoModels, videoModelLabels, modelMediaKinds, videoBaseUrl, encryptedKey }
    const videoConnections = [...connections.filter((item) => item.id !== connectionId), savedConnection]
    const nextConfig = { ...existing, model, chatModel: model, videoModel, videoBaseUrl, encryptedKey, videoConnections, activeVideoConnectionId: connectionId }
    await fs.writeFile(path.join(app.getPath('userData'), 'doubao-config.json'), JSON.stringify(nextConfig), 'utf8')
    return { ok: true, configured: Boolean(model), videoConfigured: true, model, videoModel, videoModels, videoModelLabels, modelMediaKinds, videoBaseUrl, activeVideoConnectionId: connectionId, videoConnectionName: connectionName, videoConnections: publicSeedanceConnections(nextConfig), message: createConnection ? `新连接“${connectionName}”已加密保存并启用` : `连接“${connectionName}”已保存并启用` }
  } catch (error) { return { ok: false, message: `豆包配置失败：${error.message}` } }
})

ipcMain.handle('seedance:activate-connection', async (_event, connectionId) => {
  try {
    const existing = await readDoubaoConfig()
    const connection = activeSeedanceConnection(existing, String(connectionId || ''))
    if (!connection || connection.id !== String(connectionId || '')) throw new Error('已保存的连接不存在')
    const nextConfig = { ...existing, activeVideoConnectionId: connection.id, videoBaseUrl: normalizeApiBaseUrl(connection.videoBaseUrl), videoModel: connection.videoModel || DEFAULT_SEEDANCE_MODEL, encryptedKey: connection.encryptedKey }
    await fs.writeFile(path.join(app.getPath('userData'), 'doubao-config.json'), JSON.stringify(nextConfig), 'utf8')
    return { ok: true, activeVideoConnectionId: connection.id, videoConnectionName: connection.name, videoBaseUrl: nextConfig.videoBaseUrl, videoModel: nextConfig.videoModel, videoModels: Array.isArray(connection.videoModels) && connection.videoModels.length ? connection.videoModels : [nextConfig.videoModel], videoModelLabels: connection.videoModelLabels || {}, modelMediaKinds: connection.modelMediaKinds || {}, videoConnections: publicSeedanceConnections(nextConfig), message: `已切换连接：${connection.name}` }
  } catch (error) { return { ok: false, message: `切换连接失败：${error.message}` } }
})

ipcMain.handle('doubao:chat', async (_event, messages) => {
  try {
    const config = await readDoubaoConfig()
    const model = config.chatModel || config.model
    if (!model) throw new Error('请先在设置中配置豆包 AI')
    const apiKey = arkApiKey(config)
    const safeMessages = (Array.isArray(messages) ? messages : []).slice(-20).map((item) => ({ role: item?.role === 'assistant' ? 'assistant' : 'user', content: String(item?.content || '').slice(0, 12000) })).filter((item) => item.content.trim())
    if (!safeMessages.length) throw new Error('问题不能为空')
    const result = await fetchJson('https://ark.cn-beijing.volces.com/api/v3/chat/completions', {
      method: 'POST', headers: { 'content-type': 'application/json', authorization: `Bearer ${apiKey}` },
      body: JSON.stringify({ model, messages: [{ role: 'system', content: '你是自由者应用里的豆包日常助手。请用清晰、友好、实用的中文回答用户。' }, ...safeMessages], stream: false })
    }, 90000)
    const content = result?.choices?.[0]?.message?.content
    if (typeof content !== 'string' || !content.trim()) throw new Error('豆包没有返回可显示的内容')
    return { ok: true, content: content.trim() }
  } catch (error) { return { ok: false, message: `豆包请求失败：${error.message}` } }
})

ipcMain.handle('api:image-generate', async (_event, input) => {
  try {
    const config = await readDoubaoConfig()
    const connectionId = String(input?.connectionId || config.activeVideoConnectionId || '')
    const connection = activeSeedanceConnection(config, connectionId)
    if (!connection) throw new Error('所选 API 连接不存在')
    const model = String(input?.model || '').trim()
    if (!model || !connection.videoModels?.includes(model)) throw new Error('所选图片模型不属于该 API 连接')
    if (connection.modelMediaKinds?.[model] !== 'image') throw new Error('所选模型未标记为图片生成模型')
    const prompt = String(input?.prompt || '').trim()
    if (!prompt) throw new Error('提示词不能为空')
    const size = String(input?.size || '1024x1024')
    if (!/^\d{2,5}x\d{2,5}$/.test(size)) throw new Error('图片尺寸格式无效')
    const result = await fetchJson(`${normalizeApiBaseUrl(connection.videoBaseUrl)}/images/generations`, {
      method: 'POST', headers: { 'content-type': 'application/json', authorization: `Bearer ${arkApiKey(config, connectionId)}` },
      body: JSON.stringify({ model, prompt: prompt.slice(0, 12000), size, n: 1, response_format: 'url' })
    }, 180000)
    const image = result?.data?.[0] || result?.images?.[0] || result?.output?.[0] || {}
    const imageUrl = image.url || image.image_url || image.output_url || (image.b64_json ? `data:image/png;base64,${image.b64_json}` : '')
    if (!imageUrl) throw new Error('接口未返回可识别的图片 URL 或 base64 图片')
    return { ok: true, imageUrl, model, connectionId: connection.id }
  } catch (error) { return { ok: false, message: `图片 API 生成失败：${error.message}` } }
})

function localAssetPathFromUrl(value) {
  const source = new URL(String(value || ''))
  if (source.protocol !== 'freedom-asset:' || source.hostname !== 'local') throw new Error('本地素材地址无效')
  const filename = path.basename(decodeURIComponent(source.pathname))
  if (!/^[a-f0-9]{64}\.[a-z0-9]{1,10}$/i.test(filename)) throw new Error('本地素材文件名无效')
  return path.join(assetStorageDirectory(), filename)
}

ipcMain.handle('clipboard:copy-node-content', async (_event, input) => {
  try {
    const kind = String(input?.kind || 'text')
    const value = String(input?.value || '').trim()
    if (!value) throw new Error('当前任务框没有可以复制的内容')
    if (kind === 'text') {
      clipboard.writeText(value)
      return { ok: true, message: '文本已复制' }
    }
    if (kind === 'image') {
      let image
      if (/^freedom-asset:\/\/local\//i.test(value)) image = nativeImage.createFromPath(localAssetPathFromUrl(value))
      else {
        const response = await net.fetch(value)
        if (!response.ok) throw new Error(`图片读取失败（HTTP ${response.status}）`)
        image = nativeImage.createFromBuffer(Buffer.from(await response.arrayBuffer()))
      }
      if (!image || image.isEmpty()) throw new Error('图片格式无法复制')
      clipboard.writeImage(image)
      return { ok: true, message: '图片已复制，可直接粘贴' }
    }
    if (kind === 'video') {
      if (/^freedom-asset:\/\/local\//i.test(value)) {
        const target = localAssetPathFromUrl(value)
        await execFileAsync('powershell.exe', ['-NoProfile', '-NonInteractive', '-WindowStyle', 'Hidden', '-Command', '$copyPath=[Environment]::GetEnvironmentVariable("FREEDOM_CANVAS_COPY_PATH"); Set-Clipboard -LiteralPath $copyPath'], { timeout: 8000, windowsHide: true, env: { ...process.env, FREEDOM_CANVAS_COPY_PATH: target } })
        return { ok: true, message: '视频文件已复制，可粘贴到文件夹或支持文件粘贴的位置' }
      }
      clipboard.writeText(value)
      return { ok: true, message: '远程视频地址已复制' }
    }
    throw new Error('无法识别要复制的内容类型')
  } catch (error) { return { ok: false, message: `复制失败：${error.message}` } }
})

async function seedanceReferenceUrl(value) {
  const url = String(value || '').trim()
  if (/^asset:\/\/asset-[\w-]{6,200}$/i.test(url) || /^https:\/\//i.test(url) || /^data:image\/(?:png|jpeg|jpg|webp);base64,/i.test(url)) return url
  if (/^freedom-asset:\/\/local\//i.test(url)) {
    const target = localAssetPathFromUrl(url)
    const extension = path.extname(target).toLowerCase()
    const mimeType = extension === '.png' ? 'image/png' : extension === '.webp' ? 'image/webp' : 'image/jpeg'
    return `data:${mimeType};base64,${(await fs.readFile(target)).toString('base64')}`
  }
  throw new Error('Seedance 参考图仅支持可信素材 Asset ID、HTTPS 地址或本地 PNG/JPEG/WebP 图片')
}

function seedanceErrorMessage(value) {
  const message = String(value || '').trim()
  if (/copyright restrictions/i.test(message)) return '版权审核未通过：参考人物或提示词可能涉及受保护的肖像、角色或作品；请使用已授权的原创素材，真人建议先录入火山方舟可信素材库'
  if (/input image.*may contain real person/i.test(message)) return '真人图片审核未通过：普通图片不能直接作为真人参考，请先在火山方舟可信素材库完成人像认证，然后在视频节点填写 Asset ID'
  if (/inference limit|safe experience mode/i.test(message)) return '模型推理额度已达到安心体验模式上限，请在火山方舟模型开通管理中调整额度'
  return message
}

ipcMain.handle('seedance:create-task', async (_event, input) => {
  try {
    const config = await readDoubaoConfig()
    const connectionId = String(input?.connectionId || config.activeVideoConnectionId || '')
    const connection = activeSeedanceConnection(config, connectionId)
    const apiKey = arkApiKey(config, connectionId)
    const apiBase = seedanceApiBase(config, connectionId)
    const model = String(input?.model || connection?.videoModel || config.videoModel || DEFAULT_SEEDANCE_MODEL).trim()
    if (!model || model.length > 200 || !/^[\w.:/-]+$/.test(model)) throw new Error('Seedance 模型 ID 格式无效')
    const prompt = String(input?.prompt || '').trim()
    if (!prompt) throw new Error('提示词不能为空')
    if (prompt.length > 12000) throw new Error('提示词过长，请控制在 12000 字以内')
    const ratio = String(input?.ratio || '16:9')
    if (!['16:9', '9:16', '1:1', '4:3', '3:4', '21:9', '9:21'].includes(ratio)) throw new Error('画面比例不受支持')
    const resolution = String(input?.resolution || '720p').toLowerCase()
    if (!['480p', '720p'].includes(resolution)) throw new Error('Seedance Mini 仅支持 480P 或 720P')
    const duration = Math.round(Number(input?.duration || 6))
    if (!Number.isFinite(duration) || duration < 4 || duration > 15) throw new Error('Seedance 时长需为 4–15 秒')
    const references = await Promise.all((Array.isArray(input?.references) ? input.references : []).slice(0, 9).map(async (reference) => ({
      type: 'image_url',
      image_url: { url: await seedanceReferenceUrl(reference?.dataUrl || reference?.url) },
      role: 'reference_image'
    })))
    const body = {
      model,
      content: [{ type: 'text', text: prompt }, ...references],
      generate_audio: input?.generateAudio !== false,
      ratio,
      resolution,
      duration,
      watermark: Boolean(input?.watermark)
    }
    const result = await fetchJson(`${apiBase}/contents/generations/tasks`, {
      method: 'POST', headers: { 'content-type': 'application/json', authorization: `Bearer ${apiKey}` }, body: JSON.stringify(body)
    }, 90000)
    if (!result?.id) throw new Error('火山方舟未返回任务 ID')
    return { ok: true, taskId: result.id, model, connectionId: connection?.id || connectionId }
  } catch (error) { return { ok: false, message: `Seedance 提交失败：${seedanceErrorMessage(error.message)}` } }
})

ipcMain.handle('seedance:get-task', async (_event, taskId, connectionId = '') => {
  try {
    const id = String(taskId || '').trim()
    if (!/^[\w-]{6,200}$/.test(id)) throw new Error('任务 ID 无效')
    const config = await readDoubaoConfig()
    const result = await fetchJson(`${seedanceApiBase(config, connectionId)}/contents/generations/tasks/${encodeURIComponent(id)}`, {
      headers: { authorization: `Bearer ${arkApiKey(config, connectionId)}` }
    }, 30000)
    const videoUrl = result?.content?.video_url || result?.content?.[0]?.video_url || result?.video_url || ''
    return { ok: true, status: result?.status || 'queued', videoUrl, error: seedanceErrorMessage(result?.error?.message || result?.error || '') }
  } catch (error) { return { ok: false, message: `Seedance 状态查询失败：${error.message}` } }
})

ipcMain.handle('seedance:cancel-task', async (_event, taskId, connectionId = '') => {
  try {
    const id = String(taskId || '').trim()
    if (!/^[\w-]{6,200}$/.test(id)) throw new Error('任务 ID 无效')
    const config = await readDoubaoConfig()
    await fetchJson(`${seedanceApiBase(config, connectionId)}/contents/generations/tasks/${encodeURIComponent(id)}`, {
      method: 'DELETE', headers: { authorization: `Bearer ${arkApiKey(config, connectionId)}` }
    }, 30000)
    return { ok: true, message: 'Seedance 任务已取消' }
  } catch (error) { return { ok: false, message: `Seedance 取消失败：${error.message}` } }
})

async function fetchJson(url, options = {}, timeout = 8000) {
  const response = await fetch(url, { ...options, signal: AbortSignal.timeout(timeout) })
  const text = await response.text()
  let data = null
  try { data = text ? JSON.parse(text) : {} } catch { throw new Error(`返回内容不是 JSON（HTTP ${response.status}）`) }
  if (!response.ok) {
    const parts = [data?.error?.message || data?.error || `HTTP ${response.status}`, data?.error?.details]
    for (const [nodeId, nodeError] of Object.entries(data?.node_errors || {})) {
      for (const item of nodeError?.errors || []) parts.push(`${nodeError.class_type || '节点'} #${nodeId}：${item.message}${item.details ? `（${item.details}）` : ''}`)
    }
    throw new Error([...new Set(parts.filter(Boolean))].join(' · ').slice(0, 1400))
  }
  return data
}

function isApiWorkflow(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  const entries = Object.values(value)
  return entries.length > 0 && entries.every((node) => node && typeof node === 'object' && typeof node.class_type === 'string' && node.inputs && typeof node.inputs === 'object')
}

function unwrapWorkflow(value) {
  if (isApiWorkflow(value)) return value
  if (isApiWorkflow(value?.prompt)) return value.prompt
  if (isApiWorkflow(value?.workflow)) return value.workflow
  throw new Error('请选择 ComfyUI 的 API 格式工作流 JSON')
}

function workflowSummary(workflow) {
  const classes = [...new Set(Object.values(workflow).map((node) => node.class_type))]
  const parameters = []
  const known = /^(image|text|prompt|positive|negative|ckpt_name|model_name|unet_name|diffusion_model|lora_name|vae_name|control_net_name|controlnet_name|clip_name|clip_name1|clip_name2|upscale_model|upscale_model_name|width|height|steps|cfg|seed|noise_seed|denoise|sampler_name|scheduler)$/i
  for (const [nodeId, node] of Object.entries(workflow)) {
    for (const [key, value] of Object.entries(node.inputs || {})) {
      if (!known.test(key) || (typeof value !== 'string' && typeof value !== 'number')) continue
      parameters.push({ id: `${nodeId}:${key}`, nodeId, key, label: key, value, valueType: typeof value })
    }
  }
  const priority = { image: 0, text: 1, prompt: 1, positive: 1, ckpt_name: 2, model_name: 2, unet_name: 2, diffusion_model: 2, width: 3, height: 4, steps: 5, cfg: 6, seed: 7, noise_seed: 7, negative: 8, lora_name: 9, vae_name: 9, control_net_name: 9, denoise: 10, sampler_name: 11, scheduler: 12 }
  parameters.sort((a, b) => (priority[a.key] ?? 99) - (priority[b.key] ?? 99))
  return { nodeCount: Object.keys(workflow).length, classes, parameters: parameters.slice(0, 12) }
}

function schemaOptions(schema) {
  if (!Array.isArray(schema)) return []
  if (Array.isArray(schema[0])) return schema[0]
  return Array.isArray(schema[1]?.options) && schema[1].options.every((item) => typeof item !== 'object') ? schema[1].options : []
}

function isWidgetSchema(schema) {
  if (!Array.isArray(schema)) return false
  if (schemaOptions(schema).length) return true
  const type = schema[0]
  return ['STRING', 'INT', 'FLOAT', 'BOOLEAN', 'COMBO', 'COMFY_DYNAMICCOMBO_V3', 'IMAGECOMPARE'].includes(type) || schema[1]?.socketless === true
}

function sanitizeApiWorkflow(workflow, objectInfo) {
  const prompt = structuredClone(workflow)
  let repairCount = 0
  for (const node of Object.values(prompt)) {
    const definition = objectInfo[node.class_type]?.input || {}
    for (const [name, schema] of [...Object.entries(definition.required || {}), ...Object.entries(definition.optional || {})]) {
      if (!(name in node.inputs) || !Array.isArray(schema) || (Array.isArray(node.inputs[name]) && node.inputs[name].length === 2)) continue
      const expected = schema[0]
      const options = schema[1] && typeof schema[1] === 'object' ? schema[1] : {}
      const choices = schemaOptions(schema)
      const original = node.inputs[name]
      if (choices.length && !choices.includes(original)) node.inputs[name] = choices.includes(options.default) ? options.default : choices[0]
      if (expected === 'INT') {
        const numeric = Number(original)
        let value = Number.isFinite(numeric) ? Math.max(options.min ?? -Infinity, Math.min(options.max ?? Infinity, Math.round(numeric))) : (options.default ?? 0)
        if (Number.isFinite(options.step) && options.step > 1) {
          const base = Number.isFinite(options.min) ? options.min : 0
          value = base + Math.round((value - base) / options.step) * options.step
          value = Math.max(options.min ?? -Infinity, Math.min(options.max ?? Infinity, value))
        }
        node.inputs[name] = value
      }
      if (expected === 'FLOAT') {
        const numeric = Number(original)
        node.inputs[name] = Number.isFinite(numeric) ? Math.max(options.min ?? -Infinity, Math.min(options.max ?? Infinity, numeric)) : (options.default ?? 0)
      }
      if (expected === 'BOOLEAN' && typeof original !== 'boolean') node.inputs[name] = original === 'true' || original === 1
      if (node.inputs[name] !== original) repairCount += 1
    }
  }
  return { workflow: prompt, repairCount }
}

function validateApiWorkflow(workflow, objectInfo) {
  const issues = []
  let outputCount = 0
  for (const [nodeId, node] of Object.entries(workflow)) {
    const definition = objectInfo[node.class_type]
    if (!definition) { issues.push(`${nodeId} ${node.class_type}：节点插件未安装`); continue }
    if (definition.output_node) outputCount += 1
    for (const [name, schema] of Object.entries(definition.input?.required || {})) {
      const dynamicChildPresent = schema?.[0] === 'COMFY_AUTOGROW_V3' && Object.keys(node.inputs || {}).some((key) => key.startsWith(`${name}.`))
      if (!(name in (node.inputs || {})) && !dynamicChildPresent && schema?.[1]?.socketless !== true) issues.push(`${nodeId} ${node.class_type}：缺少 ${name}`)
    }
    for (const [name, value] of Object.entries(node.inputs || {})) {
      if (Array.isArray(value) && value.length === 2 && typeof value[0] === 'string' && !workflow[String(value[0])]) issues.push(`${nodeId} ${node.class_type}：${name} 的连线已断开`)
    }
  }
  if (!outputCount) issues.push('没有可执行的输出节点')
  return issues
}

function pruneApiWorkflow(workflow, objectInfo) {
  const prompt = structuredClone(workflow)
  const outputIds = Object.entries(prompt)
    .filter(([, node]) => objectInfo[node.class_type]?.output_node === true)
    .map(([nodeId]) => String(nodeId))
  if (!outputIds.length) return { workflow: prompt, removedCount: 0 }
  const reachable = new Set()
  const pending = [...outputIds]
  while (pending.length) {
    const nodeId = String(pending.pop())
    if (reachable.has(nodeId) || !prompt[nodeId]) continue
    reachable.add(nodeId)
    for (const value of Object.values(prompt[nodeId].inputs || {})) {
      if (!Array.isArray(value) || value.length !== 2) continue
      const sourceId = String(value[0])
      if (prompt[sourceId] && !reachable.has(sourceId)) pending.push(sourceId)
    }
  }
  const executable = Object.fromEntries(Object.entries(prompt).filter(([nodeId]) => reachable.has(String(nodeId))))
  return { workflow: executable, removedCount: Object.keys(prompt).length - Object.keys(executable).length }
}

function convertUiWorkflow(value, objectInfo) {
  if (!value || !Array.isArray(value.nodes) || !Array.isArray(value.links)) throw new Error('工作流不是受支持的画布格式')
  const nodesById = new Map(value.nodes.map((node) => [String(node.id), node]))
  const linksById = new Map(value.links.map((link) => [String(link[0]), link]))
  const resolveLink = (linkId, visited = new Set()) => {
    const key = String(linkId)
    if (visited.has(key)) return null
    visited.add(key)
    const link = linksById.get(key)
    if (!link) return null
    const source = nodesById.get(String(link[1]))
    if (!source || source.mode === 2) return null
    if (source.mode === 4 || source.type === 'Reroute' || !objectInfo[source.type]) {
      const output = source.outputs?.[Number(link[2]) || 0]
      const linkedInputs = (source.inputs || []).filter((input) => input.link != null)
      const upstream = linkedInputs.find((input) => output?.name && input.name === output.name)
        || linkedInputs.find((input) => output?.type && input.type === output.type)
        || linkedInputs[Math.min(Number(link[2]) || 0, Math.max(0, linkedInputs.length - 1))]
      return upstream ? resolveLink(upstream.link, visited) : null
    }
    return [String(source.id), Number(link[2]) || 0]
  }
  const prompt = {}
  let skippedCount = 0
  let repairCount = 0
  for (const node of value.nodes) {
    if (node.mode === 2 || node.mode === 4 || node.type === 'Reroute' || !objectInfo[node.type]) { skippedCount += 1; continue }
    const inputs = {}
    const widgetValues = Array.isArray(node.widgets_values) ? node.widgets_values : []
    const namedValues = node.widgets_values_named && typeof node.widgets_values_named === 'object' ? node.widgets_values_named : {}
    const definition = objectInfo[node.type]?.input || {}
    const allowedNames = new Set([...Object.keys(definition.required || {}), ...Object.keys(definition.optional || {}), ...(node.inputs || []).map((input) => input.name)])
    let widgetIndex = 0
    for (const input of node.inputs || []) {
      if (input.link != null) {
        const resolved = resolveLink(input.link)
        if (resolved) inputs[input.name] = resolved
        continue
      }
    }
    // Current ComfyUI canvas files omit ordinary widget inputs from node.inputs.
    // Rebuild them from the server schema in widget order instead of relying on input.widget.
    for (const [name, schema] of [...Object.entries(definition.required || {}), ...Object.entries(definition.optional || {})]) {
      if (name in inputs || !isWidgetSchema(schema)) continue
      if (Object.prototype.hasOwnProperty.call(namedValues, name)) inputs[name] = namedValues[name]
      else if (widgetIndex < widgetValues.length) inputs[name] = widgetValues[widgetIndex++]
      if (schema?.[1]?.control_after_generate && typeof widgetValues[widgetIndex] === 'string') widgetIndex += 1
    }
    for (const [name, namedValue] of Object.entries(namedValues)) {
      if (allowedNames.has(name) && !(name in inputs)) inputs[name] = namedValue
    }
    for (const [name, schema] of Object.entries(definition.required || {})) {
      if (name in inputs || !Array.isArray(schema)) continue
      const configuredDefault = schema[1] && typeof schema[1] === 'object' ? schema[1].default : undefined
      if (configuredDefault !== undefined) inputs[name] = configuredDefault
      else if (schemaOptions(schema).length) inputs[name] = schemaOptions(schema)[0]
    }
    for (const [name, schema] of [...Object.entries(definition.required || {}), ...Object.entries(definition.optional || {})]) {
      if (!(name in inputs) || !Array.isArray(schema) || (Array.isArray(inputs[name]) && inputs[name].length === 2)) continue
      const expected = schema[0]
      const options = schema[1] && typeof schema[1] === 'object' ? schema[1] : {}
      const original = inputs[name]
      const choices = schemaOptions(schema)
      if (choices.length && !choices.includes(original)) inputs[name] = choices.includes(options.default) ? options.default : choices[0]
      if (expected === 'INT') {
        const numeric = Number(original)
        inputs[name] = Number.isFinite(numeric) ? Math.max(options.min ?? -Infinity, Math.min(options.max ?? Infinity, Math.round(numeric))) : (options.default ?? 0)
      }
      if (expected === 'FLOAT') {
        const numeric = Number(original)
        inputs[name] = Number.isFinite(numeric) ? Math.max(options.min ?? -Infinity, Math.min(options.max ?? Infinity, numeric)) : (options.default ?? 0)
      }
      if (expected === 'BOOLEAN' && typeof original !== 'boolean') inputs[name] = original === 'true' || original === 1
      if (inputs[name] !== original) repairCount += 1
    }
    prompt[String(node.id)] = { class_type: node.type, inputs, _meta: { title: node.title || node.type } }
  }
  if (!Object.keys(prompt).length) throw new Error('没有找到可执行节点')
  return { workflow: prompt, skippedCount, repairCount }
}

async function readManagerMappings(base) {
  const candidates = [`${base}/manager/customnode/getmappings?mode=local`, `${base}/customnode/getmappings?mode=local`]
  for (const url of candidates) {
    try { return { available: true, mappings: await fetchJson(url, {}, 4500) } } catch { /* try next route */ }
  }
  return { available: false, mappings: null }
}

function mediaFromHistory(base, history, promptId) {
  const item = history?.[promptId] || history?.history?.find?.((entry) => entry.prompt_id === promptId) || history
  const media = []
  for (const [nodeId, output] of Object.entries(item?.outputs || {})) {
    for (const key of ['images', 'gifs', 'videos', 'audio']) {
      for (const file of output?.[key] || []) {
        if (!file?.filename) continue
        const query = new URLSearchParams({ filename: file.filename, subfolder: file.subfolder || '', type: file.type || 'output' })
        const remoteUrl = `${base}/view?${query}`
        const descriptor = String(file.filename || '')
        const kind = /\.(mp4|webm|mov|mkv|avi)$/i.test(descriptor)
          ? 'videos'
          : /\.(mp3|wav|flac|ogg|m4a|aac)$/i.test(descriptor)
            ? 'audio'
            : key
        media.push({ nodeId, kind, sourceKind: key, filename: file.filename, remoteUrl, url: remoteUrl.replace(/^https?/, 'comfy-media') })
      }
    }
  }
  return media
}

async function localizeComfyMedia(media) {
  if (!media?.remoteUrl) return media
  if (comfyMediaCache.has(media.remoteUrl)) return comfyMediaCache.get(media.remoteUrl)
  const pending = (async () => {
    try {
      const source = new URL(media.remoteUrl)
      if (!['127.0.0.1', 'localhost'].includes(source.hostname) || source.pathname !== '/view') return media
      const extension = path.extname(media.filename || '').replace(/[^.a-zA-Z0-9]/g, '').slice(0, 10) || '.bin'
      const cacheDirectory = path.join(app.getPath('temp'), 'ziyouzhe-comfy-media')
      const cachePath = path.join(cacheDirectory, `${crypto.createHash('sha256').update(media.remoteUrl).digest('hex')}${extension}`)
      await fs.mkdir(cacheDirectory, { recursive: true })
      try {
        const details = await fs.stat(cachePath)
        if (details.size > 0) return { ...media, url: pathToFileURL(cachePath).href, cached: true }
      } catch { /* download missing media below */ }
      const response = await fetch(media.remoteUrl, { signal: AbortSignal.timeout(120000) })
      if (!response.ok || !response.body) return media
      const temporaryPath = `${cachePath}.${process.pid}.part`
      await pipeline(Readable.fromWeb(response.body), fsSync.createWriteStream(temporaryPath))
      await fs.rename(temporaryPath, cachePath).catch(async () => {
        await fs.copyFile(temporaryPath, cachePath)
        await fs.unlink(temporaryPath).catch(() => {})
      })
      return { ...media, url: pathToFileURL(cachePath).href, cached: true }
    } catch {
      return media
    }
  })()
  comfyMediaCache.set(media.remoteUrl, pending)
  return pending
}

function comfyActivityKind(prompt, media = []) {
  if (media.some((item) => item.kind === 'videos' || item.kind === 'gifs')) return 'video'
  const definitions = Object.values(prompt || {})
  const text = definitions.map((node) => `${node?.class_type || ''} ${node?._meta?.title || ''}`).join(' ').toLowerCase()
  return /video|minimax|\bh3\b|wan.*(i2v|t2v)|i2v|t2v|fl2v|动画|视频|首尾帧/.test(text) ? 'video' : 'image'
}

function historyTimestamp(item) {
  const created = Number(item?.prompt?.[3]?.create_time)
  if (Number.isFinite(created) && created > 0) return created
  const messages = item?.status?.messages || []
  const timestamps = messages.map((message) => Number(message?.[1]?.timestamp)).filter((value) => Number.isFinite(value) && value > 0)
  return timestamps.length ? Math.min(...timestamps) : Date.now()
}

function historyErrorMessage(item) {
  const error = [...(item?.status?.messages || [])].reverse().find((message) => message?.[0] === 'execution_error')?.[1]
  return error?.exception_message || error?.exception_type || 'ComfyUI 执行失败'
}

function queueActivities(queue, status) {
  return (queue || []).map((entry) => {
    const promptId = entry?.[1]
    const prompt = entry?.[2] || {}
    return promptId ? { promptId, status, progress: status === '生成中' ? 8 : 2, media: [], kind: comfyActivityKind(prompt), workflowName: 'ComfyUI 外部任务', createdAt: Number(entry?.[3]?.create_time) || Date.now() } : null
  }).filter(Boolean)
}

ipcMain.handle('comfy:recent-activity', async (_event, endpoint, limit = 50) => {
  try {
    const base = normalizeEndpoint(endpoint)
    const safeLimit = Math.max(10, Math.min(100, Number(limit) || 50))
    const [history, queue] = await Promise.all([
      fetchJson(`${base}/history?max_items=${safeLimit}`, {}, 9000),
      fetchJson(`${base}/queue`, {}, 9000).catch(() => ({ queue_running: [], queue_pending: [] }))
    ])
    const historyItems = await Promise.all(Object.entries(history || {}).map(async ([promptId, item]) => {
      const media = await Promise.all(mediaFromHistory(base, { [promptId]: item }, promptId).map(localizeComfyMedia))
      const failed = item?.status?.status_str === 'error'
      const complete = item?.status?.completed === true || item?.status?.status_str === 'success'
      return {
        promptId,
        status: failed ? `失败：${historyErrorMessage(item)}` : complete ? '已完成' : '生成中',
        progress: failed ? 0 : complete ? 100 : 12,
        media,
        kind: comfyActivityKind(item?.prompt?.[2], media),
        workflowName: 'ComfyUI 外部任务',
        createdAt: historyTimestamp(item)
      }
    }))
    const queued = [
      ...queueActivities(queue?.queue_running, '生成中'),
      ...queueActivities(queue?.queue_pending, '排队中')
    ]
    const byPrompt = new Map()
    for (const item of [...queued, ...historyItems]) byPrompt.set(item.promptId, { ...byPrompt.get(item.promptId), ...item })
    const items = [...byPrompt.values()].sort((left, right) => right.createdAt - left.createdAt).slice(0, safeLimit)
    return { ok: true, items }
  } catch (error) {
    return { ok: false, message: `同步 ComfyUI 任务失败：${error.message}`, items: [] }
  }
})

async function finishComfyRun(sender, base, promptId, socket) {
  try {
    await new Promise((resolve) => setTimeout(resolve, 180))
    const history = await fetchJson(`${base}/history/${encodeURIComponent(promptId)}`, {}, 12000)
    const media = await Promise.all(mediaFromHistory(base, history, promptId).map(localizeComfyMedia))
    sender.send('comfy:event', { type: 'complete', promptId, media })
  } catch (error) {
    sender.send('comfy:event', { type: 'error', promptId, message: `读取输出失败：${error.message}` })
  } finally {
    activeComfyRuns.delete(promptId)
    try { socket.close() } catch { /* already closed */ }
  }
}

ipcMain.handle('comfy:check', async (_event, endpoint) => {
  try {
    const base = normalizeEndpoint(endpoint)
    const response = await fetch(`${base}/system_stats`, { signal: AbortSignal.timeout(3500) })
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    return { ok: true, message: 'ComfyUI 连接正常' }
  } catch (error) {
    return { ok: false, message: `连接失败：${error.message}` }
  }
})

ipcMain.handle('comfy:catalog', async (_event, endpoint) => {
  try {
    const base = normalizeEndpoint(endpoint)
    const objectInfo = await fetchJson(`${base}/object_info`, {}, 18000)
    const modelKeys = /^(ckpt_name|model_name|unet_name|unet|diffusion_model|lora_name|vae_name|control_net_name|controlnet_name|clip_name|clip_name1|clip_name2|upscale_model|upscale_model_name)$/i
    const typeByKey = {
      ckpt_name: '大模型', model_name: '大模型', unet_name: '扩散模型', unet: '扩散模型', diffusion_model: '扩散模型',
      lora_name: 'LoRA', vae_name: 'VAE', control_net_name: 'ControlNet', controlnet_name: 'ControlNet',
      clip_name: 'CLIP', clip_name1: 'CLIP', clip_name2: 'CLIP', upscale_model: '放大模型', upscale_model_name: '放大模型'
    }
    const optionsByInput = {}
    for (const definition of Object.values(objectInfo)) {
      const sections = [definition?.input?.required, definition?.input?.optional]
      for (const section of sections) {
        for (const [key, schema] of Object.entries(section || {})) {
          if (!modelKeys.test(key) || !Array.isArray(schema) || !Array.isArray(schema[0])) continue
          const values = schema[0].filter((value) => typeof value === 'string' && value.trim())
          optionsByInput[key] = [...new Set([...(optionsByInput[key] || []), ...values])]
        }
      }
    }
    const seen = new Set()
    const models = []
    for (const [input, values] of Object.entries(optionsByInput)) {
      for (const name of values) {
        const key = `${typeByKey[input.toLowerCase()] || '模型'}:${name}`
        if (seen.has(key)) continue
        seen.add(key)
        models.push({ name, type: typeByKey[input.toLowerCase()] || '模型', input })
      }
    }
    models.sort((a, b) => a.type.localeCompare(b.type, 'zh-CN') || a.name.localeCompare(b.name, 'zh-CN'))
    let workflows = []
    try {
      const files = await fetchJson(`${base}/userdata?dir=workflows&recurse=true&full_info=true`, {}, 12000)
      workflows = (Array.isArray(files) ? files : []).filter((file) => String(file?.path || file).toLowerCase().endsWith('.json')).map((file) => {
        const localPath = String(file?.path || file)
        const id = `local-${crypto.createHash('sha1').update(localPath).digest('hex').slice(0, 14)}`
        return { id, name: path.posix.basename(localPath, path.posix.extname(localPath)), localPath, size: Number(file?.size || 0), modified: Number(file?.modified || 0), source: 'local' }
      }).sort((a, b) => a.name.localeCompare(b.name, 'zh-CN'))
    } catch { /* older ComfyUI versions may not expose userdata */ }
    return { ok: true, nodeTypeCount: Object.keys(objectInfo).length, models, optionsByInput, workflows, message: `已识别 ${models.length} 个本地模型 · ${workflows.length} 个本地工作流` }
  } catch (error) {
    return { ok: false, models: [], optionsByInput: {}, message: `自动识别失败：${error.message}` }
  }
})

ipcMain.handle('comfy:load-local-workflow', async (_event, endpoint, workflowPath) => {
  try {
    const base = normalizeEndpoint(endpoint)
    const localPath = String(workflowPath || '').replace(/\\/g, '/')
    if (!localPath.toLowerCase().endsWith('.json') || localPath.includes('..') || localPath.length > 500) throw new Error('本地工作流路径无效')
    const value = await fetchJson(`${base}/userdata/${encodeURIComponent(`workflows/${localPath}`)}`, {}, 30000)
    let workflow
    let convertedFromUi = false
    let skippedCount = 0
    let repairCount = 0
    try {
      workflow = unwrapWorkflow(value)
    } catch {
      const objectInfo = await fetchJson(`${base}/object_info`, {}, 18000)
      const converted = convertUiWorkflow(value, objectInfo)
      workflow = converted.workflow
      skippedCount = converted.skippedCount
      repairCount = converted.repairCount
      convertedFromUi = true
    }
    const objectInfo = await fetchJson(`${base}/object_info`, {}, 18000)
    const pruned = pruneApiWorkflow(workflow, objectInfo)
    workflow = pruned.workflow
    const validationIssues = validateApiWorkflow(workflow, objectInfo)
    return { ok: true, workflow, summary: workflowSummary(workflow), convertedFromUi, skippedCount, prunedCount: pruned.removedCount, repairCount, runnable: validationIssues.length === 0, validationIssues }
  } catch (error) {
    return { ok: false, message: `读取本地工作流失败：${error.message}` }
  }
})

ipcMain.handle('comfy:import-workflow', async (event) => {
  const window = BrowserWindow.fromWebContents(event.sender)
  const result = await dialog.showOpenDialog(window, { title: '导入 ComfyUI API 工作流', properties: ['openFile'], filters: [{ name: 'ComfyUI 工作流', extensions: ['json'] }] })
  if (result.canceled || !result.filePaths[0]) return { canceled: true }
  try {
    const filePath = result.filePaths[0]
    const raw = await fs.readFile(filePath, 'utf8')
    if (raw.length > 12 * 1024 * 1024) throw new Error('工作流文件超过 12MB')
    const workflow = unwrapWorkflow(JSON.parse(raw))
    return { ok: true, name: path.basename(filePath, path.extname(filePath)), path: filePath, workflow, summary: workflowSummary(workflow) }
  } catch (error) {
    return { ok: false, message: `导入失败：${error.message}` }
  }
})

ipcMain.handle('comfy:inspect-workflow', async (_event, endpoint, workflow) => {
  try {
    const base = normalizeEndpoint(endpoint)
    const prompt = unwrapWorkflow(workflow)
    const objectInfo = await fetchJson(`${base}/object_info`, {}, 12000)
    const classes = [...new Set(Object.values(prompt).map((node) => node.class_type))]
    const missing = classes.filter((name) => !objectInfo[name])
    const manager = await readManagerMappings(base)
    const missingPackages = missing.map((classType) => {
      const match = manager.mappings && Object.entries(manager.mappings).find(([, value]) => Array.isArray(value?.[0]) && value[0].includes(classType))
      return { classType, package: match?.[0] || null, title: match?.[1]?.[1]?.title_aux || match?.[0] || null }
    })
    return { ok: true, installedCount: Object.keys(objectInfo).length, missing, missingPackages, managerAvailable: manager.available }
  } catch (error) {
    return { ok: false, message: `检测失败：${error.message}` }
  }
})

async function uploadInputBytes(base, bytes, filename) {
  const safeName = String(filename || `reference-${Date.now()}.png`).replace(/[\\/:*?"<>|]/g, '_').slice(-160)
  const form = new FormData()
  form.append('image', new Blob([bytes]), safeName)
  form.append('type', 'input')
  form.append('overwrite', 'true')
  const uploaded = await fetchJson(`${base}/upload/image`, { method: 'POST', body: form }, 180000)
  return { ok: true, name: uploaded.name || safeName, subfolder: uploaded.subfolder || '', type: uploaded.type || 'input' }
}

async function readComfyVideoReference(base, reference) {
  let bytes
  let filename = String(reference?.name || `video-${Date.now()}.mp4`)
  const sourceValue = String(reference?.remoteUrl || reference?.url || '').trim()
  if (/^freedom-asset:\/\/local\//i.test(sourceValue)) {
    const target = localAssetPathFromUrl(sourceValue)
    bytes = await fs.readFile(target)
    filename = path.basename(target)
  } else if (/^comfy-media:/i.test(sourceValue) || /^https?:/i.test(sourceValue)) {
    const normalized = sourceValue.replace(/^comfy-media:/i, base.startsWith('https:') ? 'https:' : 'http:')
    const url = new URL(normalized)
    if (url.origin !== base || url.pathname !== '/view') throw new Error('只允许复用当前 ComfyUI 的输出视频')
    const response = await fetch(url, { signal: AbortSignal.timeout(120000) })
    if (!response.ok) throw new Error(`读取输出视频失败（HTTP ${response.status}）`)
    bytes = Buffer.from(await response.arrayBuffer())
    filename = url.searchParams.get('filename') || filename
  } else if (/^file:/i.test(sourceValue)) {
    const target = decodeURIComponent(new URL(sourceValue).pathname).replace(/^\/(?:([A-Za-z]:))/, '$1')
    const allowedDirectory = path.resolve(path.join(app.getPath('temp'), 'ziyouzhe-comfy-media'))
    const resolved = path.resolve(target)
    if (!resolved.toLowerCase().startsWith(`${allowedDirectory.toLowerCase()}${path.sep}`)) throw new Error('本地视频不在受信任的缓存目录')
    bytes = await fs.readFile(resolved)
    filename = path.basename(resolved)
  } else throw new Error('没有可上传的视频')
  if (!bytes.length || bytes.length > 1024 * 1024 * 1024) throw new Error('视频为空或超过 1GB')
  return { bytes, filename }
}

ipcMain.handle('comfy:upload-image', async (event, endpoint) => {
  const window = BrowserWindow.fromWebContents(event.sender)
  const result = await dialog.showOpenDialog(window, { title: '选择 ComfyUI 输入图片', properties: ['openFile'], filters: [{ name: '图片', extensions: ['png', 'jpg', 'jpeg', 'webp', 'bmp'] }] })
  if (result.canceled || !result.filePaths[0]) return { canceled: true }
  try {
    const base = normalizeEndpoint(endpoint)
    const filePath = result.filePaths[0]
    const stat = await fs.stat(filePath)
    if (stat.size > 80 * 1024 * 1024) throw new Error('图片超过 80MB')
    const bytes = await fs.readFile(filePath)
    return await uploadInputBytes(base, bytes, path.basename(filePath))
  } catch (error) {
    return { ok: false, message: `上传失败：${error.message}` }
  }
})

ipcMain.handle('comfy:upload-video', async (event, endpoint, reference = null) => {
  try {
    const base = normalizeEndpoint(endpoint)
    let selected = reference
    if (!selected) {
      const window = BrowserWindow.fromWebContents(event.sender)
      const result = await dialog.showOpenDialog(window, { title: '选择 ComfyUI 输入视频', properties: ['openFile'], filters: [{ name: '视频', extensions: ['mp4', 'mov', 'mkv', 'webm', 'm4v', 'avi'] }] })
      if (result.canceled || !result.filePaths[0]) return { canceled: true }
      selected = { url: pathToFileURL(result.filePaths[0]).href, name: path.basename(result.filePaths[0]), directFile: result.filePaths[0] }
    }
    let bytes
    let filename
    if (selected.directFile) {
      const stat = await fs.stat(selected.directFile)
      if (!stat.size || stat.size > 1024 * 1024 * 1024) throw new Error('视频为空或超过 1GB')
      bytes = await fs.readFile(selected.directFile)
      filename = path.basename(selected.directFile)
    } else ({ bytes, filename } = await readComfyVideoReference(base, selected))
    return await uploadInputBytes(base, bytes, `canvas-${Date.now()}-${path.basename(filename)}`)
  } catch (error) {
    return { ok: false, message: `视频上传失败：${error.message}` }
  }
})

ipcMain.handle('comfy:upload-reference', async (_event, endpoint, reference) => {
  try {
    const base = normalizeEndpoint(endpoint)
    let bytes
    let filename = String(reference?.name || `storyboard-reference-${Date.now()}.png`)
    if (typeof reference?.dataUrl === 'string') {
      const match = /^data:image\/(png|jpeg|jpg|webp|bmp);base64,([A-Za-z0-9+/=]+)$/.exec(reference.dataUrl)
      if (!match) throw new Error('参考图片格式无效')
      bytes = Buffer.from(match[2], 'base64')
      if (!/\.(png|jpe?g|webp|bmp)$/i.test(filename)) filename += `.${match[1] === 'jpeg' ? 'jpg' : match[1]}`
    } else if (typeof reference?.url === 'string' && reference.url.startsWith('freedom-asset://')) {
      const target = localAssetPathFromUrl(reference.url)
      bytes = await fs.readFile(target)
      filename = path.basename(target)
    } else if (typeof reference?.url === 'string') {
      const url = new URL(reference.url)
      if (url.origin !== base || url.pathname !== '/view') throw new Error('只允许复用当前 ComfyUI 的输出图片')
      const response = await fetch(url, { signal: AbortSignal.timeout(30000) })
      if (!response.ok) throw new Error(`读取输出图片失败（HTTP ${response.status}）`)
      bytes = Buffer.from(await response.arrayBuffer())
      filename = url.searchParams.get('filename') || filename
    } else throw new Error('没有可上传的参考图片')
    if (!bytes.length || bytes.length > 40 * 1024 * 1024) throw new Error('参考图片为空或超过 40MB')
    return await uploadInputBytes(base, bytes, `story-${Date.now()}-${path.basename(filename)}`)
  } catch (error) {
    return { ok: false, message: `参考图片上传失败：${error.message}` }
  }
})

ipcMain.handle('comfy:run-workflow', async (event, endpoint, workflow) => {
  let socket
  try {
    const base = normalizeEndpoint(endpoint)
    const rawPrompt = unwrapWorkflow(workflow)
    const objectInfo = await fetchJson(`${base}/object_info`, {}, 18000)
    const pruned = pruneApiWorkflow(rawPrompt, objectInfo)
    const prepared = sanitizeApiWorkflow(pruned.workflow, objectInfo)
    const prompt = prepared.workflow
    const validationIssues = validateApiWorkflow(prompt, objectInfo)
    if (validationIssues.length) throw new Error(`工作流暂不可运行：${validationIssues.slice(0, 4).join('；')}${validationIssues.length > 4 ? `；另有 ${validationIssues.length - 4} 项` : ''}`)
    const clientId = crypto.randomUUID()
    const wsBase = base.replace(/^http:/, 'ws:').replace(/^https:/, 'wss:')
    socket = new WebSocket(`${wsBase}/ws?clientId=${encodeURIComponent(clientId)}`)
    await new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error('WebSocket 连接超时')), 6500)
      socket.addEventListener('open', () => { clearTimeout(timer); resolve() }, { once: true })
      socket.addEventListener('error', () => { clearTimeout(timer); reject(new Error('WebSocket 连接失败')) }, { once: true })
    })
    const queued = await fetchJson(`${base}/prompt`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ prompt, client_id: clientId }) }, 15000)
    const promptId = queued.prompt_id
    if (!promptId) throw new Error(queued.error || 'ComfyUI 未返回任务编号')
    activeComfyRuns.set(promptId, { socket, base })
    socket.addEventListener('message', async (message) => {
      try {
        const raw = typeof message.data === 'string' ? message.data : Buffer.from(await message.data.arrayBuffer?.() || message.data).toString('utf8')
        const payload = JSON.parse(raw)
        const data = payload.data || {}
        if (data.prompt_id && data.prompt_id !== promptId) return
        if (payload.type === 'progress') event.sender.send('comfy:event', { type: 'progress', promptId, value: data.value || 0, max: data.max || 1, node: data.node })
        if (payload.type === 'executing') {
          event.sender.send('comfy:event', { type: 'executing', promptId, node: data.node })
          if (data.node == null) await finishComfyRun(event.sender, base, promptId, socket)
        }
        if (payload.type === 'execution_error') {
          event.sender.send('comfy:event', { type: 'error', promptId, message: data.exception_message || data.exception_type || '工作流执行失败' })
          activeComfyRuns.delete(promptId)
        }
      } catch { /* binary previews and unknown messages are ignored */ }
    })
    return { ok: true, promptId, queueNumber: queued.number, repairCount: prepared.repairCount, prunedCount: pruned.removedCount }
  } catch (error) {
    try { socket?.close() } catch { /* no-op */ }
    return { ok: false, message: `提交失败：${error.message}` }
  }
})

ipcMain.handle('comfy:interrupt', async (_event, endpoint, promptId) => {
  try {
    const base = normalizeEndpoint(endpoint)
    await fetchJson(`${base}/interrupt`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: '{}' }, 6000)
    const run = activeComfyRuns.get(promptId)
    try { run?.socket?.close() } catch { /* no-op */ }
    activeComfyRuns.delete(promptId)
    return { ok: true }
  } catch (error) {
    return { ok: false, message: `停止失败：${error.message}` }
  }
})

ipcMain.handle('comfy:cancel-task', async (_event, endpoint, promptId) => {
  try {
    const base = normalizeEndpoint(endpoint)
    if (!promptId) throw new Error('缺少 ComfyUI 任务编号')
    const queue = await fetchJson(`${base}/queue`, {}, 7000)
    const running = (queue?.queue_running || []).some((entry) => entry?.[1] === promptId)
    const pending = (queue?.queue_pending || []).some((entry) => entry?.[1] === promptId)
    if (pending) {
      await fetchJson(`${base}/queue`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ delete: [promptId] }) }, 7000)
    }
    if (running) {
      await fetchJson(`${base}/interrupt`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: '{}' }, 7000)
    }
    const activeRun = activeComfyRuns.get(promptId)
    if (activeRun) {
      try { activeRun.socket?.close() } catch { /* no-op */ }
      activeComfyRuns.delete(promptId)
    }
    if (!running && !pending) return { ok: false, inactive: true, message: '该任务已经结束，不需要取消' }
    return { ok: true, message: running ? '已中断 ComfyUI 正在执行的任务' : '已从 ComfyUI 排队列表删除任务' }
  } catch (error) {
    return { ok: false, message: `取消任务失败：${error.message}` }
  }
})

ipcMain.handle('comfy:cancel-all', async (_event, endpoint) => {
  try {
    const base = normalizeEndpoint(endpoint)
    const queue = await fetchJson(`${base}/queue`, {}, 7000)
    const runningCount = (queue?.queue_running || []).length
    const pendingCount = (queue?.queue_pending || []).length
    if (runningCount) await fetchJson(`${base}/interrupt`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: '{}' }, 7000)
    if (pendingCount) await fetchJson(`${base}/queue`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ clear: true }) }, 7000)
    for (const run of activeComfyRuns.values()) {
      try { run.socket?.close() } catch { /* no-op */ }
    }
    activeComfyRuns.clear()
    if (!runningCount && !pendingCount) return { ok: true, inactive: true, message: 'ComfyUI 后台当前没有运行或排队任务' }
    return { ok: true, message: `已取消 ComfyUI 后台 ${runningCount + pendingCount} 个任务` }
  } catch (error) {
    return { ok: false, message: `一键取消失败：${error.message}` }
  }
})

ipcMain.handle('system:stats', async () => {
  const totalMemory = os.totalmem()
  const usedMemory = totalMemory - os.freemem()
  const gpu = await readGpuStats()
  return {
    cpu: readCpuPercent(),
    memory: Math.round((usedMemory / totalMemory) * 100),
    memoryUsedGb: Number((usedMemory / 1024 ** 3).toFixed(1)),
    memoryTotalGb: Number((totalMemory / 1024 ** 3).toFixed(1)),
    gpu: gpu.utilization,
    vramUsedGb: gpu.used == null ? null : Number((gpu.used / 1024).toFixed(1)),
    vramTotalGb: gpu.total == null ? null : Number((gpu.total / 1024).toFixed(1)),
    temperature: gpu.temperature
  }
})

ipcMain.handle('system:release', async (event, endpoint, type) => {
  try {
    const base = normalizeEndpoint(endpoint)
    if (type === 'vram') await fetchJson(`${base}/free`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ unload_models: true, free_memory: false }) }, 8000)
    if (type === 'memory' || type === 'all') {
      await session.defaultSession.clearCache()
      await fetchJson(`${base}/free`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ unload_models: type === 'all', free_memory: true }) }, 8000)
    }
    BrowserWindow.fromWebContents(event.sender)?.webContents.send('system:released', type)
    return { ok: true, message: type === 'vram' ? 'ComfyUI 模型与显存缓存已释放' : type === 'all' ? '显存、内存和应用缓存已全部释放' : 'ComfyUI 内存与应用缓存已释放' }
  } catch (error) {
    return { ok: false, message: `释放失败：${error.message}` }
  }
})

ipcMain.handle('comfy:stop-all-and-release', async (event, endpoint) => {
  try {
    const base = normalizeEndpoint(endpoint)
    await fetchJson(`${base}/interrupt`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: '{}' }, 8000)
    await fetchJson(`${base}/queue`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ clear: true }) }, 8000)
    for (const run of activeComfyRuns.values()) {
      try { run.socket?.close() } catch { /* no-op */ }
    }
    activeComfyRuns.clear()
    await fetchJson(`${base}/free`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ unload_models: true, free_memory: true }) }, 8000)
    await session.defaultSession.clearCache()
    BrowserWindow.fromWebContents(event.sender)?.webContents.send('system:released', 'all')
    return { ok: true, message: '任务已停止，队列、显存和内存已释放' }
  } catch (error) {
    return { ok: false, message: `停止任务失败：${error.message}` }
  }
})

ipcMain.handle('app:quit', () => {
  isQuitting = true
  app.quit()
  return { ok: true }
})

ipcMain.handle('canvas:save', async (_event, canvasId, canvas) => {
  try {
    const id = safeStorageId(canvasId, 'canvas')
    const target = path.join(canvasStorageDirectory(), `${id}.json`)
    await writeJsonAtomic(target, { version: 2, savedAt: Date.now(), nodes: canvas?.nodes || [], edges: canvas?.edges || [] })
    const details = await fs.stat(target)
    return { ok: true, bytes: details.size }
  } catch (error) {
    return { ok: false, message: error.message }
  }
})

ipcMain.handle('canvas:load', async (_event, canvasId) => {
  try {
    const id = safeStorageId(canvasId, 'canvas')
    const target = path.join(canvasStorageDirectory(), `${id}.json`)
    const canvas = JSON.parse(await fs.readFile(target, 'utf8'))
    return { ok: true, canvas }
  } catch (error) {
    if (error.code === 'ENOENT') return { ok: true, canvas: null }
    return { ok: false, message: error.message }
  }
})

ipcMain.handle('asset:store', async (_event, input) => {
  try {
    let bytes
    let mimeType = String(input?.type || 'application/octet-stream').toLowerCase()
    if (input?.dataUrl) {
      const match = /^data:([^;,]+);base64,([a-z0-9+/=\r\n]+)$/i.exec(input.dataUrl)
      if (!match) throw new Error('素材数据格式无效')
      mimeType = match[1].toLowerCase()
      bytes = Buffer.from(match[2], 'base64')
    } else {
      bytes = Buffer.from(input?.bytes || [])
    }
    if (!bytes.length) throw new Error('素材内容为空')
    if (bytes.length > 200 * 1024 * 1024) throw new Error('单个素材不能超过 200MB')
    const extensionByType = { 'image/png': '.png', 'image/jpeg': '.jpg', 'image/jpg': '.jpg', 'image/webp': '.webp', 'image/gif': '.gif', 'image/bmp': '.bmp', 'video/mp4': '.mp4', 'video/webm': '.webm', 'video/quicktime': '.mov' }
    const originalExtension = path.extname(String(input?.name || '')).replace(/[^.a-zA-Z0-9]/g, '').slice(0, 10).toLowerCase()
    const extension = extensionByType[mimeType] || originalExtension || '.bin'
    const digest = crypto.createHash('sha256').update(bytes).digest('hex')
    const filename = `${digest}${extension}`
    const directory = assetStorageDirectory()
    const target = path.join(directory, filename)
    await fs.mkdir(directory, { recursive: true })
    if (!fsSync.existsSync(target)) await fs.writeFile(target, bytes)
    return { ok: true, url: assetUrl(filename), filename, bytes: bytes.length, type: mimeType }
  } catch (error) {
    return { ok: false, message: error.message }
  }
})

app.whenReady().then(async () => {
  await protocol.handle('freedom-asset', async (request) => {
    try {
      const source = new URL(request.url)
      if (source.hostname !== 'local') return new Response('Forbidden', { status: 403 })
      const filename = path.basename(decodeURIComponent(source.pathname))
      if (!/^[a-f0-9]{64}\.[a-z0-9]{1,10}$/i.test(filename)) return new Response('Invalid asset', { status: 400 })
      const target = path.join(assetStorageDirectory(), filename)
      if (!fsSync.existsSync(target)) return new Response('Asset not found', { status: 404 })
      return net.fetch(pathToFileURL(target).href, { method: request.method, headers: request.headers })
    } catch {
      return new Response('Invalid asset URL', { status: 400 })
    }
  })
  await protocol.handle('comfy-media', async (request) => {
    try {
      const source = new URL(request.url)
      if (!['127.0.0.1', 'localhost'].includes(source.hostname) || source.pathname !== '/view') return new Response('Forbidden', { status: 403 })
      const target = `http://${source.host}${source.pathname}${source.search}`
      return net.fetch(target, { method: request.method, headers: request.headers })
    } catch {
      return new Response('Invalid ComfyUI media URL', { status: 400 })
    }
  })
  createWindow()
  createTray()
  app.on('activate', () => {
    showMainWindow()
  })
})

app.on('before-quit', () => { isQuitting = true })
app.on('window-all-closed', () => { /* Continue running in the system tray. */ })
