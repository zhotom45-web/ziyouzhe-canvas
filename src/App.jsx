import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Background, BaseEdge, Controls, Handle, MiniMap, NodeResizer, Position, ReactFlow,
  addEdge, applyEdgeChanges, applyNodeChanges, getBezierPath
} from '@xyflow/react'

const STORAGE_KEY = 'aaa-lite-project-v1'
const PROJECTS_KEY = 'aaa-lite-projects-v1'
const ACTIVE_PROJECT_KEY = 'aaa-lite-active-project-v1'
const CANVAS_KEY_PREFIX = 'aaa-lite-canvas-v1:'
const SEEDANCE_MODELS_KEY = 'aaa-lite-seedance-models-v1'
const SEEDANCE_MODEL_LABELS_KEY = 'aaa-lite-seedance-model-labels-v1'
const ACTIVE_SEEDANCE_MODEL_KEY = 'aaa-lite-active-seedance-model-v1'
const WORKFLOW_KEY = 'aaa-lite-comfy-workflows-v1'
const LOCAL_WORKFLOW_CACHE_VERSION = 4

const nodeCatalog = {
  script: { title: '分镜师', icon: '▤', tone: 'amber', body: '人物设计、场景设计、分镜图与视频自动生成' },
  prompt: { title: '提示词', icon: '✦', tone: 'violet', body: '描述你想生成的画面' },
  image: { title: '图片', icon: '◫', tone: 'blue', body: '文生图 / 图生图' },
  video: { title: '视频', icon: '▶', tone: 'pink', body: '使用 ComfyUI 或 Seedance' },
  video_second: { title: '视频二采', icon: '✦', tone: 'violet', body: 'SeedVR2 时序修复、人脸与细节二次重建' },
  video_upscale: { title: '视频超分', icon: '▲', tone: 'blue', body: 'SeedVR2 时序一致超分至 1080P' },
  reference: { title: '通用参考', icon: '⌁', tone: 'amber', body: '角色、风格或内容参考图' },
  frame: { title: '首尾帧', icon: '◩', tone: 'cyan', body: '控制视频开始与结束画面' },
  comfy: { title: 'ComfyUI 工作流', icon: '◆', tone: 'cyan', body: '执行导入的 API 工作流' },
  output: { title: '导出结果', icon: '↓', tone: 'green', body: '预览并保存文件' }
}

const creatableNodeEntries = Object.entries(nodeCatalog).filter(([key]) => key !== 'comfy')
const comfyParamLabels = { image: '输入图片', video: '输入视频', file: '输入视频', text: '提示词', prompt: '提示词', positive: '正向提示词', negative: '负向提示词', ckpt_name: '基础模型', model_name: '模型', unet_name: '扩散模型', diffusion_model: '扩散模型', lora_name: 'LoRA', vae_name: 'VAE', control_net_name: 'ControlNet', controlnet_name: 'ControlNet', clip_name: 'CLIP', clip_name1: 'CLIP 1', clip_name2: 'CLIP 2', upscale_model: '放大模型', upscale_model_name: '放大模型', width: '宽度', height: '高度', steps: '步数', cfg: 'CFG', seed: '种子', noise_seed: '噪声种子', denoise: '重绘强度', sampler_name: '采样器', scheduler: '调度器' }
const modelParamPattern = /^(ckpt_name|model_name|unet_name|unet|diffusion_model|lora_name|vae_name|control_net_name|controlnet_name|clip_name|clip_name1|clip_name2|upscale_model|upscale_model_name)$/i

const mediaKinds = new Set(['image', 'video', 'reference', 'frame'])
const fillTaskKinds = new Set(['prompt', 'image', 'video', 'reference', 'frame'])
const cancelableTaskStatuses = new Set(['排队中', '生成中', '正在检查依赖'])
const localTaskUiKeys = new Set(['taskEditorOpen', 'taskEditorExpanded', 'taskPickerOpen'])
const MAX_SESSION_TASKS = 120

function trimSessionTasks(items) {
  if (items.length <= MAX_SESSION_TASKS) return items
  const active = items.filter((item) => cancelableTaskStatuses.has(item.status) || item.status?.endsWith('待处理'))
  const activeIds = new Set(active.map((item) => item.promptId || item.id))
  return [...active, ...items.filter((item) => !activeIds.has(item.promptId || item.id))].slice(0, MAX_SESSION_TASKS)
}

function seedVrVideoWorkflow({ resolution, maxResolution, prefix, inputNoise = 0, latentNoise = 0 }) {
  return {
    '1': { class_type: 'VHS_LoadVideo', inputs: { video: '', force_rate: 0, custom_width: 0, custom_height: 0, frame_load_cap: 0, skip_first_frames: 0, select_every_nth: 1, meta_batch: ['15', 0], format: 'None' }, _meta: { title: '输入视频' } },
    '2': { class_type: 'VHS_VideoInfoLoaded', inputs: { video_info: ['1', 3] }, _meta: { title: '自动读取原帧率' } },
    '7': { class_type: 'SeedVR2LoadDiTModel', inputs: { model: 'seedvr2_ema_3b_fp8_e4m3fn.safetensors', device: 'cuda:0', blocks_to_swap: 24, swap_io_components: true, offload_device: 'cpu', cache_model: false, attention_mode: 'sdpa' }, _meta: { title: 'SeedVR2 3B FP8 低显存模型' } },
    '8': { class_type: 'SeedVR2LoadVAEModel', inputs: { model: 'ema_vae_fp16.safetensors', device: 'cuda:0', encode_tiled: true, encode_tile_size: 512, encode_tile_overlap: 64, decode_tiled: true, decode_tile_size: 512, decode_tile_overlap: 64, tile_debug: 'false', offload_device: 'cpu', cache_model: false }, _meta: { title: 'SeedVR2 VAE 分块保真' } },
    '9': { class_type: 'SeedVR2VideoUpscaler', inputs: { image: ['1', 0], dit: ['7', 0], vae: ['8', 0], seed: 42, resolution, max_resolution: maxResolution, batch_size: 5, uniform_batch_size: true, color_correction: 'lab', temporal_overlap: 2, prepend_frames: 4, input_noise_scale: inputNoise, latent_noise_scale: latentNoise, offload_device: 'cpu', enable_debug: false }, _meta: { title: 'SeedVR2 时序细节重建' } },
    '10': { class_type: 'VHS_VideoCombine', inputs: { images: ['9', 0], audio: ['1', 2], frame_rate: ['2', 0], meta_batch: ['15', 0], loop_count: 0, filename_prefix: prefix, format: 'video/nvenc_h264-mp4', pix_fmt: 'yuv420p', bitrate: 60, megabit: true, save_metadata: true, pingpong: false, save_output: true }, _meta: { title: '保留原音频导出 MP4' } },
    '15': { class_type: 'VHS_BatchManager', inputs: { frames_per_batch: 21 }, _meta: { title: '长视频分批管理' } }
  }
}

const BUILTIN_VIDEO_TOOL_RECORDS = {
  video_second: {
    id: 'builtin-video-second-pass', name: '视频二采 · SeedVR2 720P 精修', source: 'builtin', mediaKind: 'video',
    workflow: seedVrVideoWorkflow({ resolution: 720, maxResolution: 1280, prefix: 'FreedomCanvas/SecondPass_720p', inputNoise: 0.015, latentNoise: 0.01 }),
    summary: { nodeCount: 6, classes: ['VHS_LoadVideo', 'SeedVR2VideoUpscaler', 'VHS_VideoCombine'] },
    params: [
      { id: 'second-video', nodeId: '1', key: 'video', label: '输入视频', value: '', valueType: 'string' },
      { id: 'second-resolution', nodeId: '9', key: 'resolution', label: '目标短边', value: 720, valueType: 'number' },
      { id: 'second-batch', nodeId: '9', key: 'batch_size', label: '时序批次', value: 5, valueType: 'number' },
      { id: 'second-seed', nodeId: '9', key: 'seed', label: '种子', value: 42, valueType: 'number' }
    ]
  },
  video_upscale: {
    id: 'builtin-video-upscale', name: '视频超分 · SeedVR2 1080P', source: 'builtin', mediaKind: 'video',
    workflow: seedVrVideoWorkflow({ resolution: 1080, maxResolution: 1920, prefix: 'FreedomCanvas/Upscale_1080p' }),
    summary: { nodeCount: 6, classes: ['VHS_LoadVideo', 'SeedVR2VideoUpscaler', 'VHS_VideoCombine'] },
    params: [
      { id: 'upscale-video', nodeId: '1', key: 'video', label: '输入视频', value: '', valueType: 'string' },
      { id: 'upscale-resolution', nodeId: '9', key: 'resolution', label: '目标短边', value: 1080, valueType: 'number' },
      { id: 'upscale-batch', nodeId: '9', key: 'batch_size', label: '时序批次', value: 5, valueType: 'number' },
      { id: 'upscale-seed', nodeId: '9', key: 'seed', label: '种子', value: 42, valueType: 'number' }
    ]
  }
}

function referenceDisplayUrl(reference) {
  return reference?.dataUrl || reference?.url || ''
}

async function storeAssetFile(file) {
  if (!window.aaaLite?.storeAsset) {
    return await new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve({ ok: true, url: reader.result, type: file.type, bytes: file.size })
      reader.onerror = reject
      reader.readAsDataURL(file)
    })
  }
  return window.aaaLite.storeAsset({ name: file.name, type: file.type, bytes: await file.arrayBuffer() })
}

async function externalizeCanvasMedia(nodes) {
  if (!window.aaaLite?.storeAsset) return { nodes, changed: false }
  const cache = new Map()
  let changed = false
  const visit = async (value) => {
    if (typeof value === 'string' && /^data:(?:image|video)\//i.test(value)) {
      if (!cache.has(value)) cache.set(value, window.aaaLite.storeAsset({ dataUrl: value }))
      const result = await cache.get(value)
      if (!result?.ok) throw new Error(result?.message || '素材迁移失败')
      changed = true
      return result.url
    }
    if (Array.isArray(value)) {
      let next = null
      for (let index = 0; index < value.length; index += 1) {
        const visited = await visit(value[index])
        if (visited !== value[index]) {
          if (!next) next = value.slice()
          next[index] = visited
        }
      }
      return next || value
    }
    if (!value || typeof value !== 'object') return value
    let next = null
    for (const [key, item] of Object.entries(value)) {
      const visited = await visit(item)
      if (visited !== item) {
        if (!next) next = { ...value }
        next[key] = visited
      }
    }
    return next || value
  }
  return { nodes: await visit(nodes), changed }
}

function mediaUrlLooksVideo(url) {
  return /^data:video\//i.test(String(url || '')) || /\.(mp4|webm|mov|m4v)(?:[?&#]|$)/i.test(String(url || '')) || /[?&]filename=[^&#]*\.(mp4|webm|mov|m4v)(?:[&#]|$)/i.test(String(url || ''))
}

function mediaUrlLooksImage(url) {
  return /^data:image\//i.test(String(url || '')) || /\.(png|jpe?g|webp|bmp|gif)(?:[?&#]|$)/i.test(String(url || '')) || /[?&]filename=[^&#]*\.(png|jpe?g|webp|bmp|gif)(?:[&#]|$)/i.test(String(url || ''))
}

function videoSourceFromData(data) {
  const videoMedia = data?.results?.find((result) => mediaIsVideo(result))
  return data?.frameSourceVideoUrl || (mediaUrlLooksVideo(data?.resultUrl) || videoMedia ? (data?.resultUrl || videoMedia?.url || videoMedia?.remoteUrl) : '') || (data?.uploadedMediaType === 'video' || mediaUrlLooksVideo(data?.url) ? data?.url : '') || ''
}

function captureVideoFrame(videoUrl, frameRole = '尾帧') {
  return new Promise((resolve, reject) => {
    if (!videoUrl) return reject(new Error('没有可读取的视频'))
    const video = document.createElement('video')
    video.preload = 'auto'
    video.muted = true
    video.playsInline = true
    if (/^https?:/i.test(videoUrl)) video.crossOrigin = 'anonymous'
    let settled = false
    const finish = (error, value) => {
      if (settled) return
      settled = true
      window.clearTimeout(timeout)
      video.removeAttribute('src')
      video.load()
      if (error) reject(error)
      else resolve(value)
    }
    const draw = () => {
      try {
        const width = video.videoWidth || 1280
        const height = video.videoHeight || 720
        const canvas = document.createElement('canvas')
        canvas.width = width
        canvas.height = height
        const context = canvas.getContext('2d')
        context.drawImage(video, 0, 0, width, height)
        finish(null, canvas.toDataURL('image/jpeg', 0.94))
      } catch (error) { finish(error) }
    }
    video.onerror = () => finish(new Error('视频画面读取失败'))
    video.onloadedmetadata = () => {
      const duration = Number(video.duration)
      if (!Number.isFinite(duration) || duration <= 0) return
      const offset = Math.min(0.12, Math.max(0.04, duration * 0.01))
      const target = frameRole === '首帧' ? Math.min(offset, duration / 2) : Math.max(0, duration - offset)
      if (Math.abs(video.currentTime - target) < 0.001 && video.readyState >= 2) draw()
      else video.currentTime = target
    }
    video.onseeked = draw
    video.onloadeddata = () => {
      if (frameRole === '首帧' && (!Number.isFinite(video.duration) || video.duration <= 0)) draw()
    }
    const timeout = window.setTimeout(() => finish(new Error('视频帧识别超时')), 18000)
    video.src = videoUrl
    video.load()
  })
}

function repairPersistedFrameNode(node) {
  if (node?.data?.kind !== 'frame') return node
  const persistedVideoUrl = videoSourceFromData(node.data)
  const resultIsVideo = Boolean(persistedVideoUrl)
  if (!resultIsVideo) return node
  const references = (node.data.taskReferences || []).filter((reference) => !mediaUrlLooksVideo(referenceDisplayUrl(reference)))
  const storedReference = references.find((reference) => referenceDisplayUrl(reference))
  const directImageUrl = [node.data.url, node.data.taskReferenceUrl].find((url) => mediaUrlLooksImage(url))
  const frameUrl = referenceDisplayUrl(storedReference) || directImageUrl || ''
  const frameReference = frameUrl ? { ...(storedReference || {}), url: frameUrl, dataUrl: String(frameUrl).startsWith('data:image/') ? frameUrl : storedReference?.dataUrl, name: storedReference?.name || node.data.taskReferenceName || node.data.name || `${node.data.frameRole || '首尾帧'}图片` } : null
  return { ...node, data: { ...node.data, frameRole: node.data.frameRole || '尾帧', frameSourceVideoUrl: persistedVideoUrl, resultUrl: null, results: [], promptId: null, url: frameUrl || null, uploadedMediaType: frameUrl ? 'image' : null, taskReferences: frameReference ? [frameReference, ...references.filter((reference) => referenceDisplayUrl(reference) !== frameUrl)].slice(0, 9) : [], taskReferenceUrl: frameUrl || null, taskReferenceName: frameReference?.name || '', status: `正在自动识别${node.data.frameRole || '尾帧'}` } }
}

function nodeImageReferences(node) {
  if (!node || !['image', 'reference', 'frame'].includes(node.data?.kind)) return []
  const frameRole = node.data.kind === 'frame' ? (node.data.frameRole === '首帧' ? 'first_frame' : 'last_frame') : null
  const stored = (Array.isArray(node.data.taskReferences) ? node.data.taskReferences : []).filter((reference) => !mediaUrlLooksVideo(referenceDisplayUrl(reference))).map((reference) => frameRole ? { ...reference, role: frameRole } : reference)
  const directUrl = node.data.resultUrl || node.data.url || node.data.taskReferenceUrl
  const direct = directUrl && !mediaUrlLooksVideo(directUrl) ? [{ url: directUrl, dataUrl: String(directUrl).startsWith('data:') ? directUrl : undefined, remoteUrl: node.data.remoteUrl, name: node.data.name || node.data.taskReferenceName || (frameRole === 'first_frame' ? '首帧图片' : frameRole === 'last_frame' ? '尾帧图片' : '连线参考图'), assetId: node.id, role: frameRole || undefined }] : []
  return [...direct, ...stored].filter((reference) => reference?.url || reference?.dataUrl)
}

function mergeImageReferences(...groups) {
  const indexes = new Map()
  const merged = []
  for (const reference of groups.flat()) {
    const key = reference?.remoteUrl || reference?.url || reference?.dataUrl
    if (!key) continue
    if (indexes.has(key)) {
      const index = indexes.get(key)
      if (!merged[index]?.role && reference.role) merged[index] = reference
      continue
    }
    indexes.set(key, merged.length)
    merged.push(reference)
  }
  return merged.slice(0, 9)
}

function orderedFrameReferences(references = []) {
  const first = references.find((reference) => reference.role === 'first_frame')
  const last = references.find((reference) => reference.role === 'last_frame')
  const rest = references.filter((reference) => reference !== first && reference !== last)
  return [first, last, ...rest].filter(Boolean)
}

function h3FrameInputRole(name) {
  if (/^(?:mode\.)?(?:first_frame|start_frame|start_image)$/i.test(name)) return 'first_frame'
  if (/^(?:mode\.)?(?:last_frame|end_frame|end_image)$/i.test(name)) return 'last_frame'
  return null
}

function h3FrameInputRoles(workflow) {
  const roles = new Set()
  for (const node of Object.values(workflow || {})) {
    const inputs = node.inputs || {}
    const mode = Object.entries(inputs).filter(([name]) => /mode/i.test(name)).map(([, value]) => typeof value === 'string' ? value : '').join(' ')
    const firstLastMode = /first[_\s-]*last|fl2va|首尾帧/i.test(`${mode} ${node?._meta?.title || ''}`)
    for (const [name, value] of Object.entries(inputs)) {
      const role = h3FrameInputRole(name)
      if (role && (firstLastMode || (Array.isArray(value) && value.length === 2))) roles.add(role)
    }
  }
  return roles
}

function bindH3FrameInputs(workflow, references) {
  const byRole = new Map(references.filter((reference) => reference.role && reference.comfyName).map((reference) => [reference.role, reference]))
  const boundLoaders = new Set()
  const boundRoles = new Set()
  const findLoaders = (value) => {
    const found = new Set()
    const pending = Array.isArray(value) && value.length === 2 ? [String(value[0])] : []
    const visited = new Set()
    while (pending.length) {
      const nodeId = pending.pop()
      if (visited.has(nodeId)) continue
      visited.add(nodeId)
      const node = workflow[nodeId]
      if (!node) continue
      if (/LoadImage/i.test(node.class_type || '') && Object.prototype.hasOwnProperty.call(node.inputs || {}, 'image')) {
        found.add(nodeId)
        continue
      }
      for (const input of Object.values(node.inputs || {})) {
        if (Array.isArray(input) && input.length === 2 && workflow[String(input[0])]) pending.push(String(input[0]))
      }
    }
    return found
  }
  for (const node of Object.values(workflow || {})) {
    for (const [name, value] of Object.entries(node.inputs || {})) {
      const role = h3FrameInputRole(name)
      const reference = byRole.get(role)
      if (!role || !reference) continue
      const loaderIds = findLoaders(value)
      for (const loaderId of loaderIds) {
        workflow[loaderId].inputs.image = reference.comfyName
        boundLoaders.add(loaderId)
      }
      if (loaderIds.size) boundRoles.add(role)
    }
  }
  return { boundLoaders, boundRoles }
}

function appendForcedTailFramePrompt(prompt, tailReference, duration) {
  const name = String(tailReference?.name || '尾帧图片').replace(/[\r\n@（）()]/g, ' ').trim().slice(0, 120) || '尾帧图片'
  const tag = `@尾帧（${name}）`
  if (String(prompt || '').includes(tag)) return prompt
  return `${String(prompt || '').trim()}\n\n首尾帧强制约束：视频必须从首帧图片精确开始，并在 ${duration} 秒结束时精确收束到 ${tag}。最后 0.5 秒必须逐步匹配尾帧的人物身份、姿态、构图、机位、景别、光线和背景空间；禁止忽略尾帧、提前切镜、溶解、变脸、变装或生成不相关结尾。`
}

function upstreamNodeIds(targetId, edges) {
  const ids = new Set()
  const queue = [targetId]
  while (queue.length) {
    const current = queue.shift()
    for (const edge of edges.filter((item) => item.target === current)) {
      if (ids.has(edge.source)) continue
      ids.add(edge.source)
      queue.push(edge.source)
    }
  }
  return ids
}

function semanticNodeTitle(data, fallback) {
  if (data.nodeLabel) return data.nodeLabel
  const sceneSuffix = data.storyboardScene ? ` · ${data.storyboardScene}` : ''
  if (data.storyboardStage === 'character') return `人物图片生成 · ${data.storyboardCharacterName || String(data.text || '').replace(/^人物(?:信息|设计)\s*·\s*/, '')}`
  if (data.storyboardStage === 'scene') return `场景图片生成 · ${data.storyboardSceneName || String(data.text || '').replace(/^场景(?:信息|设计)\s*·\s*/, '')}`
  if (data.storyboardStage === 'reference') return `${data.storyboardTitle || '分镜'} · 分镜图片生成${sceneSuffix}`
  if (data.storyboardStage === 'video') return `${data.storyboardTitle || '镜头'} · 视频生成${sceneSuffix}`
  if (data.kind === 'script' && data.storyboardGroup) return '剧本智能分镜'
  return fallback
}
const creativeStyles = ['无风格', '电影写实', '日系动画', '国风水墨', '柔光插画', '赛博霓虹', '复古胶片', '3D 卡通', '像素艺术', '黏土定格', '黑白漫画', '儿童绘本', '油画厚涂', '低饱和平涂', '未来科幻', '温馨治愈']
const skillTemplates = [
  { title: '分镜师', icon: '▤', template: 'video', prompt: '人物设计、场景设计、逐镜分镜图与视频连续生成' },
  { title: '角色设计', icon: '♙', template: 'image', prompt: '设计一个具有清晰轮廓、服装细节和辨识度的角色设定' },
  { title: '场景设计', icon: '▧', template: 'image', prompt: '设计一个具有前中后景层次、明确光线与氛围的场景' },
  { title: '图片转视频', icon: '▶', template: 'video', prompt: '保持主体一致，镜头缓慢推进，添加自然的环境动态' },
  { title: '商品展示', icon: '◇', template: 'image', prompt: '干净高级的商品展示，棚拍光线，突出材质与核心卖点' },
  { title: '风格探索', icon: '✦', template: 'image', prompt: '生成多种视觉风格方向用于对比，保持主体内容一致' }
]

function taskDimensions(resolution = '480P', ratio = '16:9') {
  const longSide = { '480P': 854, '720P': 1280, '1080P': 1920, '1K': 1024, '2K': 2048 }[resolution] || 1024
  const [wide, high] = ratio.split(':').map(Number)
  if (!wide || !high || wide === high) return { width: longSide, height: longSide }
  const aligned = (value) => Math.max(64, Math.round(value / 8) * 8)
  return wide > high ? { width: aligned(longSide), height: aligned(longSide * high / wide) } : { width: aligned(longSide * wide / high), height: aligned(longSide) }
}

function alignedSecondSampleSize(width, height, targetMegapixels) {
  const sourceWidth = Number(width)
  const sourceHeight = Number(height)
  const megapixels = Number(targetMegapixels)
  if (!(sourceWidth > 0) || !(sourceHeight > 0) || !(megapixels > 0)) return null
  const scale = Math.sqrt((megapixels * 1_000_000) / (sourceWidth * sourceHeight))
  const align32 = (value) => Math.max(32, Math.round(value / 32) * 32)
  return { width: align32(sourceWidth * scale), height: align32(sourceHeight * scale) }
}

function synchronizeDirectorSecondSamplePayload(workflowNode) {
  if (!/H3DirectorStudio/i.test(String(workflowNode?.class_type || ''))) return 0
  const width = Number(workflowNode.inputs.width)
  const height = Number(workflowNode.inputs.height)
  let repairCount = 0
  for (const [inputName, rawSegments] of Object.entries(workflowNode.inputs || {})) {
    if (!/segments_json$/i.test(inputName) || typeof rawSegments !== 'string' || !rawSegments.trim()) continue
    let segments
    try {
      segments = JSON.parse(rawSegments)
    } catch {
      continue
    }
    if (!Array.isArray(segments)) continue
    let inputRepairCount = 0
    for (const segment of segments) {
      const secondSample = segment?.second_sample
      if (!secondSample || secondSample.mode === 'off' || secondSample.target_size_mode !== 'megapixels') continue
      const size = alignedSecondSampleSize(width, height, secondSample.target_megapixels)
      if (!size) continue
      if (Number(secondSample.final_width) !== size.width || Number(secondSample.final_height) !== size.height) {
        secondSample.final_width = size.width
        secondSample.final_height = size.height
        inputRepairCount += 1
      }
    }
    if (inputRepairCount) workflowNode.inputs[inputName] = JSON.stringify(segments)
    repairCount += inputRepairCount
  }
  return repairCount
}

function pickDefaultWorkflowModel(workflow, catalog) {
  const preferredInputs = ['ckpt_name', 'unet_name', 'diffusion_model', 'model_name']
  for (const preferred of preferredInputs) {
    const optionEntry = Object.entries(catalog.optionsByInput || {}).find(([name]) => name.toLowerCase() === preferred)
    const options = optionEntry?.[1] || []
    if (!options.length) continue
    const current = Object.values(workflow || {}).map((node) => node.inputs?.[preferred]).find((value) => typeof value === 'string')
    const name = current && options.includes(current) ? current : options[0]
    const catalogModel = (catalog.models || []).find((model) => model.input.toLowerCase() === preferred && model.name === name)
    return { name, input: optionEntry[0], type: catalogModel?.type || (preferred === 'ckpt_name' || preferred === 'model_name' ? '大模型' : '扩散模型'), keptWorkflowDefault: current === name }
  }
  return null
}

function splitScriptIntoShots(rawScript, maxShots = 12) {
  const cleaned = String(rawScript || '').replace(/\r/g, '').trim()
  if (!cleaned) return []
  const lines = cleaned.split(/\n+/).map((line) => line.trim()).filter(Boolean)
  const tableLines = lines.filter((line) => /^\|.+\|$/.test(line) && !/^\|[\s:|-]+\|$/.test(line))
  const toCells = (row) => row.split('|').slice(1, -1).map((cell) => cell.trim())
  const headerIndex = tableLines.findIndex((line) => /镜头|镜号|序号/.test(line) && /画面|内容|动作/.test(line))
  const tableHeader = headerIndex >= 0 ? toCells(tableLines[headerIndex]) : []
  const tableRows = headerIndex >= 0 ? tableLines.slice(headerIndex + 1) : []
  const normalizedTableRows = tableRows.map((row) => {
    const cells = toCells(row)
    return cells.map((cell, index) => `${tableHeader[index] || `字段${index + 1}`}：${cell}`).join('；')
  })
  const shotNumber = '[一二三四五六七八九十百零〇两0-9]+'
  const shotTokenSource = `(?:(?:第\\s*)?${shotNumber}\\s*(?:号\\s*)?(?:镜头|镜)|(?:镜头|分镜)\\s*(?:第\\s*)?${shotNumber}|shot\\s*(?:no\\.?\\s*)?\\d+)`
  const shotMarkers = [...cleaned.matchAll(new RegExp(shotTokenSource, 'gi'))]
  const explicitBlocks = shotMarkers.map((marker, index) => cleaned.slice(marker.index, shotMarkers[index + 1]?.index ?? cleaned.length).replace(/[\s#*_【】\[\]]+$/, '').trim())
  let parts = normalizedTableRows.length ? normalizedTableRows : explicitBlocks.length ? explicitBlocks : lines
  if (!normalizedTableRows.length && !explicitBlocks.length && parts.length < 2) parts = cleaned.split(/(?<=[。！？!?；;])\s*/).map((line) => line.trim()).filter(Boolean)
  const shots = []
  const cameraMoves = ['缓慢推进，稳定中景转近景', '横向跟拍，保持人物主体居中', '低机位轻微仰拍，增强戏剧张力', '过肩镜头，聚焦人物表情', '广角建立场景后缓慢环绕', '手持感跟随，动作自然流畅']
  for (const part of parts) {
    const text = part.replace(new RegExp(`^${shotTokenSource}`, 'i'), '').replace(/^[\s#*_【】\[\]（）()：:、.\-—]+/, '').trim()
    if (!text) continue
    const segments = explicitBlocks.length || normalizedTableRows.length ? [text] : text.length > 110 ? text.split(/(?<=[。！？!?；;])/).filter(Boolean) : [text]
    for (const segment of segments) {
      if (shots.length >= maxShots) break
      const index = shots.length
      const labeledValue = (labels) => new RegExp(`(?:^|[\\n；;|])\\s*(?:${labels})\\s*[：:]\\s*([^\\n；;|]+)`, 'i').exec(segment)?.[1]?.trim() || ''
      const dialogue = labeledValue('对白|台词') || /[“\"「『](.+?)[”\"」』]/.exec(segment)?.[1] || ''
      const sound = labeledValue('音效|声音|环境声')
      const scene = labeledValue('场景|地点')
      const people = labeledValue('人物|角色|出镜人物')
      const visual = labeledValue('画面|画面内容|内容|动作|画面动作') || segment.replace(/(?:^|[\n；;|])\s*(?:镜头|镜号|序号|场景|地点|人物|角色|出镜人物|景别|运镜|镜头运动|对白|台词|音效|声音|环境声|时长)\s*[：:][^\n；;|]*/gi, '').replace(/^[；;|\s]+|[；;|\s]+$/g, '') || segment
      const camera = [labeledValue('景别|镜头景别'), labeledValue('运镜|镜头运动|机位')].filter(Boolean).join('，') || cameraMoves[index % cameraMoves.length]
      const readableLength = segment.replace(/[，。！？、；：\s]/g, '').length
      const statedDuration = Number(/(?:时长\s*[：:]?\s*)?(\d+(?:\.\d+)?)\s*(?:秒|s\b)/i.exec(segment)?.[1])
      const duration = statedDuration || Math.max(3, Math.min(15, Math.round(3 + readableLength / 13 + (dialogue ? 2 : 0))))
      shots.push({
        number: index + 1,
        title: `镜头 ${String(index + 1).padStart(2, '0')}`,
        source: segment.trim(),
        visual,
        camera,
        dialogue,
        sound,
        sceneHint: scene,
        characterHint: people,
        duration,
      })
    }
    if (shots.length >= maxShots) break
  }
  return shots
}

function analyzeStoryboardAssets(rawScript, shots) {
  const script = String(rawScript || '')
  const characterNames = new Set()
  const ignoredNames = new Set(['镜头', '场景', '画面', '内容', '动作', '旁白', '人物', '角色', '景别', '运镜', '机位', '音效', '声音', '对白', '台词', '时长', '时间', '地点', '室内', '室外', '白天', '夜晚'])
  const addCharacter = (value) => {
    const name = String(value || '').replace(/^(角色|人物|主角|配角)[：:\s]*/, '').trim()
    if (/^[\u4e00-\u9fa5]{2,4}$/.test(name) && !ignoredNames.has(name)) characterNames.add(name)
  }
  for (const match of script.matchAll(/(?:角色|人物|主角|配角)[：:\s]*([\u4e00-\u9fa5]{2,4})/g)) addCharacter(match[1])
  for (const match of script.matchAll(/(?:^|[\n。！？])\s*([\u4e00-\u9fa5]{2,4})[：:]/g)) addCharacter(match[1])
  for (const match of script.matchAll(/([\u4e00-\u9fa5]{2,4}?)(?:撑伞|奔跑|跑向|走向|走进|走出|转身|回头|开口|说道|说|问道|问|回答|答|看见|望向|抬头|低头|停下|坐下|站起|推开|拿起|笑了|哭了)/g)) addCharacter(match[1])
  if (!characterNames.size) characterNames.add('主要人物')

  const sceneNames = new Set()
  const addScene = (value) => {
    const name = String(value || '').replace(/^(场景|地点)[：:\s]*/, '').replace(/[，。！？；：].*$/, '').trim()
    if (name.length >= 2 && name.length <= 14 && !/^(人物|镜头|画面)$/.test(name)) sceneNames.add(name)
  }
  for (const match of script.matchAll(/(?:场景|地点)[：:\s]*([^\n，。！？；]{2,14})/g)) addScene(match[1])
  for (const match of script.matchAll(/(?:在|来到|走进|进入|回到|前往|跑向|走向)([\u4e00-\u9fa5]{0,6}(?:车站|房间|卧室|客厅|厨房|教室|办公室|医院|公园|机场|街道|小巷|广场|站台|天台|庭院|校园|商店|餐厅|酒吧|咖啡馆|森林|山谷|海边|河岸|桥上|路口))/gm)) addScene(match[1])
  if (!sceneNames.size) sceneNames.add('主要场景')

  const sourceLines = script.split(/\n+|(?<=[。！？!?])/).map((line) => line.trim()).filter(Boolean)
  const summarizeMatches = (name, fallback, definitionPattern) => {
    const matches = sourceLines.filter((line) => line.includes(name) && !/^(?:第?\s*[一二三四五六七八九十百\d]+\s*(?:镜|镜头)|shot\s*\d+)\s*$/i.test(line))
    const definitions = matches.filter((line) => definitionPattern.test(line))
    return [...new Set(definitions.length ? definitions : matches)].slice(0, definitions.length ? 2 : 4).join('；').slice(0, 520) || fallback
  }
  const characters = [...characterNames].slice(0, 30).map((name, index) => ({ id: `character-${index + 1}`, name, description: summarizeMatches(name, `${name}，根据剧本建立固定的年龄、面部、发型、服装、体型与配色设定。`, /^(?:角色|人物|主角|配角)\s*[：:]/) }))
  const scenes = [...sceneNames].slice(0, 30).map((name, index) => ({ id: `scene-${index + 1}`, name, description: summarizeMatches(name, `${name}，根据剧本建立固定的时代、空间结构、陈设、天气、时间与光线。`, /^(?:场景|地点)\s*[：:]/) }))
  let activeScene = scenes[0]
  const shotScenes = shots.map((shot) => {
    const matched = scenes.find((scene) => shot.sceneHint?.includes(scene.name) || (shot.sceneHint && scene.name.includes(shot.sceneHint)) || shot.source.includes(scene.name))
    if (matched) activeScene = matched
    return activeScene
  })
  const shotCharacters = shots.map((shot) => {
    const selected = characters.filter((character) => shot.source.includes(character.name) || shot.characterHint?.includes(character.name))
    return selected.length ? selected : characters.length === 1 ? characters : []
  })
  return { characters, scenes, shotScenes, shotCharacters }
}

function buildStoryboardShotPrompt(shot, scene, characters) {
  const characterInfo = characters?.length ? characters.map((character) => `${character.name}（${character.description}）`).join('；') : '本镜头无明确出镜人物，保持前后镜头角色连续性'
  return `integrated_multimodal_description: [Shot ${shot.number}]\n场景信息：${scene?.name || shot.sceneHint || '主要场景'}；${scene?.description || '延续前一镜头的时间、空间与光线。'}\n人物信息：${characterInfo}\n镜头画面：${shot.visual}\n景别与运镜：${shot.camera}\n连续性要求：只沿用本镜头所需人物和场景设定，身份、面部、发型、服装、道具、空间方位与前后镜头一致；自然连续动作，电影级构图，画面稳定，无文字、字幕、标志或水印。\noverall_soundscape: ${shot.dialogue ? `对白：“${shot.dialogue}”；` : ''}${shot.sound ? `音效：${shot.sound}；` : ''}补充与当前场景和动作匹配的环境声，空间层次清晰。\nnon_diegetic_music: 连贯的电影配乐，音量适中，不遮盖对白与环境声。`
}

function miniMaxWorkflowScore(item) {
  const text = `${item.name || ''} ${item.localPath || ''}`.toLowerCase()
  if (!/minimax|mini.?max|h3/.test(text)) return -1
  let score = 0
  if (/h3_manju.*01_|快速生产.*fl2va.*8步/.test(text)) score += 100
  if (/文生视频|t2v/.test(text)) score += 70
  if (/fl2va|8步|加速/.test(text)) score += 30
  if (/图生视频|i2v|首尾帧|参考|r2v/.test(text)) score -= 25
  return score
}

function imageWorkflowScore(item) {
  const text = `${item.name || ''} ${item.localPath || ''}`.toLowerCase()
  if (/video|视频|minimax|h3|i2v|t2v/.test(text)) return -1
  let score = 0
  if (/(^|[\\/])文生图(\.json)?$/.test(text)) score += 140
  if (/文生图|text.?to.?image|txt2img/.test(text)) score += 90
  if (/zimage.*turbo|z-image.*turbo/.test(text)) score += 70
  if (/高清|upscale/.test(text)) score += 15
  if (/图生图|img2img/.test(text)) score -= 25
  return score
}

function workflowMediaKind(item) {
  const classes = item.summary?.classes || Object.values(item.workflow || {}).map((node) => node.class_type || '')
  const text = `${item.name || ''} ${item.localPath || ''} ${classes.join(' ')}`.toLowerCase()
  if (/minimax|\bh3\b|video|videocombine|animatediff|wan.*(i2v|t2v|video)|i2v|t2v|fl2v|图生视频|文生视频|视频|首尾帧|动态|动画|导演台|多分镜|\d+秒/.test(text)) return 'video'
  if (/saveimage|previewimage|文生图|图生图|生图|txt2img|img2img|t2i|z.?image|flux|sdxl|stable.?diffusion|image|图片|图像|人物.*设计|角色.*设计|放大|根据图形|多图/.test(text)) return 'image'
  return 'unknown'
}

function linkedPromptNodeIds(workflow, inputPattern) {
  const ids = new Set()
  for (const node of Object.values(workflow || {})) {
    for (const [key, value] of Object.entries(node?.inputs || {})) {
      if (inputPattern.test(key) && Array.isArray(value) && value[0] != null) ids.add(String(value[0]))
    }
  }
  return ids
}

function injectPositivePrompt(workflow, prompt) {
  const text = String(prompt || '').trim()
  if (!text) return workflow
  const positiveNodeIds = linkedPromptNodeIds(workflow, /^(positive|positive_prompt|prompt_positive)$/i)
  const negativeNodeIds = linkedPromptNodeIds(workflow, /^(negative|negative_prompt|prompt_negative)$/i)
  for (const [nodeId, node] of Object.entries(workflow || {})) {
    const descriptor = `${node?.class_type || ''} ${node?._meta?.title || ''}`
    const looksNegative = negativeNodeIds.has(String(nodeId)) || /negative|负向|反向/i.test(descriptor)
    for (const [key, value] of Object.entries(node?.inputs || {})) {
      if (Array.isArray(value) || !/^(text|prompt|positive|positive_prompt)$/i.test(key) || looksNegative) continue
      const isPromptTextNode = positiveNodeIds.has(String(nodeId)) || /text.*encode|prompt|caption|conditioning|文本|提示词/i.test(descriptor)
      if (isPromptTextNode) node.inputs[key] = text
    }
  }
  return workflow
}

function mediaIsVideo(media) {
  const descriptor = `${media?.filename || ''} ${media?.url || ''}`
  return media?.kind === 'videos' || /\.(mp4|webm|mov|mkv|avi)(?:[?&#\s]|$)/i.test(descriptor)
}

function mediaIsAudio(media) {
  const descriptor = `${media?.filename || ''} ${media?.url || ''}`
  return media?.kind === 'audio' || /\.(mp3|wav|flac|ogg|m4a|aac)(?:[?&#\s]|$)/i.test(descriptor)
}

function nodeContentMedia(node) {
  const data = node?.data || {}
  const candidates = [...(data.results || []), data.resultUrl ? { url: data.resultUrl, filename: data.name || '生成结果' } : null, data.url ? { url: data.url, filename: data.name || '任务素材' } : null, data.taskReferenceUrl ? { url: data.taskReferenceUrl, filename: data.taskReferenceName || '参考图片' } : null].filter((item) => item?.url)
  const seen = new Set()
  return candidates.filter((item) => {
    if (seen.has(item.url)) return false
    seen.add(item.url)
    return mediaIsVideo(item) || mediaUrlLooksImage(item.url) || mediaIsAudio(item)
  }).map((item) => ({ ...item, kind: mediaIsVideo(item) ? 'videos' : mediaIsAudio(item) ? 'audio' : 'images' }))
}

function nodeContentDescriptor(node) {
  const media = nodeContentMedia(node)
  const preferred = media.find(mediaIsVideo) || media.find((item) => !mediaIsAudio(item)) || media[0]
  if (preferred) return { kind: mediaIsVideo(preferred) ? 'video' : mediaIsAudio(preferred) ? 'audio' : 'image', value: preferred.url, media, index: Math.max(0, media.indexOf(preferred)) }
  const text = String(node?.data?.taskPrompt || node?.data?.text || node?.data?.name || '').trim()
  return text ? { kind: 'text', value: text, media: [] } : { kind: 'empty', value: '', media: [] }
}

function TaskMediaThumb({ media, alt = '生成结果' }) {
  if (mediaIsVideo(media)) return <video src={media.url} muted playsInline preload="metadata" />
  if (mediaIsAudio(media)) return <span className="task-audio-thumb"><b>♫</b><small>音频</small></span>
  return <img src={media.url} alt={alt} />
}

function ImeTextarea({ value = '', onChange, onKeyDown, ...props }) {
  const inputRef = useRef(null)
  const composingRef = useRef(false)

  useEffect(() => {
    const input = inputRef.current
    if (!input || composingRef.current || document.activeElement === input) return
    const next = String(value ?? '')
    if (input.value !== next) input.value = next
  }, [value])

  const stopInputEvent = (event) => event.stopPropagation()
  const commit = (event) => onChange?.(event.target.value, event)

  return <textarea
    {...props}
    ref={inputRef}
    defaultValue={String(value ?? '')}
    onMouseDown={stopInputEvent}
    onPointerDown={stopInputEvent}
    onKeyDown={(event) => { event.stopPropagation(); onKeyDown?.(event) }}
    onKeyUp={stopInputEvent}
    onBeforeInput={stopInputEvent}
    onCompositionStart={(event) => { composingRef.current = true; event.stopPropagation() }}
    onCompositionUpdate={stopInputEvent}
    onCompositionEnd={(event) => { composingRef.current = false; event.stopPropagation(); commit(event) }}
    onInput={(event) => { event.stopPropagation(); if (!composingRef.current && !event.nativeEvent?.isComposing) commit(event) }}
    onBlur={(event) => { composingRef.current = false; commit(event); props.onBlur?.(event) }}
  />
}

const templates = {
  blank: { name: '空白画布', nodes: [], edges: [] },
  image: {
    name: '快速生图',
    nodes: [
      { id: 'prompt-1', type: 'workflow', position: { x: 130, y: 180 }, data: { kind: 'prompt', text: '一只在霓虹城市散步的橘猫' } },
      { id: 'image-1', type: 'workflow', position: { x: 460, y: 180 }, data: { kind: 'image' } },
      { id: 'output-1', type: 'workflow', position: { x: 790, y: 180 }, data: { kind: 'output' } }
    ],
    edges: [
      { id: 'e1', source: 'prompt-1', target: 'image-1' },
      { id: 'e2', source: 'image-1', target: 'output-1' }
    ]
  },
  video: {
    name: '图片转视频',
    nodes: [
      { id: 'prompt-1', type: 'workflow', position: { x: 90, y: 130 }, data: { kind: 'prompt', text: '镜头缓慢推进，柔和自然光' } },
      { id: 'image-1', type: 'workflow', position: { x: 390, y: 280 }, data: { kind: 'image' } },
      { id: 'video-1', type: 'workflow', position: { x: 690, y: 180 }, data: { kind: 'video' } },
      { id: 'output-1', type: 'workflow', position: { x: 990, y: 180 }, data: { kind: 'output' } }
    ],
    edges: [
      { id: 'e1', source: 'prompt-1', target: 'video-1' },
      { id: 'e2', source: 'image-1', target: 'video-1' },
      { id: 'e3', source: 'video-1', target: 'output-1' }
    ]
  }
}

function WorkflowNode({ id, data, selected }) {
  const item = nodeCatalog[data.kind] || nodeCatalog.prompt
  const nodeTitle = semanticNodeTitle(data, item.title)
  const isMedia = mediaKinds.has(data.kind)
  const isFrame = data.kind === 'frame'
  const isFillTask = fillTaskKinds.has(data.kind)
  const mediaUploadRef = useRef(null)
  const taskModelOptions = data.kind === 'video'
    ? ['本地 ComfyUI 视频', '图生视频工作流', '首尾帧视频工作流']
    : ['F2K-9B-文生+图生', '本地 ComfyUI 生图', '自定义生图工作流']
  const workflowInputs = data.taskWorkflowInputs || []
  const taskModels = (data.availableModels || []).filter((model) => ['大模型', '扩散模型'].includes(model.type) && (!workflowInputs.length || workflowInputs.some((input) => input.toLowerCase() === model.input.toLowerCase())))
  const visibleTaskModels = workflowInputs.length ? taskModels : (taskModels.length ? taskModels : (data.availableModels || []))
  const desiredWorkflowKind = data.kind === 'video' ? 'video' : 'image'
  const compatibleWorkflows = (data.availableWorkflows || []).filter((workflow) => workflow.mediaKind === desiredWorkflowKind)
  const modelOptionsFor = (key) => Object.entries(data.comfyModelOptions || {}).find(([name]) => name.toLowerCase() === key.toLowerCase())?.[1] || []
  const taskReferences = data.taskReferences?.length ? data.taskReferences : data.taskReferenceUrl ? [{ url: data.taskReferenceUrl, name: data.taskReferenceName || '参考图片' }] : []
  const isSeedance = data.taskProvider === 'api' || data.taskProvider === 'seedance'
  const compatibleApiModels = (data.apiModels || []).filter((entry) => entry.kind === desiredWorkflowKind)
  const selectedSeedanceModel = data.taskApiModel || data.taskSeedanceModel || data.activeSeedanceModel || ''
  const selectedApiEntry = compatibleApiModels.find((entry) => entry.model === selectedSeedanceModel && (!data.taskApiConnectionId || entry.connectionId === data.taskApiConnectionId))
  const selectedSeedanceLabel = selectedApiEntry?.label || data.seedanceModelLabels?.[selectedSeedanceModel] || selectedSeedanceModel || '尚未选择 API 模型'
  const previewMedia = data.results?.find((result) => result?.url) || null
  const previewReference = taskReferences.find((reference) => referenceDisplayUrl(reference))
  const previewUrl = data.resultUrl || previewMedia?.url || data.url || referenceDisplayUrl(previewReference) || null
  const previewName = data.resultUrl ? '生成结果' : data.name || taskReferences[0]?.name || nodeTitle
  const previewIsVideo = mediaIsVideo(previewMedia) || data.uploadedMediaType === 'video' || mediaUrlLooksVideo(previewUrl)
  const acceptTaskReferences = (files) => {
    const images = [...(files || [])].filter((file) => file?.type?.startsWith('image/'))
    if (images.length) data.onTaskReferenceFiles?.(id, images)
  }
  const openMediaPicker = (event) => {
    event?.preventDefault?.()
    event?.stopPropagation?.()
    const input = mediaUploadRef.current
    if (!input) return
    input.value = ''
    try {
      if (typeof input.showPicker === 'function') input.showPicker()
      else input.click()
    } catch {
      input.click()
    }
  }
  return (
    <div className={`workflow-node ${item.tone} ${isFillTask ? 'fill-task' : ''} ${data.kind === 'prompt' ? 'prompt-task' : ''} ${selected ? 'selected' : ''}`} onContextMenu={(event) => data.onElementContextMenu?.(event, 'node', { id, data })}>
      <NodeResizer isVisible={selected} minWidth={220} minHeight={data.kind === 'comfy' ? 260 : data.kind === 'script' ? 300 : data.kind === 'prompt' ? 150 : 190} lineClassName={data.kind === 'prompt' ? 'prompt-resize-line' : 'task-resize-line'} handleClassName={data.kind === 'prompt' ? 'prompt-resize-handle' : 'task-resize-handle'} />
      {isFillTask && data.kind !== 'prompt' && <div className="node-drag-grip" title="按住鼠标左键拖动任务框"><i /><span>拖动</span></div>}
      {data.kind === 'prompt' && <div className="prompt-border-drag-zones" aria-label="按住任务框边框拖动"><i className="top" /><i className="right" /><i className="bottom" /><i className="left" /></div>}
      {selected && isMedia && <div className="node-quickbar">
        {['智能编辑', '联动编辑', '复制节点', '设为首帧', '设为尾帧'].map((action) => <button key={action} title={action} onMouseDown={(event) => event.stopPropagation()} onClick={() => data.onQuickAction?.(id, action)}>{action}</button>)}
      </div>}
      <Handle type="target" position={Position.Left} />
      <div className="node-head">
        <button className="node-head-avatar nodrag" title={isMedia ? '选择任务框并显示参数面板' : '任务类型'}><span>{item.icon}</span></button>
        <input className="node-title-input nodrag" aria-label="任务框标题" title="点击修改任务框名称" value={data.nodeLabel ?? nodeTitle} onMouseDown={(event) => event.stopPropagation()} onChange={(event) => data.onTaskConfig?.(id, 'nodeLabel', event.target.value)} onBlur={() => { if (!String(data.nodeLabel ?? nodeTitle).trim()) data.onTaskConfig?.(id, 'nodeLabel', item.title) }} />
        {data.model && <em>{data.model}</em>}
      </div>
      {isMedia && <div className={`media-preview uploadable ${data.kind}`} title={data.kind === 'video' ? '双击上传视频或参考图片' : '双击上传或更换图片'} onDoubleClickCapture={openMediaPicker}>
        {previewUrl ? (previewIsVideo ? <video src={previewUrl} controls muted playsInline preload={selected || data.taskEditorOpen ? 'metadata' : 'none'} /> : <img src={previewUrl} alt={previewName} loading="lazy" decoding="async" />) : <><span>{item.icon}</span><small>{data.kind === 'video' ? '视频预览' : data.kind === 'frame' ? '首帧 · 尾帧' : '素材预览'}</small></>}
        {data.kind === 'video' && !previewIsVideo && <i>▶</i>}
        {previewUrl && <b className="media-preview-label">{data.resultUrl ? '生成结果' : previewIsVideo ? '素材视频' : '素材图片'}</b>}
        <button className="media-upload-action nodrag" type="button" onDoubleClick={(event) => event.stopPropagation()} onClick={openMediaPicker}>{previewUrl ? '更换素材' : '上传素材'}</button>
        <em className="media-upload-hint">{data.kind === 'video' ? '双击任务框上传视频 / 图片' : '双击任务框上传图片'}</em>
        <input ref={mediaUploadRef} className="media-upload-input nodrag" type="file" accept={data.kind === 'video' ? 'video/mp4,video/webm,video/quicktime,image/*' : 'image/*'} onClick={(event) => event.stopPropagation()} onChange={(event) => { const file = event.target.files?.[0]; if (file) data.onMediaUpload?.(id, file); event.target.value = '' }} />
      </div>}
      {data.kind === 'comfy' && <div className="comfy-node-content">
        <div className="comfy-node-meta"><span>{data.nodeCount || 0} 个 ComfyUI 节点</span><em className={data.missingCount ? 'warn' : ''}>{data.missingCount ? `缺少 ${data.missingCount} 项` : '依赖待检测'}</em></div>
        {(data.params || []).slice(0, 4).map((param) => <label className="comfy-param nodrag" key={param.id} onMouseDown={(event) => event.stopPropagation()}>
          <span>{comfyParamLabels[param.key] || param.label}</span>
          {param.key === 'image' ? <button className="comfy-image-picker" onClick={() => data.onUploadComfyImage?.(id, param.id)}>{param.value || '选择图片'}</button> : /^(video|file)$/i.test(param.key) ? <button className="comfy-image-picker comfy-video-picker" onClick={() => data.onUploadComfyVideo?.(id, param.id)}>{param.value || '选择或上传视频'}</button> : modelParamPattern.test(param.key) && modelOptionsFor(param.key).length ? <select value={param.value} onChange={(event) => data.onComfyParam?.(id, param.id, event.target.value)}>{!modelOptionsFor(param.key).includes(param.value) && <option>{param.value}</option>}{modelOptionsFor(param.key).map((model) => <option key={model}>{model}</option>)}</select> : <input type={param.valueType === 'number' ? 'number' : 'text'} value={param.value} onChange={(event) => data.onComfyParam?.(id, param.id, event.target.value)} />}
        </label>)}
        {(data.params || []).length > 4 && <small className="comfy-more">其余 {(data.params || []).length - 4} 个参数沿用工作流原值</small>}
        {data.resultUrl && <div className="comfy-result">{data.results?.[0]?.kind === 'videos' ? <video src={data.resultUrl} controls muted playsInline /> : <img src={data.resultUrl} alt="ComfyUI 输出" />}</div>}
        <div className="comfy-actions nodrag" onMouseDown={(event) => event.stopPropagation()}>
          <button onClick={() => data.onInspectComfy?.(id)}>检查依赖</button>
          {data.status === '生成中' || data.status === '排队中' ? <button className="danger" onClick={() => data.onStopComfy?.(id)}>停止</button> : <button className="run" onClick={() => data.onRunComfy?.(id)}>运行工作流</button>}
        </div>
        {data.progress > 0 && data.progress < 100 && <i className="comfy-progress"><b style={{ width: `${data.progress}%` }} /></i>}
      </div>}
      {data.kind === 'script'
        ? <div className="story-script nodrag nowheel" onMouseDown={(event) => event.stopPropagation()} onDoubleClick={(event) => event.stopPropagation()}>
          <textarea value={data.text || ''} placeholder="在这里粘贴剧本…" onChange={(event) => data.onTaskConfig?.(id, 'text', event.target.value)} />
          <div className="story-runtime"><span>分镜师 · 生图 + MiniMax H3</span><small>{data.storyboardCharacterCount || 1} 个人物 · {data.storyboardSceneCount || 1} 个场景 · {data.storyboardCount || 0} 个镜头</small></div>
          <div className="story-actions"><button onClick={() => data.onStoryboardEdit?.(id)}>重新分镜</button><button className="run" onClick={() => data.onStoryboardRunAll?.(data.storyboardGroup)}>▶ 全部生成</button></div>
        </div>
        : data.kind === 'prompt'
        ? <div className="node-body prompt-body nodrag nowheel" onMouseDown={(event) => event.stopPropagation()} onDoubleClick={(event) => event.stopPropagation()}>
          <ImeTextarea aria-label="编辑提示词" value={data.text ?? ''} placeholder="点击这里编辑提示词…" onChange={(value) => data.onPromptText?.(id, value)} />
        </div>
        : isMedia ? null : <div className="node-body">{data.text || item.body}</div>}
      {data.storyboardStage === 'video' && <button className="shot-run-button nodrag" disabled={['生成中', '排队中'].includes(data.status)} onMouseDown={(event) => event.stopPropagation()} onClick={(event) => { event.stopPropagation(); data.onRunTask?.(id, { autoRunSequence: false }) }}>▶ 单镜头生成 · {data.taskDuration || 6}秒</button>}
      {isMedia
        ? <button type="button" className={`node-status node-task-toggle nodrag ${data.taskEditorOpen ? 'open' : ''}`} title={data.taskEditorOpen ? '点击收起任务参数' : '点击展开任务参数'} onMouseDown={(event) => event.stopPropagation()} onClick={(event) => { event.stopPropagation(); data.onTaskConfig?.(id, 'taskEditorOpen', !data.taskEditorOpen) }}><i className={data.status === '生成中' ? 'working' : ''} /><span>{data.status || '就绪'}</span>{data.linkGroup && <em>联动</em>}{data.taskStyle && data.taskStyle !== '无风格' && <em>{data.taskStyle}</em>}<b>{data.taskEditorOpen ? '⌃' : '⌄'}</b></button>
        : data.kind === 'prompt' ? null : <div className={`node-status ${isFillTask ? 'node-drag-footer' : ''}`} title={isFillTask ? '按住鼠标左键拖动任务框' : undefined}><i className={data.status === '生成中' ? 'working' : ''} /><span>{data.status || '就绪'}</span>{data.linkGroup && <em>联动</em>}{data.taskStyle && data.taskStyle !== '无风格' && <em>{data.taskStyle}</em>}</div>}
      {data.taskEditorOpen && isMedia && <div className={`task-editor nodrag nowheel ${data.taskEditorExpanded ? 'expanded' : ''}`} onMouseDown={(event) => event.stopPropagation()} onDoubleClick={(event) => event.stopPropagation()} onKeyDown={(event) => event.stopPropagation()}>
        <div className="task-editor-head"><span>{isFrame ? '首尾帧任务 · 自动识别视频画面' : data.kind === 'video' ? '视频任务 · 自动匹配视频工作流' : '图片任务 · 自动匹配生图工作流'} · 拖动右下角调整大小</span><div><button title={data.taskEditorExpanded ? '恢复大小' : '放大面板'} onClick={() => data.onTaskConfig?.(id, 'taskEditorExpanded', !data.taskEditorExpanded)}>{data.taskEditorExpanded ? '↙' : '⛶'}</button><button title="收起参数面板" onClick={() => data.onTaskConfig?.(id, 'taskEditorOpen', false)}>×</button></div></div>
        {isFrame ? <div className="frame-task-editor">
          <div className="frame-task-controls"><label><span>识别位置</span><select value={data.frameRole || '尾帧'} onChange={(event) => data.onFrameExtract?.(id, event.target.value)}><option>首帧</option><option>尾帧</option></select></label><button onClick={() => data.onFrameExtract?.(id, data.frameRole || '尾帧')}>↻ 重新识别视频{data.frameRole || '尾帧'}</button></div>
          <div className="frame-task-note"><b>{data.frameRole || '尾帧'}专用任务</b><span>连接视频后自动截取{data.frameRole === '首帧' ? '开头' : '结束前'}画面；此任务不会调用生图模型。</span></div>
        </div> : <>
        <div className="task-editor-compose">
          <div className="task-reference-grid" title="最多添加 9 张人物、服装或场景参考图片" onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); acceptTaskReferences(event.dataTransfer.files) }}>
            {taskReferences.slice(0, 9).map((reference, index) => <button className="task-reference-thumb" key={`${reference.name}-${index}`} title={reference.name || `参考图 ${index + 1}`} onClick={() => data.onTaskConfig?.(id, 'taskReferences', taskReferences.filter((_, itemIndex) => itemIndex !== index))}><img src={referenceDisplayUrl(reference)} alt={reference.name || `参考图 ${index + 1}`} /><i>×</i></button>)}
            {taskReferences.length < 9 && <label className="task-reference-add"><span>＋</span><small>{taskReferences.length}/9</small><input type="file" accept="image/*" multiple onChange={(event) => { acceptTaskReferences(event.target.files); event.target.value = '' }} /></label>}
          </div>
          <ImeTextarea aria-label="任务提示词" value={data.taskPrompt || ''} onChange={(value) => data.onTaskConfig?.(id, 'taskPrompt', value)} placeholder="在这里输入提示词…（输入 @ 引用角色/素材）" />
        </div>
        <div className="task-style-row"><span>视觉风格</span><select value={data.taskStyle || '无风格'} onChange={(event) => data.onTaskConfig?.(id, 'taskStyle', event.target.value)}>{creativeStyles.map((style) => <option key={style}>{style}</option>)}</select><small>{data.linkGroup ? '已同步关联节点' : '可在风格库中选择'}</small></div>
        {isSeedance && data.kind === 'video' && <label className="seedance-asset-row"><span>真人可信素材</span><input value={data.taskSeedanceAssetId || ''} onChange={(event) => data.onTaskConfig?.(id, 'taskSeedanceAssetId', event.target.value)} placeholder="填写 Asset ID（真人图片必须认证）" /><small>{data.taskSeedanceAssetId ? '将优先使用已认证素材，不发送普通连线人物图' : '非真人角色可留空'}</small></label>}
        <div className="task-editor-footer">
          <select className="video-provider-select" title="生成引擎" value={isSeedance ? 'api' : 'comfy'} onChange={(event) => data.onTaskProvider?.(id, event.target.value)}><option value="comfy">本地 ComfyUI</option><option value="api">已接入 API</option></select>
          <button className="task-model-select task-model-trigger" onClick={() => data.onTaskConfig?.(id, 'taskPickerOpen', !data.taskPickerOpen)}><span>{isSeedance ? `${selectedSeedanceLabel} · 云端` : data.taskWorkflowName || data.taskModel || taskModelOptions[0]}</span><b>⌄</b></button>
          <select value={data.taskRatio || '16:9'} onChange={(event) => data.onTaskConfig?.(id, 'taskRatio', event.target.value)}><option>16:9</option><option>1:1</option><option>9:16</option><option>4:3</option><option>3:4</option></select>
          <select value={data.taskResolution || '480P'} onChange={(event) => data.onTaskConfig?.(id, 'taskResolution', event.target.value)}><option>480P</option><option>720P</option>{(data.kind === 'image' || !isSeedance) && <><option>1080P</option><option>1K</option><option>2K</option></>}</select>
          {data.kind === 'video' && <select title="视频时长（分镜师会自动估算，也可手动调整）" value={Math.max(isSeedance ? 4 : 1, Number(data.taskDuration || 6))} onChange={(event) => data.onTaskConfig?.(id, 'taskDuration', Number(event.target.value))}>{Array.from({ length: isSeedance ? 12 : 15 }, (_, index) => index + (isSeedance ? 4 : 1)).map((seconds) => <option key={seconds} value={seconds}>{seconds}秒</option>)}</select>}
          <select value={data.taskSeedMode || '随机'} onChange={(event) => data.onTaskConfig?.(id, 'taskSeedMode', event.target.value)}><option>随机</option><option>固定</option></select>
          <select value={data.taskCount || '1x'} onChange={(event) => data.onTaskConfig?.(id, 'taskCount', event.target.value)}><option>1x</option><option>2x</option><option>4x</option></select>
          <button className="task-run" title="加入生成任务" onClick={() => data.onRunTask?.(id, { autoRunSequence: false, autoRunVideo: false, autoRunPipeline: false })}>▷</button>
        </div>
        {data.taskPickerOpen && <div className="task-model-picker">
          <div className="picker-head"><span>{isSeedance ? `已接入的${desiredWorkflowKind === 'video' ? '视频' : '图片'} API` : data.comfyCatalogStatus || '本地 ComfyUI'}</span><button onClick={() => data.onRefreshComfy?.()}>↻ 重新识别本地工作流</button></div>
          <section>
            <div className="picker-column"><h4>{desiredWorkflowKind === 'video' ? '视频 API / 视频工作流' : '图片 API / 图片工作流'} <small>{compatibleApiModels.length + compatibleWorkflows.length} 个</small></h4>{compatibleApiModels.map((entry) => <button className={isSeedance && selectedSeedanceModel === entry.model && data.taskApiConnectionId === entry.connectionId ? 'active' : ''} key={`${entry.connectionId}:${entry.model}`} onClick={() => data.onSeedanceModelSelect?.(id, entry)}><span>{entry.label}</span><small>{entry.connectionName} · {entry.model}</small></button>)}{!compatibleApiModels.length && <div className="picker-empty"><b>尚未接入{desiredWorkflowKind === 'video' ? '视频' : '图片'} API 模型</b><small>请到连接设置的新建 API 页面添加</small></div>}{compatibleWorkflows.map((workflow) => <button className={!isSeedance && data.taskWorkflowId === workflow.id ? 'active' : ''} key={workflow.id} onClick={() => data.onTaskSelect?.(id, 'workflow', workflow)}><span>{workflow.name}</span><small>{workflow.nodeCount ? `${workflow.nodeCount} 个节点 · 本地 ComfyUI` : '本地工作流 · 点击自动加载'}</small></button>)}</div>
            <div className="picker-column picker-auto-model"><h4>{isSeedance ? `当前${desiredWorkflowKind === 'video' ? '视频' : '图片'} API 模型` : '自动匹配本地模型'}</h4><div className="auto-model-result"><b>{isSeedance ? selectedSeedanceLabel : data.taskModel || (data.taskWorkflowId ? '使用工作流内置模型' : `选择左侧${desiredWorkflowKind === 'video' ? '视频' : '图片'}工作流`)}</b><small>{isSeedance ? `${selectedApiEntry?.connectionName || '已保存 API'} · ${selectedSeedanceModel || '请先添加匹配模型'}` : data.taskWorkflowId ? '工作流确认后自动采用可用默认模型，无需手动选择' : `已扫描 ${visibleTaskModels.length} 个本地可用模型`}</small></div></div>
          </section>
        </div>}
        </>}
      </div>}
      <Handle type="source" position={Position.Right} />
    </div>
  )
}

const nodeTypes = { workflow: WorkflowNode }

function AtmosphereEdge({ id, sourceX, sourceY, sourcePosition, targetX, targetY, targetPosition, markerEnd, markerStart, interactionWidth, data }) {
  const [path] = getBezierPath({ sourceX, sourceY, sourcePosition, targetX, targetY, targetPosition })
  return <>
    <BaseEdge id={id} path={path} markerEnd={markerEnd} markerStart={markerStart} interactionWidth={interactionWidth || 28} />
    <path className="edge-context-hit" d={path} onContextMenu={(event) => data?.onElementContextMenu?.(event, 'edge', { id })} />
    {!data?.lowMotion && <path className="edge-travel-light" d={path} aria-hidden="true" />}
  </>
}

const edgeTypes = { atmosphere: AtmosphereEdge }

function App() {
  const initial = useMemo(() => {
    try {
      const storedProjects = JSON.parse(localStorage.getItem(PROJECTS_KEY) || '[]')
      if (Array.isArray(storedProjects) && storedProjects.length) {
        const activeId = localStorage.getItem(ACTIVE_PROJECT_KEY)
        const active = storedProjects.find((item) => item.id === activeId) || storedProjects[0]
        const legacyHeavyFormat = storedProjects.some((item) => Array.isArray(item.nodes) || Array.isArray(item.edges))
        if (legacyHeavyFormat) {
          const metadata = storedProjects.map((item) => ({ id: item.id, name: item.name || '未命名画布', nodeCount: item.nodes?.length || 0, updatedAt: item.updatedAt || Date.now() }))
          localStorage.removeItem(STORAGE_KEY)
          localStorage.removeItem(PROJECTS_KEY)
          for (const item of storedProjects) localStorage.setItem(`${CANVAS_KEY_PREFIX}${item.id}`, JSON.stringify({ nodes: item.nodes || [], edges: item.edges || [] }))
          localStorage.setItem(PROJECTS_KEY, JSON.stringify(metadata))
          localStorage.setItem(ACTIVE_PROJECT_KEY, active.id)
          return { ...metadata.find((item) => item.id === active.id), projects: metadata, nodes: (active.nodes || []).map(repairPersistedFrameNode), edges: active.edges || [] }
        }
        const canvas = JSON.parse(localStorage.getItem(`${CANVAS_KEY_PREFIX}${active.id}`) || '{}')
        return { ...active, projects: storedProjects, nodes: (canvas.nodes || []).map(repairPersistedFrameNode), edges: canvas.edges || [] }
      }
      const stored = JSON.parse(localStorage.getItem(STORAGE_KEY)) || templates.image
      const metadata = { id: 'canvas-1', name: '画布 1', nodeCount: stored.nodes?.length || 0, updatedAt: Date.now() }
      localStorage.removeItem(STORAGE_KEY)
      localStorage.setItem(`${CANVAS_KEY_PREFIX}${metadata.id}`, JSON.stringify({ nodes: stored.nodes || [], edges: stored.edges || [] }))
      localStorage.setItem(PROJECTS_KEY, JSON.stringify([metadata]))
      localStorage.setItem(ACTIVE_PROJECT_KEY, metadata.id)
      return { ...metadata, projects: [metadata], nodes: (stored.nodes || []).map(repairPersistedFrameNode), edges: stored.edges || [] }
    } catch { return templates.image }
  }, [])
  const [nodes, setNodes] = useState(initial.nodes)
  const [edges, setEdges] = useState(initial.edges)
  const [projectId, setProjectId] = useState(initial.id || 'canvas-1')
  const [projectName, setProjectName] = useState(initial.name || '画布 1')
  const [projects, setProjects] = useState(initial.projects || [{ id: initial.id || 'canvas-1', name: initial.name || '画布 1', nodeCount: initial.nodes?.length || 0, updatedAt: Date.now() }])
  const [saved, setSaved] = useState(true)
  const [canvasHydrated, setCanvasHydrated] = useState(!window.aaaLite?.loadCanvas)
  const [panel, setPanel] = useState('chat')
  const [modal, setModal] = useState(null)
  const [message, setMessage] = useState('')
  const [chat, setChat] = useState([{ role: 'assistant', text: '你好，我是自由者里的豆包助手。配置后可以直接问我日常问题。' }])
  const [doubaoConfig, setDoubaoConfig] = useState({ configured: false, videoConfigured: false, model: '', videoModel: '', videoBaseUrl: '' })
  const [doubaoModel, setDoubaoModel] = useState('')
  const [seedanceModels, setSeedanceModels] = useState(() => {
    try {
      const stored = localStorage.getItem(SEEDANCE_MODELS_KEY)
      return stored == null ? ['doubao-seedance-2-0-mini-260615'] : JSON.parse(stored)
    } catch { return ['doubao-seedance-2-0-mini-260615'] }
  })
  const [seedanceModelLabels, setSeedanceModelLabels] = useState(() => {
    try {
      return { 'doubao-seedance-2-0-mini-260615': 'Seedance 2.0 Mini', ...(JSON.parse(localStorage.getItem(SEEDANCE_MODEL_LABELS_KEY)) || {}) }
    } catch { return { 'doubao-seedance-2-0-mini-260615': 'Seedance 2.0 Mini' } }
  })
  const [seedanceModel, setSeedanceModel] = useState(localStorage.getItem(ACTIVE_SEEDANCE_MODEL_KEY) || 'doubao-seedance-2-0-mini-260615')
  const [newSeedanceModel, setNewSeedanceModel] = useState('')
  const [newSeedanceModelLabel, setNewSeedanceModelLabel] = useState('')
  const [newSeedanceModelKind, setNewSeedanceModelKind] = useState('video')
  const [seedanceModelKinds, setSeedanceModelKinds] = useState({ 'doubao-seedance-2-0-mini-260615': 'video' })
  const [seedanceBaseUrl, setSeedanceBaseUrl] = useState('https://ark.cn-beijing.volces.com/api/v3')
  const [seedanceConnections, setSeedanceConnections] = useState([])
  const [activeSeedanceConnectionId, setActiveSeedanceConnectionId] = useState('')
  const [seedanceConnectionName, setSeedanceConnectionName] = useState('')
  const [seedanceConnectionMode, setSeedanceConnectionMode] = useState('saved')
  const availableApiModels = useMemo(() => seedanceConnections.flatMap((connection) => (connection.videoModels || []).map((model) => ({ connectionId: connection.id, connectionName: connection.name, baseUrl: connection.videoBaseUrl, model, label: connection.videoModelLabels?.[model] || model, kind: connection.modelMediaKinds?.[model] === 'image' ? 'image' : 'video' }))), [seedanceConnections])
  const [doubaoKey, setDoubaoKey] = useState('')
  const [doubaoLoading, setDoubaoLoading] = useState(false)
  const [endpoint, setEndpoint] = useState(localStorage.getItem('aaa-lite-comfy') || 'http://127.0.0.1:8188')
  const [connection, setConnection] = useState(null)
  const [comfyCatalog, setComfyCatalog] = useState({ ok: null, models: [], optionsByInput: {}, message: '等待识别本地模型' })
  const [flow, setFlow] = useState(null)
  const [menu, setMenu] = useState(null)
  const [rightSelectBox, setRightSelectBox] = useState(null)
  const [dockMode, setDockMode] = useState('image')
  const [dockPrompt, setDockPrompt] = useState('')
  const [dockModel, setDockModel] = useState('本地 ComfyUI')
  const [dockStyle, setDockStyle] = useState('无风格')
  const [toast, setToast] = useState(null)
  const [storyboardScript, setStoryboardScript] = useState('')
  const [storyboardStyle, setStoryboardStyle] = useState('电影写实')
  const [storyboardRatio, setStoryboardRatio] = useState('16:9')
  const [storyboardResolution, setStoryboardResolution] = useState('720P')
  const [storyboardMaxShots, setStoryboardMaxShots] = useState(8)
  const [storyboardEditingGroup, setStoryboardEditingGroup] = useState(null)
  const [stats, setStats] = useState({ cpu: 0, memory: 0, gpu: null, vramUsedGb: null, vramTotalGb: null, temperature: null })
  const [chatOpen, setChatOpen] = useState(false)
  const [chatMode, setChatMode] = useState('web')
  const [chatGptWebKey, setChatGptWebKey] = useState(0)
  const [qwenWebKey, setQwenWebKey] = useState(0)
  const [doubaoWebKey, setDoubaoWebKey] = useState(0)
  const [taskOpen, setTaskOpen] = useState(false)
  const [taskMenu, setTaskMenu] = useState(null)
  const [taskPreview, setTaskPreview] = useState(null)
  const [sessionTasks, setSessionTasks] = useState([])
  const [theme, setTheme] = useState(localStorage.getItem('aaa-lite-theme') || 'dark')
  const [workflowLibrary, setWorkflowLibrary] = useState(() => {
    try { return JSON.parse(localStorage.getItem(WORKFLOW_KEY)) || [] } catch { return [] }
  })
  const [localWorkflows, setLocalWorkflows] = useState([])
  const canvasRef = useRef(null)
  const glowRef = useRef(null)
  const rightSelectRef = useRef(null)
  const suppressRightContextRef = useRef(false)
  const progressEventRef = useRef(new Map())
  const taskRunMetaRef = useRef(new Map())
  const comfyActivitySignatureRef = useRef(new Map())
  const externalAssignedPromptRef = useRef(new Set())
  const sessionStartedAtRef = useRef(Date.now())
  const cancelledPromptRef = useRef(new Set())
  const frameCaptureInFlightRef = useRef(new Set())
  const hadSeedanceModelsRef = useRef(localStorage.getItem(SEEDANCE_MODELS_KEY) != null)
  const projectsRef = useRef(projects)
  const legacyMigrationStartedRef = useRef(false)

  useEffect(() => {
    let active = true
    const hydrate = async () => {
      if (!window.aaaLite?.loadCanvas) return
      try {
        const result = await window.aaaLite.loadCanvas(initial.id || 'canvas-1')
        if (!active) return
        if (result?.ok && result.canvas) {
          setNodes((result.canvas.nodes || []).map(repairPersistedFrameNode))
          setEdges(result.canvas.edges || [])
        }
      } catch (error) {
        console.error('画布文件读取失败', error)
      } finally {
        if (active) setCanvasHydrated(true)
      }
    }
    void hydrate()
    return () => { active = false }
  }, [initial.id])

  useEffect(() => {
    if (!canvasHydrated || legacyMigrationStartedRef.current || !window.aaaLite?.saveCanvas) return
    legacyMigrationStartedRef.current = true
    const migrate = async () => {
      let migrated = 0
      for (const project of projectsRef.current) {
        const key = `${CANVAS_KEY_PREFIX}${project.id}`
        const raw = localStorage.getItem(key)
        if (!raw) continue
        try {
          const canvas = JSON.parse(raw)
          const optimized = await externalizeCanvasMedia(canvas.nodes || [])
          const result = await window.aaaLite.saveCanvas(project.id, { nodes: optimized.nodes, edges: canvas.edges || [] })
          if (!result?.ok) throw new Error(result?.message || '画布文件保存失败')
          localStorage.removeItem(key)
          migrated += 1
        } catch (error) {
          console.error(`历史画布迁移失败：${project.id}`, error)
        }
      }
      if (migrated) {
        setToast(`已优化 ${migrated} 个历史画布的数据存储`)
        window.setTimeout(() => setToast(null), 3000)
      }
    }
    void migrate()
  }, [canvasHydrated])

  useEffect(() => { setSaved(false) }, [nodes, edges])

  useEffect(() => {
    setNodes((items) => items.map((node) => node.data.promptId && !node.data.submittedByApp && mediaKinds.has(node.data.kind) ? { ...node, data: { ...node.data, promptId: null, resultUrl: null, results: [], status: '就绪' } } : node))
  }, [])

  useEffect(() => {
    try { localStorage.setItem(WORKFLOW_KEY, JSON.stringify(workflowLibrary)) } catch { /* large libraries remain available for this session */ }
  }, [workflowLibrary])

  useEffect(() => {
    localStorage.setItem(SEEDANCE_MODELS_KEY, JSON.stringify(seedanceModels))
  }, [seedanceModels])

  useEffect(() => {
    localStorage.setItem(SEEDANCE_MODEL_LABELS_KEY, JSON.stringify(seedanceModelLabels))
  }, [seedanceModelLabels])

  useEffect(() => {
    if (seedanceModel) localStorage.setItem(ACTIVE_SEEDANCE_MODEL_KEY, seedanceModel)
    else localStorage.removeItem(ACTIVE_SEEDANCE_MODEL_KEY)
  }, [seedanceModel])

  useEffect(() => {
    window.aaaLite?.getDoubaoConfig?.().then((result) => {
      if (!result?.ok) return
      setDoubaoConfig({ configured: result.configured, videoConfigured: result.videoConfigured, model: result.model || '', videoModel: result.videoModel || '', videoBaseUrl: result.videoBaseUrl || '', activeVideoConnectionId: result.activeVideoConnectionId || '' })
      setDoubaoModel(result.model || '')
      const configuredModel = result.videoModel || 'doubao-seedance-2-0-mini-260615'
      setSeedanceModels(result.videoModels?.length ? result.videoModels : [configuredModel])
      setSeedanceModelLabels((items) => ({ ...items, ...(result.videoModelLabels || {}) }))
      setSeedanceModelKinds(result.modelMediaKinds || Object.fromEntries((result.videoModels || [configuredModel]).map((model) => [model, 'video'])))
      setSeedanceModel(configuredModel)
      setSeedanceBaseUrl(result.videoBaseUrl || 'https://ark.cn-beijing.volces.com/api/v3')
      setSeedanceConnections(result.videoConnections || [])
      setActiveSeedanceConnectionId(result.activeVideoConnectionId || '')
      setSeedanceConnectionName(result.videoConnectionName || '默认连接')
    })
  }, [])

  useEffect(() => {
    if (!window.aaaLite?.onComfyEvent) return
    return window.aaaLite.onComfyEvent((event) => {
      if (event.type === 'progress') {
        const now = performance.now()
        const previous = progressEventRef.current.get(event.promptId) || 0
        if (now - previous < 180) return
        progressEventRef.current.set(event.promptId, now)
      } else if (event.type === 'complete' || event.type === 'error') progressEventRef.current.delete(event.promptId)
      const runMeta = taskRunMetaRef.current.get(event.promptId)
      setSessionTasks((records) => {
        const existing = records.find((record) => record.promptId === event.promptId)
        const progress = event.type === 'progress' ? Math.round((event.value / Math.max(1, event.max)) * 100) : event.type === 'complete' ? 100 : existing?.progress || 0
        const status = event.type === 'complete' ? '已完成' : event.type === 'error' && cancelledPromptRef.current.has(event.promptId) ? '已取消' : event.type === 'error' ? `失败：${event.message}` : event.type === 'progress' || event.type === 'executing' ? '生成中' : existing?.status || '排队中'
        const next = {
          id: existing?.id || event.promptId,
          promptId: event.promptId,
          nodeId: existing?.nodeId || runMeta?.nodeId || null,
          kind: existing?.kind || runMeta?.kind || 'comfy',
          title: existing?.title || runMeta?.title || 'ComfyUI 任务',
          workflowName: existing?.workflowName || runMeta?.workflowName || '本地 ComfyUI',
          status,
          progress,
          media: event.type === 'complete' ? (event.media || []) : existing?.media || [],
          createdAt: existing?.createdAt || runMeta?.createdAt || Date.now(),
          updatedAt: Date.now()
        }
        if (event.type === 'complete' || event.type === 'error') taskRunMetaRef.current.delete(event.promptId)
        return trimSessionTasks(existing ? records.map((record) => record.promptId === event.promptId ? next : record) : [next, ...records])
      })
      setNodes((items) => {
        const source = items.find((node) => node.data.promptId === event.promptId)
        const completedReference = event.type === 'complete' && source?.data.storyboardStage === 'reference'
        const completedCharacter = event.type === 'complete' && source?.data.storyboardStage === 'character'
        const completedScene = event.type === 'complete' && source?.data.storyboardStage === 'scene'
        const completedVideo = event.type === 'complete' && source?.data.storyboardStage === 'video'
        const allCharactersComplete = completedCharacter && items.filter((node) => node.data.storyboardGroup === source.data.storyboardGroup && node.data.storyboardStage === 'character').every((node) => node.id === source.id || node.data.status === '已完成')
        const allScenesComplete = completedScene && items.filter((node) => node.data.storyboardGroup === source.data.storyboardGroup && node.data.storyboardStage === 'scene').every((node) => node.id === source.id || node.data.status === '已完成')
        const media = event.media || []
        const generatedReference = media[0]?.url ? { url: media[0].url, remoteUrl: media[0].remoteUrl, name: media[0].filename || (completedCharacter ? `${source.data.storyboardCharacterName || '人物'}设计图` : completedScene ? `${source.data.storyboardSceneName || '场景'}设计图` : `${source?.data.storyboardTitle || '镜头'}分镜图`), role: completedCharacter ? 'character' : completedScene ? 'scene' : 'storyboard', assetId: source?.id } : null
        return items.map((node) => {
        if (completedCharacter && generatedReference && node.data.storyboardGroup === source.data.storyboardGroup && ['scene', 'reference'].includes(node.data.storyboardStage)) {
          const characterLimit = node.data.storyboardStage === 'reference' ? 8 : 9
          const taskReferences = [generatedReference, ...(node.data.taskReferences || []).filter((reference) => reference.assetId !== source.id)].filter((reference) => reference.role === 'character').slice(0, characterLimit)
          const shouldRun = node.data.storyboardStage === 'scene' && allCharactersComplete && source.data.autoRunPipeline
          return { ...node, data: { ...node.data, taskReferences, taskReferenceUrl: taskReferences[0].url, taskReferenceName: taskReferences[0].name, status: node.data.storyboardStage === 'scene' ? `${taskReferences.length} 个人物素材已识别${allCharactersComplete ? '，准备设计场景' : ''}` : `多图片素材参考 ${taskReferences.length}/9`, autoRunRequested: shouldRun ? Date.now() : node.data.autoRunRequested, autoRunHandled: shouldRun ? false : node.data.autoRunHandled, autoRunPipeline: source.data.autoRunPipeline || node.data.autoRunPipeline } }
        }
        if (completedScene && generatedReference && node.data.storyboardGroup === source.data.storyboardGroup && node.data.storyboardStage === 'reference') {
          const isAssignedScene = node.data.storyboardScene === source.data.storyboardSceneName
          const existing = (node.data.taskReferences || []).filter((reference) => reference.assetId !== source.id)
          const taskReferences = isAssignedScene ? [...existing.filter((reference) => reference.role === 'character').slice(0, 8), generatedReference].slice(0, 9) : existing
          const shouldRun = allScenesComplete && source.data.autoRunPipeline
          return { ...node, data: { ...node.data, taskReferences, taskReferenceUrl: taskReferences[0]?.url || null, taskReferenceName: taskReferences[0]?.name || '', status: `多图片素材参考 ${taskReferences.length}/9${allScenesComplete ? ' · 准备生成分镜图' : ''}`, autoRunRequested: shouldRun ? Date.now() : node.data.autoRunRequested, autoRunHandled: shouldRun ? false : node.data.autoRunHandled, autoRunVideo: source.data.autoRunPipeline || node.data.autoRunVideo } }
        }
        if (completedReference && generatedReference && node.data.storyboardStage === 'video' && node.data.storyboardGroup === source.data.storyboardGroup && node.data.storyboardOrder === source.data.storyboardOrder) {
          const firstShot = node.data.storyboardOrder === 1
          return { ...node, data: { ...node.data, taskReferences: [generatedReference], taskReferenceUrl: generatedReference.url, taskReferenceName: generatedReference.name, status: source.data.autoRunVideo ? (firstShot ? '分镜图完成，准备生成第 1 段视频' : '分镜图已完成，等待前一段视频') : '分镜图已就绪，可单镜头生成', autoRunRequested: source.data.autoRunVideo && firstShot ? Date.now() : null, autoRunHandled: source.data.autoRunVideo && firstShot ? false : node.data.autoRunHandled, autoRunSequence: Boolean(source.data.autoRunVideo) } }
        }
        if (completedVideo && source.data.autoRunSequence && node.data.storyboardStage === 'video' && node.data.storyboardGroup === source.data.storyboardGroup && node.data.storyboardOrder === source.data.storyboardOrder + 1) {
          return { ...node, data: { ...node.data, status: `第 ${source.data.storyboardOrder} 段完成，准备生成当前视频`, autoRunRequested: Date.now(), autoRunHandled: false, autoRunSequence: true } }
        }
        if (node.data.promptId !== event.promptId) return node
        if (event.type === 'progress') return { ...node, data: { ...node.data, status: '生成中', progress: Math.round((event.value / Math.max(1, event.max)) * 100) } }
        if (event.type === 'executing') return { ...node, data: { ...node.data, status: event.node ? '生成中' : node.data.status } }
        if (event.type === 'complete') {
          return { ...node, data: { ...node.data, status: '已完成', progress: 100, resultUrl: media[0]?.url || null, results: media } }
        }
        if (event.type === 'error') return { ...node, data: { ...node.data, status: `失败：${event.message}`, progress: 0 } }
        return node
        })
      })
    })
  }, [])

  useEffect(() => {
    if (!window.aaaLite?.getRecentComfyActivity) return
    let active = true
    let syncing = false
    const sync = async () => {
      if (syncing) return
      syncing = true
      let result
      try { result = await window.aaaLite.getRecentComfyActivity(endpoint, 40) }
      finally { syncing = false }
      if (!active || !result?.ok) return
      const cutoff = sessionStartedAtRef.current - 60 * 60 * 1000
      const activities = (result.items || []).filter((item) => item.createdAt >= cutoff)
      if (comfyActivitySignatureRef.current.size > 160) {
        const recentIds = new Set(activities.map((item) => item.promptId))
        for (const promptId of comfyActivitySignatureRef.current.keys()) if (!recentIds.has(promptId)) comfyActivitySignatureRef.current.delete(promptId)
        for (const promptId of cancelledPromptRef.current) if (!recentIds.has(promptId)) cancelledPromptRef.current.delete(promptId)
        for (const promptId of externalAssignedPromptRef.current) if (!recentIds.has(promptId)) externalAssignedPromptRef.current.delete(promptId)
      }
      const changed = activities.filter((item) => {
        const signature = `${item.status}|${item.progress}|${(item.media || []).map((media) => `${media.kind}:${media.url}`).join('|')}`
        if (comfyActivitySignatureRef.current.get(item.promptId) === signature) return false
        comfyActivitySignatureRef.current.set(item.promptId, signature)
        return true
      })
      if (!changed.length) return
      setSessionTasks((records) => {
        const next = [...records]
        for (const activity of changed) {
          const index = next.findIndex((record) => record.promptId === activity.promptId)
          const existing = index >= 0 ? next[index] : null
          const record = {
            id: activity.promptId,
            promptId: activity.promptId,
            nodeId: existing?.nodeId || null,
            kind: existing?.kind || activity.kind || 'image',
            title: existing?.title || nodeCatalog[activity.kind]?.title || 'ComfyUI 任务',
            workflowName: existing?.workflowName || activity.workflowName || 'ComfyUI 外部任务',
            status: cancelledPromptRef.current.has(activity.promptId) ? '已取消' : activity.status,
            progress: cancelledPromptRef.current.has(activity.promptId) ? 0 : activity.progress,
            media: activity.media || [],
            createdAt: existing?.createdAt || activity.createdAt || Date.now(),
            updatedAt: Date.now(),
            external: existing?.external ?? !existing
          }
          if (index >= 0) next[index] = record
          else next.push(record)
        }
        return trimSessionTasks(next.sort((left, right) => right.createdAt - left.createdAt))
      })
      const mediaActivities = changed.filter((item) => item.media?.length).sort((left, right) => Number(right.status === '已完成') - Number(left.status === '已完成') || right.createdAt - left.createdAt)
      if (mediaActivities.length) {
        setNodes((items) => {
          let next = items
          for (const activity of mediaActivities) {
            const target = next.find((node) => node.data.promptId === activity.promptId && node.data.submittedByApp)
            if (!target) continue
            externalAssignedPromptRef.current.add(activity.promptId)
            const resultUrl = activity.media[0]?.url || null
            const unchanged = target.data.resultUrl === resultUrl && target.data.status === activity.status
            if (unchanged) continue
            next = next.map((node) => node.id === target.id ? { ...node, data: { ...node.data, promptId: activity.promptId, status: activity.status, progress: activity.progress, resultUrl, results: activity.media, workflowName: node.data.workflowName || activity.workflowName } } : node)
          }
          return next
        })
      }
    }
    sync()
    const timer = window.setInterval(() => { if (!document.hidden) void sync() }, 3500)
    return () => { active = false; window.clearInterval(timer) }
  }, [endpoint])

  useEffect(() => {
    if (!window.aaaLite?.getSystemStats) return
    let active = true
    const update = async () => {
      if (document.hidden) return
      try {
        const next = await window.aaaLite.getSystemStats()
        if (active) setStats((current) => Object.keys(next || {}).every((key) => current?.[key] === next[key]) && Object.keys(current || {}).every((key) => current[key] === next?.[key]) ? current : next)
      } catch { /* 系统监控不可用时保留上次数据 */ }
    }
    update()
    const timer = window.setInterval(update, 2000)
    return () => { active = false; window.clearInterval(timer) }
  }, [])

  useEffect(() => {
    localStorage.setItem('aaa-lite-theme', theme)
  }, [theme])

  const persistCurrentProject = useCallback(async (nextNodes = nodes, nextEdges = edges, nextName = projectName) => {
    const record = { id: projectId, name: nextName || '未命名画布', nodeCount: nextNodes.length, updatedAt: Date.now() }
    try {
      const optimized = await externalizeCanvasMedia(nextNodes)
      if (window.aaaLite?.saveCanvas) {
        const result = await window.aaaLite.saveCanvas(projectId, { nodes: optimized.nodes, edges: nextEdges })
        if (!result?.ok) throw new Error(result?.message || '画布文件保存失败')
        localStorage.removeItem(`${CANVAS_KEY_PREFIX}${projectId}`)
      } else {
        localStorage.setItem(`${CANVAS_KEY_PREFIX}${projectId}`, JSON.stringify({ nodes: optimized.nodes, edges: nextEdges }))
      }
      const items = projectsRef.current
      const next = items.some((item) => item.id === projectId) ? items.map((item) => item.id === projectId ? record : item) : [...items, record]
      localStorage.setItem(PROJECTS_KEY, JSON.stringify(next))
      localStorage.setItem(ACTIVE_PROJECT_KEY, projectId)
      localStorage.removeItem(STORAGE_KEY)
      projectsRef.current = next
      setProjects(next)
      if (optimized.changed) setNodes(optimized.nodes)
      return true
    } catch (error) {
      console.error('画布保存失败', error)
      setSaved(false)
      setToast(`画布保存失败：${error.message}`)
      window.setTimeout(() => setToast(null), 4200)
      return false
    }
  }, [nodes, edges, projectId, projectName])

  const save = useCallback(async () => {
    if (await persistCurrentProject()) setSaved(true)
  }, [persistCurrentProject])

  useEffect(() => {
    if (!canvasHydrated) return
    const timer = window.setTimeout(() => {
      void persistCurrentProject().then((ok) => { if (ok) setSaved(true) })
    }, 1200)
    return () => window.clearTimeout(timer)
  }, [nodes, edges, projectName, persistCurrentProject, canvasHydrated])

  useEffect(() => {
    const listener = (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') {
        event.preventDefault(); save()
      }
    }
    window.addEventListener('keydown', listener)
    return () => window.removeEventListener('keydown', listener)
  }, [save])

  const createCanvas = useCallback(async () => {
    if (!await persistCurrentProject()) return
    const id = `canvas-${Date.now()}`
    const name = `画布 ${projectsRef.current.length + 1}`
    const template = structuredClone(templates.image)
    const record = { id, name, nodeCount: template.nodes.length, updatedAt: Date.now() }
    try {
      const next = [...projectsRef.current, record]
      if (window.aaaLite?.saveCanvas) {
        const result = await window.aaaLite.saveCanvas(id, { nodes: template.nodes, edges: template.edges })
        if (!result?.ok) throw new Error(result?.message || '画布文件创建失败')
      } else localStorage.setItem(`${CANVAS_KEY_PREFIX}${id}`, JSON.stringify({ nodes: template.nodes, edges: template.edges }))
      localStorage.setItem(PROJECTS_KEY, JSON.stringify(next))
      localStorage.setItem(ACTIVE_PROJECT_KEY, id)
      projectsRef.current = next
      setProjects(next)
    } catch (error) {
      console.error('新建画布失败', error)
      setToast('本机画布存储空间不足，无法新建画布')
      window.setTimeout(() => setToast(null), 3600)
      return
    }
    setProjectId(id)
    setProjectName(name)
    setNodes(template.nodes)
    setEdges(template.edges)
    setModal(null)
    setSaved(true)
    setToast(`已新建 ${name}`)
    window.setTimeout(() => setToast(null), 2200)
  }, [persistCurrentProject])

  const openCanvas = useCallback(async (id) => {
    if (id === projectId) { setModal(null); return }
    if (!await persistCurrentProject()) return
    const target = projectsRef.current.find((item) => item.id === id)
    if (!target) return
    let canvas
    try {
      if (window.aaaLite?.loadCanvas) {
        const result = await window.aaaLite.loadCanvas(id)
        if (!result?.ok) throw new Error(result?.message || '画布文件读取失败')
        canvas = result.canvas || JSON.parse(localStorage.getItem(`${CANVAS_KEY_PREFIX}${id}`) || '{}')
      } else canvas = JSON.parse(localStorage.getItem(`${CANVAS_KEY_PREFIX}${id}`) || '{}')
      localStorage.setItem(ACTIVE_PROJECT_KEY, id)
    } catch (error) {
      console.error('打开画布失败', error)
      setToast('画布读取失败，原画布仍保持不变')
      window.setTimeout(() => setToast(null), 3200)
      return
    }
    setProjectId(target.id)
    setProjectName(target.name)
    setNodes((canvas.nodes || []).map(repairPersistedFrameNode))
    setEdges(canvas.edges || [])
    setModal(null)
    setSaved(true)
  }, [persistCurrentProject, projectId])

  const renameCanvas = useCallback((name) => {
    setProjectName(String(name || '').slice(0, 60))
  }, [])

  const addSeedanceModel = useCallback(() => {
    const model = newSeedanceModel.trim()
    const label = newSeedanceModelLabel.trim() || model
    if (!model) return
    if (!/^[\w.:/-]{2,200}$/.test(model)) {
      setToast('模型 ID 格式无效')
      window.setTimeout(() => setToast(null), 2600)
      return
    }
    setSeedanceModels((items) => items.includes(model) ? items : [...items, model])
    setSeedanceModelLabels((items) => ({ ...items, [model]: label.slice(0, 80) }))
    setSeedanceModelKinds((items) => ({ ...items, [model]: newSeedanceModelKind }))
    setSeedanceModel(model)
    setNewSeedanceModel('')
    setNewSeedanceModelLabel('')
    setToast(`已接入模型：${label}`)
    window.setTimeout(() => setToast(null), 2200)
  }, [newSeedanceModel, newSeedanceModelLabel, newSeedanceModelKind])

  const deleteSeedanceModel = useCallback((model) => {
    setSeedanceModels((items) => {
      const next = items.filter((item) => item !== model)
      if (seedanceModel === model) setSeedanceModel(next[0] || '')
      return next
    })
    setSeedanceModelLabels((items) => {
      const next = { ...items }
      delete next[model]
      return next
    })
    setSeedanceModelKinds((items) => {
      const next = { ...items }
      delete next[model]
      return next
    })
  }, [seedanceModel])

  const addNode = (kind, options = {}) => {
    const id = `${kind}-${Date.now()}`
    const position = options.position || { x: 180 + nodes.length * 45, y: 140 + nodes.length * 35 }
    const builtinRecord = BUILTIN_VIDEO_TOOL_RECORDS[kind]
    if (builtinRecord) {
      setWorkflowLibrary((items) => [...items.filter((item) => item.id !== builtinRecord.id), { ...builtinRecord, params: undefined }])
    }
    const nodeData = builtinRecord
      ? { kind: 'comfy', builtinVideoTool: kind, workflowId: builtinRecord.id, workflowName: builtinRecord.name, model: builtinRecord.name, nodeCount: builtinRecord.summary.nodeCount, params: structuredClone(builtinRecord.params), missingCount: 0, status: '就绪 · 请选择输入视频' }
      : { kind, ...options.data }
    setNodes((items) => [...items, { id, type: 'workflow', position, data: nodeData }])
    if (options.connection?.fromNode && options.connection?.fromHandle) {
      const { fromNode, fromHandle } = options.connection
      const edge = fromHandle.type === 'target'
        ? { id: `edge-${Date.now()}`, type: 'atmosphere', source: id, target: fromNode.id, targetHandle: fromHandle.id || undefined, animated: false }
        : { id: `edge-${Date.now()}`, type: 'atmosphere', source: fromNode.id, target: id, sourceHandle: fromHandle.id || undefined, animated: false }
      setEdges((items) => addEdge(edge, items))
    }
    setModal(null)
    setMenu(null)
  }

  const eventPoint = (event) => {
    if ('clientX' in event) return { x: event.clientX, y: event.clientY }
    const touch = event.changedTouches?.[0] || event.touches?.[0]
    return touch ? { x: touch.clientX, y: touch.clientY } : null
  }

  const beginRightSelection = useCallback((event) => {
    if (event.button !== 2 || !event.target.closest?.('.react-flow__pane') || event.target.closest?.('.react-flow__node, .react-flow__edge, button, input, textarea, select')) return
    const bounds = canvasRef.current?.getBoundingClientRect()
    if (!bounds) return
    event.preventDefault()
    event.stopPropagation()
    event.currentTarget.setPointerCapture?.(event.pointerId)
    rightSelectRef.current = { pointerId: event.pointerId, startX: event.clientX, startY: event.clientY, currentX: event.clientX, currentY: event.clientY, moved: false }
    setMenu(null)
    setRightSelectBox({ left: event.clientX - bounds.left, top: event.clientY - bounds.top, width: 0, height: 0 })
  }, [])

  const moveRightSelection = useCallback((event) => {
    const active = rightSelectRef.current
    if (!active || active.pointerId !== event.pointerId) return
    const bounds = canvasRef.current?.getBoundingClientRect()
    if (!bounds) return
    event.preventDefault()
    event.stopPropagation()
    active.currentX = event.clientX
    active.currentY = event.clientY
    active.moved = active.moved || Math.hypot(event.clientX - active.startX, event.clientY - active.startY) > 6
    setRightSelectBox({
      left: Math.min(active.startX, event.clientX) - bounds.left,
      top: Math.min(active.startY, event.clientY) - bounds.top,
      width: Math.abs(event.clientX - active.startX),
      height: Math.abs(event.clientY - active.startY)
    })
  }, [])

  const finishRightSelection = useCallback((event) => {
    const active = rightSelectRef.current
    if (!active || active.pointerId !== event.pointerId) return
    event.preventDefault()
    event.stopPropagation()
    event.currentTarget.releasePointerCapture?.(event.pointerId)
    rightSelectRef.current = null
    setRightSelectBox(null)
    if (!active.moved) return
    const selection = {
      left: Math.min(active.startX, event.clientX),
      right: Math.max(active.startX, event.clientX),
      top: Math.min(active.startY, event.clientY),
      bottom: Math.max(active.startY, event.clientY)
    }
    const selectedIds = new Set()
    canvasRef.current?.querySelectorAll('.react-flow__node[data-id]').forEach((element) => {
      const rect = element.getBoundingClientRect()
      if (rect.right >= selection.left && rect.left <= selection.right && rect.bottom >= selection.top && rect.top <= selection.bottom) selectedIds.add(element.dataset.id)
    })
    setNodes((items) => items.map((node) => ({ ...node, selected: selectedIds.has(node.id) })))
    suppressRightContextRef.current = true
    window.setTimeout(() => { suppressRightContextRef.current = false }, 300)
    setToast(selectedIds.size ? `已框选 ${selectedIds.size} 个任务框 · 再单击右键可批量删除` : '框选范围内没有任务框')
    window.setTimeout(() => setToast(null), 2600)
  }, [])

  const openCreateMenu = useCallback((event, connection = null) => {
    event.preventDefault?.()
    if (suppressRightContextRef.current) { suppressRightContextRef.current = false; return }
    const point = eventPoint(event)
    if (!point || !flow) return
    const position = flow.screenToFlowPosition(point)
    const bounds = canvasRef.current?.getBoundingClientRect()
    const fromScreen = connection?.from ? flow.flowToScreenPosition(connection.from) : null
    setMenu({
      x: Math.min(point.x, window.innerWidth - 250),
      y: Math.min(point.y, window.innerHeight - (connection ? 430 : 500)),
      position: { x: position.x - 115, y: position.y - 52 },
      connection,
      line: bounds && fromScreen ? {
        x1: fromScreen.x - bounds.left,
        y1: fromScreen.y - bounds.top,
        x2: point.x - bounds.left,
        y2: point.y - bounds.top
      } : null
    })
  }, [flow])

  const handleConnectEnd = useCallback((event, connectionState) => {
    if (connectionState?.isValid || !connectionState?.fromNode || !connectionState?.fromHandle) return
    openCreateMenu(event, connectionState)
  }, [openCreateMenu])

  const handleCanvasDoubleClick = useCallback((event) => {
    if (!event.target.closest?.('.react-flow__pane')) return
    openCreateMenu(event)
  }, [openCreateMenu])

  const openElementMenu = useCallback((event, elementType, element) => {
    event.preventDefault()
    event.stopPropagation()
    if (suppressRightContextRef.current) { suppressRightContextRef.current = false; return }
    const point = eventPoint(event)
    if (!point) return
    const selectedNodes = elementType === 'node' ? nodes.filter((node) => node.selected) : []
    if (elementType === 'node' && selectedNodes.length > 1 && selectedNodes.some((node) => node.id === element.id)) {
      setMenu({ kind: 'nodes', x: Math.min(point.x, window.innerWidth - 250), y: Math.min(point.y, window.innerHeight - 155), elements: selectedNodes })
      return
    }
    setMenu({
      kind: elementType,
      x: Math.min(point.x, window.innerWidth - 224),
      y: Math.min(point.y, window.innerHeight - 150),
      element
    })
  }, [nodes])

  const openPaneContextMenu = useCallback((event) => {
    event.preventDefault()
    if (suppressRightContextRef.current) { suppressRightContextRef.current = false; return }
    const point = eventPoint(event)
    const selectedNodes = nodes.filter((node) => node.selected)
    if (point && selectedNodes.length > 1) {
      setMenu({ kind: 'nodes', x: Math.min(point.x, window.innerWidth - 250), y: Math.min(point.y, window.innerHeight - 155), elements: selectedNodes })
      return
    }
    openCreateMenu(event)
  }, [nodes, openCreateMenu])

  const closeTaskNode = useCallback(async (node) => {
    setMenu(null)
    if (node?.data?.promptId && window.aaaLite?.interruptComfy) {
      try { await window.aaaLite.interruptComfy(endpoint, node.data.promptId) } catch { /* 节点仍可从画布关闭 */ }
    }
    setNodes((items) => items.filter((item) => item.id !== node.id))
    setEdges((items) => items.filter((edge) => edge.source !== node.id && edge.target !== node.id))
    setToast('任务框及其连接线已关闭')
    window.setTimeout(() => setToast(null), 2200)
  }, [endpoint])

  const closeSelectedTaskNodes = useCallback(async (selectedNodes) => {
    const targets = selectedNodes?.length ? selectedNodes : nodes.filter((node) => node.selected)
    const selectedIds = new Set(targets.map((node) => node.id))
    setMenu(null)
    if (!selectedIds.size) return
    if (window.aaaLite?.interruptComfy) {
      await Promise.allSettled(targets.filter((node) => node.data?.promptId).map((node) => window.aaaLite.interruptComfy(endpoint, node.data.promptId)))
    }
    setNodes((items) => items.filter((node) => !selectedIds.has(node.id)))
    setEdges((items) => items.filter((edge) => !selectedIds.has(edge.source) && !selectedIds.has(edge.target)))
    setToast(`已删除 ${selectedIds.size} 个任务框及相关连接线`)
    window.setTimeout(() => setToast(null), 2400)
  }, [endpoint, nodes])

  const closeConnectionLine = useCallback((edge) => {
    setEdges((items) => items.filter((item) => item.id !== edge.id))
    setMenu(null)
    setToast('连接线已删除')
    window.setTimeout(() => setToast(null), 1800)
  }, [])

  const applyTemplate = (key) => {
    const item = templates[key]
    setNodes(structuredClone(item.nodes)); setEdges(structuredClone(item.edges)); setModal(null); setMenu(null)
  }

  const saveDoubaoSettings = async () => {
    const createConnection = seedanceConnectionMode === 'new'
    const result = await window.aaaLite?.saveDoubaoConfig?.({ apiKey: doubaoKey, model: doubaoModel, videoModel: seedanceModel, videoModels: seedanceModels, videoModelLabels: seedanceModelLabels, modelMediaKinds: seedanceModelKinds, videoBaseUrl: seedanceBaseUrl, connectionId: createConnection ? '' : activeSeedanceConnectionId, connectionName: seedanceConnectionName, createConnection })
    if (!result) return
    setToast(result.message)
    if (result.ok) {
      setDoubaoConfig({ configured: result.configured, videoConfigured: result.videoConfigured, model: result.model, videoModel: result.videoModel, videoBaseUrl: result.videoBaseUrl, activeVideoConnectionId: result.activeVideoConnectionId })
      setSeedanceBaseUrl(result.videoBaseUrl)
      setSeedanceConnections(result.videoConnections || [])
      setActiveSeedanceConnectionId(result.activeVideoConnectionId || '')
      setSeedanceConnectionName(result.videoConnectionName || '')
      setSeedanceConnectionMode('saved')
      setSeedanceModels(result.videoModels || [result.videoModel])
      setSeedanceModelLabels((items) => ({ ...items, ...(result.videoModelLabels || {}) }))
      setSeedanceModelKinds(result.modelMediaKinds || {})
      setSeedanceModel(result.videoModel || '')
      setDoubaoKey('')
    }
    window.setTimeout(() => setToast(null), 3200)
  }

  const beginNewSeedanceConnection = useCallback(() => {
    setSeedanceConnectionMode('new')
    setSeedanceConnectionName('')
    setSeedanceBaseUrl('https://ark.cn-beijing.volces.com/api/v3')
    setDoubaoKey('')
    setSeedanceModels([])
    setSeedanceModelLabels({})
    setSeedanceModelKinds({})
    setSeedanceModel('')
  }, [])

  const selectSavedSeedanceConnection = useCallback(async (connectionId) => {
    const result = await window.aaaLite?.activateSeedanceConnection?.(connectionId)
    if (!result) return
    setToast(result.message)
    if (result.ok) {
      setSeedanceConnectionMode('saved')
      setSeedanceConnections(result.videoConnections || [])
      setActiveSeedanceConnectionId(result.activeVideoConnectionId || '')
      setSeedanceConnectionName(result.videoConnectionName || '')
      setSeedanceBaseUrl(result.videoBaseUrl || 'https://ark.cn-beijing.volces.com/api/v3')
      setSeedanceModel(result.videoModel || '')
      setSeedanceModels(result.videoModels || (result.videoModel ? [result.videoModel] : []))
      setSeedanceModelLabels(result.videoModelLabels || {})
      setSeedanceModelKinds(result.modelMediaKinds || {})
      setDoubaoKey('')
      setDoubaoConfig((current) => ({ ...current, videoConfigured: true, videoModel: result.videoModel, videoBaseUrl: result.videoBaseUrl, activeVideoConnectionId: result.activeVideoConnectionId }))
    }
    window.setTimeout(() => setToast(null), 2600)
  }, [])

  const sendMessage = async () => {
    const text = message.trim(); if (!text || doubaoLoading) return
    setMessage('')
    if (!doubaoConfig.configured) {
      setChat((items) => [...items, { role: 'user', text }, { role: 'assistant', text: '请先打开“设置”，填写火山方舟 API Key 和豆包模型 ID / Endpoint ID。密钥会使用 Windows 安全存储加密保存在本机。' }])
      setModal('settings')
      return
    }
    const requestId = `doubao-${Date.now()}`
    const history = [...chat, { role: 'user', text }]
    setChat([...history, { id: requestId, role: 'assistant', text: '豆包正在思考…' }])
    setDoubaoLoading(true)
    const result = await window.aaaLite?.chatWithDoubao?.(history.map((item) => ({ role: item.role, content: item.text })))
    setDoubaoLoading(false)
    setChat((items) => items.map((item) => item.id === requestId ? { ...item, text: result?.ok ? result.content : result?.message || '豆包请求失败，请检查设置。' } : item))
  }

  const createSuggestedFlow = () => {
    applyTemplate(/视频|动效|镜头/.test(chat.at(-2)?.text || '') ? 'video' : 'image')
  }

  const refreshComfyCatalog = useCallback(async (showToast = true) => {
    if (!window.aaaLite?.scanComfyCatalog) return
    setComfyCatalog((catalog) => ({ ...catalog, ok: null, message: '正在识别本地模型…' }))
    const result = await window.aaaLite.scanComfyCatalog(endpoint)
    setComfyCatalog(result)
    if (result.ok) setLocalWorkflows(result.workflows || [])
    if (result.ok) setConnection({ ok: true, message: `ComfyUI 连接正常 · ${result.message}` })
    if (showToast) {
      setToast(result.message)
      window.setTimeout(() => setToast(null), 3200)
    }
    return result
  }, [endpoint])

  useEffect(() => {
    const timer = window.setTimeout(() => refreshComfyCatalog(false), 900)
    return () => window.clearTimeout(timer)
  }, [refreshComfyCatalog])

  const checkConnection = async () => {
    setConnection({ ok: null, message: '正在连接…' })
    localStorage.setItem('aaa-lite-comfy', endpoint)
    if (!window.aaaLite) return setConnection({ ok: false, message: '请在桌面应用中测试连接' })
    const result = await window.aaaLite.checkComfy(endpoint)
    setConnection(result)
    if (result.ok) await refreshComfyCatalog(false)
  }

  const extractFrameForNode = useCallback(async (id, roleOverride = null, explicitVideoUrl = '') => {
    const frameNode = nodes.find((node) => node.id === id)
    if (!frameNode) return false
    const frameRole = roleOverride || frameNode.data.frameRole || '尾帧'
    const incomingVideo = edges.map((edge) => edge.target === id ? nodes.find((node) => node.id === edge.source) : null).find((node) => videoSourceFromData(node?.data))
    const videoUrl = explicitVideoUrl || frameNode.data.frameSourceVideoUrl || videoSourceFromData(incomingVideo?.data)
    if (!videoUrl) {
      setNodes((items) => items.map((node) => node.id === id ? { ...node, data: { ...node.data, kind: 'frame', frameRole, status: `未识别到视频，请连接视频或双击上传${frameRole}图片` } } : node))
      setToast('首尾帧节点需要连接一个视频任务')
      window.setTimeout(() => setToast(null), 2800)
      return false
    }
    const captureKey = `${id}:${frameRole}:${videoUrl}`
    if (frameCaptureInFlightRef.current.has(captureKey)) return false
    frameCaptureInFlightRef.current.add(captureKey)
    setNodes((items) => items.map((node) => node.id === id ? { ...node, data: { ...node.data, kind: 'frame', frameRole, frameSourceVideoUrl: videoUrl, status: `正在自动识别视频${frameRole}` } } : node))
    try {
      const dataUrl = await captureVideoFrame(videoUrl, frameRole)
      const name = `${frameRole}-${Date.now()}.jpg`
      const reference = { url: dataUrl, dataUrl, name, role: frameRole === '首帧' ? 'first_frame' : 'last_frame' }
      setNodes((items) => items.map((node) => node.id === id ? { ...node, data: { ...node.data, kind: 'frame', frameRole, frameSourceVideoUrl: videoUrl, frameCaptureSource: videoUrl, frameCaptureRole: frameRole, resultUrl: null, results: [], promptId: null, url: dataUrl, name, uploadedMediaType: 'image', taskReferences: [reference], taskReferenceUrl: dataUrl, taskReferenceName: name, text: `${frameRole} · 自动提取`, status: `已自动识别视频${frameRole}` } } : node))
      setToast(`已自动提取视频${frameRole}`)
      window.setTimeout(() => setToast(null), 2200)
      return true
    } catch (error) {
      setNodes((items) => items.map((node) => node.id === id ? { ...node, data: { ...node.data, frameCaptureFailedSource: videoUrl, status: `${frameRole}识别失败，可双击上传图片` } } : node))
      setToast(`${frameRole}识别失败：${error.message}`)
      window.setTimeout(() => setToast(null), 3600)
      return false
    } finally {
      frameCaptureInFlightRef.current.delete(captureKey)
    }
  }, [edges, nodes])

  const quickAction = useCallback(async (id, action) => {
    if (action === '复制节点') {
      setNodes((items) => {
        const source = items.find((node) => node.id === id)
        if (!source) return items
        return [...items.map((node) => ({ ...node, selected: false })), { ...structuredClone(source), id: `${source.data.kind}-${Date.now()}`, position: { x: source.position.x + 45, y: source.position.y + 45 }, selected: true, data: { ...structuredClone(source.data), status: '副本就绪' } }]
      })
      setToast('已复制任务框，可独立编辑')
    } else if (action === '联动编辑') {
      setNodes((items) => {
        const source = items.find((node) => node.id === id)
        const linkedIds = new Set([id, ...items.filter((node) => node.selected && mediaKinds.has(node.data.kind)).map((node) => node.id)])
        edges.forEach((edge) => { if (edge.source === id) linkedIds.add(edge.target); if (edge.target === id) linkedIds.add(edge.source) })
        const group = source?.data.linkGroup || `link-${Date.now()}`
        return items.map((node) => linkedIds.has(node.id) && mediaKinds.has(node.data.kind) ? { ...node, data: { ...node.data, linkGroup: group, status: '联动就绪' } } : node)
      })
      setToast('已将选中或直接连接的素材设为联动编辑')
    } else if (action === '设为首帧' || action === '设为尾帧') {
      const frameRole = action === '设为首帧' ? '首帧' : '尾帧'
      const source = nodes.find((node) => node.id === id)
      const sourceVideoUrl = videoSourceFromData(source?.data)
      if (sourceVideoUrl) {
        setNodes((items) => items.map((node) => node.id === id ? { ...node, data: { ...node.data, kind: 'frame', frameRole, frameSourceVideoUrl: sourceVideoUrl, resultUrl: null, results: [], promptId: null, url: null, uploadedMediaType: null, taskReferences: [], taskReferenceUrl: null, taskReferenceName: '', text: `${frameRole} · 自动提取`, status: `正在自动识别视频${frameRole}` } } : node))
        await extractFrameForNode(id, frameRole, sourceVideoUrl)
        return
      }
      const references = (source?.data.taskReferences || []).filter((reference) => !mediaUrlLooksVideo(referenceDisplayUrl(reference)))
      const storedReference = references.find((reference) => referenceDisplayUrl(reference))
      const directImageUrl = [source?.data.resultUrl, source?.data.url, source?.data.taskReferenceUrl].find((url) => mediaUrlLooksImage(url))
      const frameUrl = referenceDisplayUrl(storedReference) || directImageUrl || ''
      const frameReference = frameUrl ? { ...(storedReference || {}), url: frameUrl, dataUrl: String(frameUrl).startsWith('data:image/') ? frameUrl : storedReference?.dataUrl, name: storedReference?.name || source?.data.taskReferenceName || source?.data.name || `${frameRole}图片` } : null
      setNodes((items) => items.map((node) => {
        if (node.id !== id) return node
        return { ...node, data: { ...node.data, kind: 'frame', frameRole, resultUrl: null, results: [], promptId: null, url: frameUrl || null, uploadedMediaType: frameUrl ? 'image' : null, taskReferences: frameReference ? [frameReference, ...references.filter((reference) => referenceDisplayUrl(reference) !== frameUrl)].slice(0, 9) : [], taskReferenceUrl: frameUrl || null, taskReferenceName: frameReference?.name || '', text: `${frameRole} · ${frameReference?.name || '画面参考'}`, status: frameUrl ? `${frameRole}图片就绪` : `请双击上传${frameRole}图片` } }
      }))
      setToast(frameUrl ? `已使用参考图片设为${frameRole}` : `视频不能直接作为${frameRole}图片，请双击任务框上传图片`)
    } else {
      setNodes((items) => items.map((node) => node.id === id ? { ...node, data: { ...node.data, status: `${action}待处理` } } : node))
      setToast(`${action} 已加入当前素材的编辑步骤`)
    }
    window.setTimeout(() => setToast(null), 2400)
  }, [edges, nodes, extractFrameForNode])

  useEffect(() => {
    const nodeById = new Map(nodes.map((node) => [node.id, node]))
    const incomingByTarget = new Map()
    for (const edge of edges) {
      if (!incomingByTarget.has(edge.target)) incomingByTarget.set(edge.target, [])
      incomingByTarget.get(edge.target).push(edge.source)
    }
    for (const frameNode of nodes.filter((node) => node.data.kind === 'frame')) {
      const incomingVideo = (incomingByTarget.get(frameNode.id) || []).map((sourceId) => nodeById.get(sourceId)).find((node) => videoSourceFromData(node?.data))
      const videoUrl = frameNode.data.frameSourceVideoUrl || videoSourceFromData(incomingVideo?.data)
      const frameRole = frameNode.data.frameRole || '尾帧'
      if (!videoUrl || (frameNode.data.frameCaptureSource === videoUrl && frameNode.data.frameCaptureRole === frameRole) || frameNode.data.frameCaptureFailedSource === videoUrl) continue
      void extractFrameForNode(frameNode.id, frameRole, videoUrl)
    }
  }, [edges, extractFrameForNode, nodes])

  const changeTaskConfig = useCallback((id, key, value) => {
    setNodes((items) => {
      const source = items.find((node) => node.id === id)
      return items.map((node) => {
        if (key === 'taskEditorOpen' && value && node.id !== id && (node.data.taskEditorOpen || node.data.taskPickerOpen)) return { ...node, data: { ...node.data, taskEditorOpen: false, taskEditorExpanded: false, taskPickerOpen: false } }
        if (node.id !== id && (localTaskUiKeys.has(key) || !(source?.data.linkGroup && node.data.linkGroup === source.data.linkGroup))) return node
        const nextData = { ...node.data, [key]: value }
        if (key === 'taskReferences') {
          nextData.taskReferenceUrl = value?.[0]?.url || value?.[0]?.dataUrl || null
          nextData.taskReferenceName = value?.[0]?.name || ''
        }
        return { ...node, data: nextData }
      })
    })
  }, [])

  const selectTaskRuntime = useCallback(async (id, type, item) => {
    let selectedItem = item
    if (type === 'workflow' && !item.localPath) {
      const stored = workflowLibrary.find((record) => record.id === item.id && record.workflow)
      if (stored) selectedItem = { ...item, ...stored }
    }
    if (type === 'workflow' && item.localPath) {
      const loaded = workflowLibrary.find((record) => record.id === item.id && record.workflow && record.conversionVersion === LOCAL_WORKFLOW_CACHE_VERSION)
      if (loaded) selectedItem = { ...item, ...loaded }
      else {
        if (!window.aaaLite?.loadLocalComfyWorkflow) return
        setToast(`正在加载本地工作流：${item.name}`)
        const result = await window.aaaLite.loadLocalComfyWorkflow(endpoint, item.localPath)
        if (!result.ok) {
          setToast(result.message)
          window.setTimeout(() => setToast(null), 3800)
          return
        }
        selectedItem = { ...item, workflow: result.workflow, summary: result.summary, inspection: null, source: 'local', conversionVersion: LOCAL_WORKFLOW_CACHE_VERSION, runnable: result.runnable, validationIssues: result.validationIssues || [] }
        setWorkflowLibrary((items) => [...items.filter((record) => record.id !== item.id), selectedItem])
        setToast(result.runnable === false ? `${item.name} 暂不可运行：${(result.validationIssues || []).slice(0, 2).join('；')}` : `已自动加载 ${item.name}${result.convertedFromUi ? ` · 已转换画布格式${result.skippedCount ? ` · 跳过 ${result.skippedCount} 个非执行节点` : ''}${result.repairCount ? ` · 修复 ${result.repairCount} 个参数` : ''}` : ''}`)
        window.setTimeout(() => setToast(null), 3200)
      }
    }
    const modelInputs = selectedItem.modelInputs || [...new Set(Object.values(selectedItem.workflow || {}).flatMap((node) => Object.keys(node.inputs || {})).filter((key) => modelParamPattern.test(key)))]
    const defaultModel = type === 'workflow' ? pickDefaultWorkflowModel(selectedItem.workflow, comfyCatalog) : null
    setNodes((items) => {
      const source = items.find((node) => node.id === id)
      return items.map((node) => {
        const linkedTarget = source?.data.linkGroup && node.data.linkGroup === source.data.linkGroup
        const storyboardVideoTarget = source?.data.kind === 'video' && source.data.storyboardGroup && source.data.storyboardOrder === 1 && node.data.storyboardGroup === source.data.storyboardGroup && node.data.storyboardStage === 'video'
        if (node.id !== id && !linkedTarget && !storyboardVideoTarget) return node
        if (type === 'workflow') {
          const inherited = storyboardVideoTarget && node.id !== id
          return { ...node, data: { ...node.data, taskProvider: 'comfy', taskWorkflowId: selectedItem.id, taskWorkflowName: selectedItem.name, taskWorkflowInputs: modelInputs, taskModel: defaultModel?.name || '', taskModelInput: defaultModel?.input || '', taskModelType: defaultModel?.type || '', taskPickerOpen: false, status: selectedItem.runnable === false ? '工作流不完整，请在 ComfyUI 中修复连线' : inherited ? `已继承镜头 01 · ${selectedItem.name}${defaultModel ? ` · ${defaultModel.name}` : ''}` : defaultModel ? `工作流与模型已自动确认 · ${defaultModel.name}` : '工作流已确认 · 使用内置模型' } }
        }
        return { ...node, data: { ...node.data, taskModel: item.name, taskModelType: item.type, taskModelInput: item.input, taskPickerOpen: false, status: storyboardVideoTarget && node.id !== id ? '已继承镜头 01 的模型' : '模型已选择' } }
      })
    })
    if (type === 'workflow' && selectedItem.runnable !== false) {
      setToast(defaultModel ? `已确认 ${selectedItem.name} · 自动匹配 ${defaultModel.name}` : `已确认 ${selectedItem.name} · 使用工作流内置模型`)
      window.setTimeout(() => setToast(null), 2600)
    }
  }, [endpoint, workflowLibrary, comfyCatalog])

  const selectTaskProvider = useCallback((id, provider) => {
    setNodes((items) => items.map((node) => node.id === id ? { ...node, data: { ...node.data, taskProvider: provider, taskPickerOpen: provider === 'api', status: provider === 'api' ? '请选择已接入的 API 模型' : `本地 ComfyUI ${node.data.kind === 'video' ? '视频' : '图片'}工作流已选择` } } : node))
    setToast(provider === 'api' ? '请在模型列表选择与当前任务类型匹配的 API 模型' : '已选择本地 ComfyUI')
    window.setTimeout(() => setToast(null), 2800)
  }, [])

  const selectSeedanceTaskModel = useCallback((id, entry) => {
    const model = typeof entry === 'string' ? entry : entry.model
    const label = typeof entry === 'string' ? (seedanceModelLabels[model] || model) : entry.label
    const connectionId = typeof entry === 'string' ? activeSeedanceConnectionId : entry.connectionId
    const kind = typeof entry === 'string' ? 'video' : entry.kind
    setNodes((items) => items.map((node) => node.id === id ? { ...node, data: { ...node.data, taskProvider: 'api', taskApiConnectionId: connectionId, taskApiModel: model, taskSeedanceModel: model, taskPickerOpen: false, taskResolution: kind === 'video' && !['480P', '720P'].includes(node.data.taskResolution) ? '720P' : (node.data.taskResolution || (kind === 'image' ? '1K' : '720P')), taskDuration: kind === 'video' ? Math.max(4, Number(node.data.taskDuration || 6)) : node.data.taskDuration, status: `${label} 已选择 · 点击生成时调用 API` } } : node))
    setToast(`任务框模型已切换为：${label}`)
    window.setTimeout(() => setToast(null), 2400)
  }, [activeSeedanceConnectionId, seedanceModelLabels])

  const createStoryboard = useCallback(async () => {
    const shots = splitScriptIntoShots(storyboardScript, Number.MAX_SAFE_INTEGER)
    const storyboardAssets = analyzeStoryboardAssets(storyboardScript, shots)
    if (!shots.length) {
      setToast('请先粘贴剧本内容')
      window.setTimeout(() => setToast(null), 2600)
      return
    }
    setToast('正在识别人物参考图与 MiniMax H3 工作流…')
    const catalog = comfyCatalog.ok ? comfyCatalog : await refreshComfyCatalog(false)
    if (!catalog?.ok) {
      setToast(catalog?.message || 'ComfyUI 未连接，无法建立分镜任务')
      window.setTimeout(() => setToast(null), 4200)
      return
    }
    const merged = [...workflowLibrary, ...(catalog.workflows || [])]
    const seen = new Set()
    const candidates = merged.filter((item) => {
      const key = item.localPath || item.id
      if (!key || seen.has(key)) return false
      seen.add(key)
      return miniMaxWorkflowScore(item) >= 0
    }).sort((a, b) => miniMaxWorkflowScore(b) - miniMaxWorkflowScore(a))
    let runtime = null
    let lastIssue = ''
    for (const candidate of candidates) {
      let loaded = workflowLibrary.find((item) => item.id === candidate.id && item.workflow && item.conversionVersion === LOCAL_WORKFLOW_CACHE_VERSION)
      if (!loaded && candidate.localPath && window.aaaLite?.loadLocalComfyWorkflow) {
        const result = await window.aaaLite.loadLocalComfyWorkflow(endpoint, candidate.localPath)
        if (!result.ok || result.runnable === false) {
          lastIssue = result.message || result.validationIssues?.[0] || `${candidate.name} 不可运行`
          continue
        }
        loaded = { ...candidate, workflow: result.workflow, summary: result.summary, inspection: null, source: 'local', conversionVersion: LOCAL_WORKFLOW_CACHE_VERSION, runnable: true, validationIssues: [] }
      }
      const hasH3Node = loaded?.summary?.classes?.some((name) => /minimax.*h3/i.test(name))
      if (loaded?.workflow && loaded.runnable !== false && hasH3Node) { runtime = loaded; break }
    }
    if (!runtime) {
      setToast(`没有找到可执行的 MiniMax H3 工作流${lastIssue ? `：${lastIssue}` : '，请确认模型与节点已安装'}`)
      window.setTimeout(() => setToast(null), 5200)
      return
    }
    const imageCandidates = merged.filter((item) => imageWorkflowScore(item) >= 0).sort((a, b) => imageWorkflowScore(b) - imageWorkflowScore(a))
    let imageRuntime = null
    let imageIssue = ''
    for (const candidate of imageCandidates) {
      let loaded = workflowLibrary.find((item) => item.id === candidate.id && item.workflow && item.conversionVersion === LOCAL_WORKFLOW_CACHE_VERSION)
      if (!loaded && candidate.localPath && window.aaaLite?.loadLocalComfyWorkflow) {
        const result = await window.aaaLite.loadLocalComfyWorkflow(endpoint, candidate.localPath)
        if (!result.ok || result.runnable === false) { imageIssue = result.message || result.validationIssues?.[0] || `${candidate.name} 不可运行`; continue }
        loaded = { ...candidate, workflow: result.workflow, summary: result.summary, inspection: null, source: 'local', conversionVersion: LOCAL_WORKFLOW_CACHE_VERSION, runnable: true, validationIssues: [] }
      }
      const classes = loaded?.summary?.classes || []
      if (loaded?.workflow && loaded.runnable !== false && classes.some((name) => /SaveImage|PreviewImage/i.test(name)) && !classes.some((name) => /Video|MiniMaxH3/i.test(name))) { imageRuntime = loaded; break }
    }
    if (!imageRuntime) {
      setToast(`没有找到可执行的人物参考图工作流${imageIssue ? `：${imageIssue}` : '，请确认本地存在文生图工作流'}`)
      window.setTimeout(() => setToast(null), 5200)
      return
    }
    const referenceCandidates = [...imageCandidates].sort((a, b) => {
      const score = (item) => /根据图形|图生图|多图|参考|人物|角色|img2img/i.test(`${item.name || ''} ${item.localPath || ''}`) ? 100 : 0
      return score(b) - score(a) || imageWorkflowScore(b) - imageWorkflowScore(a)
    })
    let referenceImageRuntime = null
    let referenceRuntimeScore = -1
    for (const candidate of referenceCandidates) {
      let loaded = workflowLibrary.find((item) => item.id === candidate.id && item.workflow && item.conversionVersion === LOCAL_WORKFLOW_CACHE_VERSION)
      if (!loaded && candidate.localPath && window.aaaLite?.loadLocalComfyWorkflow) {
        const result = await window.aaaLite.loadLocalComfyWorkflow(endpoint, candidate.localPath)
        if (!result.ok || result.runnable === false) continue
        loaded = { ...candidate, workflow: result.workflow, summary: result.summary, inspection: null, source: 'local', conversionVersion: LOCAL_WORKFLOW_CACHE_VERSION, runnable: true, validationIssues: [] }
      }
      const imageInputCount = Object.values(loaded?.workflow || {}).filter((node) => /LoadImage/i.test(node.class_type || '') || Object.entries(node.inputs || {}).some(([key, value]) => /^(image|image_\d+|image\d+|reference_image|start_image)$/i.test(key) && !Array.isArray(value))).length
      const hasPromptInput = Object.values(loaded?.workflow || {}).some((node) => Object.keys(node.inputs || {}).some((key) => /^(text|prompt|positive)$/i.test(key)))
      const nameBonus = /多图|多图片|多参考|人物|角色/i.test(`${candidate.name || ''} ${candidate.localPath || ''}`) ? 20 : 0
      const candidateScore = imageInputCount * 20 + nameBonus
      if (loaded?.workflow && loaded.runnable !== false && imageInputCount && hasPromptInput && candidateScore > referenceRuntimeScore) { referenceImageRuntime = loaded; referenceRuntimeScore = candidateScore }
    }
    referenceImageRuntime ||= imageRuntime
    setWorkflowLibrary((items) => {
      const runtimes = [runtime, imageRuntime, referenceImageRuntime].filter((item, index, list) => list.findIndex((other) => other.id === item.id) === index)
      return [...items.filter((item) => !runtimes.some((runtimeItem) => runtimeItem.id === item.id)), ...runtimes]
    })
    const defaultModel = pickDefaultWorkflowModel(runtime.workflow, catalog)
    const modelInputs = [...new Set(Object.values(runtime.workflow || {}).flatMap((node) => Object.keys(node.inputs || {})).filter((key) => modelParamPattern.test(key)))]
    const imageDefaultModel = pickDefaultWorkflowModel(imageRuntime.workflow, catalog)
    const imageModelInputs = [...new Set(Object.values(imageRuntime.workflow || {}).flatMap((node) => Object.keys(node.inputs || {})).filter((key) => modelParamPattern.test(key)))]
    const referenceImageDefaultModel = pickDefaultWorkflowModel(referenceImageRuntime.workflow, catalog)
    const referenceImageModelInputs = [...new Set(Object.values(referenceImageRuntime.workflow || {}).flatMap((node) => Object.keys(node.inputs || {})).filter((key) => modelParamPattern.test(key)))]
    const group = storyboardEditingGroup || `storyboard-${Date.now()}`
    const rect = canvasRef.current?.getBoundingClientRect()
    const center = flow && rect ? flow.screenToFlowPosition({ x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 }) : { x: 620, y: 360 }
    const startX = center.x - 560
    const startY = center.y - 260
    const scriptId = `script-${group}`
    const scriptNode = {
      id: scriptId, type: 'workflow', position: { x: startX, y: startY }, style: { width: 340, height: 390 },
      data: { kind: 'script', nodeLabel: '剧本智能分镜', text: storyboardScript, storyboardGroup: group, storyboardCount: shots.length, storyboardCharacterCount: storyboardAssets.characters.length, storyboardSceneCount: storyboardAssets.scenes.length, taskStyle: storyboardStyle, taskRatio: storyboardRatio, taskResolution: storyboardResolution, status: `已识别 ${storyboardAssets.characters.length} 个人物 · ${storyboardAssets.scenes.length} 个场景 · ${shots.length} 个镜头` }
    }
    const characterNodes = storyboardAssets.characters.map((character, index) => ({
      id: `character-${group}-${index + 1}`, type: 'workflow', position: { x: startX + 400, y: startY + index * 310 }, style: { width: 290, height: 285 },
      data: { kind: 'image', nodeLabel: `人物图片生成 · ${character.name}`, text: `人物信息 · ${character.name}`, taskPrompt: `人物信息：${character.description}\n任务：仅为人物“${character.name}”制作独立角色设定图，不混入其他人物或场景。输出正面、侧面和关键表情，固定年龄、面部、发型、服装、体型和配色，纯净背景，无文字、标志或水印。`, storyboardCharacterInfo: character.description, storyboardGroup: group, storyboardStage: 'character', storyboardCharacterName: character.name, taskWorkflowId: imageRuntime.id, taskWorkflowName: imageRuntime.name, taskWorkflowInputs: imageModelInputs, taskModel: imageDefaultModel?.name || '', taskModelInput: imageDefaultModel?.input || '', taskModelType: imageDefaultModel?.type || '', taskStyle: storyboardStyle, taskRatio: storyboardRatio, taskResolution: storyboardResolution, taskSeedMode: '随机', taskCount: '1x', status: `人物信息 ${index + 1}/${storyboardAssets.characters.length} · 已独立提取` }
    }))
    const sceneNodes = storyboardAssets.scenes.map((scene, index) => ({
      id: `scene-${group}-${index + 1}`, type: 'workflow', position: { x: startX + 720, y: startY + index * 310 }, style: { width: 290, height: 285 },
      data: { kind: 'image', nodeLabel: `场景图片生成 · ${scene.name}`, text: `场景信息 · ${scene.name}`, taskPrompt: `场景信息：${scene.description}\n任务：仅为场景“${scene.name}”制作独立场景设定图，不混入人物。固定时代、建筑与空间结构、陈设道具、天气、时间和光线，建立连续一致的电影空间，无文字、标志或水印。`, storyboardSceneInfo: scene.description, storyboardGroup: group, storyboardStage: 'scene', storyboardSceneName: scene.name, taskWorkflowId: referenceImageRuntime.id, taskWorkflowName: referenceImageRuntime.name, taskWorkflowInputs: referenceImageModelInputs, taskModel: referenceImageDefaultModel?.name || '', taskModelInput: referenceImageDefaultModel?.input || '', taskModelType: referenceImageDefaultModel?.type || '', taskStyle: storyboardStyle, taskRatio: storyboardRatio, taskResolution: storyboardResolution, taskSeedMode: '随机', taskCount: '1x', status: `场景信息 ${index + 1}/${storyboardAssets.scenes.length} · 已独立提取` }
    }))
    const referenceNodes = shots.map((shot, index) => ({
      id: `reference-${group}-${index + 1}`, type: 'workflow', position: { x: startX + 1040 + (index % 2) * 660, y: startY + Math.floor(index / 2) * 330 }, style: { width: 300, height: 285 },
      data: { kind: 'image', nodeLabel: `${shot.title} · 分镜图片生成 · ${storyboardAssets.shotScenes[index]?.name || '主要场景'}`, text: `镜头提示词 · ${shot.title}`, taskPrompt: `${buildStoryboardShotPrompt(shot, storyboardAssets.shotScenes[index], storyboardAssets.shotCharacters[index])}\n输出要求：生成本镜头的单幅电影分镜关键帧。`, storyboardShotPrompt: buildStoryboardShotPrompt(shot, storyboardAssets.shotScenes[index], storyboardAssets.shotCharacters[index]), storyboardGroup: group, storyboardOrder: index + 1, storyboardStage: 'reference', storyboardTitle: shot.title, storyboardScene: storyboardAssets.shotScenes[index]?.name, storyboardCharacters: storyboardAssets.shotCharacters[index]?.map((item) => item.name) || [], taskWorkflowId: referenceImageRuntime.id, taskWorkflowName: referenceImageRuntime.name, taskWorkflowInputs: referenceImageModelInputs, taskModel: referenceImageDefaultModel?.name || '', taskModelInput: referenceImageDefaultModel?.input || '', taskModelType: referenceImageDefaultModel?.type || '', taskStyle: storyboardStyle, taskRatio: storyboardRatio, taskResolution: storyboardResolution, taskSeedMode: '随机', taskCount: '1x', status: `镜头 ${index + 1}/${shots.length} · 提示词已独立生成` }
    }))
    const videoNodes = shots.map((shot, index) => ({
      id: `video-${group}-${index + 1}`, type: 'workflow', position: { x: startX + 1360 + (index % 2) * 660, y: startY + Math.floor(index / 2) * 330 }, style: { width: 300, height: 285 },
      data: { kind: 'video', nodeLabel: `${shot.title} · 视频生成 · ${storyboardAssets.shotScenes[index]?.name || '主要场景'}`, text: `${shot.title} · ${shot.visual}`, taskPrompt: `${buildStoryboardShotPrompt(shot, storyboardAssets.shotScenes[index], storyboardAssets.shotCharacters[index])}\n镜头时长：约 ${shot.duration} 秒。`, taskDuration: shot.duration, storyboardGroup: group, storyboardOrder: index + 1, storyboardStage: 'video', storyboardTitle: shot.title, storyboardScene: storyboardAssets.shotScenes[index]?.name, storyboardCharacters: storyboardAssets.shotCharacters[index]?.map((item) => item.name) || [], taskWorkflowId: runtime.id, taskWorkflowName: runtime.name, taskWorkflowInputs: modelInputs, taskModel: defaultModel?.name || '', taskModelInput: defaultModel?.input || '', taskModelType: defaultModel?.type || '', taskStyle: storyboardStyle, taskRatio: storyboardRatio, taskResolution: storyboardResolution, taskSeedMode: '随机', taskCount: '1x', status: `等待分镜图 · 独立提示词 · ${shot.duration} 秒` }
    }))
    const shotEdges = [
      ...characterNodes.map((node, index) => ({ id: `edge-${group}-character-${index + 1}`, type: 'atmosphere', source: scriptId, target: node.id, animated: false })),
      ...sceneNodes.map((node, index) => ({ id: `edge-${group}-scene-${index + 1}`, type: 'atmosphere', source: characterNodes[index % characterNodes.length].id, target: node.id, animated: false })),
      ...shots.flatMap((shot, index) => [
      { id: `edge-${group}-reference-${index + 1}`, type: 'atmosphere', source: sceneNodes.find((node) => node.data.storyboardSceneName === storyboardAssets.shotScenes[index]?.name)?.id || sceneNodes[0].id, target: referenceNodes[index].id, animated: false },
      { id: `edge-${group}-video-${index + 1}`, type: 'atmosphere', source: referenceNodes[index].id, target: videoNodes[index].id, animated: false }
    ])]
    setNodes((items) => [...items.filter((node) => node.data.storyboardGroup !== group).map((node) => ({ ...node, selected: false })), scriptNode, ...characterNodes, ...sceneNodes, ...referenceNodes, ...videoNodes])
    setEdges((items) => [...items.filter((edge) => !String(edge.id).includes(group)), ...shotEdges])
    setStoryboardEditingGroup(null)
    setModal(null)
    setToast(`分镜师已识别 ${characterNodes.length} 个人物、${sceneNodes.length} 个场景、${shots.length} 个镜头`)
    window.setTimeout(() => {
      if (shots.length > 30) flow?.setCenter(startX + 760, startY + 320, { zoom: .62, duration: 0 })
      else flow?.fitView({ padding: .12, duration: 420 })
      setToast(null)
    }, 700)
  }, [storyboardScript, storyboardMaxShots, storyboardStyle, storyboardRatio, storyboardResolution, storyboardEditingGroup, comfyCatalog, refreshComfyCatalog, workflowLibrary, endpoint, flow])

  const changePromptText = useCallback((id, value) => {
    setNodes((items) => items.map((node) => node.id === id ? { ...node, data: { ...node.data, text: value } } : node))
  }, [])

  const addTaskReferences = useCallback(async (id, files) => {
    const source = nodes.find((node) => node.id === id)
    const existing = source?.data.taskReferences || []
    const available = Math.max(0, 9 - existing.length)
    const selected = [...(files || [])].filter((file) => file?.type?.startsWith('image/')).slice(0, available)
    if (!selected.length) return
    if (selected.some((file) => file.size > 8 * 1024 * 1024)) {
      setToast('每张参考图片不能超过 8MB')
      window.setTimeout(() => setToast(null), 2600)
      return
    }
    try {
      const stored = await Promise.all(selected.map(storeAssetFile))
      const failed = stored.find((item) => !item?.ok)
      if (failed) throw new Error(failed.message || '素材保存失败')
      const added = stored.map((item, index) => ({ url: item.url, name: selected[index].name }))
      setNodes((items) => items.map((node) => {
        if (node.id !== id) return node
        const current = node.data.taskReferences || []
        const taskReferences = [...current, ...added].slice(0, 9)
        return { ...node, data: { ...node.data, taskReferences, taskReferenceUrl: taskReferences[0]?.url || null, taskReferenceName: taskReferences[0]?.name || '' } }
      }))
      setToast(`已添加 ${added.length} 张参考图片（最多 9 张）`)
    } catch { setToast('参考图片读取失败，请重新选择') }
    window.setTimeout(() => setToast(null), 2600)
  }, [nodes])

  const uploadMediaPreview = useCallback(async (id, file) => {
    const isVideoFile = file?.type?.startsWith('video/')
    const isImageFile = file?.type?.startsWith('image/')
    if (!isImageFile && !isVideoFile) return
    const maxBytes = isVideoFile ? 50 * 1024 * 1024 : 12 * 1024 * 1024
    if (file.size > maxBytes) {
      setToast(isVideoFile ? '视频超过 50MB，请压缩后再上传' : '图片超过 12MB，请压缩后再上传')
      window.setTimeout(() => setToast(null), 2800)
      return
    }
    try {
      const stored = await storeAssetFile(file)
      if (!stored?.ok) throw new Error(stored?.message || '素材保存失败')
      setNodes((items) => items.map((node) => {
        if (node.id !== id) return node
        if (isVideoFile) return { ...node, data: { ...node.data, url: stored.url, name: file.name, text: node.data.text || file.name, uploadedMediaType: 'video', status: '已识别并加载视频素材' } }
        const reference = { url: stored.url, name: file.name }
        const taskReferences = [reference, ...(node.data.taskReferences || [])].slice(0, 9)
        return { ...node, data: { ...node.data, url: stored.url, name: file.name, text: node.data.text || file.name, uploadedMediaType: 'image', taskReferences, taskReferenceUrl: stored.url, taskReferenceName: file.name, status: '已识别图片并设为首张参考图' } }
      }))
      setToast(`${isVideoFile ? '视频' : '图片'}素材已自动识别：${file.name}`)
      window.setTimeout(() => setToast(null), 2400)
    } catch (error) {
      setToast(`${isVideoFile ? '视频' : '图片'}保存失败：${error.message}`)
      window.setTimeout(() => setToast(null), 2600)
    }
  }, [])

  const runTaskEditor = useCallback(async (id, overrides = {}) => {
    const node = nodes.find((item) => item.id === id)
    let taskData = { ...(node?.data || {}), ...overrides }
    const upstreamIds = upstreamNodeIds(id, edges)
    const connectedPrompt = nodes.filter((item) => upstreamIds.has(item.id) && item.data.kind === 'prompt' && item.data.text?.trim()).map((item) => item.data.text.trim()).join('\n')
    if (connectedPrompt) taskData = { ...taskData, taskPrompt: connectedPrompt }
    const connectedReferences = nodes.filter((item) => upstreamIds.has(item.id)).flatMap(nodeImageReferences)
    const explicitReferences = taskData.taskReferences?.length ? taskData.taskReferences : taskData.taskReferenceUrl ? [{ url: taskData.taskReferenceUrl, dataUrl: String(taskData.taskReferenceUrl).startsWith('data:') ? taskData.taskReferenceUrl : undefined, name: taskData.taskReferenceName || '参考图片' }] : []
    const resolvedReferences = orderedFrameReferences(mergeImageReferences(explicitReferences, connectedReferences))
    if (resolvedReferences.length) taskData = { ...taskData, taskReferences: resolvedReferences, taskReferenceUrl: resolvedReferences[0].url || resolvedReferences[0].dataUrl, taskReferenceName: resolvedReferences[0].name || '参考图片' }
    const isApiTask = taskData.taskProvider === 'api' || taskData.taskProvider === 'seedance'
    if (taskData.kind === 'image' && isApiTask) {
      const model = String(taskData.taskApiModel || taskData.taskSeedanceModel || '').trim()
      const prompt = taskData.taskPrompt || taskData.text || connectedPrompt
      if (!model || !prompt || !window.aaaLite?.createApiImage) {
        setToast(!model ? '请先选择已接入的图片 API 模型' : '图片 API 任务需要提示词')
        window.setTimeout(() => setToast(null), 3200)
        return
      }
      const dimensions = taskDimensions(taskData.taskResolution || '1K', taskData.taskRatio || '1:1')
      const label = availableApiModels.find((entry) => entry.model === model && entry.connectionId === taskData.taskApiConnectionId)?.label || model
      setNodes((items) => items.map((item) => item.id === id ? { ...item, data: { ...item.data, ...taskData, submittedByApp: true, submittedPrompt: prompt, status: `正在调用 ${label}`, progress: 15, resultUrl: null, results: [], workflowName: label } } : item))
      const result = await window.aaaLite.createApiImage({ connectionId: taskData.taskApiConnectionId || activeSeedanceConnectionId, model, prompt, size: `${dimensions.width}x${dimensions.height}` })
      const media = result?.ok ? [{ kind: 'images', url: result.imageUrl, remoteUrl: result.imageUrl, filename: 'api-image.png' }] : []
      setNodes((items) => items.map((item) => item.id === id ? { ...item, data: { ...item.data, status: result?.ok ? '已完成' : result?.message || '图片 API 生成失败', progress: result?.ok ? 100 : 0, resultUrl: result?.imageUrl || null, results: media } } : item))
      if (result?.ok) setSessionTasks((items) => trimSessionTasks([{ id: `api-image-${Date.now()}`, provider: 'api-image', nodeId: id, kind: 'image', title: semanticNodeTitle(taskData, '图片生成'), workflowName: label, status: '已完成', progress: 100, media, createdAt: Date.now(), updatedAt: Date.now() }, ...items]))
      else { setToast(result?.message || '图片 API 生成失败'); window.setTimeout(() => setToast(null), 3600) }
      return
    }
    if (taskData.kind === 'video' && isApiTask) {
      if (!window.aaaLite?.createSeedanceTask || !window.aaaLite?.getSeedanceTask) return
      const taskSeedanceModel = String(taskData.taskApiModel || taskData.taskSeedanceModel || seedanceModel || '').trim()
      const taskSeedanceLabel = availableApiModels.find((entry) => entry.model === taskSeedanceModel && entry.connectionId === taskData.taskApiConnectionId)?.label || seedanceModelLabels[taskSeedanceModel] || taskSeedanceModel || '视频 API'
      if (!taskSeedanceModel) {
        setToast('请先为任务框选择一个已接入的 Seedance 模型')
        window.setTimeout(() => setToast(null), 3200)
        return
      }
      const prompt = taskData.taskPrompt || taskData.text || ''
      const duration = Math.max(4, Math.min(15, Number(taskData.taskDuration || 6)))
      const resolution = ['480P', '720P'].includes(taskData.taskResolution) ? taskData.taskResolution : '720P'
      const assetValue = String(taskData.taskSeedanceAssetId || '').trim()
      const trustedAssetUrl = assetValue ? (assetValue.startsWith('asset://') ? assetValue : `asset://${assetValue}`) : ''
      const references = trustedAssetUrl ? [{ url: trustedAssetUrl, name: '火山方舟可信人物素材' }] : resolvedReferences
      setNodes((items) => items.map((item) => item.id === id ? { ...item, data: { ...item.data, ...taskData, taskSeedanceModel, taskDuration: duration, taskResolution: resolution, submittedByApp: true, submittedPrompt: prompt, status: `正在提交 ${taskSeedanceLabel}`, progress: 2, resultUrl: null, results: [], workflowName: taskSeedanceLabel } } : item))
      setTaskOpen(true)
      const result = await window.aaaLite.createSeedanceTask({ connectionId: taskData.taskApiConnectionId || activeSeedanceConnectionId, model: taskSeedanceModel, prompt, references, ratio: taskData.taskRatio || '16:9', resolution: resolution.toLowerCase(), duration, generateAudio: true, watermark: false })
      if (!result.ok) {
        setNodes((items) => items.map((item) => item.id === id ? { ...item, data: { ...item.data, status: result.message, progress: 0 } } : item))
        setToast(result.message)
        window.setTimeout(() => setToast(null), 4200)
        return
      }
      const sessionRecord = { id: result.taskId, promptId: result.taskId, provider: 'seedance', connectionId: result.connectionId || activeSeedanceConnectionId, nodeId: id, kind: 'video', title: semanticNodeTitle(taskData, '视频生成'), workflowName: taskSeedanceLabel, status: '排队中', progress: 5, media: [], createdAt: Date.now(), updatedAt: Date.now() }
      setNodes((items) => items.map((item) => item.id === id ? { ...item, data: { ...item.data, promptId: result.taskId, status: '排队中', progress: 5 } } : item))
      setSessionTasks((items) => trimSessionTasks([sessionRecord, ...items.filter((item) => item.promptId !== result.taskId)]))
      setToast('Seedance 任务已提交，正在云端生成')
      window.setTimeout(() => setToast(null), 3000)
      void (async () => {
        let consecutiveErrors = 0
        for (let attempt = 0; attempt < 600; attempt += 1) {
          await new Promise((resolve) => window.setTimeout(resolve, 3000))
          if (cancelledPromptRef.current.has(result.taskId)) return
          const statusResult = await window.aaaLite.getSeedanceTask(result.taskId, result.connectionId || activeSeedanceConnectionId)
          if (!statusResult.ok) {
            consecutiveErrors += 1
            if (consecutiveErrors < 3) continue
            const failedStatus = statusResult.message || 'Seedance 状态查询失败'
            setSessionTasks((items) => items.map((item) => item.promptId === result.taskId ? { ...item, status: `失败：${failedStatus}`, progress: 0, updatedAt: Date.now() } : item))
            setNodes((items) => items.map((item) => item.data.promptId === result.taskId ? { ...item, data: { ...item.data, status: `失败：${failedStatus}`, progress: 0 } } : item))
            return
          }
          consecutiveErrors = 0
          if (statusResult.status === 'succeeded' && statusResult.videoUrl) {
            const media = [{ kind: 'videos', url: statusResult.videoUrl, remoteUrl: statusResult.videoUrl, filename: 'seedance.mp4' }]
            setSessionTasks((items) => items.map((item) => item.promptId === result.taskId ? { ...item, status: '已完成', progress: 100, media, updatedAt: Date.now() } : item))
            setNodes((items) => items.map((item) => item.data.promptId === result.taskId ? { ...item, data: { ...item.data, status: '已完成', progress: 100, resultUrl: statusResult.videoUrl, results: media } } : item))
            return
          }
          if (['failed', 'cancelled', 'expired'].includes(statusResult.status)) {
            const terminalStatus = statusResult.status === 'cancelled' ? '已取消' : `失败：${statusResult.error || (statusResult.status === 'expired' ? '任务已过期' : '云端生成失败')}`
            setSessionTasks((items) => items.map((item) => item.promptId === result.taskId ? { ...item, status: terminalStatus, progress: 0, updatedAt: Date.now() } : item))
            setNodes((items) => items.map((item) => item.data.promptId === result.taskId ? { ...item, data: { ...item.data, status: terminalStatus, progress: 0 } } : item))
            return
          }
          const running = statusResult.status === 'running'
          setSessionTasks((items) => items.map((item) => item.promptId === result.taskId ? { ...item, status: running ? '生成中' : '排队中', progress: running ? Math.min(90, Math.max(15, item.progress + 2)) : 5, updatedAt: Date.now() } : item))
          setNodes((items) => items.map((item) => item.data.promptId === result.taskId ? { ...item, data: { ...item.data, status: running ? '生成中' : '排队中', progress: running ? Math.min(90, Math.max(15, item.data.progress + 2)) : 5 } } : item))
        }
      })()
      return
    }
    let record = workflowLibrary.find((item) => item.id === taskData.taskWorkflowId)
    if (record?.source === 'local' && record.conversionVersion !== LOCAL_WORKFLOW_CACHE_VERSION && record.localPath && window.aaaLite?.loadLocalComfyWorkflow) {
      setToast(`正在更新本地工作流：${record.name}`)
      const refreshed = await window.aaaLite.loadLocalComfyWorkflow(endpoint, record.localPath)
      if (!refreshed.ok) {
        setToast(refreshed.message)
        window.setTimeout(() => setToast(null), 3800)
        return
      }
      record = { ...record, workflow: refreshed.workflow, summary: refreshed.summary, conversionVersion: LOCAL_WORKFLOW_CACHE_VERSION, runnable: refreshed.runnable, validationIssues: refreshed.validationIssues || [] }
      setWorkflowLibrary((items) => [...items.filter((item) => item.id !== record.id), record])
      const defaultModel = pickDefaultWorkflowModel(record.workflow, comfyCatalog)
      if (defaultModel) taskData = { ...taskData, taskModel: defaultModel.name, taskModelInput: defaultModel.input, taskModelType: defaultModel.type }
    }
    if (!record) {
      setNodes((items) => items.map((item) => item.id === id ? { ...item, data: { ...item.data, taskPickerOpen: true, status: '请先选择工作流' } } : item))
      setToast('请先在双栏面板中选择已导入的 ComfyUI 工作流')
      window.setTimeout(() => setToast(null), 3200)
      return
    }
    if (record.runnable === false) {
      const detail = (record.validationIssues || []).slice(0, 3).join('；')
      setNodes((items) => items.map((item) => item.id === id ? { ...item, data: { ...item.data, status: '工作流不完整，未提交任务', progress: 0 } } : item))
      setToast(`该工作流暂不可运行：${detail || '存在断开的必要连线'}`)
      window.setTimeout(() => setToast(null), 4800)
      return
    }
    if (!window.aaaLite?.runComfyWorkflow) return
    let taskReferences = taskData.taskReferences?.length ? taskData.taskReferences.slice(0, 9) : taskData.taskReferenceUrl ? [{ url: taskData.taskReferenceUrl, name: taskData.taskReferenceName || '参考图片' }] : []
    if (taskReferences.length && window.aaaLite?.uploadComfyReference) {
      const uploaded = []
      for (const reference of taskReferences) {
        if (reference.comfyName) { uploaded.push(reference); continue }
        const result = await window.aaaLite.uploadComfyReference(endpoint, {
          name: reference.name,
          dataUrl: reference.dataUrl || (String(reference.url || '').startsWith('data:') ? reference.url : undefined),
          url: /^(?:https?|freedom-asset):/i.test(String(reference.remoteUrl || reference.url || '')) ? (reference.remoteUrl || reference.url) : undefined
        })
        if (!result.ok) {
          setToast(result.message)
          window.setTimeout(() => setToast(null), 3800)
          return
        }
        uploaded.push({ ...reference, comfyName: result.subfolder ? `${result.subfolder}/${result.name}` : result.name })
      }
      taskReferences = uploaded
      taskData = { ...taskData, taskReferences, taskReferenceUrl: taskReferences[0]?.url || null, taskReferenceName: taskReferences[0]?.name || '' }
    }
    const workflow = structuredClone(record.workflow)
    const requiredFrameRoles = h3FrameInputRoles(workflow)
    if (requiredFrameRoles.has('first_frame') && requiredFrameRoles.has('last_frame')) {
      if (!taskReferences.some((reference) => reference.role === 'first_frame') && !taskReferences.some((reference) => reference.role === 'last_frame') && taskReferences.length >= 2) {
        taskReferences = taskReferences.map((reference, index) => index === 0 ? { ...reference, role: 'first_frame' } : index === 1 ? { ...reference, role: 'last_frame' } : reference)
      }
      const firstFrame = taskReferences.find((reference) => reference.role === 'first_frame')
      const lastFrame = taskReferences.find((reference) => reference.role === 'last_frame')
      if (!firstFrame || !lastFrame) {
        const missing = [!firstFrame ? '首帧' : '', !lastFrame ? '尾帧' : ''].filter(Boolean).join('和')
        setNodes((items) => items.map((item) => item.id === id ? { ...item, data: { ...item.data, status: `缺少${missing}，已阻止误提交`, progress: 0 } } : item))
        setToast(`首尾帧工作流必须连接${missing}图片后才能生成`)
        window.setTimeout(() => setToast(null), 4200)
        return
      }
    }
    let prompt = taskData.taskPrompt || taskData.text || ''
    const tailReference = taskReferences.find((reference) => reference.role === 'last_frame')
    if (requiredFrameRoles.has('last_frame') && tailReference) prompt = appendForcedTailFramePrompt(prompt, tailReference, Math.max(1, Math.min(15, Number(taskData.taskDuration || 6))))
    injectPositivePrompt(workflow, prompt)
    const dimensions = taskDimensions(taskData.taskResolution, taskData.taskRatio)
    const duration = Math.max(1, Math.min(15, Number(taskData.taskDuration || 6)))
    const workflowFps = Number(Object.values(workflow).map((item) => Object.entries(item.inputs || {}).find(([key, value]) => /^(fps|frame_rate)$/i.test(key) && Number(value) > 0)?.[1]).find(Boolean) || 24)
    const durationFrames = Math.max(5, Math.floor((duration * workflowFps) / 4) * 4 + 1)
    const h3DurationFrames = Math.max(5, Math.ceil((duration * 24 - 5) / 17) * 17 + 5)
    const frameBinding = bindH3FrameInputs(workflow, taskReferences)
    if (requiredFrameRoles.size && [...requiredFrameRoles].some((role) => !frameBinding.boundRoles.has(role))) {
      setNodes((items) => items.map((item) => item.id === id ? { ...item, data: { ...item.data, status: '首尾帧输入绑定失败，已阻止误提交', progress: 0 } } : item))
      setToast('工作流的首尾帧输入无法自动绑定，请检查首尾帧 LoadImage 连线')
      window.setTimeout(() => setToast(null), 4600)
      return
    }
    const genericTaskReferences = requiredFrameRoles.size ? taskReferences.filter((reference) => !['first_frame', 'last_frame'].includes(reference.role)) : taskReferences
    let imageReferenceCursor = 0
    let payloadRepairCount = 0
    for (const [workflowNodeId, workflowNode] of Object.entries(workflow)) {
      if (taskData.storyboardGroup && !taskReferences.length && /MiniMaxH3.*ToVideo/i.test(workflowNode.class_type || '')) {
        delete workflowNode.inputs.first_frame
        delete workflowNode.inputs.last_frame
      }
      for (const key of Object.keys(workflowNode.inputs || {})) {
        if (genericTaskReferences.length && !frameBinding.boundLoaders.has(workflowNodeId) && /^(image|image_\d+|image\d+|reference_image|start_image)$/i.test(key) && !Array.isArray(workflowNode.inputs[key])) {
          workflowNode.inputs[key] = genericTaskReferences[imageReferenceCursor % genericTaskReferences.length].comfyName
          imageReferenceCursor += 1
        }
        if (modelParamPattern.test(key) && taskData.taskModel && (!taskData.taskModelInput || taskData.taskModelInput.toLowerCase() === key.toLowerCase())) {
          const choices = Object.entries(comfyCatalog.optionsByInput || {}).find(([name]) => name.toLowerCase() === key.toLowerCase())?.[1] || []
          if (!choices.length || choices.includes(taskData.taskModel)) workflowNode.inputs[key] = taskData.taskModel
        }
        if (/^width$/i.test(key)) workflowNode.inputs[key] = dimensions.width
        if (/^height$/i.test(key)) workflowNode.inputs[key] = dimensions.height
        if (taskData.kind === 'video' && /^(duration|seconds|video_seconds|duration_seconds|length_seconds)$/i.test(key) && !Array.isArray(workflowNode.inputs[key])) workflowNode.inputs[key] = duration
        if (taskData.kind === 'video' && /^(num_frames|frame_count|video_frames)$/i.test(key) && !Array.isArray(workflowNode.inputs[key])) workflowNode.inputs[key] = durationFrames
        if (taskData.kind === 'video' && /^length$/i.test(key) && /MiniMaxH3.*ToVideo/i.test(workflowNode.class_type || '') && !Array.isArray(workflowNode.inputs[key])) workflowNode.inputs[key] = h3DurationFrames
        if (/^(seed|noise_seed)$/i.test(key) && taskData.taskSeedMode !== '固定') workflowNode.inputs[key] = Math.floor(Math.random() * 2147483647)
      }
      payloadRepairCount += synchronizeDirectorSecondSamplePayload(workflowNode)
    }
    setNodes((items) => items.map((item) => item.id === id ? { ...item, data: { ...item.data, ...taskData, submittedByApp: true, submittedPrompt: prompt, status: '排队中', progress: 2, resultUrl: null, results: [], workflowName: record.name } } : item))
    setTaskOpen(true)
    const result = await window.aaaLite.runComfyWorkflow(endpoint, workflow)
    setNodes((items) => items.map((item) => item.id === id ? { ...item, data: { ...item.data, status: result.ok ? '排队中' : result.message, promptId: result.promptId || null, progress: result.ok ? 3 : 0 } } : item))
    if (result.ok && result.promptId) {
    const sessionRecord = { id: result.promptId, promptId: result.promptId, nodeId: id, kind: taskData.kind || 'image', title: semanticNodeTitle(taskData, nodeCatalog[taskData.kind]?.title || '生成任务'), workflowName: record.name, status: '排队中', progress: 3, media: [], createdAt: Date.now(), updatedAt: Date.now() }
      taskRunMetaRef.current.set(result.promptId, sessionRecord)
      setSessionTasks((items) => {
        const existing = items.find((item) => item.promptId === result.promptId)
        return trimSessionTasks(existing ? items.map((item) => item.promptId === result.promptId ? { ...sessionRecord, ...item, nodeId: id, kind: sessionRecord.kind, title: sessionRecord.title, workflowName: sessionRecord.workflowName } : item) : [sessionRecord, ...items])
      })
    }
    const totalRepairCount = Number(result.repairCount || 0) + payloadRepairCount
    setToast(result.ok ? `已提交：${record.name}${taskData.taskModel ? ` · ${taskData.taskModel}` : ''}${totalRepairCount ? ` · 自动修正 ${totalRepairCount} 个参数` : ''}` : result.message)
    window.setTimeout(() => setToast(null), 3500)
  }, [nodes, edges, workflowLibrary, comfyCatalog, endpoint, seedanceModel, seedanceModelLabels, activeSeedanceConnectionId, availableApiModels])

  const runStoryboardAll = useCallback(async (group) => {
    const characters = nodes.filter((node) => node.data.storyboardGroup === group && node.data.storyboardStage === 'character')
    const shots = nodes.filter((node) => node.data.storyboardGroup === group && node.data.storyboardStage === 'reference')
    if (!characters.length || !shots.length) return
    setToast(`分镜师开始工作：${characters.length} 个人物 → 场景 → ${shots.length} 个分镜 → 顺序视频`)
    for (const character of characters) await runTaskEditor(character.id, { autoRunPipeline: true })
    window.setTimeout(() => setToast(null), 3600)
  }, [nodes, runTaskEditor])

  useEffect(() => {
    const target = nodes.find((node) => node.data.autoRunRequested && !node.data.autoRunHandled && ['scene', 'reference', 'video'].includes(node.data.storyboardStage))
    if (!target) return
    setNodes((items) => items.map((node) => node.id === target.id ? { ...node, data: { ...node.data, autoRunHandled: true } } : node))
    window.setTimeout(() => runTaskEditor(target.id), 180)
  }, [nodes, runTaskEditor])

  const editStoryboard = useCallback((id) => {
    const source = nodes.find((node) => node.id === id)
    if (!source) return
    setStoryboardScript(source.data.text || '')
    setStoryboardStyle(source.data.taskStyle || '电影写实')
    setStoryboardRatio(source.data.taskRatio || '16:9')
    setStoryboardResolution(source.data.taskResolution || '720P')
    setStoryboardMaxShots(source.data.storyboardCount || 8)
    setStoryboardEditingGroup(source.data.storyboardGroup || null)
    setModal('storyboard')
  }, [nodes])

  const changeComfyParam = useCallback((id, paramId, value) => {
    setNodes((items) => items.map((node) => node.id === id ? { ...node, data: { ...node.data, params: (node.data.params || []).map((param) => param.id === paramId ? { ...param, value } : param) } } : node))
  }, [])

  const uploadComfyVideo = useCallback(async (id, paramId, reference = null) => {
    if (!window.aaaLite?.uploadComfyVideo) return
    setNodes((items) => items.map((node) => node.id === id ? { ...node, data: { ...node.data, status: reference ? '正在同步上游视频' : '正在上传视频' } } : node))
    const result = await window.aaaLite.uploadComfyVideo(endpoint, reference)
    if (!result || result.canceled) {
      setNodes((items) => items.map((node) => node.id === id ? { ...node, data: { ...node.data, status: '就绪 · 请选择输入视频' } } : node))
      return null
    }
    if (!result.ok) {
      setNodes((items) => items.map((node) => node.id === id ? { ...node, data: { ...node.data, status: result.message } } : node))
      setToast(result.message)
      window.setTimeout(() => setToast(null), 3500)
      return null
    }
    const value = result.subfolder ? `${result.subfolder}/${result.name}` : result.name
    changeComfyParam(id, paramId, value)
    setNodes((items) => items.map((node) => node.id === id ? { ...node, data: { ...node.data, status: '输入视频已就绪' } } : node))
    return value
  }, [changeComfyParam, endpoint])

  const inspectComfyNode = useCallback(async (id) => {
    const node = nodes.find((item) => item.id === id)
    const record = workflowLibrary.find((item) => item.id === node?.data.workflowId) || Object.values(BUILTIN_VIDEO_TOOL_RECORDS).find((item) => item.id === node?.data.workflowId)
    if (!record || !window.aaaLite?.inspectComfyWorkflow) return
    setNodes((items) => items.map((item) => item.id === id ? { ...item, data: { ...item.data, status: '正在检查依赖' } } : item))
    const result = await window.aaaLite.inspectComfyWorkflow(endpoint, record.workflow)
    setNodes((items) => items.map((item) => item.id === id ? { ...item, data: { ...item.data, status: result.ok ? (result.missing.length ? `缺少 ${result.missing.length} 个节点` : '依赖完整') : result.message, missingCount: result.missing?.length || 0, managerAvailable: result.managerAvailable } } : item))
    setWorkflowLibrary((items) => items.map((item) => item.id === record.id ? { ...item, inspection: result } : item))
    const missingText = result.missingPackages?.map((item) => item.title || item.package || item.classType).join('、')
    setToast(result.ok ? (result.missing.length ? `缺少插件：${missingText || result.missing.join('、')}` : `依赖完整，检测到 ${result.installedCount} 类节点`) : result.message)
    window.setTimeout(() => setToast(null), 3500)
  }, [endpoint, nodes, workflowLibrary])

  const runComfyNode = useCallback(async (id, parameterOverrides = null, promptOverride = '') => {
    const node = nodes.find((item) => item.id === id)
    const record = workflowLibrary.find((item) => item.id === node?.data.workflowId) || Object.values(BUILTIN_VIDEO_TOOL_RECORDS).find((item) => item.id === node?.data.workflowId)
    if (!record || !window.aaaLite?.runComfyWorkflow) return
    const workflow = structuredClone(record.workflow)
    const runtimeParams = structuredClone(parameterOverrides || node.data.params || [])
    const videoParam = runtimeParams.find((param) => /^(video|file)$/i.test(param.key))
    if (node.data.builtinVideoTool && videoParam && !String(videoParam.value || '').trim()) {
      const upstreamIds = upstreamNodeIds(id, edges)
      const source = nodes.find((item) => upstreamIds.has(item.id) && videoSourceFromData(item.data))
      const media = source?.data.results?.find((item) => mediaIsVideo(item))
      const sourceUrl = source && videoSourceFromData(source.data)
      if (sourceUrl) videoParam.value = await uploadComfyVideo(id, videoParam.id, { url: sourceUrl, remoteUrl: media?.remoteUrl, name: source.data.name || media?.filename || '上游视频.mp4' })
    }
    if (videoParam && !String(videoParam.value || '').trim()) {
      const message = '请先选择输入视频，或把一个已有视频任务框连接到当前任务框'
      setNodes((items) => items.map((item) => item.id === id ? { ...item, data: { ...item.data, status: message } } : item))
      setToast(message)
      window.setTimeout(() => setToast(null), 3500)
      return
    }
    for (const param of runtimeParams) {
      if (!workflow[param.nodeId]?.inputs) continue
      const value = param.valueType === 'number' ? Number(param.value) : param.value
      workflow[param.nodeId].inputs[param.key] = Number.isNaN(value) ? param.value : value
    }
    injectPositivePrompt(workflow, promptOverride)
    setNodes((items) => items.map((item) => item.id === id ? { ...item, data: { ...item.data, submittedByApp: true, submittedPrompt: promptOverride || item.data.submittedPrompt || '', status: '排队中', progress: 2, resultUrl: null, results: [] } } : item))
    const result = await window.aaaLite.runComfyWorkflow(endpoint, workflow)
    setNodes((items) => items.map((item) => item.id === id ? { ...item, data: { ...item.data, status: result.ok ? '排队中' : result.message, promptId: result.promptId || null, progress: result.ok ? 3 : 0 } } : item))
    if (result.ok && result.promptId) {
      const taskRecord = { id: result.promptId, promptId: result.promptId, nodeId: id, kind: workflowMediaKind(record), title: record.name || 'ComfyUI 工作流', workflowName: record.name, status: '排队中', progress: 3, media: [], createdAt: Date.now(), updatedAt: Date.now() }
      taskRunMetaRef.current.set(result.promptId, taskRecord)
      setSessionTasks((items) => {
        const existing = items.find((item) => item.promptId === result.promptId)
        return trimSessionTasks(existing ? items.map((item) => item.promptId === result.promptId ? { ...taskRecord, ...item, nodeId: id, kind: taskRecord.kind, title: taskRecord.title, workflowName: taskRecord.workflowName } : item) : [taskRecord, ...items])
      })
      setTaskOpen(true)
    }
    if (!result.ok) { setToast(result.message); window.setTimeout(() => setToast(null), 3500) }
  }, [edges, endpoint, nodes, uploadComfyVideo, workflowLibrary])

  const oneClickGenerate = useCallback(async () => {
    const target = nodes.find((node) => node.selected && (node.data.kind === 'comfy' || mediaKinds.has(node.data.kind)))
      || nodes.find((node) => mediaKinds.has(node.data.kind) && (['api', 'seedance'].includes(node.data.taskProvider) || node.data.taskWorkflowId))
      || nodes.find((node) => node.data.kind === 'comfy')
    if (!target) {
      setToast('请先选中一个图片、视频或 ComfyUI 工作流任务框')
      window.setTimeout(() => setToast(null), 3000)
      return
    }
    if (['排队中', '生成中'].includes(target.data.status)) {
      setToast('当前任务正在运行，请等待完成或先停止任务')
      window.setTimeout(() => setToast(null), 2600)
      return
    }
    const upstreamIds = new Set()
    const queue = [target.id]
    while (queue.length) {
      const current = queue.shift()
      for (const edge of edges.filter((item) => item.target === current)) {
        if (upstreamIds.has(edge.source)) continue
        upstreamIds.add(edge.source)
        queue.push(edge.source)
      }
    }
    const upstream = nodes.filter((node) => upstreamIds.has(node.id))
    const promptText = upstream.filter((node) => node.data.kind === 'prompt' && node.data.text?.trim()).map((node) => node.data.text.trim()).join('\n')
    const reference = upstream.find((node) => mediaKinds.has(node.data.kind) && node.data.url)
    setNodes((items) => items.map((node) => node.id === target.id ? { ...node, selected: true } : { ...node, selected: false }))
    if (target.data.kind === 'comfy') {
      const params = target.data.params || []
      setNodes((items) => items.map((node) => node.id === target.id ? { ...node, data: { ...node.data, status: promptText ? '已同步连接提示词' : node.data.status } } : node))
      await runComfyNode(target.id, params, promptText)
      return
    }
    const synced = {
      taskPrompt: promptText || target.data.taskPrompt?.trim() || target.data.text || '',
      taskProvider: ['api', 'seedance'].includes(target.data.taskProvider) ? target.data.taskProvider : 'comfy',
      ...(reference ? { taskReferenceUrl: target.data.taskReferenceUrl || reference.data.url, taskReferenceName: target.data.taskReferenceName || reference.data.name || '已连接参考图' } : {})
    }
    await runTaskEditor(target.id, synced)
  }, [nodes, edges, runComfyNode, runTaskEditor])

  const stopComfyNode = useCallback(async (id) => {
    const node = nodes.find((item) => item.id === id)
    if (!node?.data.promptId || !window.aaaLite?.interruptComfy) return
    const result = await window.aaaLite.interruptComfy(endpoint, node.data.promptId)
    setNodes((items) => items.map((item) => item.id === id ? { ...item, data: { ...item.data, status: result.ok ? '已停止' : result.message, progress: 0 } } : item))
    setSessionTasks((items) => items.map((item) => item.promptId === node.data.promptId ? { ...item, status: result.ok ? '已停止' : result.message, progress: 0, updatedAt: Date.now() } : item))
  }, [endpoint, nodes])

  const cancelTask = useCallback(async (task) => {
    setTaskMenu(null)
    if (!task?.promptId || !cancelableTaskStatuses.has(task.status)) {
      setToast('该任务已经结束，不需要取消')
      window.setTimeout(() => setToast(null), 2600)
      return
    }
    setSessionTasks((items) => items.map((item) => item.promptId === task.promptId ? { ...item, status: '正在取消…', updatedAt: Date.now() } : item))
    const result = task.provider === 'seedance' ? await window.aaaLite?.cancelSeedanceTask?.(task.promptId, task.connectionId) : await window.aaaLite?.cancelComfyTask?.(endpoint, task.promptId)
    if (!result) return
    if (result.ok) cancelledPromptRef.current.add(task.promptId)
    const status = result.ok ? '已取消' : result.inactive ? '已结束' : result.message
    setSessionTasks((items) => items.map((item) => item.promptId === task.promptId ? { ...item, status, progress: result.ok ? 0 : item.progress, updatedAt: Date.now() } : item))
    setNodes((items) => items.map((item) => item.data.promptId === task.promptId ? { ...item, data: { ...item.data, status, progress: result.ok ? 0 : item.data.progress, promptId: result.ok ? null : item.data.promptId } } : item))
    setToast(result.message)
    window.setTimeout(() => setToast(null), 3200)
  }, [endpoint])

  const cancelAllTasks = useCallback(async () => {
    setTaskMenu(null)
    const activeTasks = sessionTasks.filter((task) => cancelableTaskStatuses.has(task.status) && task.promptId)
    const activeIds = activeTasks.map((task) => task.promptId)
    setSessionTasks((items) => items.map((item) => cancelableTaskStatuses.has(item.status) ? { ...item, status: '正在取消…', updatedAt: Date.now() } : item))
    const cloudTasks = activeTasks.filter((task) => task.provider === 'seedance')
    const localTasks = activeTasks.filter((task) => task.provider !== 'seedance')
    const cloudResults = await Promise.all(cloudTasks.map((task) => window.aaaLite?.cancelSeedanceTask?.(task.promptId, task.connectionId)))
    const localResult = localTasks.length ? await window.aaaLite?.cancelAllComfyTasks?.(endpoint) : { ok: true }
    const ok = localResult?.ok !== false && cloudResults.every((result) => result?.ok)
    if (ok) activeIds.forEach((promptId) => cancelledPromptRef.current.add(promptId))
    const message = ok ? '全部活动任务已取消' : '部分任务取消失败，请稍后重试'
    setSessionTasks((items) => items.map((item) => activeIds.includes(item.promptId) ? { ...item, status: ok ? '已取消' : message, progress: ok ? 0 : item.progress, updatedAt: Date.now() } : item))
    setNodes((items) => items.map((item) => activeIds.includes(item.data.promptId) ? { ...item, data: { ...item.data, status: ok ? '已取消' : message, progress: ok ? 0 : item.data.progress, promptId: ok ? null : item.data.promptId } } : item))
    setToast(message)
    window.setTimeout(() => setToast(null), 3400)
  }, [endpoint, sessionTasks])

  const releaseNow = useCallback(async (type) => {
    if (!window.aaaLite?.releaseResources) return
    setToast(type === 'all' ? '正在释放显存和内存…' : type === 'vram' ? '正在释放显存…' : '正在释放内存…')
    const result = await window.aaaLite.releaseResources(endpoint, type)
    setToast(result.message)
    window.setTimeout(() => setToast(null), 3200)
  }, [endpoint])

  const openComfyWeb = useCallback(async () => {
    if (!window.aaaLite?.openComfyWeb) return
    const result = await window.aaaLite.openComfyWeb(endpoint)
    setToast(result.message)
    window.setTimeout(() => setToast(null), 2800)
  }, [endpoint])

  const stopAllTasks = useCallback(async () => {
    if (!window.aaaLite?.stopAllAndRelease) return
    setToast('正在停止任务并释放显存和内存…')
    const result = await window.aaaLite.stopAllAndRelease(endpoint)
    if (result.ok) {
      setNodes((items) => items.map((item) => ['排队中', '生成中', '正在检查依赖'].includes(item.data.status) || item.data.status?.endsWith('待处理') ? { ...item, data: { ...item.data, status: '已停止并释放资源', progress: 0, promptId: null } } : item))
      setSessionTasks((items) => items.map((item) => ['排队中', '生成中', '正在检查依赖'].includes(item.status) || item.status?.endsWith('待处理') ? { ...item, status: '已停止并释放资源', progress: 0, updatedAt: Date.now() } : item))
    }
    setToast(result.message)
    window.setTimeout(() => setToast(null), 4000)
  }, [endpoint])

  const runContextNode = useCallback(async (menuNode) => {
    setMenu(null)
    const source = nodes.find((node) => node.id === menuNode?.id)
    if (!source) return
    if (['排队中', '生成中', '正在检查依赖'].includes(source.data.status)) {
      setToast('该节点已经在运行中')
      window.setTimeout(() => setToast(null), 2200)
      return
    }
    let target = source
    if (!['script', 'comfy', 'image', 'video'].includes(target.data.kind)) {
      const visited = new Set([source.id])
      const queue = [source.id]
      while (queue.length && target === source) {
        const current = queue.shift()
        for (const edge of edges.filter((item) => item.source === current)) {
          if (visited.has(edge.target)) continue
          visited.add(edge.target)
          const candidate = nodes.find((node) => node.id === edge.target)
          if (!candidate) continue
          if (['comfy', 'image', 'video'].includes(candidate.data.kind)) { target = candidate; break }
          queue.push(candidate.id)
        }
      }
    }
    if (target.data.kind === 'script') await runStoryboardAll(target.data.storyboardGroup)
    else if (target.data.kind === 'comfy') await runComfyNode(target.id)
    else if (['image', 'video'].includes(target.data.kind)) await runTaskEditor(target.id, { autoRunSequence: false, autoRunVideo: false, autoRunPipeline: false })
    else {
      setToast('该节点需要连接到图片、视频或 ComfyUI 任务后才能运行')
      window.setTimeout(() => setToast(null), 3000)
    }
  }, [edges, nodes, runComfyNode, runStoryboardAll, runTaskEditor])

  const stopContextNode = useCallback(async (menuNode) => {
    setMenu(null)
    const source = nodes.find((node) => node.id === menuNode?.id)
    if (!source || !['排队中', '生成中', '正在检查依赖'].includes(source.data.status)) {
      setToast('该节点当前没有运行任务')
      window.setTimeout(() => setToast(null), 2200)
      return
    }
    if (source.data.promptId) await stopComfyNode(source.id)
    else setNodes((items) => items.map((node) => node.id === source.id ? { ...node, data: { ...node.data, status: '已停止', progress: 0 } } : node))
    setToast('已停止该节点')
    window.setTimeout(() => setToast(null), 2200)
  }, [nodes, stopComfyNode])

  const uploadComfyImage = useCallback(async (id, paramId) => {
    if (!window.aaaLite?.uploadComfyImage) return
    const result = await window.aaaLite.uploadComfyImage(endpoint)
    if (result.canceled) return
    if (!result.ok) { setToast(result.message); window.setTimeout(() => setToast(null), 3200); return }
    changeComfyParam(id, paramId, result.name)
    setToast(`已上传输入图片：${result.name}`)
    window.setTimeout(() => setToast(null), 2400)
  }, [endpoint, changeComfyParam])

  const importComfyWorkflow = useCallback(async () => {
    if (!window.aaaLite?.importComfyWorkflow) return
    const result = await window.aaaLite.importComfyWorkflow()
    if (result.canceled) return
    if (!result.ok) { setToast(result.message); window.setTimeout(() => setToast(null), 3200); return }
    const record = { id: `workflow-${Date.now()}`, name: result.name, path: result.path, workflow: result.workflow, summary: result.summary, inspection: null }
    setWorkflowLibrary((items) => [...items.filter((item) => item.path !== record.path), record])
    setToast(`已导入 ${record.name}，共 ${record.summary.nodeCount} 个节点`)
    window.setTimeout(() => setToast(null), 2600)
  }, [])

  const addComfyWorkflowNode = useCallback((record) => {
    addNode('comfy', { data: { workflowId: record.id, workflowName: record.name, model: record.name, nodeCount: record.summary.nodeCount, params: structuredClone(record.summary.parameters || []), missingCount: record.inspection?.missing?.length || 0, status: '就绪' } })
  }, [nodes])

  const availableWorkflowChoices = useMemo(() => {
    const loadedIds = new Set(workflowLibrary.map((record) => record.id))
    const loaded = workflowLibrary.map((record) => ({ id: record.id, name: record.name, nodeCount: record.summary?.nodeCount || Object.keys(record.workflow || {}).length, source: record.source || 'manual', localPath: record.localPath, summary: record.summary, mediaKind: workflowMediaKind(record), modelInputs: [...new Set(Object.values(record.workflow || {}).flatMap((item) => Object.keys(item.inputs || {})).filter((key) => modelParamPattern.test(key)))] }))
    const discovered = localWorkflows.filter((record) => !loadedIds.has(record.id)).map((record) => ({ ...record, nodeCount: 0, mediaKind: workflowMediaKind(record), modelInputs: [] }))
    return [...loaded, ...discovered]
  }, [workflowLibrary, localWorkflows])

  const latestNodeActionsRef = useRef({})
  latestNodeActionsRef.current = { openElementMenu, refreshComfyCatalog, selectTaskRuntime, selectTaskProvider, selectSeedanceTaskModel, uploadMediaPreview, quickAction, extractFrameForNode, changePromptText, changeTaskConfig, addTaskReferences, runTaskEditor, runStoryboardAll, editStoryboard, changeComfyParam, uploadComfyImage, uploadComfyVideo, inspectComfyNode, runComfyNode, stopComfyNode }
  const stableNodeActions = useMemo(() => {
    const call = (name) => (...args) => latestNodeActionsRef.current[name]?.(...args)
    return {
      onElementContextMenu: call('openElementMenu'), onRefreshComfy: call('refreshComfyCatalog'), onTaskSelect: call('selectTaskRuntime'), onTaskProvider: call('selectTaskProvider'), onSeedanceModelSelect: call('selectSeedanceTaskModel'), onMediaUpload: call('uploadMediaPreview'), onQuickAction: call('quickAction'), onFrameExtract: call('extractFrameForNode'), onPromptText: call('changePromptText'), onTaskConfig: call('changeTaskConfig'), onTaskReferenceFiles: call('addTaskReferences'), onRunTask: call('runTaskEditor'), onStoryboardRunAll: call('runStoryboardAll'), onStoryboardEdit: call('editStoryboard'), onComfyParam: call('changeComfyParam'), onUploadComfyImage: call('uploadComfyImage'), onUploadComfyVideo: call('uploadComfyVideo'), onInspectComfy: call('inspectComfyNode'), onRunComfy: call('runComfyNode'), onStopComfy: call('stopComfyNode')
    }
  }, [])
  const sharedNodeData = useMemo(() => ({ availableWorkflows: availableWorkflowChoices, apiModels: availableApiModels, availableModels: comfyCatalog.models, comfyModelOptions: comfyCatalog.optionsByInput, comfyCatalogStatus: comfyCatalog.message, seedanceModels, seedanceModelLabels, activeSeedanceModel: seedanceModel, ...stableNodeActions }), [availableWorkflowChoices, availableApiModels, comfyCatalog, seedanceModels, seedanceModelLabels, seedanceModel, stableNodeActions])
  const displayNodeCacheRef = useRef(new Map())
  const displayNodes = useMemo(() => {
    const nodeById = new Map(nodes.map((node) => [node.id, node]))
    const incoming = new Map()
    for (const edge of edges) {
      if (!incoming.has(edge.target)) incoming.set(edge.target, [])
      incoming.get(edge.target).push(edge.source)
    }
    const collectUpstream = (targetId) => {
      const found = new Set()
      const pending = [...(incoming.get(targetId) || [])]
      while (pending.length) {
        const sourceId = pending.pop()
        if (found.has(sourceId)) continue
        found.add(sourceId)
        pending.push(...(incoming.get(sourceId) || []))
      }
      return found
    }
    const previous = displayNodeCacheRef.current
    const nextCache = new Map()
    const result = nodes.map((node) => {
      const connectedReferences = node.data.kind === 'video' ? [...collectUpstream(node.id)].flatMap((sourceId) => nodeImageReferences(nodeById.get(sourceId))) : []
      const taskReferences = node.data.kind === 'video' ? mergeImageReferences(node.data.taskReferences || [], connectedReferences) : (node.data.taskReferences || [])
      const referenceSignature = taskReferences.map((reference) => `${reference?.remoteUrl || reference?.url || reference?.dataUrl || ''}|${reference?.role || ''}|${reference?.name || ''}`).join('\n')
      const cached = previous.get(node.id)
      if (cached?.source === node && cached.shared === sharedNodeData && cached.referenceSignature === referenceSignature) {
        nextCache.set(node.id, cached)
        return cached.display
      }
      const defaultHeight = node.data.kind === 'comfy' ? 420 : node.data.kind === 'script' ? 390 : node.data.kind === 'prompt' ? 250 : mediaKinds.has(node.data.kind) ? 270 : 135
      const display = { ...node, style: { width: node.data.kind === 'comfy' ? 330 : node.data.kind === 'script' ? 340 : 260, height: defaultHeight, ...node.style }, data: { ...node.data, ...sharedNodeData, taskReferences, taskReferenceUrl: taskReferences[0]?.url || taskReferences[0]?.dataUrl || node.data.taskReferenceUrl, taskReferenceName: taskReferences[0]?.name || node.data.taskReferenceName } }
      nextCache.set(node.id, { source: node, shared: sharedNodeData, referenceSignature, display })
      return display
    })
    displayNodeCacheRef.current = nextCache
    return result
  }, [nodes, edges, sharedNodeData])

  const performanceMode = nodes.length > 24 || edges.length > 40
  const displayEdges = useMemo(() => edges.map((edge) => ({ ...edge, data: { ...edge.data, onElementContextMenu: openElementMenu, lowMotion: performanceMode } })), [edges, openElementMenu, performanceMode])

  const createFromDock = () => {
    const text = dockPrompt.trim()
    if (!text || !flow || !canvasRef.current) return
    const rect = canvasRef.current.getBoundingClientRect()
    const center = flow.screenToFlowPosition({ x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 })
    let position = { x: center.x - 130, y: center.y - 120 }
    for (let attempt = 0; attempt < 16; attempt += 1) {
      const occupied = nodes.some((node) => Math.abs(node.position.x - position.x) < 285 && Math.abs(node.position.y - position.y) < 220)
      if (!occupied) break
      const slot = attempt + 1
      position = {
        x: center.x - 430 + (slot % 3) * 300,
        y: center.y + 110 + Math.floor(slot / 3) * 245
      }
    }
    addNode(dockMode, {
      position,
      data: { text, model: dockModel, taskStyle: dockStyle, status: '生成中' }
    })
    setDockPrompt('')
    setToast(`${dockMode === 'video' ? '视频' : '图片'}任务已放到画布，可继续并行创建`)
    window.setTimeout(() => setToast(null), 2200)
  }

  const handleAssetDrop = useCallback(async (event) => {
    event.preventDefault()
    const file = [...(event.dataTransfer?.files || [])].find((item) => item.type.startsWith('image/'))
    if (!file || !flow) return
    if (file.size > 4 * 1024 * 1024) {
      setToast('图片超过 4MB，请压缩后再拖入')
      window.setTimeout(() => setToast(null), 2600)
      return
    }
    const point = flow.screenToFlowPosition({ x: event.clientX, y: event.clientY })
    try {
      const stored = await storeAssetFile(file)
      if (!stored?.ok) throw new Error(stored?.message || '素材保存失败')
      const id = `reference-${Date.now()}`
      setNodes((items) => [...items, { id, type: 'workflow', position: { x: point.x - 130, y: point.y - 130 }, data: { kind: 'reference', url: stored.url, name: file.name, text: file.name } }])
      setToast('参考图片已添加到画布')
      window.setTimeout(() => setToast(null), 2200)
    } catch (error) {
      setToast(`参考图片保存失败：${error.message}`)
      window.setTimeout(() => setToast(null), 2800)
    }
  }, [flow])

  useEffect(() => {
    const pasteImage = async (event) => {
      if (!flow || event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) return
      const directFile = [...(event.clipboardData?.files || [])].find((item) => item.type.startsWith('image/'))
      const imageItem = [...(event.clipboardData?.items || [])].find((item) => item.type.startsWith('image/'))
      const file = directFile || imageItem?.getAsFile()
      if (!file || file.size > 4 * 1024 * 1024) return
      try {
        const stored = await storeAssetFile(file)
        if (!stored?.ok) throw new Error(stored?.message || '素材保存失败')
        const rect = canvasRef.current?.getBoundingClientRect()
        if (!rect) return
        const point = flow.screenToFlowPosition({ x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 })
        setNodes((items) => [...items, { id: `reference-${Date.now()}`, type: 'workflow', position: { x: point.x - 130, y: point.y - 130 }, data: { kind: 'reference', url: stored.url, name: file.name || '剪贴板图片', text: file.name || '剪贴板图片' } }])
        setToast('剪贴板图片已添加到画布')
        window.setTimeout(() => setToast(null), 2200)
      } catch (error) {
        setToast(`剪贴板图片保存失败：${error.message}`)
        window.setTimeout(() => setToast(null), 2800)
      }
    }
    window.addEventListener('paste', pasteImage)
    return () => window.removeEventListener('paste', pasteImage)
  }, [flow])

  const assetItems = useMemo(() => nodes.flatMap((node) => {
    const items = []
    if (node.data.url) items.push({ id: `${node.id}-main`, url: node.data.url, name: node.data.name || nodeCatalog[node.data.kind]?.title || '画布素材' })
    if (node.data.taskReferenceUrl) items.push({ id: `${node.id}-reference`, url: node.data.taskReferenceUrl, name: node.data.taskReferenceName || '任务参考图' })
    if (node.data.resultUrl) items.push({ id: `${node.id}-result`, url: node.data.resultUrl, name: `${node.data.workflowName || 'ComfyUI'} 输出` })
    return items
  }), [nodes])

  const applySkillTemplate = (skill) => {
    const item = structuredClone(templates[skill.template])
    const prompt = item.nodes.find((node) => node.data.kind === 'prompt')
    if (prompt) prompt.data.text = skill.prompt
    setNodes(item.nodes); setEdges(item.edges); setModal(null)
    setToast(`${skill.title}技能已放到画布`)
    window.setTimeout(() => setToast(null), 2200)
  }

  const addAssetFromLibrary = (asset) => {
    addNode('reference', { data: { url: asset.url, name: asset.name, text: asset.name } })
    setToast('资产副本已添加到画布')
    window.setTimeout(() => setToast(null), 2200)
  }

  const taskItems = sessionTasks
  const activeTaskCount = taskItems.filter((task) => cancelableTaskStatuses.has(task.status) || task.status?.endsWith('待处理')).length
  const mediaEditorOpen = nodes.some((node) => node.selected && mediaKinds.has(node.data.kind))
  const formatStat = (value, suffix = '%') => value == null ? '—' : `${value}${suffix}`
  const vramPercent = stats.vramUsedGb == null || !stats.vramTotalGb ? null : Math.max(0, Math.min(100, Math.round((stats.vramUsedGb / stats.vramTotalGb) * 100)))

  const moveCanvasGlow = useCallback((event) => {
    if (!canvasRef.current || !glowRef.current) return
    const bounds = canvasRef.current.getBoundingClientRect()
    glowRef.current.style.setProperty('--glow-x', `${event.clientX - bounds.left}px`)
    glowRef.current.style.setProperty('--glow-y', `${event.clientY - bounds.top}px`)
    glowRef.current.style.opacity = '.72'
  }, [])
  const openContextNodeContent = useCallback((node) => {
    const descriptor = nodeContentDescriptor(node)
    if (!descriptor.media.length) {
      setToast('当前任务框没有可打开的图片或视频')
      window.setTimeout(() => setToast(null), 2400)
      return
    }
    setTaskPreview({ task: { title: semanticNodeTitle(node.data, nodeCatalog[node.data.kind]?.title || '任务内容'), workflowName: node.data.workflowName || node.data.name || '画布素材', media: descriptor.media }, index: descriptor.index || 0 })
    setMenu(null)
  }, [])
  const copyContextNodeContent = useCallback(async (node) => {
    const descriptor = nodeContentDescriptor(node)
    if (descriptor.kind === 'empty') {
      setToast('当前任务框没有可以复制的内容')
      window.setTimeout(() => setToast(null), 2400)
      return
    }
    const result = await window.aaaLite?.copyNodeContent?.({ kind: descriptor.kind === 'audio' ? 'text' : descriptor.kind, value: descriptor.value })
    if (!result && descriptor.kind === 'text') {
      try { await navigator.clipboard.writeText(descriptor.value) } catch { /* Electron clipboard is preferred */ }
    }
    setToast(result?.message || (descriptor.kind === 'text' ? '文本已复制' : '内容已复制'))
    window.setTimeout(() => setToast(null), 2800)
    setMenu(null)
  }, [])
  const handleNodesChange = useCallback((changes) => setNodes((items) => applyNodeChanges(changes, items)), [])
  const handleEdgesChange = useCallback((changes) => setEdges((items) => applyEdgeChanges(changes, items)), [])
  const handleEdgeConnect = useCallback((connection) => setEdges((items) => addEdge({ ...connection, type: 'atmosphere', animated: false }, items)), [])
  const contextNodeContent = menu?.kind === 'node' ? nodeContentDescriptor(menu.element) : null

  return (
    <main className={`app-shell ${chatOpen ? '' : 'chat-collapsed'} ${chatOpen && chatMode !== 'api' ? 'chat-web-mode' : ''} ${theme === 'light' ? 'theme-light' : ''}`}>
      <header className="topbar">
        <div className="brand compact"><span className="brand-mark"><img src="./icon.png" alt="" /></span><div><b>自由者</b><small>无限画布 · v0.18.57</small></div></div>
        <div className="project-save"><span>项目</span><button className="new-canvas-button" onClick={createCanvas}>＋ 新建画布</button><button className="canvas-list-button" onClick={() => setModal('canvases')}>{projectName}<b>{projects.length}</b></button><em>{saved ? '已保存' : '未保存'}</em><button className="save-compact" onClick={save} disabled={saved}>保存</button><button className="history-button" title="保存历史" onClick={() => setModal('history')}>◴</button></div>
        <div className="system-strip">
          <div className="live-metric cpu" title="Windows 全部处理器核心的实时平均使用率"><i><em style={{ width: `${stats.cpu || 0}%` }} /></i><b>{formatStat(stats.cpu)}</b><span>CPU</span></div>
          <div className="live-metric memory" title={`${stats.memoryUsedGb ?? '—'}GB / ${stats.memoryTotalGb ?? '—'}GB`}><i><em style={{ width: `${stats.memory || 0}%` }} /></i><b>{formatStat(stats.memory)}</b><span>内存</span></div>
          <div className="live-metric gpu" title="NVIDIA 驱动报告的实时 GPU 使用率"><i><em style={{ width: `${stats.gpu || 0}%` }} /></i><b>{formatStat(stats.gpu)}</b><span>GPU</span></div>
          <div className="live-metric vram" title={`${stats.vramUsedGb ?? '—'}GB / ${stats.vramTotalGb ?? '—'}GB`}><i><em style={{ width: `${vramPercent || 0}%` }} /></i><b>{formatStat(vramPercent)}</b><span>显存</span></div>
          <div className="live-metric temperature" title="NVIDIA GPU 核心温度"><i><em style={{ width: `${Math.max(0, Math.min(100, stats.temperature || 0))}%` }} /></i><b>{stats.temperature == null ? '—' : `${stats.temperature}°`}</b><span>温度</span></div>
          <div className="resource-actions">
            <button className="stop-all" title="停止当前任务、清空排队任务并释放显存和内存" onClick={stopAllTasks}>■ 停止任务</button>
            <button title="同时卸载 ComfyUI 模型并清理显存、内存和应用缓存" onClick={() => releaseNow('all')}>释放显存和内存</button>
            <button className="open-comfy" title="恢复并弹出现有 ComfyUI 网页，不新建页面" onClick={openComfyWeb}>◆ 打开 ComfyUI</button>
          </div>
        </div>
        <div className="top-actions">
          <button className="icon-action" title="应用日志" onClick={() => setModal('logs')}>▧</button>
          <button className="icon-action" title="切换主题" onClick={() => setTheme((value) => value === 'dark' ? 'light' : 'dark')}>☼</button>
          <button className={`task-button ${taskOpen ? 'active' : ''}`} onClick={() => setTaskOpen((value) => !value)}>任务中心 <b>{taskItems.length}</b></button>
        </div>
      </header>

      <aside className="rail">
        <button className="rail-add" title="添加节点" onClick={() => setModal('nodes')}>＋</button>
        <div className="rail-group">
          <button className="storyboard-entry" onClick={() => { setStoryboardEditingGroup(null); setModal('storyboard') }} title="分镜师">▤</button>
          <button onClick={() => setModal('templates')} title="工作流与模板">⌘</button>
          <button onClick={() => setModal('assets')} title="本地资产库">▣</button>
          <button onClick={() => setModal('skills')} title="本地技能库">✦</button>
          <button onClick={() => setModal('styles')} title="视觉风格库">◈</button>
          <button onClick={() => setModal('comfy')} title="ComfyUI 工作流库">◆</button>
          <button onClick={() => setModal('models')} title="模型与服务">◎</button>
          <button onClick={() => setModal('info')} title="关于与帮助">ⓘ</button>
          <button onClick={() => setModal('settings')} title="设置">⚙</button>
          <button onClick={() => setModal('more')} title="更多">•••</button>
        </div>
        <div className="rail-bottom"><span>A</span></div>
      </aside>

      <section className={`canvas-wrap ${performanceMode ? 'dense-canvas' : ''}`} ref={canvasRef} onDoubleClick={handleCanvasDoubleClick} onPointerDownCapture={beginRightSelection} onPointerMoveCapture={moveRightSelection} onPointerUpCapture={finishRightSelection} onPointerCancelCapture={finishRightSelection} onPointerMove={performanceMode ? undefined : moveCanvasGlow} onPointerLeave={() => { if (glowRef.current) glowRef.current.style.opacity = '0' }} onDragOver={(event) => event.preventDefault()} onDrop={handleAssetDrop}>
        <div className="canvas-title"><span>项目 / 我的工作流</span><em>{nodes.length} 个节点</em><small>拖入图片作为参考</small></div>
        <button className={`canvas-one-click ${activeTaskCount ? 'running' : ''}`} title="自动识别当前任务使用本地 ComfyUI 或 Seedance（SD），同步连接节点后生成" onMouseDown={(event) => event.stopPropagation()} onDoubleClick={(event) => event.stopPropagation()} onClick={oneClickGenerate}><span>✦</span><b>一键生成</b>{activeTaskCount > 0 && <small>{activeTaskCount} 个任务运行中</small>}</button>
        <div ref={glowRef} className="canvas-hover-glow" aria-hidden="true" />
        <ReactFlow
          nodes={displayNodes} edges={displayEdges} nodeTypes={nodeTypes} edgeTypes={edgeTypes} fitView={nodes.length <= 80}
          onInit={setFlow}
          onNodesChange={handleNodesChange}
          onEdgesChange={handleEdgesChange}
          onConnect={handleEdgeConnect}
          onConnectEnd={handleConnectEnd}
          onPaneContextMenu={openPaneContextMenu}
          onNodeContextMenu={(event, node) => openElementMenu(event, 'node', node)}
          onEdgeContextMenu={(event, edge) => openElementMenu(event, 'edge', edge)}
          onPaneClick={() => setMenu(null)}
          onMoveStart={() => setMenu(null)}
          deleteKeyCode={['Backspace', 'Delete']}
          minZoom={0.1}
          maxZoom={3}
          zoomOnDoubleClick={false}
          onlyRenderVisibleElements
          fitViewOptions={{ padding: 0.2 }}
          defaultEdgeOptions={{ type: 'atmosphere', animated: false }}
        >
          <Background className="fixed-dot-field" color="#50617e" gap={32} size={0.72} />
          <Controls showInteractive={false} />
          {!performanceMode && <MiniMap pannable zoomable nodeColor="#7285ff" maskColor="rgba(8,10,15,.75)" />}
        </ReactFlow>
        {rightSelectBox && <div className="right-selection-box" style={rightSelectBox}><span>右键框选</span></div>}
        {menu?.line && <svg className="pending-connection" aria-hidden="true">
          <defs><filter id="green-glow" x="-100%" y="-100%" width="300%" height="300%"><feGaussianBlur stdDeviation="5" result="blur" /><feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge></filter></defs>
          <path className="pending-halo" d={`M ${menu.line.x1} ${menu.line.y1} C ${menu.line.x1 + Math.max(80, Math.abs(menu.line.x2 - menu.line.x1) * .45)} ${menu.line.y1}, ${menu.line.x2 - Math.max(80, Math.abs(menu.line.x2 - menu.line.x1) * .45)} ${menu.line.y2}, ${menu.line.x2} ${menu.line.y2}`} />
          <path className="pending-core" filter="url(#green-glow)" d={`M ${menu.line.x1} ${menu.line.y1} C ${menu.line.x1 + Math.max(80, Math.abs(menu.line.x2 - menu.line.x1) * .45)} ${menu.line.y1}, ${menu.line.x2 - Math.max(80, Math.abs(menu.line.x2 - menu.line.x1) * .45)} ${menu.line.y2}, ${menu.line.x2} ${menu.line.y2}`} />
          <circle className="pending-orbit" cx={menu.line.x2} cy={menu.line.y2} r="10" /><circle className="pending-dot" cx={menu.line.x2} cy={menu.line.y2} r="4" />
        </svg>}
        {nodes.length === 0 && <div className="empty-canvas"><span>＋</span><h2>从一个节点开始</h2><p>添加节点，或选择一个简单模板</p><button onClick={() => setModal('nodes')}>添加节点</button></div>}
        <div className={`creation-dock ${mediaEditorOpen ? 'media-editor-open' : ''}`} onMouseDown={(event) => event.stopPropagation()}>
          <div className="dock-modes"><button className={dockMode === 'image' ? 'active' : ''} onClick={() => setDockMode('image')}>◫ 图片</button><button className={dockMode === 'video' ? 'active' : ''} onClick={() => setDockMode('video')}>▶ 视频</button></div>
          <textarea value={dockPrompt} onChange={(event) => setDockPrompt(event.target.value)} onKeyDown={(event) => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') createFromDock() }} placeholder={`描述要生成的${dockMode === 'video' ? '视频和镜头运动' : '图片内容'}…`} />
          <div className="dock-footer"><select value={dockModel} onChange={(event) => setDockModel(event.target.value)}><option>本地 ComfyUI</option><option>自定义工作流</option><option>仅创建占位卡</option></select><select value={dockStyle} onChange={(event) => setDockStyle(event.target.value)}>{creativeStyles.map((style) => <option key={style}>{style}</option>)}</select><span>可连续提交多个任务</span><button onClick={createFromDock} disabled={!dockPrompt.trim()}>生成到画布 ↗</button></div>
        </div>
      </section>

      {menu?.kind === 'node' && <div className="context-menu element-context-menu" style={{ left: menu.x, top: menu.y }} onMouseDown={(event) => event.stopPropagation()}>
        <button className="context-action" disabled={!contextNodeContent?.media.length} onClick={() => openContextNodeContent(menu.element)}><span>{contextNodeContent?.kind === 'video' ? '▶' : '▣'}</span><b>{contextNodeContent?.kind === 'video' ? '打开视频' : contextNodeContent?.kind === 'image' ? '打开图片' : '没有图片或视频'}</b></button>
        <button className="context-action" disabled={contextNodeContent?.kind === 'empty'} onClick={() => copyContextNodeContent(menu.element)}><span>⧉</span><b>{contextNodeContent?.kind === 'video' ? '复制视频' : contextNodeContent?.kind === 'image' ? '复制图片' : '复制文本'}</b></button>
        <div className="context-divider" />
        <button className="context-action" onClick={() => { setMenu(null); quickAction(menu.element.id, '复制节点') }}><span>▣</span><b>复制一份</b></button>
        <button className="context-action danger" onClick={() => closeTaskNode(menu.element)}><span>♲</span><b>删除</b></button>
        <div className="context-divider" />
        <button className="context-action" onClick={() => runContextNode(menu.element)}><span>▷</span><b>运行该节点</b></button>
        <button className="context-action" onClick={() => stopContextNode(menu.element)}><span>□</span><b>停止该节点</b></button>
      </div>}

      {menu?.kind === 'nodes' && <div className="context-menu element-context-menu multi-node-menu" style={{ left: menu.x, top: menu.y }} onMouseDown={(event) => event.stopPropagation()}>
        <div className="context-title"><span>已选择 {menu.elements.length} 个任务框</span><small>操作会同时应用到所有选中任务</small></div>
        <button className="context-action danger" onClick={() => closeSelectedTaskNodes(menu.elements)}><span>♲</span><b>删除所选任务框</b></button>
      </div>}

      {menu?.kind === 'edge' && <div className="context-menu element-context-menu" style={{ left: menu.x, top: menu.y }} onMouseDown={(event) => event.stopPropagation()}>
        <div className="context-title"><span>连接线操作</span><small>只影响当前这条连接</small></div>
        <button className="context-delete" onClick={() => closeConnectionLine(menu.element)}><span>⌁</span><div><b>删除连接线</b><small>两端任务框会保留</small></div></button>
      </div>}

      {menu && !menu.kind && <div className="context-menu" style={{ left: menu.x, top: menu.y }} onMouseDown={(event) => event.stopPropagation()}>
        <div className="context-title"><span>{menu.connection ? '连接到新节点' : '在这里创建'}</span><small>{menu.connection ? '选择后自动连接' : '选择一个功能节点'}</small></div>
        <div className="context-nodes">
          {creatableNodeEntries.map(([key, item]) => <button key={key} onClick={() => addNode(key, { position: menu.position, connection: menu.connection })}><span className={item.tone}>{item.icon}</span><div><b>{item.title}</b><small>{item.body}</small></div></button>)}
        </div>
        {!menu.connection && <><div className="context-divider" /><div className="context-label">快速模板</div><div className="context-templates"><button onClick={() => applyTemplate('image')}>快速生图</button><button onClick={() => applyTemplate('video')}>图片转视频</button></div></>}
      </div>}

      <aside className={`assistant-panel ${chatOpen ? '' : 'is-collapsed'}`} aria-hidden={!chatOpen}>
        <div className="chat-edge-hover-zone">
          <button className="chat-collapse-handle" type="button" title="收回聊天框" aria-label="收回聊天框" onClick={() => setChatOpen(false)}><span>›</span></button>
        </div>
        <div className="assistant-head"><div><span>✦</span><b>{chatMode === 'web' ? 'ChatGPT 网页版' : chatMode === 'qwen' ? '千问 AI 网页版' : chatMode === 'doubao-web' ? '豆包网页版' : '豆包助手'}</b></div><div className="assistant-mode-switch"><button className={chatMode === 'web' ? 'active' : ''} onClick={() => setChatMode('web')}>GPT</button><button className={chatMode === 'qwen' ? 'active' : ''} onClick={() => setChatMode('qwen')}>千问</button><button className={chatMode === 'doubao-web' ? 'active' : ''} onClick={() => setChatMode('doubao-web')}>豆包</button><button className={chatMode === 'api' ? 'active' : ''} onClick={() => setChatMode('api')}>API</button></div>{chatMode === 'api' && <small>{doubaoConfig.configured ? `已连接 · ${doubaoConfig.model}` : '未配置'}</small>}<button title="收起聊天" onClick={() => setChatOpen(false)}>×</button></div>
        {chatOpen && chatMode === 'web' && <div className="chat-web-shell chat-mode-pane is-active"><div className="chat-web-toolbar"><span>ChatGPT 官方网页 · 登录状态仅保存在本机</span><button onClick={() => setChatGptWebKey((value) => value + 1)}>↻ 刷新</button></div><webview key={`gpt-${chatGptWebKey}`} className="chat-webview" src="https://chatgpt.com/" partition="persist:chatgpt-web" webpreferences="contextIsolation=yes, nodeIntegration=no, sandbox=yes" /></div>}
        {chatOpen && chatMode === 'qwen' && <div className="chat-web-shell chat-mode-pane is-active"><div className="chat-web-toolbar"><span>千问官方网页 · 登录状态仅保存在本机</span><button onClick={() => setQwenWebKey((value) => value + 1)}>↻ 刷新</button></div><webview key={`qwen-${qwenWebKey}`} className="chat-webview" src="https://chat.qwen.ai/" partition="persist:qwen-web" useragent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/142.0.0.0 Safari/537.36" webpreferences="contextIsolation=yes, nodeIntegration=no, sandbox=yes" /></div>}
        {chatOpen && chatMode === 'doubao-web' && <div className="chat-web-shell chat-mode-pane is-active"><div className="chat-web-toolbar"><span>豆包官方网页 · 原登录状态继续保留</span><button onClick={() => setDoubaoWebKey((value) => value + 1)}>↻ 刷新</button></div><webview key={`doubao-${doubaoWebKey}`} className="chat-webview" src="https://www.doubao.com/chat/" partition="persist:doubao-web" webpreferences="contextIsolation=yes, nodeIntegration=no, sandbox=yes" /></div>}
        <div className={`chat-mode-pane api-chat-pane ${chatMode === 'api' ? 'is-active' : ''}`} aria-hidden={chatMode !== 'api'}><div className="chat-list">
          {chat.map((item, index) => <div className={`bubble ${item.role}`} key={index}>{item.text}{item.action && <button onClick={createSuggestedFlow}>生成到画布</button>}</div>)}
        </div>
        <div className="composer">
          <textarea value={message} onChange={(e) => setMessage(e.target.value)} placeholder="向豆包提问日常问题…" onKeyDown={(e) => { if (e.ctrlKey && e.key === 'Enter') sendMessage() }} />
          <div><small>{doubaoLoading ? '豆包正在回复…' : 'Ctrl + Enter 发送'}</small><button onClick={sendMessage} disabled={!message.trim() || doubaoLoading}>➜</button></div>
        </div></div>
      </aside>

      {!chatOpen && <button className="floating-chat" title="打开聊天" onClick={() => setChatOpen(true)}><span>▱</span><small>聊天</small></button>}

      {taskOpen && <aside className="task-drawer">
        <div className="drawer-head"><div><b>任务中心</b><small>本次运行 {taskItems.length} 条记录 · {activeTaskCount} 个进行中</small></div><section className="drawer-actions"><button className="cancel-all-tasks" disabled={!activeTaskCount} onClick={cancelAllTasks}>■ 一键取消任务</button><button className="drawer-close" onClick={() => { setTaskOpen(false); setTaskMenu(null) }}>×</button></section></div>
        <div className="task-list">{taskItems.length ? taskItems.map((task) => {
          const primaryMedia = task.media?.[0]
          const isVideoResult = mediaIsVideo(primaryMedia)
          const isAudioResult = mediaIsAudio(primaryMedia)
          return <article className={`task-item ${task.status === '已完成' ? 'complete' : task.status === '已取消' ? 'cancelled' : task.status?.startsWith('失败') ? 'failed' : ''}`} key={task.id} onContextMenu={(event) => { event.preventDefault(); event.stopPropagation(); setTaskMenu({ x: Math.min(event.clientX, window.innerWidth - 250), y: Math.min(event.clientY, window.innerHeight - 222), task }) }}>
            <div className={`task-result-preview ${primaryMedia ? 'has-media' : ''}`}>{primaryMedia ? (isVideoResult ? <video src={primaryMedia.url} controls muted playsInline /> : isAudioResult ? <span className="task-audio-result">♫<small>音频结果</small></span> : <img src={primaryMedia.url} alt={primaryMedia.filename || task.title} />) : <span>{nodeCatalog[task.kind]?.icon || '✦'}</span>}</div>
            <div className="task-item-info"><header><b>{task.title}</b><time>{new Date(task.createdAt).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}</time></header><small>{task.workflowName}</small><em>{task.status}</em><i><u style={{ width: `${Math.max(0, Math.min(100, task.progress || 0))}%` }} /></i>{task.media?.length > 1 && <div className="task-media-strip">{task.media.slice(1, 5).map((media, index) => <TaskMediaThumb key={`${media.url}-${index}`} media={media} alt={media.filename || `结果 ${index + 2}`} />)}<small>{task.media.length} 个结果</small></div>}</div>
          </article>
        }) : <div className="empty-tasks"><span>✓</span><b>本次运行还没有生成记录</b><small>ComfyUI 图片和视频完成后会实时显示在这里</small></div>}</div>
      </aside>}

      {taskMenu && <div className="task-cancel-menu" style={{ left: taskMenu.x, top: taskMenu.y }} onMouseDown={(event) => event.stopPropagation()}>
        <div><span>{nodeCatalog[taskMenu.task.kind]?.icon || '✦'}</span><section><b>{taskMenu.task.title}</b><small>{taskMenu.task.status}</small></section><button onClick={() => setTaskMenu(null)}>×</button></div>
        <p>{taskMenu.task.provider === 'seedance' ? '取消自由者任务与火山方舟云端任务。' : '同时取消自由者任务与 ComfyUI 后台执行/排队任务。'}</p>
        <button className="view-task-result" disabled={!taskMenu.task.media?.length} onClick={() => { setTaskPreview({ task: taskMenu.task, index: 0 }); setTaskMenu(null) }}>▣ {taskMenu.task.media?.some(mediaIsVideo) ? '查看图片 / 视频' : taskMenu.task.media?.some(mediaIsAudio) ? '查看图片 / 音频' : '查看图片'}</button>
        <button className="cancel-one-task" disabled={!cancelableTaskStatuses.has(taskMenu.task.status)} onClick={() => cancelTask(taskMenu.task)}>■ {cancelableTaskStatuses.has(taskMenu.task.status) ? '取消此任务' : '任务已经结束'}</button>
      </div>}

      {taskPreview && (() => {
        const media = taskPreview.task.media?.[taskPreview.index]
        const isVideo = mediaIsVideo(media)
        const isAudio = mediaIsAudio(media)
        return <div className="task-preview-backdrop" onMouseDown={() => setTaskPreview(null)}>
          <section className="task-preview-modal" onMouseDown={(event) => event.stopPropagation()}>
            <header><div><b>{taskPreview.task.title}</b><small>{media?.filename || taskPreview.task.workflowName}</small></div><em>{taskPreview.index + 1} / {taskPreview.task.media.length}</em><button onClick={() => setTaskPreview(null)}>×</button></header>
            <div className="task-preview-stage">{isVideo ? <video src={media?.url} controls autoPlay muted playsInline /> : isAudio ? <div className="task-audio-stage"><span>♫</span><b>{media?.filename || '音频结果'}</b><audio src={media?.url} controls autoPlay /></div> : <img src={media?.url} alt={media?.filename || taskPreview.task.title} />}{taskPreview.task.media.length > 1 && <><button className="preview-previous" onClick={() => setTaskPreview((value) => ({ ...value, index: (value.index - 1 + value.task.media.length) % value.task.media.length }))}>‹</button><button className="preview-next" onClick={() => setTaskPreview((value) => ({ ...value, index: (value.index + 1) % value.task.media.length }))}>›</button></>}</div>
            {taskPreview.task.media.length > 1 && <div className="task-preview-thumbs">{taskPreview.task.media.map((item, index) => <button className={index === taskPreview.index ? 'active' : ''} key={`${item.url}-${index}`} onClick={() => setTaskPreview((value) => ({ ...value, index }))}><TaskMediaThumb media={item} alt={item.filename || `结果 ${index + 1}`} /></button>)}</div>}
          </section>
        </div>
      })()}

      {modal && <div className="modal-backdrop" onMouseDown={() => setModal(null)}>
        <section className="modal" onMouseDown={(e) => e.stopPropagation()}>
          <button className="modal-close" onClick={() => setModal(null)}>×</button>
          {modal === 'nodes' && <><h2>添加节点</h2><p>选择工作流中需要的一步。</p><div className="card-grid">{creatableNodeEntries.map(([key, item]) => <button className="choice-card" key={key} onClick={() => addNode(key)}><span className={item.tone}>{item.icon}</span><b>{item.title}</b><small>{item.body}</small></button>)}</div></>}
          {modal === 'canvases' && <><h2>我的画布</h2><p>像浏览器页面一样新建并切换独立画布，内容会自动保存在本机。</p><button className="primary full" onClick={createCanvas}>＋ 新建画布</button><label className="field"><span>当前画布名称</span><input value={projectName} onChange={(event) => renameCanvas(event.target.value)} placeholder="输入画布名称" /></label><div className="canvas-library">{projects.map((project, index) => <button className={project.id === projectId ? 'active' : ''} key={project.id} onClick={() => openCanvas(project.id)}><span>{index + 1}</span><section><b>{project.id === projectId ? projectName : project.name}</b><small>{project.nodeCount || 0} 个节点 · {project.id === projectId ? '当前画布' : '点击打开'}</small></section><em>{project.id === projectId ? '使用中' : '打开'}</em></button>)}</div></>}
          {modal === 'templates' && <><h2>工作流模板</h2><p>选一个模板快速开始，当前画布会被替换。</p><div className="template-list">{Object.entries(templates).map(([key, item]) => <button key={key} onClick={() => applyTemplate(key)}><span>◇</span><div><b>{item.name}</b><small>{item.nodes.length} 个节点</small></div><i>使用</i></button>)}</div></>}
          {modal === 'assets' && <><h2>本地资产库</h2><p>自动汇总画布中的上传图片、参考图和 ComfyUI 输出，点击可再次放到画布。</p>{assetItems.length ? <div className="asset-grid">{assetItems.map((asset) => <button key={asset.id} onClick={() => addAssetFromLibrary(asset)}><img src={asset.url} alt={asset.name} /><span>{asset.name}</span><small>＋ 添加副本</small></button>)}</div> : <div className="empty-library">拖入、粘贴图片或运行 ComfyUI 后，素材会显示在这里</div>}</>}
          {modal === 'skills' && <><h2>本地技能库</h2><p>选择创作目标，自动建立对应的节点工作流。</p><div className="skill-grid">{skillTemplates.map((skill) => <button key={skill.title} onClick={() => skill.title === '分镜师' ? setModal('storyboard') : applySkillTemplate(skill)}><span>{skill.icon}</span><b>{skill.title}</b><small>{skill.prompt}</small><em>使用技能</em></button>)}</div></>}
          {modal === 'storyboard' && <div className="storyboard-modal">
            <h2>分镜师</h2>
            <p>自动把人物信息、场景信息和逐镜头提示词分开；每次换镜头都会生成独立的画面、人物、场景、运镜、对白与音效提示词。</p>
            <label className="storyboard-script-field"><span>我的完整剧本</span><textarea autoFocus value={storyboardScript} onChange={(event) => setStoryboardScript(event.target.value)} placeholder={'示例：\n夜雨中的旧车站，林夏撑伞跑向站台。\n她看见多年未见的周辰，停下脚步。\n周辰转身说：“你终于来了。”'} /></label>
            <div className="storyboard-options">
              <label><span>视觉风格</span><select value={storyboardStyle} onChange={(event) => setStoryboardStyle(event.target.value)}>{creativeStyles.map((style) => <option key={style}>{style}</option>)}</select></label>
              <label><span>画面比例</span><select value={storyboardRatio} onChange={(event) => setStoryboardRatio(event.target.value)}><option>16:9</option><option>9:16</option><option>1:1</option><option>4:3</option><option>3:4</option></select></label>
              <label><span>清晰度</span><select value={storyboardResolution} onChange={(event) => setStoryboardResolution(event.target.value)}><option>480P</option><option>720P</option><option>1080P</option><option>1K</option><option>2K</option></select></label>
              <label><span>镜头识别</span><select value={splitScriptIntoShots(storyboardScript, Number.MAX_SAFE_INTEGER).length} disabled><option value={splitScriptIntoShots(storyboardScript, Number.MAX_SAFE_INTEGER).length}>已识别 {splitScriptIntoShots(storyboardScript, Number.MAX_SAFE_INTEGER).length} 个镜头</option></select></label>
            </div>
            <div className="storyboard-runtime-check"><span>◆</span><div><b>本地 ComfyUI · 生图 + MiniMax H3</b><small>{comfyCatalog.ok ? `${comfyCatalog.models.length} 个模型 · ${localWorkflows.length} 个工作流已识别` : comfyCatalog.message}</small></div><em className={comfyCatalog.ok ? 'online' : ''}>{comfyCatalog.ok ? '已连接' : '等待连接'}</em></div>
            <button className="primary full storyboard-create" onClick={createStoryboard}>✦ 一键生成全部分镜</button>
          </div>}
          {modal === 'styles' && <><h2>视觉风格库</h2><p>选择后应用到已选中的媒体任务；未选中节点时作为底部生成栏的默认风格。</p><div className="style-grid">{creativeStyles.map((style, index) => <button key={style} className={`style-${index % 8}`} onClick={() => { const selected = nodes.filter((node) => node.selected && mediaKinds.has(node.data.kind)); if (selected.length) selected.forEach((node) => changeTaskConfig(node.id, 'taskStyle', style)); else setDockStyle(style); setModal(null); setToast(`已选择风格：${style}`); window.setTimeout(() => setToast(null), 2000) }}><i /><span>{style}</span></button>)}</div></>}
          {modal === 'comfy' && <><h2>ComfyUI 工作流库</h2><p>导入“API 格式”JSON，检测自定义节点后放到画布运行。</p><button className="primary full" onClick={importComfyWorkflow}>＋ 导入 API 工作流</button><div className="workflow-library">{workflowLibrary.length ? workflowLibrary.map((record) => <div className="workflow-record" key={record.id}><span>◆</span><section><b>{record.name}</b><small>{record.summary.nodeCount} 个节点 · {record.summary.classes.length} 种节点类型</small><em className={record.inspection?.missing?.length ? 'warn' : ''}>{record.inspection ? (record.inspection.ok ? (record.inspection.missing.length ? `缺少 ${record.inspection.missing.length} 项` : `依赖完整${record.inspection.managerAvailable ? ' · Manager 可用' : ''}`) : record.inspection.message) : '尚未检查依赖'}</em></section><div><button onClick={async () => { const result = await window.aaaLite.inspectComfyWorkflow(endpoint, record.workflow); setWorkflowLibrary((items) => items.map((item) => item.id === record.id ? { ...item, inspection: result } : item)) }}>检查</button><button className="run" onClick={() => addComfyWorkflowNode(record)}>放到画布</button></div></div>) : <div className="empty-library">尚未导入工作流</div>}</div></>}
          {modal === 'settings' && <><h2>连接设置</h2><p>本地 ComfyUI 与云端 API 分开管理；图片、视频模型会自动分类到对应任务框。</p><label className="field"><span>ComfyUI 地址</span><input value={endpoint} onChange={(e) => setEndpoint(e.target.value)} placeholder="http://127.0.0.1:8188" /></label><button className="primary full" onClick={checkConnection}>连接并自动识别本地模型</button>{connection && <div className={`connection ${connection.ok === true ? 'success' : connection.ok === false ? 'error' : ''}`}>{connection.message}</div>}<div className="settings-divider"><b>云端模型 API</b><small>兼容 OpenAI 图片生成接口与 Seedance 视频任务接口</small></div><div className="connection-profile-switch api-page-switch"><button className={seedanceConnectionMode === 'new' ? 'active' : ''} onClick={beginNewSeedanceConnection}><span>＋ 新建 API</span><small>填写、添加模型并保存</small></button><button className={seedanceConnectionMode === 'saved' ? 'active' : ''} onClick={() => setSeedanceConnectionMode('saved')}><span>已保存 API</span><small>{seedanceConnections.length} 个连接</small></button></div>{seedanceConnectionMode === 'saved' && <div className="saved-api-page"><div className="saved-connection-list">{seedanceConnections.length ? seedanceConnections.map((item) => <button className={item.id === activeSeedanceConnectionId ? 'active' : ''} key={item.id} onClick={() => selectSavedSeedanceConnection(item.id)}><span>◆</span><section><b>{item.name}</b><small>{item.videoBaseUrl}</small><small>{(item.videoModels || []).filter((model) => item.modelMediaKinds?.[model] === 'image').length} 个图片模型 · {(item.videoModels || []).filter((model) => item.modelMediaKinds?.[model] !== 'image').length} 个视频模型</small></section><em>{item.id === activeSeedanceConnectionId ? '当前连接' : '切换使用'}</em></button>) : <div className="empty-library">暂无已保存 API，请切换到“新建 API”</div>}</div><div className="security-note"><b>已保存 API 独立查看</b><span>此页面不会显示编辑表单，点击连接即可切换；API Key 只保存在 Windows 加密存储中。</span></div></div>}{seedanceConnectionMode === 'new' && <div className="new-api-page"><label className="field"><span>API 连接名称</span><input value={seedanceConnectionName} onChange={(event) => setSeedanceConnectionName(event.target.value)} placeholder="例如：火山方舟、OpenAI 兼容图片接口" /></label><label className="field"><span>API 基础地址</span><input value={seedanceBaseUrl} onChange={(e) => setSeedanceBaseUrl(e.target.value)} placeholder="例如 https://服务商地址/v1 或 /api/v3" /></label><label className="field"><span>API Key</span><input type="password" value={doubaoKey} onChange={(e) => setDoubaoKey(e.target.value)} placeholder="新 API 必须填写密钥" autoComplete="new-password" /></label><label className="field"><span>聊天模型 ID（可选）</span><input value={doubaoModel} onChange={(e) => setDoubaoModel(e.target.value)} placeholder="不用于聊天可留空" /></label><div className="model-manager custom-model-manager api-model-manager"><label className="field"><span>模型显示名称</span><input value={newSeedanceModelLabel} onChange={(event) => setNewSeedanceModelLabel(event.target.value)} placeholder="例如 SD2.0、Seedance 2.0" /></label><label className="field"><span>模型用途</span><select value={newSeedanceModelKind} onChange={(event) => setNewSeedanceModelKind(event.target.value)}><option value="image">图片生成</option><option value="video">视频生成</option></select></label><label className="field"><span>Model ID / Endpoint ID</span><input value={newSeedanceModel} onChange={(event) => setNewSeedanceModel(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter') addSeedanceModel() }} placeholder="填写服务商提供的真实 ID" /></label><button className="primary" onClick={addSeedanceModel}>＋ 添加到当前 API</button></div><div className="managed-model-list">{seedanceModels.length ? seedanceModels.map((model) => <div className={model === seedanceModel ? 'active' : ''} key={model}><button onClick={() => setSeedanceModel(model)}><span>{seedanceModelKinds[model] === 'image' ? '◫' : '▶'}</span><section><b>{seedanceModelLabels[model] || model}</b><small>{seedanceModelKinds[model] === 'image' ? '图片生成' : '视频生成'} · {model}</small></section></button><button className="delete-model" title="删除模型" onClick={() => deleteSeedanceModel(model)}>删除</button></div>) : <div className="empty-library">请先添加至少一个图片或视频模型</div>}</div><button className="primary full" onClick={saveDoubaoSettings} disabled={!seedanceModel || !seedanceConnectionName.trim() || !doubaoKey.trim()}>加密保存新 API</button><div className="security-note"><b>自动分类与调用</b><span>图片模型只会出现在图片任务框，视频模型只会出现在视频任务框；保存后任务框会自动识别。</span></div></div>}</>}
          {modal === 'history' && <><h2>保存历史</h2><p>当前项目保存在本机浏览器数据中。</p><div className="history-list"><div><span>{projects.findIndex((item) => item.id === projectId) + 1}</span><section><b>{projectName}</b><small>{nodes.length} 个节点 · {edges.length} 条连接</small></section><em>{saved ? '已保存' : '有修改'}</em></div></div><button className="primary full" onClick={() => { save(); setModal(null) }}>立即保存快照</button></>}
          {modal === 'logs' && <><h2>应用日志</h2><p>显示本次运行中的主要状态。</p><div className="log-list"><div><time>现在</time><span>系统监控运行中</span></div><div><time>画布</time><span>已加载 {nodes.length} 个节点、{edges.length} 条连接</span></div><div><time>ComfyUI</time><span>{connection?.message || '尚未测试连接'}</span></div><div><time>任务</time><span>{taskItems.length} 条生成记录 · {activeTaskCount} 个进行中</span></div></div></>}
          {modal === 'models' && <><h2>本地模型与服务</h2><p>自动识别 ComfyUI 当前可用的模型，任务框选择后可直接写入工作流运行。</p><div className="service-list"><div><span>◎</span><section><b>本地 ComfyUI</b><small>{endpoint}</small></section><em>{comfyCatalog.ok ? '在线' : comfyCatalog.ok === false ? '离线' : '检测中'}</em></div><div><span>◆</span><section><b>已导入工作流</b><small>{workflowLibrary.length} 个 API 工作流</small></section><em>可选择</em></div></div><div className="model-summary"><b>{comfyCatalog.message}</b>{Object.entries(comfyCatalog.models.reduce((types, model) => ({ ...types, [model.type]: (types[model.type] || 0) + 1 }), {})).map(([type, count]) => <span key={type}>{type} {count}</span>)}</div><button className="primary full" onClick={() => refreshComfyCatalog(true)}>↻ 重新识别本地模型</button><button className="secondary full" onClick={() => setModal('settings')}>配置连接地址</button></>}
          {modal === 'info' && <><h2>自由者 0.8</h2><p>本地优先的 AI 图片与视频无限画布。</p><div className="about-grid"><div><b>鼠标右键</b><small>快速创建节点</small></div><div><b>拖出连接</b><small>松开后添加并连接</small></div><div><b>联动编辑</b><small>同步关联任务的创作参数</small></div><div><b>本地模型</b><small>自动识别 ComfyUI 可用模型</small></div><div><b>ComfyUI 工作流</b><small>选择模板、模型并直接运行</small></div><div><b>Ctrl + S</b><small>保存当前项目</small></div></div></>}
          {modal === 'more' && <><h2>更多功能</h2><p>右上角 × 只隐藏窗口；需要彻底关闭时请点击下方“退出自由者”或使用系统托盘菜单。</p><div className="template-list"><button onClick={() => { flow?.fitView({ padding: .2 }); setModal(null) }}><span>⌗</span><div><b>适应全部内容</b><small>缩放画布以显示所有节点</small></div><i>执行</i></button><button onClick={() => setModal('history')}><span>◴</span><div><b>保存历史</b><small>查看和保存当前快照</small></div><i>打开</i></button><button onClick={() => setTheme((value) => value === 'dark' ? 'light' : 'dark')}><span>☼</span><div><b>切换主题</b><small>当前为{theme === 'dark' ? '深色' : '浅色'}主题</small></div><i>切换</i></button><button className="danger" onClick={() => window.aaaLite?.quitApp?.()}><span>⏻</span><div><b>退出自由者</b><small>保存后彻底结束后台应用</small></div><i>退出</i></button></div></>}
        </section>
      </div>}
      {toast && <div className={`toast ${/失败|错误|缺少|无效|invalid|failed|not found/i.test(String(toast)) ? 'error' : ''}`}>{/失败|错误|缺少|无效|invalid|failed|not found/i.test(String(toast)) ? '!' : '✓'} {toast}</div>}
    </main>
  )
}

export default App
