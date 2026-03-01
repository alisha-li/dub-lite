import { useState, useRef, useEffect } from 'react'
import './App.css'

// Same origin on real domain (Nginx proxies /api/). Localhost dev uses backend URL.
function getApiBase() {
  if (typeof import.meta.env?.VITE_API_BASE === 'string' && import.meta.env.VITE_API_BASE) return import.meta.env.VITE_API_BASE
  if (typeof window !== 'undefined' && window.location?.hostname !== 'localhost') return ''
  return 'http://localhost:8000'
}
const API_BASE = getApiBase()

function App() {
  const [file, setFile] = useState(null)
  const [targetLanguage, setTargetLanguage] = useState('en')
  const [isDragging, setIsDragging] = useState(false)
  const [jobId, setJobId] = useState(null)
  const [jobStatus, setJobStatus] = useState(null) // 'pending' | 'completed' | 'failed'
  const [jobError, setJobError] = useState(null)
  const [jobProgress, setJobProgress] = useState(0)
  const [jobStage, setJobStage] = useState('')
  const [outputUrl, setOutputUrl] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [translationProvider, setTranslationProvider] = useState('translategemma') // 'groq' | 'gemini' | 'translategemma'
  const [hfToken, setHfToken] = useState('')
  const [pyannoteKey, setPyannoteKey] = useState('')
  const [geminiApi, setGeminiApi] = useState('')
  const [geminiModel, setGeminiModel] = useState('')
  const [groqApi, setGroqApi] = useState('')
  const [groqModel, setGroqModel] = useState('')
  const fileInputRef = useRef(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setJobError(null)
    setJobStatus(null)
    setOutputUrl(null)

    try {
      // 1. Get presigned upload URL from API
      setUploading(true)
      setJobStage('Getting upload URL...')
      const urlForm = new FormData()
      urlForm.append('filename', file.name)
      const urlRes = await fetch(`${API_BASE}/api/upload-url`, { method: 'POST', body: urlForm })
      const urlData = await urlRes.json()
      if (!urlRes.ok) throw new Error(urlData.detail || 'Failed to get upload URL')

      // 2. Upload file directly to Spaces via presigned PUT
      setJobStage('Uploading video...')
      const putRes = await fetch(urlData.upload_url, {
        method: 'PUT',
        headers: { 'Content-Type': 'video/mp4' },
        body: file,
      })
      if (!putRes.ok) throw new Error(`Upload failed (${putRes.status})`)
      setUploading(false)

      // 3. Create job with the Spaces object key (no file bytes)
      setJobStage('Starting...')
      const formData = new FormData()
      formData.append('spaces_object_key', urlData.object_key)
      formData.append('target_language', targetLanguage)
      if (translationProvider === 'groq') {
        if (groqApi.trim()) formData.append('groq_api', groqApi.trim())
        if (groqModel.trim()) formData.append('groq_model', groqModel.trim())
      }
      if (translationProvider === 'gemini') {
        if (geminiApi.trim()) formData.append('gemini_api', geminiApi.trim())
        if (geminiModel.trim()) formData.append('gemini_model', geminiModel.trim())
      }
      if (translationProvider === 'translategemma') formData.append('translation_provider', 'translategemma')
      if (pyannoteKey.trim()) formData.append('pyannote_key', pyannoteKey.trim())

      const res = await fetch(`${API_BASE}/api/jobs`, { method: 'POST', body: formData })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || data.message || `Request failed (${res.status})`)

      setJobId(data.job_id)
      setJobStatus('pending')
      setJobProgress(0)
      setJobStage('Starting...')
    } catch (err) {
      setUploading(false)
      const msg = err.message || 'Network error — is the API running?'
      setJobError(msg)
      setJobStatus('failed')
      console.error('Submit failed:', msg, err)
    }
  }

  // Poll job status until completed or failed; update progress and stage while processing
  useEffect(() => {
    if (!jobId || jobStatus !== 'pending') return
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/jobs/${jobId}`)
        const data = await res.json()
        if (data.status === 'completed') {
          setJobStatus('completed')
          setJobProgress(100)
          setJobStage('Done')
          if (data.output_url) setOutputUrl(data.output_url)
          clearInterval(interval)
        } else if (data.status === 'failed') {
          setJobStatus('failed')
          setJobError(data.error || 'Job failed')
          clearInterval(interval)
        } else if (data.status === 'processing') {
          setJobProgress(data.progress ?? 0)
          setJobStage(data.stage ?? '')
        }
      } catch (err) {
        console.error('Polling job status failed:', err)
      }
    }, 2000)
    return () => clearInterval(interval)
  }, [jobId, jobStatus])

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    const dropped = e.dataTransfer.files[0]
    if (dropped?.type.startsWith('video/')) {
      setFile(dropped)
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => setIsDragging(false)

  const handleFileChange = (e) => {
    const chosen = e.target.files?.[0]
    if (chosen) {
      setFile(chosen)
    }
  }

  const clearSource = () => {
    setFile(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const hasSource = !!file
  const noKeysProvided = !groqApi.trim() && !geminiApi.trim() && !hfToken.trim()
  const hasRequiredApiKey =
    (translationProvider === 'groq' && groqApi.trim()) ||
    (translationProvider === 'gemini' && geminiApi.trim()) ||
    (translationProvider === 'helsinki' && hfToken.trim())
  // Allow submit with just source + language (use host default); or with a chosen provider + key
  const canSubmit = hasSource && targetLanguage && (noKeysProvided || hasRequiredApiKey)

  return (
    <>
      <a
        href="https://github.com/alisha-li/dub-lite"
        target="_blank"
        rel="noopener noreferrer"
        className="github-corner"
        aria-label="View source on GitHub"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
          <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
        </svg>
        <span>GitHub</span>
      </a>
      <div className="app">
      <h1>Dub-Lite</h1>
      <p className="tagline">Upload a video. Pick a target language. Sit back, and let it dub.</p>

      <form onSubmit={handleSubmit} className="form">
        <div className="section">
          <div
            className={`dropzone ${isDragging ? 'dropzone--active' : ''} ${file ? 'dropzone--has-file' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*"
              onChange={handleFileChange}
              className="dropzone-input"
            />
            {file ? (
              <span className="dropzone-file">{file.name}</span>
            ) : (
              <>
                <span className="dropzone-icon">↓</span>
                <span className="dropzone-text">Drop a video here or click to browse</span>
              </>
            )}
          </div>

          {file && (
            <button type="button" className="clear-btn" onClick={clearSource}>
              Clear
            </button>
          )}
        </div>

        <div className="row">
          <label className="label">Target language</label>
          <select
            value={targetLanguage}
            onChange={(e) => setTargetLanguage(e.target.value)}
            className="select"
          >
            <option value="en">English</option>
            <option value="es">Spanish</option>
            <option value="fr">French</option>
            <option value="de">German</option>
            <option value="it">Italian</option>
            <option value="ru">Russian</option>
            <option value="zh">Chinese</option>
          </select>
          <p className="field-desc">Must be different from the source language.</p>
        </div>

        <div className="advanced">
          <button
            type="button"
            className="advanced-toggle"
            onClick={() => setAdvancedOpen((o) => !o)}
            aria-expanded={advancedOpen}
          >
            {advancedOpen ? '▼' : '▶'} Advanced settings
          </button>
          {advancedOpen && (
            <div className="advanced-fields">
              <div className="advanced-field">
                <label className="label">Translation / API</label>
                <p className="field-desc">Pick one. Only the selected option’s field(s) are used.</p>
                <div className="provider-options">
                  <label className="provider-option">
                    <input
                      type="radio"
                      name="translationProvider"
                      value="groq"
                      checked={translationProvider === 'groq'}
                      onChange={() => setTranslationProvider('groq')}
                    />
                    <span>Groq - faster translations</span>
                  </label>
                  <label className="provider-option">
                    <input
                      type="radio"
                      name="translationProvider"
                      value="gemini"
                      checked={translationProvider === 'gemini'}
                      onChange={() => setTranslationProvider('gemini')}
                    />
                    <span>Gemini - highest quality translation</span>
                  </label>
                  <label className="provider-option">
                    <input
                      type="radio"
                      name="translationProvider"
                      value="translategemma"
                      checked={translationProvider === 'translategemma'}
                      onChange={() => setTranslationProvider('translategemma')}
                    />
                    <span>TranslateGemma - free translations</span>
                  </label>
                </div>
              </div>

              {translationProvider === 'groq' && (
                <>
                  <div className="advanced-field">
                    <label className="label">Groq API key</label>
                    <input
                      type="password"
                      className="input url-input"
                      value={groqApi}
                      onChange={(e) => setGroqApi(e.target.value)}
                      placeholder="Required"
                    />
                    <p className="field-desc">Faster translation.</p>
                  </div>
                  <div className="advanced-field">
                    <label className="label">Groq model</label>
                    <input
                      type="text"
                      className="input url-input"
                      value={groqModel}
                      onChange={(e) => setGroqModel(e.target.value)}
                      placeholder="e.g. openai/gpt-oss-120b"
                    />
                    <p className="field-desc">Model used for translation when Groq is selected.</p>
                  </div>
                </>
              )}

              {translationProvider === 'gemini' && (
                <>
                  <div className="advanced-field">
                    <label className="label">Gemini API key</label>
                    <input
                      type="password"
                      className="input url-input"
                      value={geminiApi}
                      onChange={(e) => setGeminiApi(e.target.value)}
                      placeholder="Required"
                    />
                    <p className="field-desc">Better translation quality.</p>
                  </div>
                  <div className="advanced-field">
                    <label className="label">Gemini model</label>
                    <input
                      type="text"
                      className="input url-input"
                      value={geminiModel}
                      onChange={(e) => setGeminiModel(e.target.value)}
                      placeholder="e.g. gemini-2.5-flash-lite"
                    />
                    <p className="field-desc">Model used for translation when Gemini is selected.</p>
                  </div>
                </>
              )}

              {translationProvider === 'translategemma' && (
                <div className="advanced-field">
                  <p className="field-desc">Free translation powered by Google's TranslateGemma model. No API key required.</p>
                </div>
              )}
            </div>
          )}
        </div>

        <button type="submit" className="submit" disabled={!canSubmit || uploading || jobStatus === 'pending'}>
          {uploading ? 'Uploading...' : jobStatus === 'pending' ? 'Processing...' : 'Start dubbing'}
        </button>

        {(uploading || jobStatus === 'pending') && (
          <p className="job-result-note job-result-note-inline">
            This may take a few minutes. Feel free to do something else and check back later.
          </p>
        )}
      </form>

      {jobError && !jobId && (
        <div className="job-result job-result-error-wrap">
          <p className="job-result-error">Error: {jobError}</p>
        </div>
      )}

      {jobId && (
        <div className="job-result">
          <div className="status">
            {jobStatus === 'pending' && (
              <>
                <p className="job-result-stage">{jobStage}</p>
                <div className="progress-bar-wrap">
                  <div className="progress-bar" style={{ width: `${jobProgress}%` }} />
                </div>
                <p className="progress-percent">{jobProgress}%</p>
                <p className="job-id-note">Job ID: <code>{jobId}</code></p>
              </>
            )}
            {jobStatus === 'failed' && (
              <p className="job-result-error">Failed: {jobError}</p>
            )}
            {jobStatus === 'completed' && outputUrl && (
              <>
                <p>Done. Watch below or download.</p>
                <div className="video-container">
                  <video
                    src={outputUrl}
                    controls
                    className="result-video"
                  >
                    Your browser does not support the video tag.
                  </video>
                </div>
                <a
                  href={outputUrl}
                  download={`dubbed_${jobId}.mp4`}
                  className="download-btn"
                >
                  Download video
                </a>
              </>
            )}
          </div>
        </div>
      )}
    </div>
    </>
  )
}

export default App
