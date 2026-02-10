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
  const [youtubeUrl, setYoutubeUrl] = useState('')
  const [targetLanguage, setTargetLanguage] = useState('en')
  const [isDragging, setIsDragging] = useState(false)
  const [jobId, setJobId] = useState(null)
  const [jobStatus, setJobStatus] = useState(null) // 'pending' | 'completed' | 'failed'
  const [jobError, setJobError] = useState(null)
  const [jobProgress, setJobProgress] = useState(0)
  const [jobStage, setJobStage] = useState('')
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [translationProvider, setTranslationProvider] = useState('helsinki') // 'groq' | 'gemini' | 'helsinki'
  const [hfToken, setHfToken] = useState('')
  const [pyannoteKey, setPyannoteKey] = useState('')
  const [geminiApi, setGeminiApi] = useState('')
  const [geminiModel, setGeminiModel] = useState('')
  const [groqApi, setGroqApi] = useState('')
  const [groqModel, setGroqModel] = useState('')
  const fileInputRef = useRef(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    const formData = new FormData()
    if (file) {
      formData.append('file', file)
    } else if (youtubeUrl.trim()) {
      formData.append('source', youtubeUrl.trim())
    }
    formData.append('target_language', targetLanguage)
    // Only send keys for the selected translation provider (mutually exclusive)
    if (translationProvider === 'groq') {
      if (groqApi.trim()) formData.append('groq_api', groqApi.trim())
      if (groqModel.trim()) formData.append('groq_model', groqModel.trim())
    }
    if (translationProvider === 'gemini') {
      if (geminiApi.trim()) formData.append('gemini_api', geminiApi.trim())
      if (geminiModel.trim()) formData.append('gemini_model', geminiModel.trim())
    }
    if (translationProvider === 'helsinki' && hfToken.trim()) formData.append('hf_token', hfToken.trim())
    if (pyannoteKey.trim()) formData.append('pyannote_key', pyannoteKey.trim())

    try {
      const res = await fetch(`${API_BASE}/api/jobs`, {
        method: 'POST',
        body: formData,
      })
      const data = await res.json()
      if (!res.ok) {
        const msg = data.detail || data.message || `Request failed (${res.status})`
        setJobError(msg)
        setJobStatus('failed')
        return
      }
      setJobId(data.job_id)
      setJobStatus('pending')
      setJobError(null)
      setJobProgress(0)
      setJobStage('Starting...')
    } catch (err) {
      const msg = err.message || 'Network error — is the API running?'
      setJobError(msg)
      setJobStatus('failed')
      console.error('Create job failed:', msg, err)
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
          clearInterval(interval)
        } else if (data.status === 'failed') {
          setJobStatus('failed')
          setJobError(data.error || 'Job failed')
          clearInterval(interval)
        } else if (data.status === 'processing') {
          setJobProgress(data.progress ?? 0)
          setJobStage(data.stage ?? '')
        }
      } catch {
        clearInterval(interval)
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
      setYoutubeUrl('')
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
      setYoutubeUrl('')
    }
  }

  const clearSource = () => {
    setFile(null)
    setYoutubeUrl('')
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const hasSource = file || youtubeUrl.trim()
  const noKeysProvided = !groqApi.trim() && !geminiApi.trim() && !hfToken.trim()
  const hasRequiredApiKey =
    (translationProvider === 'groq' && groqApi.trim()) ||
    (translationProvider === 'gemini' && geminiApi.trim()) ||
    (translationProvider === 'helsinki' && hfToken.trim())
  // Allow submit with just source + language (use host default); or with a chosen provider + key
  const canSubmit = hasSource && targetLanguage && (noKeysProvided || hasRequiredApiKey)

  return (
    <div className="app">
      <h1>Video Dubbing</h1>
      <p className="tagline">Upload a video or paste a YouTube link. Pick a language. We’ll dub it.</p>

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

          <p className="or">or paste a URL</p>
          <input
            type="url"
            className="input url-input"
            value={youtubeUrl}
            onChange={(e) => {
              setYoutubeUrl(e.target.value)
              if (e.target.value.trim()) setFile(null)
            }}
            placeholder="https://www.youtube.com/watch?v=..."
          />

          {(file || youtubeUrl.trim()) && (
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
                    <span>Gemini - higher quality translations</span>
                  </label>
                  <label className="provider-option">
                    <input
                      type="radio"
                      name="translationProvider"
                      value="helsinki"
                      checked={translationProvider === 'helsinki'}
                      onChange={() => setTranslationProvider('helsinki')}
                    />
                    <span>Helsinki - free translations</span>
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

              {translationProvider === 'helsinki' && (
                <div className="advanced-field">
                  <label className="label">Hugging Face token</label>
                  <input
                    type="password"
                    className="input url-input"
                    value={hfToken}
                    onChange={(e) => setHfToken(e.target.value)}
                    placeholder="Required"
                  />
                  <p className="field-desc">Free speaker diarization and translation.</p>
                </div>
              )}

              <div className="advanced-field advanced-field--optional">
                <label className="label">Pyannote API key (optional)</label>
                <input
                  type="password"
                  className="input url-input"
                  value={pyannoteKey}
                  onChange={(e) => setPyannoteKey(e.target.value)}
                  placeholder="Better diarization (paid)"
                />
                <p className="field-desc">Better diarization quality. Can be used with any translation option above.</p>
              </div>
            </div>
          )}
        </div>

        <button type="submit" className="submit" disabled={!canSubmit}>
          Start dubbing
        </button>
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
            {jobStatus === 'completed' && (
              <>
                <p>Done. Watch below or download.</p>
                <div className="video-container">
                  <video
                    src={`${API_BASE}/api/jobs/${jobId}/download`}
                    controls
                    className="result-video"
                  >
                    Your browser does not support the video tag.
                  </video>
                </div>
                <a
                  href={`${API_BASE}/api/jobs/${jobId}/download`}
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
  )
}

export default App
