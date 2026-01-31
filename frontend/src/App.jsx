import { useState, useRef } from 'react'
import './App.css'

function App() {
  const [file, setFile] = useState(null)
  const [youtubeUrl, setYoutubeUrl] = useState('')
  const [targetLanguage, setTargetLanguage] = useState('en')
  const [isDragging, setIsDragging] = useState(false)
  const [jobId, setJobId] = useState(null)
  const [jobStatus, setJobStatus] = useState(null)
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [hfToken, setHfToken] = useState('')
  const [pyannoteKey, setPyannoteKey] = useState('')
  const [geminiApi, setGeminiApi] = useState('')
  const [geminiModel, setGeminiModel] = useState('')
  const [groqApi, setGroqApi] = useState('')
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
    if (hfToken.trim()) formData.append('hf_token', hfToken.trim())
    if (pyannoteKey.trim()) formData.append('pyannote_key', pyannoteKey.trim())
    if (geminiApi.trim()) formData.append('gemini_api', geminiApi.trim())
    if (geminiModel.trim()) formData.append('gemini_model', geminiModel.trim())
    if (groqApi.trim()) formData.append('groq_api', groqApi.trim())

    const res = await fetch('http://localhost:8000/api/jobs', {
      method: 'POST',
      body: formData,
    })
    const data = await res.json()
    setJobId(data.job_id)
    setJobStatus('pending')
  }

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
  const canSubmit = hasSource && targetLanguage

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
                <label className="label">Hugging Face token</label>
                <input
                  type="password"
                  className="input url-input"
                  value={hfToken}
                  onChange={(e) => setHfToken(e.target.value)}
                  placeholder="Optional"
                />
                <p className="field-desc">Free speaker diarization (who spoke when).</p>
              </div>
              <div className="advanced-field">
                <label className="label">Pyannote API key</label>
                <input
                  type="password"
                  className="input url-input"
                  value={pyannoteKey}
                  onChange={(e) => setPyannoteKey(e.target.value)}
                  placeholder="Optional"
                />
                <p className="field-desc">Better diarization quality (paid).</p>
              </div>
              <div className="advanced-field">
                <label className="label">Gemini API key</label>
                <input
                  type="password"
                  className="input url-input"
                  value={geminiApi}
                  onChange={(e) => setGeminiApi(e.target.value)}
                  placeholder="Optional"
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
                <p className="field-desc">Model used for translation when Gemini key is set.</p>
              </div>
              <div className="advanced-field">
                <label className="label">Groq API key</label>
                <input
                  type="password"
                  className="input url-input"
                  value={groqApi}
                  onChange={(e) => setGroqApi(e.target.value)}
                  placeholder="Optional"
                />
                <p className="field-desc">Faster translation (alternative to Gemini).</p>
              </div>
            </div>
          )}
        </div>

        <button type="submit" className="submit" disabled={!canSubmit}>
          Start dubbing
        </button>
      </form>

      {jobId && (
        <div className="status">
          <p>Job started. ID: <code>{jobId}</code></p>
        </div>
      )}
    </div>
  )
}

export default App
