import { useState } from 'react'
import './App.css'

function App() {
  const [file, setFile] = useState(null)
  const [youtubeUrl, setYoutubeUrl] = useState('')
  const [targetLanguage, setTargetLanguage] = useState('en')
  const [jobId, setJobId] = useState(null)
  const [jobStatus, setJobStatus] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    const formData = new FormData()
    if (file) {
      formData.append('file', file)
    } else if (youtubeUrl) {
      formData.append('source', youtubeUrl)
    }

    formData.append('target_language', targetLanguage)
    const res = await fetch('http://localhost:8000/api/jobs', {
      method: 'POST',
      body: formData
    })

    const data = await res.json()
    setJobId(data.job_id)
    setJobStatus('pending')
  }

  return (
    <div className="app">
      <h1>Video Dubbing</h1>

      <form onSubmit={handleSubmit}>
        {/* File upload */}
        <div>
          <label>Upload Video</label>
          <input
            type="file"
            accept="video/*"
            onChange={(e) => setFile(e.target.files[0])}
          />
        </div>

        {/* OR Youtube URL */}
        <div>
          <label>Or enter a YouTube URL:</label>
          <input
            type="text"
            value={youtubeUrl}
            onChange={(e) => setYoutubeUrl(e.target.value)}
            placeholder = "https://www.youtube.com/watch?v=..."
          />
        </div>

        {/* Language selector */}
        <div>
          <label>Target Language:</label>
          <select
            value={targetLanguage}
            onChange={(e) => setTargetLanguage(e.target.value)}
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
        <button type="submit">Start Dubbing</button>
      </form>
      
      {/* Form will go here */}
      
      {/* Status display will go here */}
    </div>
  )
}

export default App
