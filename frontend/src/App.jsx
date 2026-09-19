import { useEffect, useState } from 'react'
import UploadZone from './components/UploadZone'
import ResultCard from './components/ResultCard'
import { predictImage, getModelInfo } from './services/api'
import './App.css'

function App() {
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [modelInfo, setModelInfo] = useState(null)

  useEffect(() => {
    getModelInfo().then(setModelInfo).catch(() => setModelInfo(null))
  }, [])

  const handleFile = async (file) => {
    setError(null)
    setResult(null)
    setPreview(URL.createObjectURL(file))
    setLoading(true)
    try {
      setResult(await predictImage(file))
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleReset = () => {
    if (preview) URL.revokeObjectURL(preview)
    setPreview(null)
    setResult(null)
    setError(null)
  }

  return (
    <div className="app">
      <header className="header">
        <h1>AI Image Detector</h1>
        <p>Is this image real or AI-generated?</p>
        {modelInfo && (
          <span className="model-badge">
            {modelInfo.model} · {modelInfo.test_accuracy ? (modelInfo.test_accuracy * 100).toFixed(1) + '% test accuracy' : ''}
          </span>
        )}
      </header>

      <main className="content">
        {!preview && <UploadZone onFile={handleFile} onError={setError} />}

        {preview && (
          <div className="analysis">
            <div className="preview-card">
              <img src={preview} alt="Uploaded preview" />
            </div>

            {loading && (
              <div className="status-card">
                <div className="spinner" />
                <p>Analyzing image…</p>
              </div>
            )}

            {error && (
              <div className="status-card error">
                <p>⚠️ {error}</p>
                <button className="btn-primary" onClick={handleReset}>Try again</button>
              </div>
            )}

            {result && <ResultCard result={result} onReset={handleReset} />}
          </div>
        )}
      </main>

      <footer className="footer">
        <p>Trained on CIFAKE (Stable Diffusion vs CIFAR-10) — see README for limitations</p>
      </footer>
    </div>
  )
}

export default App