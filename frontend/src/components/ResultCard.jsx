export default function ResultCard({ result, onReset }) {
  const isFake = result.label === 'FAKE'
  const pct = (x) => (x * 100).toFixed(1)

  return (
    <div className={`result-card ${isFake ? 'fake' : 'real'}`}>
      <div className="result-verdict">
        <span className="result-icon">{isFake ? '🤖' : '📷'}</span>
        <div>
          <h2>{isFake ? 'AI-Generated' : 'Real Image'}</h2>
          <p>The model predicts this image is {isFake ? 'AI-generated' : 'real'}.</p>
        </div>
      </div>

      <div className="confidence-section">
        <div className="confidence-header">
          <span>Confidence</span>
          <span>{pct(result.confidence)}%</span>
        </div>
        <div className="confidence-bar">
          <div className="confidence-fill" style={{ width: `${result.confidence * 100}%` }} />
        </div>
      </div>

      <div className="probability-row">
        <div className="probability-chip fake">FAKE {pct(result.probabilities.FAKE)}%</div>
        <div className="probability-chip real">REAL {pct(result.probabilities.REAL)}%</div>
      </div>

      <div className="result-meta">Analyzed in {result.prediction_time_ms} ms</div>

      <button className="btn-primary" onClick={onReset}>Analyze another image</button>
    </div>
  )
}