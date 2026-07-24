import './App.css'
import NetworkCanvas from './components/NetworkCanvas'
import PerceptronForward from './components/PerceptronForward'
import BackpropagationPanel from './components/BackpropagationPanel'
import DatasetPanel from './components/DatasetPanel'

function App() {
  return (
    <main className="app-shell">
      <section className="hero-card">
        <p className="eyebrow">Phase 3 • Perceptron forward pass</p>
        <h1>ANN Visualizer</h1>
        <p className="intro">
          The forward pass is now computed manually with NumPy and rendered step by
          step so the mathematical path is visible.
        </p>
        <div className="pill-row">
          <span className="pill">Manual math</span>
          <span className="pill">NumPy</span>
          <span className="pill">Step-by-step</span>
        </div>
      </section>

      <NetworkCanvas />
      <PerceptronForward />
      <BackpropagationPanel />
      <DatasetPanel />
    </main>
  )
}

export default App
