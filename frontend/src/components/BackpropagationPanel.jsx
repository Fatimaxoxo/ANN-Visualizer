import React, { useState } from 'react'
import axios from 'axios'

function BackpropagationPanel() {
  const [inputs, setInputs] = useState('0.2,0.8')
  const [weightsOutput, setWeightsOutput] = useState('0.6,0.4')
  const [weightsHidden, setWeightsHidden] = useState('0.2,0.5,0.3,0.7')
  const [target, setTarget] = useState('1')
  const [learningRate, setLearningRate] = useState('0.1')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  const runBackpropagation = async () => {
    setError('')
    try {
      const response = await axios.post('http://127.0.0.1:5000/api/perceptron/backprop', {
        inputs: inputs.split(',').map((value) => Number(value.trim())),
        weights_output: weightsOutput.split(',').map((value) => Number(value.trim())),
        weights_hidden: weightsHidden.split(',').map((value) => Number(value.trim())),
        target: Number(target),
        learning_rate: Number(learningRate),
      })
      setResult(response.data)
    } catch (err) {
  console.error("Backpropagation Error:", err);

  if (err.response) {
    console.error("Status:", err.response.status);
    console.error("Data:", err.response.data);
    setError(`Backend Error (${err.response.status}): ${JSON.stringify(err.response.data)}`);
  } else if (err.request) {
    setError("No response received from backend.");
  } else {
    setError(`Error: ${err.message}`);
  }
}
  }

  return (
    <section className="card">
      <h2>Backpropagation</h2>
      <p className="muted">Output error, delta, gradients, and weight updates are shown step by step.</p>

      <div className="controls-grid">
        <label>
          Inputs
          <input value={inputs} onChange={(event) => setInputs(event.target.value)} />
        </label>
        <label>
          Output weights
          <input value={weightsOutput} onChange={(event) => setWeightsOutput(event.target.value)} />
        </label>
        <label>
          Hidden weights
          <input value={weightsHidden} onChange={(event) => setWeightsHidden(event.target.value)} />
        </label>
        <label>
          Target
          <input value={target} onChange={(event) => setTarget(event.target.value)} />
        </label>
        <label>
          Learning rate
          <input value={learningRate} onChange={(event) => setLearningRate(event.target.value)} />
        </label>
      </div>

      <button type="button" onClick={runBackpropagation}>Run Backpropagation</button>
      {error ? <p className="error">{error}</p> : null}

      {result ? (
        <div className="results">
          <div className="step-block">
            <h3>1. Output Error</h3>
            <p>error = prediction - target = {result.output_activation.toFixed(3)} - {result.target.toFixed(3)} = {result.error.toFixed(3)}</p>
          </div>

          <div className="step-block">
            <h3>2. Output Delta</h3>
            <p>δ = error × y_hat × (1 - y_hat) = {result.error.toFixed(3)} × {result.output_activation.toFixed(3)} × {(1 - result.output_activation).toFixed(3)} = {result.delta_output.toFixed(3)}</p>
          </div>

          <div className="step-block">
            <h3>3. Gradient</h3>
            <p>grad = δ × activation_prev = [{result.gradients_output.map((value) => value.toFixed(3)).join(', ')}]</p>
          </div>

          <div className="step-block">
            <h3>4. Weight Update</h3>
            <p>new weight = old weight - learning rate × gradient = [{result.updated_output_weights.map((value) => value.toFixed(3)).join(', ')}]</p>
          </div>

          <div className="step-block">
            <h3>5. Hidden Delta</h3>
            <p>δ_hidden = δ_output × weight × activation'(hidden) = [{result.hidden_delta.map((value) => value.toFixed(3)).join(', ')}]</p>
          </div>

          <div className="step-block highlight">
            <h3>6. Hidden Weight Update</h3>
            <p>new hidden weights = [{result.updated_hidden_weights.map((value) => value.toFixed(3)).join(', ')}]</p>
          </div>
        </div>
      ) : null}
    </section>
  )
}

export default BackpropagationPanel
