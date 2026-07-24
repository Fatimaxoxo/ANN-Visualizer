import React, { useMemo, useState } from 'react'
import axios from 'axios'

function PerceptronForward() {
  const [inputs, setInputs] = useState('0.20,0.80,0.40')
  const [weights, setWeights] = useState('0.50,0.12,0.70')
  const [bias, setBias] = useState('0.20')
  const [activation, setActivation] = useState('sigmoid')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  const parsedInputs = useMemo(() => inputs.split(',').map((value) => Number(value.trim())), [inputs])
  const parsedWeights = useMemo(() => weights.split(',').map((value) => Number(value.trim())), [weights])

  const runForwardPass = async () => {
    setError('')
    try {
      const response = await axios.post('http://127.0.0.1:5000/api/perceptron/forward', {
        inputs: parsedInputs,
        weights: parsedWeights,
        bias: Number(bias),
        activation,
      })
      setResult(response.data)
    } catch (err) {
  console.error("Forward Pass Error:", err);

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
      <h2>Perceptron Forward Pass</h2>
      <p className="muted">Manual NumPy computation with visible intermediate steps.</p>

      <div className="controls-grid">
        <label>
          Inputs
          <input value={inputs} onChange={(event) => setInputs(event.target.value)} />
        </label>
        <label>
          Weights
          <input value={weights} onChange={(event) => setWeights(event.target.value)} />
        </label>
        <label>
          Bias
          <input value={bias} onChange={(event) => setBias(event.target.value)} />
        </label>
        <label>
          Activation
          <select value={activation} onChange={(event) => setActivation(event.target.value)}>
            <option value="sigmoid">Sigmoid</option>
            <option value="relu">ReLU</option>
            <option value="tanh">Tanh</option>
            <option value="linear">Linear</option>
          </select>
        </label>
      </div>

      <button type="button" onClick={runForwardPass}>Run Forward Pass</button>
      {error ? <p className="error">{error}</p> : null}

      {result ? (
        <div className="results">
          <div className="step-block">
            <h3>1. Input</h3>
            <p>{result.inputs.join(', ')}</p>
          </div>

          <div className="step-block">
            <h3>2. Multiplications</h3>
            <p>{result.products.map((value, index) => `${result.inputs[index]} × ${result.weights[index]} = ${value.toFixed(3)}`).join(' · ')}</p>
          </div>

          <div className="step-block">
            <h3>3. Weighted Sum</h3>
            <p>{result.products.map((value) => value.toFixed(3)).join(' + ')} + {result.bias.toFixed(3)} = {result.weighted_sum.toFixed(3)}</p>
          </div>

          <div className="step-block">
            <h3>4. Activation</h3>
            <p>{result.activation}({result.weighted_sum.toFixed(3)}) = {result.activation_value.toFixed(3)}</p>
          </div>

          <div className="step-block highlight">
            <h3>5. Prediction</h3>
            <p>{result.activation_value.toFixed(3)}</p>
          </div>
        </div>
      ) : null}
    </section>
  )
}

export default PerceptronForward
