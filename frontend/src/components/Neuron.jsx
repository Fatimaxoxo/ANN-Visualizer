import React from 'react'

function Neuron({ label, layerName }) {
  return (
    <div className={`neuron-card ${layerName.toLowerCase()}`}>
      <div className="neuron-circle">○</div>
      <span className="neuron-label">{label}</span>
    </div>
  )
}

export default Neuron
