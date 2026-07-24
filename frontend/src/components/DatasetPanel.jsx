import React, { useEffect, useState } from 'react'

function DatasetPanel() {
  const [selectedDataset, setSelectedDataset] = useState('toy_classification.csv')
  const [rows, setRows] = useState([])
  const [message, setMessage] = useState('')

  useEffect(() => {
    loadDataset(selectedDataset)
  }, [selectedDataset])

  const loadDataset = async (filename) => {
    try {
      const response = await fetch(`../../datasets/${filename}`)
      const text = await response.text()
      const parsedRows = parseCsv(text)
      setRows(parsedRows)
      setMessage(`Loaded ${filename} with ${parsedRows.length} rows.`)
    } catch (error) {
      setMessage('Default dataset could not be loaded.')
    }
  }

  const parseCsv = (text) => {
    const lines = text.trim().split(/\r?\n/)
    const headers = lines[0].split(',')
    return lines.slice(1).filter(Boolean).map((line) => {
      const values = line.split(',')
      return headers.reduce((acc, header, index) => {
        acc[header] = values[index]
        return acc
      }, {})
    })
  }

  const handleUpload = async (event) => {
    const file = event.target.files?.[0]
    if (!file) return

    const text = await file.text()
    const parsedRows = parseCsv(text)
    setRows(parsedRows)
    setSelectedDataset(file.name)
    setMessage(`Uploaded ${file.name} with ${parsedRows.length} rows.`)
  }

  return (
    <section className="card">
      <h2>Dataset Loader</h2>
      <p className="muted">The app loads a bundled dataset by default and supports CSV upload as an override.</p>

      <div className="controls-grid">
        <label>
          Default dataset
          <select value={selectedDataset} onChange={(event) => setSelectedDataset(event.target.value)}>
            <option value="toy_classification.csv">toy_classification.csv</option>
            <option value="toy_regression.csv">toy_regression.csv</option>
          </select>
        </label>
        <label>
          Upload your own CSV
          <input type="file" accept=".csv" onChange={handleUpload} />
        </label>
      </div>

      <p className="muted">{message}</p>

      <div className="table-card">
        <h3>Preview</h3>
        <pre>{JSON.stringify(rows.slice(0, 5), null, 2)}</pre>
      </div>
    </section>
  )
}

export default DatasetPanel
