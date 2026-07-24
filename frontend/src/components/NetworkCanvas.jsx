import React, { useMemo } from 'react'
import { ReactFlow, Background, Controls, Position, BaseEdge, EdgeLabelRenderer, getStraightPath } from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import Neuron from './Neuron'
import Edge from './Edge'

const nodeDefaults = {
  sourcePosition: Position.Right,
  targetPosition: Position.Left,
}

function CustomEdge({ id, sourceX, sourceY, targetX, targetY, label, markerEnd, style }) {
  const [path, labelX, labelY] = getStraightPath({ sourceX, sourceY, targetX, targetY })

  return (
    <>
      <BaseEdge id={id} path={path} markerEnd={markerEnd} style={style} />
      <EdgeLabelRenderer>
        <div
          style={{
            position: 'absolute',
            transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
            pointerEvents: 'none',
          }}
        >
          <Edge label={label} />
        </div>
      </EdgeLabelRenderer>
    </>
  )
}

function NetworkCanvas() {
  const nodes = useMemo(
    () => [
      {
        id: 'input-1',
        type: 'input',
        position: { x: 40, y: 120 },
        data: { label: <Neuron label="x1" layerName="Input" /> },
        ...nodeDefaults,
      },
      {
        id: 'input-2',
        position: { x: 40, y: 220 },
        data: { label: <Neuron label="x2" layerName="Input" /> },
        ...nodeDefaults,
      },
      {
        id: 'input-3',
        position: { x: 40, y: 320 },
        data: { label: <Neuron label="x3" layerName="Input" /> },
        ...nodeDefaults,
      },
      {
        id: 'hidden-1',
        position: { x: 320, y: 120 },
        data: { label: <Neuron label="h1" layerName="Hidden" /> },
        ...nodeDefaults,
      },
      {
        id: 'hidden-2',
        position: { x: 320, y: 220 },
        data: { label: <Neuron label="h2" layerName="Hidden" /> },
        ...nodeDefaults,
      },
      {
        id: 'hidden-3',
        position: { x: 320, y: 320 },
        data: { label: <Neuron label="h3" layerName="Hidden" /> },
        ...nodeDefaults,
      },
      {
        id: 'output-1',
        position: { x: 600, y: 220 },
        data: { label: <Neuron label="y" layerName="Output" /> },
        ...nodeDefaults,
      },
    ],
    []
  )

  const edges = useMemo(
    () => [
      { id: 'e1', source: 'input-1', target: 'hidden-1', type: 'customEdge', label: 'w1' },
      { id: 'e2', source: 'input-2', target: 'hidden-1', type: 'customEdge', label: 'w2' },
      { id: 'e3', source: 'input-3', target: 'hidden-1', type: 'customEdge', label: 'w3' },
      { id: 'e4', source: 'input-1', target: 'hidden-2', type: 'customEdge', label: 'w4' },
      { id: 'e5', source: 'input-2', target: 'hidden-2', type: 'customEdge', label: 'w5' },
      { id: 'e6', source: 'input-3', target: 'hidden-2', type: 'customEdge', label: 'w6' },
      { id: 'e7', source: 'input-1', target: 'hidden-3', type: 'customEdge', label: 'w7' },
      { id: 'e8', source: 'input-2', target: 'hidden-3', type: 'customEdge', label: 'w8' },
      { id: 'e9', source: 'input-3', target: 'hidden-3', type: 'customEdge', label: 'w9' },
      { id: 'e10', source: 'hidden-1', target: 'output-1', type: 'customEdge', label: 'w10' },
      { id: 'e11', source: 'hidden-2', target: 'output-1', type: 'customEdge', label: 'w11' },
      { id: 'e12', source: 'hidden-3', target: 'output-1', type: 'customEdge', label: 'w12' },
    ],
    []
  )

  const edgeTypes = useMemo(
    () => ({ customEdge: CustomEdge }),
    []
  )

  return (
    <div className="canvas-card">
      <div className="canvas-header">
        <div>
          <p className="eyebrow">Phase 2 • Visualization</p>
          <h2>Network Structure</h2>
        </div>
      </div>
      <div className="react-flow-wrapper">
        <ReactFlow nodes={nodes} edges={edges} edgeTypes={edgeTypes} fitView proOptions={{ hideAttribution: true }}>
          <Background gap={16} size={1} />
          <Controls />
        </ReactFlow>
      </div>
    </div>
  )
}

export default NetworkCanvas
