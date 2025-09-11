import React from 'react'

export default function Submit() {
  const onProposal = () => {
    alert('Submit proposal clicked')
  }
  const onShip = () => {
    alert('Ship clicked')
  }

  const baseBtn = {
    padding: '18px 28px',
    borderRadius: 14,
    border: '2px solid #111',
    background: '#fff',
    color: '#111',
    fontWeight: 700,
    textTransform: 'uppercase',
    transition: 'transform 160ms ease',
    cursor: 'pointer',
    width: '100%',
    maxWidth: 320
  }

  return (
    <div>
      <h2>Submit</h2>
      <div style={{ display: 'flex', gap: 24, alignItems: 'stretch', justifyContent: 'space-between', flexWrap: 'wrap' }}>
        <div style={{ flex: '1 1 300px', display: 'flex', justifyContent: 'center' }}>
          <button
            onClick={onProposal}
            onMouseEnter={(e) => (e.currentTarget.style.transform = 'scale(1.04)')}
            onMouseLeave={(e) => (e.currentTarget.style.transform = 'scale(1.0)')}
            style={baseBtn}
          >
            submit proposal
          </button>
        </div>
        <div style={{ flex: '1 1 300px', display: 'flex', justifyContent: 'center' }}>
          <button
            onClick={onShip}
            onMouseEnter={(e) => (e.currentTarget.style.transform = 'scale(1.04)')}
            onMouseLeave={(e) => (e.currentTarget.style.transform = 'scale(1.0)')}
            style={baseBtn}
          >
            ship
          </button>
        </div>
      </div>
    </div>
  )
}
