import React, { useState, useEffect, useRef } from 'react'

export default function Submit() {
  const [hovered, setHovered] = useState(null)
  const [tick, setTick] = useState(0)
  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 100)
    return () => clearInterval(id)
  }, [])

  // Arrow-around-OR that points toward the mouse
  const orRef = useRef(null)
  const [mouse, setMouse] = useState({ x: 0, y: 0 })
  const [orCenter, setOrCenter] = useState({ x: 0, y: 0 })
  useEffect(() => {
    const onMove = (e) => {
      setMouse({ x: e.clientX, y: e.clientY })
    }
    window.addEventListener('mousemove', onMove)
    const updateCenter = () => {
      const el = orRef.current
      if (!el) return
      const r = el.getBoundingClientRect()
      setOrCenter({ x: r.left + r.width / 2, y: r.top + r.height / 2 })
    }
    updateCenter()
    window.addEventListener('resize', updateCenter)
    const obs = new MutationObserver(updateCenter)
    if (orRef.current) obs.observe(orRef.current, { attributes: true, childList: true, subtree: true })
    return () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('resize', updateCenter)
      obs.disconnect()
    }
  }, [])

  // Prefer simple centered OR on mobile/touch or narrow screens
  const [showArrow, setShowArrow] = useState(true)
  useEffect(() => {
    const compute = () => {
      const fine = typeof window.matchMedia === 'function' && window.matchMedia('(pointer: fine)').matches
      const wide = window.innerWidth >= 768
      setShowArrow(Boolean(fine && wide))
    }
    compute()
    window.addEventListener('resize', compute)
    return () => window.removeEventListener('resize', compute)
  }, [])
  const onProposal = () => {
    alert('Submit proposal clicked')
  }
  const onShip = () => {
    alert('Ship clicked')
  }

  const baseBtn = {
    padding: '28px 36px',
    borderRadius: 16,
    border: '3px solid #111',
    background: '#fff',
    color: '#111',
    fontWeight: 800,
    fontSize: 28,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    transition: 'transform 160ms ease, box-shadow 160ms ease',
    cursor: 'pointer',
    width: '80%',
    maxWidth: 720,
    height: 200,
    boxShadow: '0 10px 28px rgba(0,0,0,0.08)'
  }

  return (
    <div>
      <div style={{ display: 'flex', gap: 24, alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', minHeight: '70vh' }}>
        <div style={{ flex: '1 1 480px', display: 'flex', justifyContent: 'center', position: 'relative' }}>
          <button
            onClick={onProposal}
            onMouseEnter={(e) => { e.currentTarget.style.transform = 'scale(1.06)'; setHovered('propose') }}
            onMouseLeave={(e) => { e.currentTarget.style.transform = 'scale(1.0)'; setHovered(null) }}
            style={baseBtn}
          >
            propose
          </button>
          <div style={{ position: 'absolute', marginTop: 220, width: '80%', maxWidth: 720, textAlign: 'center', color: '#333', opacity: hovered === 'propose' ? 1 : 0, transition: 'opacity 180ms ease' }}>
            {(() => {
              const word = 'Pitch'
              const rest = ' your new project in a 1-2 minute video with voiceover'
              const letters = Array.from(word)
              const n = letters.length
              const period = Math.max(1, 2 * n - 2)
              let k = tick % period
              if (k >= n) k = period - k
              return (
                <span style={{ fontFamily: 'ui-monospace, Menlo, Monaco, Consolas, "Liberation Mono", monospace', letterSpacing: 0.2 }}>
                  {letters.map((ch, i) => {
                    const active = i === k
                    const weight = active ? 900 : 700
                    const shade = active ? '#000' : '#333'
                    const glow = active ? '0 0 12px rgba(0,0,0,0.25)' : 'none'
                    return (
                      <span key={i} style={{ fontWeight: weight, color: shade, textShadow: glow }}>
                        {ch}
                      </span>
                    )
                  })}
                  <span style={{ fontWeight: 500, color: '#333' }}>{rest}</span>
                </span>
              )
            })()}
          </div>
        </div>
        <div ref={orRef} style={{ position: 'relative', flex: '0 0 auto', alignSelf: 'center', fontWeight: 700, color: '#000', opacity: 0.9, fontSize: 18, letterSpacing: 1, width: 64, height: 64, display: 'inline-flex', alignItems: 'center', justifyContent: 'center' }}>
          <span>OR</span>
          {showArrow && (() => {
            const dx = mouse.x - orCenter.x
            const dy = mouse.y - orCenter.y
            const ang = Math.atan2(dy, dx) * 180 / Math.PI
            const radius = 40
            return (
              <div style={{ position: 'absolute', top: '50%', left: '50%', transform: `translate(-50%, -50%) rotate(${ang}deg) translate(${radius}px)`, transformOrigin: 'center', pointerEvents: 'none' }}>
                <svg width="26" height="26" viewBox="0 0 24 24" style={{ display: 'block' }} aria-hidden="true">
                  <path d="M2 12 H16 M10 6 L16 12 L10 18" stroke="#000" strokeWidth="3" fill="none" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </div>
            )
          })()}
        </div>
        <div style={{ flex: '1 1 480px', display: 'flex', justifyContent: 'center', position: 'relative' }}>
          <button
            onClick={onShip}
            onMouseEnter={(e) => { e.currentTarget.style.transform = 'scale(1.06)'; setHovered('ship') }}
            onMouseLeave={(e) => { e.currentTarget.style.transform = 'scale(1.0)'; setHovered(null) }}
            style={baseBtn}
          >
            ship
          </button>
          <div style={{ position: 'absolute', marginTop: 220, width: '80%', maxWidth: 720, textAlign: 'center', color: '#333', opacity: hovered === 'ship' ? 1 : 0, transition: 'opacity 180ms ease' }}>
            <span style={{ fontFamily: 'ui-monospace, Menlo, Monaco, Consolas, "Liberation Mono", monospace', letterSpacing: 0.2 }}>
              Submit your shipped project along with a 5 minute video w/ voiceover for your{' '}
              {(() => {
                const colors = ['#ff4757', '#ffa502', '#fffa65', '#2ed573', '#1e90ff', '#a29bfe', '#e056fd']
                const letters = ['p', 'r', 'i', 'z', 'e']
                return (
                  <span aria-label="prize animated" style={{ display: 'inline-block' }}>
                    {letters.map((ch, i) => {
                      const color = colors[(tick + i) % colors.length]
                      const bounce = Math.sin((tick + i) * 0.5) * 3
                      return (
                        <span key={i} style={{ color, display: 'inline-block', transform: `translateY(${bounce}px)` }}>
                          {ch}
                        </span>
                      )
                    })}
                  </span>
                )
              })()}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
