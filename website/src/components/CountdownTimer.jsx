import React, { useEffect, useMemo, useState } from 'react'

// Floating countdown timer with a digital-clock feel (bottom-right on all pages)
export default function CountdownTimer() {
  const detectDeadline = () => {
    try {
      const url = new URL(window.location.href)
      const q = url.searchParams.get('deadline')
      if (q) {
        const t = new Date(q)
        if (!isNaN(t.getTime())) return t
      }
    } catch {}
    try {
      const s = localStorage.getItem('countdownDeadline')
      if (s) {
        const t = new Date(s)
        if (!isNaN(t.getTime())) return t
      }
    } catch {}
    const fallback = new Date(Date.now() + 30 * 24 * 60 * 60 * 1000)
    try { localStorage.setItem('countdownDeadline', fallback.toISOString()) } catch {}
    return fallback
  }

  const [deadline] = useState(() => detectDeadline())
  const [now, setNow] = useState(Date.now())
  const [blink, setBlink] = useState(true)
  const [hover, setHover] = useState(false)

  useEffect(() => {
    const id = setInterval(() => {
      setNow(Date.now())
      setBlink((b) => !b)
    }, 1000)
    return () => clearInterval(id)
  }, [])

  const parts = useMemo(() => {
    const ms = Math.max(0, deadline.getTime() - now)
    const totalSec = Math.floor(ms / 1000)
    const days = Math.floor(totalSec / 86400)
    const hours = Math.floor((totalSec % 86400) / 3600)
    const mins = Math.floor((totalSec % 3600) / 60)
    const secs = totalSec % 60
    const pad2 = (n) => String(n).padStart(2, '0')
    return { days, hours: pad2(hours), mins: pad2(mins), secs: pad2(secs) }
  }, [deadline, now])

  const segmentStyle = {
    padding: '6px 8px',
    background: '#ffffff',
    border: '1px solid rgba(0,0,0,0.12)',
    borderRadius: 6,
    boxShadow: '0 6px 18px rgba(0,0,0,0.08) inset',
    color: '#111111',
    fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace',
    fontWeight: 800,
    fontSize: 20,
    lineHeight: 1,
    letterSpacing: 1
  }

  const labelStyle = { color: '#666', fontSize: 10, marginTop: 4, textAlign: 'center' }

  return (
    <div
      aria-label="countdown"
      title={`Deadline: ${deadline.toLocaleString()}`}
      style={{
        position: 'fixed',
        right: 12,
        bottom: 12,
        zIndex: 3,
        background: 'rgba(255,255,255,0.85)',
        backdropFilter: 'blur(6px)',
        WebkitBackdropFilter: 'blur(6px)',
        padding: 8,
        borderRadius: 10,
        border: '1px solid rgba(0,0,0,0.08)',
        boxShadow: '0 10px 28px rgba(0,0,0,0.08)',
        color: '#111',
        userSelect: 'none'
      }}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
    >
      <div style={{ position: 'absolute', right: 0, bottom: '100%', marginBottom: 6, padding: '6px 8px', background: 'rgba(255,255,255,0.95)', border: '1px solid rgba(0,0,0,0.12)', borderRadius: 8, boxShadow: '0 6px 18px rgba(0,0,0,0.08)', color: '#111', fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace', fontSize: 12, letterSpacing: 0.5, opacity: hover ? 1 : 0, transition: 'opacity 160ms ease', pointerEvents: 'none', whiteSpace: 'nowrap' }}>
        The deadline for submissions is {deadline.toLocaleString()}
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <div style={{ textAlign: 'center' }}>
          <div style={segmentStyle}>{parts.days}</div>
          <div style={labelStyle}>DAYS</div>
        </div>
        <div style={{ fontSize: 22, fontWeight: 900, color: '#999', opacity: blink ? 1 : 0.25 }}>:</div>
        <div style={{ textAlign: 'center' }}>
          <div style={segmentStyle}>{parts.hours}</div>
          <div style={labelStyle}>HRS</div>
        </div>
        <div style={{ fontSize: 22, fontWeight: 900, color: '#999', opacity: blink ? 1 : 0.25 }}>:</div>
        <div style={{ textAlign: 'center' }}>
          <div style={segmentStyle}>{parts.mins}</div>
          <div style={labelStyle}>MIN</div>
        </div>
        <div style={{ fontSize: 22, fontWeight: 900, color: '#999', opacity: blink ? 1 : 0.25 }}>:</div>
        <div style={{ textAlign: 'center' }}>
          <div style={segmentStyle}>{parts.secs}</div>
          <div style={labelStyle}>SEC</div>
        </div>
      </div>
    </div>
  )
}


