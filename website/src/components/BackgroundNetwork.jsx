import React, { useEffect, useRef } from 'react'

// Subtle canvas-2D background of nodes connected with edges
export default function BackgroundNetwork() {
  const canvasRef = useRef(null)
  const rafRef = useRef(0)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const state = {
      dpr: Math.min(2, window.devicePixelRatio || 1),
      nodes: [],
      phases: [],
      edges: [],
      worldW: 0,
      worldH: 0,
      x0: 0,
      y0: 0,
      lastDocW: 0,
      lastDocH: 0
    }

    // Hard-coded large content area so background doesn't reflow between routes
    const FIXED_DOC_WIDTH = 8000
    const FIXED_DOC_HEIGHT = 12000

    const fit = () => {
      state.dpr = Math.min(2, window.devicePixelRatio || 1)
      const w = Math.floor(window.innerWidth)
      const h = Math.floor(window.innerHeight)
      canvas.width = Math.floor(w * state.dpr)
      canvas.height = Math.floor(h * state.dpr)
      canvas.style.width = w + 'px'
      canvas.style.height = h + 'px'
      ctx.setTransform(state.dpr, 0, 0, state.dpr, 0, 0)
      // World dimensions for seamless scrolling (cover large fixed area regardless of page content)
      const docW = Math.max(w, FIXED_DOC_WIDTH)
      const docH = Math.max(h, FIXED_DOC_HEIGHT)
      const padX = Math.ceil(w * 0.75)
      const padY = Math.ceil(h * 0.75)
      state.x0 = -padX
      state.y0 = -padY
      state.worldW = docW + padX * 2
      state.worldH = docH + padY * 2
      state.lastDocW = docW
      state.lastDocH = docH
      initNodes()
    }

    const initNodes = () => {
      const area = state.worldW * state.worldH
      const count = Math.min(420, Math.max(100, Math.floor(area / 16000)))
      state.nodes = new Array(count)
      state.phases = new Array(count)
      for (let i = 0; i < count; i++) {
        state.nodes[i] = {
          x: state.x0 + Math.random() * state.worldW,
          y: state.y0 + Math.random() * state.worldH,
          r: 2.5 + Math.random() * 3.5, // smaller base radius (~2.5..6px)
          sp: 12 + Math.floor(Math.random() * 6), // more segments for rounder shape
          wob: Math.random() * Math.PI * 2 // wobble phase
        }
        state.phases[i] = Math.random()
      }
      // Build simple KNN edges (k=3)
      const K = 3
      state.edges = []
      for (let i = 0; i < count; i++) {
        const { x: px, y: py } = state.nodes[i]
        const dists = []
        for (let j = 0; j < count; j++) {
          if (i === j) continue
          const q = state.nodes[j]
          const dx = px - q.x
          const dy = py - q.y
          dists.push([dx * dx + dy * dy, j])
        }
        dists.sort((a, b) => a[0] - b[0])
        for (let k = 0; k < K; k++) state.edges.push([i, dists[k][1]])
      }
    }

    fit()
    window.addEventListener('resize', fit)

    const render = (tMs) => {
      const t = tMs / 1000
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      ctx.save()
      // Scroll-follow with rubber-band/bounce support via VisualViewport deltas
      const baseX = window.scrollX || window.pageXOffset || 0
      const baseY = window.scrollY || window.pageYOffset || 0
      let extraX = 0, extraY = 0
      const vv = window.visualViewport
      if (vv) {
        const pageLeft = typeof vv.pageLeft === 'number' ? vv.pageLeft : baseX
        const pageTop = typeof vv.pageTop === 'number' ? vv.pageTop : baseY
        extraX = pageLeft - baseX
        extraY = pageTop - baseY
      }
      ctx.translate(-(baseX + extraX), -(baseY + extraY))
      ctx.lineWidth = 2
      ctx.strokeStyle = 'rgba(170, 175, 190, 0.10)'
      const amp = 10 // px subtle drift

      // Compute animated positions
      const ax = new Array(state.nodes.length)
      const ay = new Array(state.nodes.length)
      for (let i = 0; i < state.nodes.length; i++) {
        const base = state.nodes[i]
        const ph = state.phases[i]
        ax[i] = base.x + amp * Math.sin(t * 0.2 + ph * 6.2831)
        ay[i] = base.y + amp * Math.cos(t * 0.14 + ph * 6.2831)
      }

      // Draw edges as soft quadratic curves (biological feel)
      for (const [i, j] of state.edges) {
        const x1 = ax[i], y1 = ay[i]
        const x2 = ax[j], y2 = ay[j]
        const mx = (x1 + x2) * 0.5
        const my = (y1 + y2) * 0.5
        const dx = x2 - x1
        const dy = y2 - y1
        const len = Math.hypot(dx, dy) || 1
        const nx = -dy / len
        const ny = dx / len
        const phase = (state.phases[i] + state.phases[j]) * Math.PI
        const wobble = Math.sin(t * 0.6 + phase) * Math.min(40, len * 0.25)
        const cx = mx + nx * wobble
        const cy = my + ny * wobble
        ctx.beginPath()
        ctx.moveTo(x1, y1)
        ctx.quadraticCurveTo(cx, cy, x2, y2)
        ctx.stroke()
      }

      // Draw nodes
      // Wobbly blobs instead of perfect circles
      for (let i = 0; i < ax.length; i++) {
        const n = state.nodes[i]
        const cx = ax[i]
        const cy = ay[i]
        const segs = n.sp
        const wobAmp = n.r * 0.12 // subtler wobble for rounder look
        const speed = 0.6
        ctx.beginPath()
        for (let k = 0; k <= segs; k++) {
          const tAng = (k / segs) * Math.PI * 2
          const rad = n.r + wobAmp * Math.sin(tAng * 3.0 + n.wob + t * speed)
          const px = cx + rad * Math.cos(tAng)
          const py = cy + rad * Math.sin(tAng)
          if (k === 0) ctx.moveTo(px, py)
          else ctx.lineTo(px, py)
        }
        ctx.closePath()
        ctx.fillStyle = 'rgba(120, 125, 140, 0.10)'
        ctx.fill()
      }
      ctx.restore()

      // If document size grows (e.g., content added), expand world and rebuild nodes/edges
      if (tMs % 1000 < 16) {
        const docW = Math.max(window.innerWidth, FIXED_DOC_WIDTH)
        const docH = Math.max(window.innerHeight, FIXED_DOC_HEIGHT)
        if (docW !== state.lastDocW || docH !== state.lastDocH) {
          fit()
        }
      }

      rafRef.current = requestAnimationFrame(render)
    }
    rafRef.current = requestAnimationFrame(render)

    return () => {
      cancelAnimationFrame(rafRef.current)
      window.removeEventListener('resize', fit)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      aria-hidden="true"
      style={{ position: 'fixed', inset: 0, zIndex: -1, pointerEvents: 'none' }}
    />
  )
}


