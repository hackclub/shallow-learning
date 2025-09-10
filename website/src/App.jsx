import React from 'react'
import { NavLink, Routes, Route, Navigate } from 'react-router-dom'
import Overview from './pages/Overview'
import Example from './pages/Example'
import FAQ from './pages/FAQ'
import Submit from './pages/Submit'

export default function App() {
  const navbarHeight = 75
  return (
    <div style={{ backgroundColor: '#fff', minHeight: '100vh' }}>
      <div
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          height: navbarHeight,
          backgroundColor: '#e9ecef'
        }}
      >
        <div style={{ position: 'relative', height: '100%' }}>
          <div
            className="nav-center"
            style={{
              height: '100%',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}
          >
            <nav className="nav-menu" style={{ display: 'flex', gap: 12 }}>
              <NavLink
                to="/overview"
                className="nav-link"
                style={({ isActive }) => ({
                  textDecoration: 'none',
                  color: '#111',
                  fontWeight: 600,
                  textTransform: 'uppercase',
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  borderRadius: 12,
                  backgroundColor: isActive ? '#fff' : 'transparent'
                })}
              >
                Overview
              </NavLink>
              <NavLink
                to="/example"
                className="nav-link"
                style={({ isActive }) => ({
                  textDecoration: 'none',
                  color: '#111',
                  fontWeight: 600,
                  textTransform: 'uppercase',
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  borderRadius: 12,
                  backgroundColor: isActive ? '#fff' : 'transparent'
                })}
              >
                Example
              </NavLink>
              <NavLink
                to="/faq"
                className="nav-link"
                style={({ isActive }) => ({
                  textDecoration: 'none',
                  color: '#111',
                  fontWeight: 600,
                  textTransform: 'uppercase',
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  borderRadius: 12,
                  backgroundColor: isActive ? '#fff' : 'transparent'
                })}
              >
                FAQ
              </NavLink>
              <NavLink
                to="/submit"
                className="nav-link"
                style={({ isActive }) => ({
                  textDecoration: 'none',
                  color: '#111',
                  fontWeight: 600,
                  textTransform: 'uppercase',
                  display: 'inline-flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  borderRadius: 12,
                  backgroundColor: isActive ? '#fff' : 'transparent'
                })}
              >
                Submit
              </NavLink>
            </nav>
          </div>
          <NavLink to="/" aria-label="Home">
            <img
              id="logo-img"
              src="/img/logo.png"
              alt="Logo"
              width="75"
              height="75"
              style={{ position: 'absolute', top: (navbarHeight - 75) / 2, right: 11, display: 'block' }}
            />
          </NavLink>
        </div>
      </div>

      <main
        style={{
          fontFamily: 'Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif',
          lineHeight: 1.5,
          padding: 24,
          paddingTop: navbarHeight
        }}
      >
        <Routes>
          <Route path="/" element={<Navigate to="/overview" replace />} />
          <Route path="/overview" element={<Overview />} />
          <Route path="/example" element={<Example />} />
          <Route path="/faq" element={<FAQ />} />
          <Route path="/submit" element={<Submit />} />
        </Routes>
      </main>
    </div>
  )
}


