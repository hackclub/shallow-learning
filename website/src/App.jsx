import React from 'react'
import { NavLink, Routes, Route, Navigate } from 'react-router-dom'
import Overview from './pages/Overview'
import Example from './pages/Example'
import FAQ from './pages/FAQ'
import Submit from './pages/Submit'
import BackgroundNetwork from './components/BackgroundNetwork'

export default function App() {
  const navbarHeight = 75
  return (
    <div style={{ backgroundColor: 'transparent', minHeight: '100vh', position: 'relative', zIndex: 0 }}>
      <BackgroundNetwork />
      <div
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          height: navbarHeight,
          backgroundColor: '#e9ecef',
          zIndex: 2
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
          <a
            id="github-link"
            href="https://github.com/hackclub/shallow-learning"
            aria-label="GitHub repository"
            target="_blank"
            rel="noreferrer noopener"
            style={{ position: 'absolute', top: (navbarHeight - 36) / 2, right: 100, width: 36, height: 36, display: 'block' }}
          >
            <svg viewBox="0 0 16 16" width="36" height="36" fill="#111" aria-hidden="true">
              <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z" />
            </svg>
          </a>
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


