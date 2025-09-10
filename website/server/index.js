import express from 'express'
import compression from 'compression'
import path from 'path'
import { fileURLToPath } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)

const app = express()
app.disable('x-powered-by')
app.use(compression())

const isDev = process.env.NODE_ENV !== 'production'
const rootDir = path.join(__dirname, '..')
const distPath = path.join(rootDir, 'dist')
const imgPath = path.join(rootDir, 'img')

// serve images in both dev and prod
app.use('/img', express.static(imgPath))

if (isDev) {
  const { createServer: createViteServer } = await import('vite')
  const vite = await createViteServer({
    root: rootDir,
    server: { middlewareMode: true }
  })
  app.use(vite.middlewares)
} else {
  app.use(express.static(distPath))
  // SPA fallback to index.html in production
  app.get('*', (_req, res) => {
    res.sendFile(path.join(distPath, 'index.html'))
  })
}

app.get('/healthz', (_req, res) => {
  res.status(200).json({ ok: true })
})

const PORT = 3000
const server = app.listen(PORT)
server.on('listening', () => {
  console.log(`server listening on http://localhost:${PORT}`)
})
server.on('error', (err) => {
  console.error('Failed to bind server on port 3000:', err.message)
  process.exit(1)
})


