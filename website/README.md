# Website

Minimal React (Vite) + Express server. The server listens on port 3000 and exits if binding fails.

## Scripts

- `npm run dev` — build client in watch mode and run server with nodemon
- `npm run build` — build client to `dist/`
- `npm start` — start server (serves `dist/`)

## Local development

```bash
cd website
npm install
npm run build
npm start
# open http://localhost:3000
```

For dev with auto-reload of server and client build:

```bash
npm run dev
```

## Docker

```bash
docker build -t platformer-website:latest .
docker run --rm -p 3000:3000 platformer-website:latest
```


