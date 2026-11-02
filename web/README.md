# Web site

The project site: an overview, the measured results, the kernel stories and a dashboard that replays a recorded serving run or connects to your own `engine serve`. It is a static Next.js export, so it needs no server of its own; the plan and design decisions are in `docs/claude/web_plan.md` (local) and the data formats in `docs/api.md`.

## Run it

```bash
cd web
npm ci
npm run dev
```

`npm run dev`, `npm run build` and `npm run typecheck` first run `scripts/sync-data.mjs`, which reads `../results/` (summary CSVs, `environment.txt`, `traces/*.json`) into `src/data/*.json` and `public/traces/`. Those files are generated and not committed; to show new numbers, rerun `scripts/reproduce.sh` or `engine bench --trace` and rebuild.

`npm run build` writes the site to `out/`. Before a commit: `npm run lint -- --max-warnings=0`, `npm run typecheck` and `npm run build`, the same steps as the `web` CI job.

## Deploy on Vercel

1. On [vercel.com](https://vercel.com), choose Add New, then Project, and import this GitHub repository (allow the Vercel GitHub app access to it if asked).
2. Under Root Directory choose `web`. Leave the framework preset on Next.js and the build and output settings on their defaults; they come from `package.json` and `next.config.ts`.
3. After the project exists, open Settings, then Build and Deployment. Keep "Include files outside of the Root Directory in the Build Step" on, because the build reads `../results/` and `../docs/`, and set Node.js to 22, the version CI uses.
4. Add no environment variables. Vercel sets `VERCEL_PROJECT_PRODUCTION_URL` itself, and `src/app/layout.tsx` uses it for the absolute URLs of the social cards.
5. Deploy. Every push to `main` then redeploys, and pull requests get preview URLs.
6. Open `/`, `/results`, `/kernels` and `/dashboard` on the production URL. The replay needs nothing else.

If Vercel builds from a shallow clone, the dashboard's recording label leaves out the trace's commit; the rest of the label and the commit in the footer come from the data files.

### Live mode from the deployed site

The site is live at https://llm-inference-engine-jz.vercel.app. It is public but the engine stays on your machine, so each visitor's browser connects to their own `localhost`. `engine serve` allows that origin by default (`kDefaultCorsOrigins` in `include/engine/server/cors.h`; a different domain needs `--cors-origin https://<domain>`, which replaces the list):

```bash
build/release/bin/engine serve --model models/tinyllama --graphs
```

Then open `/dashboard`, choose live and connect to `http://localhost:8000`.

An HTTPS page reaching `localhost` needs the browser's local network permission. In Chrome 156 the deployed dashboard connected and streamed a chat answer from a local engine once that permission was granted, and without it Chrome blocked every request; the dashboard then shows a how-to. Other browsers may refuse outright, in which case run the site yourself with `npm run dev`. Never expose or tunnel the server to get around it.
