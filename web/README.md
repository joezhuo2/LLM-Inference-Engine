# Web site

The project site: an overview, the measured results, the kernel stories and a dashboard that replays recorded serving runs. It is a static Next.js export, so it needs no server; the plan and design decisions are in `docs/claude/web_plan.md` (local) and the data formats in `docs/api.md`.

## Run it

```bash
cd web
npm ci
npm run dev
```

`npm run dev`, `npm run build` and `npm run typecheck` first run `scripts/sync-data.mjs`, which reads `../results/` (summary CSVs, `environment.txt`, `traces/*.json`) into `src/data/*.json` and `public/traces/`. Those files are generated and not committed; to show new numbers, rerun `scripts/reproduce.sh` or `engine bench --trace` and rebuild.

`npm run build` writes the site to `out/`. Before a commit: `npm run lint -- --max-warnings=0`, `npm run typecheck` and `npm run build`, the same steps as the `web` CI job.

## Deploy on Vercel

Import the repository, set the Root Directory to `web` and keep "Include files outside of the Root Directory in the Build Step" enabled, because the build reads `../results/`. The framework preset is Next.js; the build command and output come from `package.json` and `next.config.ts`.
