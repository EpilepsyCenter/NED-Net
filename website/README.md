# NED-Net website

Static site for [NED-Net](https://github.com/EpilepsyCenter/NED-Net), built with
[Astro](https://astro.build) and deployed to
<https://epilepsycenter.github.io/NED-Net/> by
`.github/workflows/deploy-website.yml`.

## Pages

- `/` — homepage
- `/features` — the detection pipeline and the app's pages, expanded
- `/workflow` — the five stages from recording to group statistics
- `/docs` — the full user manual, rendered from
  `eeg_seizure_analyzer/dash_app/assets/USER_MANUAL.md`

## Develop

```bash
cd website
npm install        # one-time
npm run dev        # http://localhost:4321/NED-Net
```

## Build

```bash
npm run build      # → website/dist/
npm run preview    # serve the built site
```

## Conventions

- **Design language.** `src/styles/global.css` is the shared token layer of the
  Epilepsy Center sites — the same block as the hub at
  <https://epilepsycenter.github.io/>. Paper `#f3f2ee` on ink `#15181f`, one
  interactive accent (`#007d93`), and `#0b2c61` from NED-Net's own wordmark as
  the identity colour. Change a palette or a typeface here and nowhere else.
- **Manual and screenshots** are read straight out of the app's asset
  directory, so the site and the in-app manual can never drift apart. Adding a
  screenshot to `assets/screenshots/` is enough — the docs page picks it up.
- **Accuracy first.** This site describes real features only: no invented
  benchmarks, no placeholder lab logos, no fabricated quotes or numbers. The
  figures are real screenshots of the app.
- **Version string.** `v2.0.0` appears in the nav, the footer and the page
  headers — search and replace on release.
