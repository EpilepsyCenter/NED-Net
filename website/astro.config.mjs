import { defineConfig } from 'astro/config';

// Deployed to GitHub Pages from the EpilepsyCenter org.
// Final URL: https://epilepsycenter.github.io/NED-Net/
//
// The base must match the repository name exactly, capitals included —
// GitHub Pages project paths are case-sensitive.
//
// If you point a custom domain at this site, drop the `base` line and set
// `site` to the apex — Astro will rewrite all internal links accordingly.
export default defineConfig({
  site: 'https://epilepsycenter.github.io',
  base: '/NED-Net',
  trailingSlash: 'ignore',
});
