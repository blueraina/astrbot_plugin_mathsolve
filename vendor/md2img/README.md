# Bundled Markdown Rendering Assets

This directory contains browser-side assets used by the Markdown-to-image renderer.

- `mathjax-2.7.7/`: MathJax 2.7.7, Apache-2.0 license. The upstream license file is kept at `mathjax-2.7.7/LICENSE`.
- `pagedjs/paged.polyfill.js`: PagedJS 0.4.3, MIT license. The bundled file includes its upstream license banner.

These files are committed so normal plugin installs can render LaTeX and `/pc` paginated output without depending on CDN access.
