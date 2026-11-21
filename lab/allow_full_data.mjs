// Lets net_parity.mjs read public/model/nn_data.json (the page's own split, which includes FUTURE rows)
// to prove that net.mjs reproduces src/lib/nn.js. Nothing trained here enters the archive.
globalThis.LAB_ALLOW_FULL_DATA = true
