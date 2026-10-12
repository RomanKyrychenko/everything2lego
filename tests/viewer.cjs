// Offline smoke test for the three.js brick viewer.
//
// Builds its own fixture scene (a small packed sphere exported through
// viewer_export.write_viewer), checks that the committed bundle.js is not
// stale relative to main.js, then drives the exported page in headless
// Chromium at three viewport sizes.
//
// Environment:
//   CHROME_PATH  explicit Chromium/Chrome executable (optional).
//   PYTHON       Python interpreter with everything2lego importable (optional;
//                defaults to .venv/bin/python when present, else python3).
//   VIEWER_OUT   directory for the fixture and screenshots (optional;
//                defaults to a fresh temp dir that is removed afterwards).
const { chromium } = require('@playwright/test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const { pathToFileURL } = require('node:url');

const repo = path.resolve(__dirname, '..');
const viewerSrc = path.join(repo, 'src', 'everything2lego', 'viewer');

function findChromium() {
  if (process.env.CHROME_PATH) {
    assert(fs.existsSync(process.env.CHROME_PATH), `CHROME_PATH does not exist: ${process.env.CHROME_PATH}`);
    return process.env.CHROME_PATH;
  }
  // Playwright's own download (npx playwright install chromium): let it resolve.
  try {
    const bundled = chromium.executablePath();
    if (bundled && fs.existsSync(bundled)) return bundled;
  } catch { /* fall through to manual search */ }
  // Playwright cache directories that may hold a browser revision the installed
  // @playwright/test does not pin (e.g. PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers).
  const home = os.homedir();
  const caches = [process.env.PLAYWRIGHT_BROWSERS_PATH];
  if (process.platform === 'linux') caches.push(path.join(home, '.cache', 'ms-playwright'));
  if (process.platform === 'darwin') caches.push(path.join(home, 'Library', 'Caches', 'ms-playwright'));
  if (process.platform === 'win32') caches.push(path.join(process.env.LOCALAPPDATA || path.join(home, 'AppData', 'Local'), 'ms-playwright'));
  const leaf = {
    linux: ['chrome-linux', 'chrome'],
    darwin: ['chrome-mac', 'Chromium.app', 'Contents', 'MacOS', 'Chromium'],
    win32: ['chrome-win', 'chrome.exe'],
  }[process.platform];
  for (const cache of caches.filter(Boolean)) {
    if (!fs.existsSync(cache)) continue;
    const revisions = fs.readdirSync(cache).filter(name => /^chromium-\d+$/.test(name)).sort().reverse();
    for (const rev of revisions) {
      const candidate = path.join(cache, rev, ...leaf);
      if (fs.existsSync(candidate)) return candidate;
    }
  }
  // System browsers.
  const system = {
    linux: ['/usr/bin/chromium', '/usr/bin/chromium-browser', '/usr/bin/google-chrome', '/usr/bin/google-chrome-stable'],
    darwin: ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', '/Applications/Chromium.app/Contents/MacOS/Chromium'],
    win32: [
      path.join(process.env.PROGRAMFILES || 'C:\\Program Files', 'Google', 'Chrome', 'Application', 'chrome.exe'),
      path.join(process.env['PROGRAMFILES(X86)'] || 'C:\\Program Files (x86)', 'Google', 'Chrome', 'Application', 'chrome.exe'),
    ],
  }[process.platform] || [];
  for (const candidate of system) if (fs.existsSync(candidate)) return candidate;
  throw new Error('No Chromium found. Set CHROME_PATH or run `npx playwright install chromium`.');
}

function findPython() {
  if (process.env.PYTHON) return process.env.PYTHON;
  const venv = process.platform === 'win32'
    ? path.join(repo, '.venv', 'Scripts', 'python.exe')
    : path.join(repo, '.venv', 'bin', 'python');
  if (fs.existsSync(venv)) return venv;
  return process.platform === 'win32' ? 'python' : 'python3';
}

function checkBundleFresh() {
  const bundle = path.join(viewerSrc, 'bundle.js');
  const main = path.join(viewerSrc, 'main.js');
  assert(fs.existsSync(bundle), 'viewer/bundle.js is missing; run `npm run build:viewer`');
  const bundleStat = fs.statSync(bundle);
  assert(bundleStat.size > 100000, `viewer/bundle.js is suspiciously small (${bundleStat.size} bytes)`);
  const bundleText = fs.readFileSync(bundle, 'utf8');
  // Element ids the page script wires up must survive bundling.
  for (const id of ['scene-data', 'brick-count', 'layer', 'rotate', 'reset']) {
    assert(bundleText.includes(id), `viewer/bundle.js does not reference #${id}; rebuild with \`npm run build:viewer\``);
  }
  // Rebuild into a temp file when esbuild is available and require a byte-identical result.
  const esbuild = path.join(repo, 'node_modules', '.bin', process.platform === 'win32' ? 'esbuild.cmd' : 'esbuild');
  if (fs.existsSync(esbuild)) {
    const tmp = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'e2l-bundle-')), 'bundle.js');
    const result = spawnSync(esbuild, [main, '--bundle', '--minify', '--format=iife', `--outfile=${tmp}`], { cwd: repo, encoding: 'utf8', shell: process.platform === 'win32' });
    assert.equal(result.status, 0, `esbuild failed:\n${result.stderr}`);
    const rebuilt = fs.readFileSync(tmp, 'utf8');
    fs.rmSync(path.dirname(tmp), { recursive: true, force: true });
    assert(rebuilt === bundleText, 'viewer/bundle.js is stale relative to main.js; run `npm run build:viewer` and commit the result');
    console.log('Bundle check passed: bundle.js matches a fresh esbuild of main.js');
  } else {
    const mainStat = fs.statSync(main);
    assert(bundleStat.mtimeMs >= mainStat.mtimeMs, 'viewer/bundle.js is older than main.js; run `npm run build:viewer`');
    console.log('Bundle check passed: bundle.js is non-empty and newer than main.js (esbuild not installed, skipped rebuild)');
  }
}

function buildFixture(outDir) {
  const script = `
import sys
from pathlib import Path
import numpy as np
from everything2lego import colors, io_bricklink, io_ldraw, pack, viewer_export, voxelize
out = Path(sys.argv[1])
# A sphere recoloured in 3x3x3-voxel blocks across six palette colours, so the
# render has enough colour variation to exercise the viewer even at 320px.
grid = voxelize.dummy_grid_from_primitive("sphere", radius_studs=5)
pal = colors.palette()
wanted = ["Red", "Blue", "Yellow", "Green", "White", "Orange"]
idx = [next(i for i, c in enumerate(pal) if c.name == n) for n in wanted]
y, x, z = np.nonzero(grid.palette_idx >= 0)
grid.palette_idx[y, x, z] = np.array(idx, dtype=np.int16)[(y // 3 + x // 3 + z // 3) % len(idx)]
placements = pack.pack(grid, use_tiles=True)
assert placements, "fixture produced no bricks"
io_ldraw.write_ldr(placements, out / "model.ldr")
io_bricklink.write_wanted_list(placements, out / "wanted.xml")
(out / "instructions.pdf").write_bytes(b"%PDF-1.4\\n%fixture placeholder\\n")
viewer_export.write_viewer(placements, out / "viewer", report={"fixture": True})
print(len(placements))
`;
  const env = { ...process.env, PYTHONPATH: [path.join(repo, 'src'), process.env.PYTHONPATH].filter(Boolean).join(path.delimiter) };
  const result = spawnSync(findPython(), ['-c', script, outDir], { cwd: repo, encoding: 'utf8', env });
  assert.equal(result.status, 0, `Fixture export failed:\n${result.stdout}\n${result.stderr}`);
  const index = path.join(outDir, 'viewer', 'index.html');
  assert(fs.existsSync(index), 'exporter did not write viewer/index.html');
  console.log(`Fixture scene exported with ${result.stdout.trim()} bricks to ${outDir}`);
  return index;
}

(async () => {
  checkBundleFresh();
  const keepOut = Boolean(process.env.VIEWER_OUT);
  const outDir = process.env.VIEWER_OUT
    ? path.resolve(process.env.VIEWER_OUT)
    : fs.mkdtempSync(path.join(os.tmpdir(), 'e2l-viewer-'));
  fs.mkdirSync(outDir, { recursive: true });
  const indexPath = buildFixture(outDir);
  const viewerDir = path.dirname(indexPath);

  const browser = await chromium.launch({
    executablePath: findChromium(),
    headless: true,
    args: ['--enable-webgl', '--use-gl=angle', '--use-angle=swiftshader', '--allow-file-access-from-files'],
  });
  try {
    for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }, { width: 320, height: 568 }]) {
      const context = await browser.newContext({ viewport, offline: true });
      const page = await context.newPage();
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.goto(pathToFileURL(indexPath).href);
      await page.waitForFunction(() => Number(document.getElementById('brick-count').textContent) > 0);
      await page.waitForTimeout(300);
      assert.equal(await page.locator('#error').isVisible(), false);
      assert.deepEqual(errors, []);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
      const canvas = page.locator('#canvas');
      const initial = await canvas.screenshot();
      assert(initial.length > 10000, 'Model canvas must have rendered detail');
      const coloredPixels = await page.evaluate(async encoded => {
        const img = new Image();
        img.src = `data:image/png;base64,${encoded}`;
        await img.decode();
        const probe = document.createElement('canvas');
        probe.width = img.width; probe.height = img.height;
        const ctx = probe.getContext('2d');
        ctx.drawImage(img, 0, 0);
        const pixels = ctx.getImageData(0, 0, img.width, img.height).data;
        let colored = 0;
        for (let i = 0; i < pixels.length; i += 4) {
          if (Math.max(pixels[i], pixels[i + 1], pixels[i + 2]) - Math.min(pixels[i], pixels[i + 1], pixels[i + 2]) > 40) colored++;
        }
        return colored;
      }, initial.toString('base64'));
      assert(coloredPixels > 1000, 'Canvas pixels must include the colored model');
      await page.screenshot({ path: path.join(outDir, `viewer-${viewport.width}.png`) });
      const bounds = await canvas.boundingBox();
      await page.mouse.move(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2);
      await page.mouse.down();
      await page.mouse.move(bounds.x + bounds.width / 2 + 80, bounds.y + bounds.height / 2 + 20, { steps: 8 });
      await page.mouse.up();
      await page.waitForTimeout(200);
      assert.notDeepEqual(await canvas.screenshot(), initial, 'Orbit must change the rendered model');
      await page.locator('#reset').click();
      const slider = page.locator('#layer');
      await slider.fill('0');
      await slider.dispatchEvent('input');
      await page.waitForTimeout(150);
      const empty = await canvas.screenshot();
      assert.notDeepEqual(empty, initial, 'Step zero must hide the model');
      await slider.fill(await slider.getAttribute('max'));
      await slider.dispatchEvent('input');
      await page.waitForTimeout(150);
      assert.notDeepEqual(await canvas.screenshot(), empty, 'Last step must restore the model');
      await page.locator('#rotate').check();
      const before = await canvas.screenshot();
      await page.waitForTimeout(300);
      assert.notDeepEqual(await canvas.screenshot(), before, 'Auto-rotate must animate');
      for (const link of await page.locator('a[download]').all()) {
        const href = await link.getAttribute('href');
        assert(fs.existsSync(path.resolve(viewerDir, href)), `download link target missing: ${href}`);
      }
      console.log(`Offline viewer passed: ${viewport.width} x ${viewport.height}`);
      await context.close();
    }
  } finally {
    await browser.close();
    if (!keepOut) fs.rmSync(outDir, { recursive: true, force: true });
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
