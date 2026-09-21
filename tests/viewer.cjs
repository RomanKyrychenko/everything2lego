const { chromium } = require('@playwright/test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { pathToFileURL } = require('node:url');

(async () => {
  const browser = await chromium.launch({
    executablePath: process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    headless: true,
    args: ['--enable-webgl', '--use-gl=angle', '--use-angle=swiftshader', '--allow-file-access-from-files'],
  });
  try {
    for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }, { width: 320, height: 568 }]) {
      const context = await browser.newContext({ viewport, offline: true });
      const page = await context.newPage();
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.goto(pathToFileURL(path.resolve('out_finished/viewer/index.html')).href);
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
      await page.screenshot({ path: `out_finished/viewer-${viewport.width}.png` });
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
        assert(require('node:fs').existsSync(path.resolve('out_finished/viewer', href)));
      }
      console.log(`Offline viewer passed: ${viewport.width} x ${viewport.height}`);
      await context.close();
    }
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
