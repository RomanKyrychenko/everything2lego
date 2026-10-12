import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { createIcons, Download, FileText, RotateCcw, List } from 'lucide';

async function main() {
  createIcons({ icons: { Download, FileText, RotateCcw, List } });
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf1f3f2);

  const canvas = document.getElementById('canvas');
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
  renderer.shadowMap.enabled = true;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;

  const data = JSON.parse(document.getElementById('scene-data').textContent);
  if (!data.bricks.length) throw new Error('This model has no parts.');
  const S = data.mm_per_stud;
  const P = data.mm_per_plate;

  // Compute bounds for camera framing.
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (const b of data.bricks) {
    const x0 = b.x * S, y0 = b.y * P, z0 = b.z * S;
    const x1 = (b.x + b.w) * S, y1 = (b.y + b.h) * P, z1 = (b.z + b.d) * S;
    if (x0 < minX) minX = x0; if (y0 < minY) minY = y0; if (z0 < minZ) minZ = z0;
    if (x1 > maxX) maxX = x1; if (y1 > maxY) maxY = y1; if (z1 > maxZ) maxZ = z1;
  }
  const cx = (minX + maxX) / 2, cz = (minZ + maxZ) / 2;
  const modelHeight = maxY - minY;
  const maxHorizontal = Math.max(maxX - minX, maxZ - minZ);

  const camera = new THREE.PerspectiveCamera(45, canvas.clientWidth / canvas.clientHeight, 0.1, 100000);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  const radius = Math.hypot(maxX - minX, modelHeight, maxZ - minZ) / 2;
  function resetView() {
    const vertical = THREE.MathUtils.degToRad(camera.fov / 2);
    const horizontal = Math.atan(Math.tan(vertical) * camera.aspect);
    const dist = radius / Math.sin(Math.min(vertical, horizontal)) * 1.15;
    controls.target.set(cx, (minY + maxY) / 2, cz);
    camera.position.copy(controls.target).add(new THREE.Vector3(1, 0.85, 1).normalize().multiplyScalar(dist));
    controls.update();
  }
  resetView();
  document.getElementById('reset').addEventListener('click', resetView);
  document.getElementById('rotate').addEventListener('change', e => { controls.autoRotate = e.target.checked; });

  // Lights.
  scene.add(new THREE.AmbientLight(0xffffff, 2.2));
  const dir = new THREE.DirectionalLight(0xffffff, 2.8);
  dir.position.set(1, 2, 1).normalize();
  scene.add(dir);
  const rim = new THREE.DirectionalLight(0xffffff, 1.0);
  rim.position.set(-1, 0.5, -1).normalize();
  scene.add(rim);

  // Ground plane.
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(maxHorizontal * 4, maxHorizontal * 4),
    new THREE.MeshBasicMaterial({ color: 0xf1f3f2, toneMapped: false })
  );
  ground.rotation.x = -Math.PI / 2;
  ground.position.y = minY - 0.1;
  scene.add(ground);

  // ---------------------------------------------------------------------
  // Part geometry.
  //
  // Every body geometry is built in "unit space": it fits the unit cube
  // [-0.5, 0.5]^3, with the part's long footprint axis along x and (for
  // slopes) the low side at +x. Each instance matrix is T · S · R: the unit
  // geometry is first rotated about Y in unit space (a 90° rotation maps the
  // unit cube onto itself) and only then scaled to the brick's *world*
  // extents (w·S, h·P, d·S). Scaling after rotating means the rendered
  // footprint always equals (w, d) in world studs no matter which way the
  // part is turned. (The old T · R · S order scaled the geometry's x axis by
  // w and then rotated it into z, so a 2x1 slope with low_dir ±z drew as a
  // 1x2 footprint.)
  // ---------------------------------------------------------------------

  // Axis-aligned box spanning [x0,x1]×[y0,y1]×[z0,z1] in unit space.
  function box(x0, x1, y0, y1, z0, z1) {
    const g = new THREE.BoxGeometry(x1 - x0, y1 - y0, z1 - z0);
    g.translate((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2);
    return g;
  }
  function merged(parts) {
    const g = mergeGeometries(parts, false);
    for (const p of parts) p.dispose();
    return g;
  }

  // Regular slope: flat bottom, slanted top, low side at +x. `top` is the
  // fraction of the run that stays flat at full height (real 45° and 33°
  // slopes keep one stud of flat top; cheese slopes have none).
  function createSlopeGeom(top) {
    const shape = new THREE.Shape();
    shape.moveTo(-0.5, -0.5);        // bottom-left
    shape.lineTo(0.5, -0.5);         // bottom-right (LOW at +x)
    shape.lineTo(-0.5 + top, 0.5);   // top edge of the slanted face
    shape.lineTo(-0.5, 0.5);         // top-left (HIGH at -x)
    shape.lineTo(-0.5, -0.5);
    const g = new THREE.ExtrudeGeometry(shape, { depth: 1, bevelEnabled: false });
    g.translate(0, 0, -0.5);
    g.computeVertexNormals();
    return g;
  }
  // Inverted slope: full flat top, vertical face at +x, slanted underside
  // rising towards -x. The material "droops" to full depth at +x.
  function createSlopeInvGeom(top) {
    const shape = new THREE.Shape();
    shape.moveTo(-0.5, 0.5);         // top-left
    shape.lineTo(0.5, 0.5);          // top-right
    shape.lineTo(0.5, -0.5);         // bottom-right (LOW at +x)
    shape.lineTo(0.5 - top, -0.5);   // flat bottom under the full-depth end
    shape.lineTo(-0.5, 0.5);         // slanted hypotenuse
    const g = new THREE.ExtrudeGeometry(shape, { depth: 1, bevelEnabled: false });
    g.translate(0, 0, -0.5);
    g.computeVertexNormals();
    return g;
  }
  // Corner slope: pyramid, high corner at (-x, +y, -z), low corner at (+x, +z).
  function createCornerGeom() {
    const positions = new Float32Array([
      -0.5, -0.5, -0.5,   0.5, -0.5, -0.5,   0.5, -0.5, 0.5,   -0.5, -0.5, 0.5,
      -0.5, 0.5, -0.5,
    ]);
    // Counter-clockwise seen from outside (three.js front faces).
    const indices = [
      0, 1, 2,   0, 2, 3,   // bottom
      0, 4, 1,              // back (z=-0.5)
      0, 3, 4,              // left (x=-0.5)
      4, 2, 1,              // sloped face towards +x
      4, 3, 2,              // sloped face towards +z
    ];
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    g.setIndex(indices);
    g.computeVertexNormals();
    return g;
  }
  // Window / door frame: four rails around an opening, `tx`/`ty` thick.
  function createFrameGeom(tx, ty) {
    return merged([
      box(-0.5, 0.5, -0.5, -0.5 + ty, -0.5, 0.5),       // sill
      box(-0.5, 0.5, 0.5 - ty, 0.5, -0.5, 0.5),         // lintel
      box(-0.5, -0.5 + tx, -0.5 + ty, 0.5 - ty, -0.5, 0.5),
      box(0.5 - tx, 0.5, -0.5 + ty, 0.5 - ty, -0.5, 0.5),
    ]);
  }
  // Recessed pane filling the frame opening, `tz` of the depth thick.
  function createPaneGeom(tx, ty, tz) {
    return box(-0.5 + tx, 0.5 - tx, -0.5 + ty, 0.5 - ty, -tz / 2, tz / 2);
  }
  // Panel: one-plate base with a thin wall along the -z edge.
  function createPanelGeom(baseY, wallZ) {
    return merged([
      box(-0.5, 0.5, -0.5, -0.5 + baseY, -0.5, 0.5),
      box(-0.5, 0.5, -0.5 + baseY, 0.5, -0.5, -0.5 + wallZ),
    ]);
  }
  // Arch: lintel on two one-stud legs.
  function createArchGeom(legX, lintelY) {
    return merged([
      box(-0.5, 0.5, 0.5 - lintelY, 0.5, -0.5, 0.5),
      box(-0.5, -0.5 + legX, -0.5, 0.5 - lintelY, -0.5, 0.5),
      box(0.5 - legX, 0.5, -0.5, 0.5 - lintelY, -0.5, 0.5),
    ]);
  }

  const SLOPE_SHAPES = new Set(['slope', 'slope_inv', 'slope_corner']);
  const FRAMED_SHAPES = new Set(['window', 'door']);
  // Shapes whose top is not a flat studded plate.
  const NO_STUD_SHAPES = new Set(['tile', 'slope', 'slope_inv', 'slope_corner', 'window', 'door', 'panel', 'cone']);
  const hasStuds = (b) => !NO_STUD_SHAPES.has(b.shape);

  // Length (in studs) of the part along its slope run, i.e. along low_dir.
  function slopeRun(b) {
    if (!b.low_dir) return b.w;
    return b.low_dir.includes('x') && !b.low_dir.includes('z') ? b.w
         : b.low_dir.includes('z') && !b.low_dir.includes('x') ? b.d
         : Math.max(b.w, b.d);
  }
  // Long and short footprint sides in the part's own (unrotated) frame.
  const longSide = (b) => Math.max(b.w, b.d);

  // Geometry key and builder per brick. Geometries are cached by key and
  // shared by every instance with the same key so instancing still works.
  function geomSpec(b) {
    const h = b.h;
    switch (b.shape) {
      case 'slope': {
        const run = slopeRun(b);
        const top = run >= 2 && h >= 3 ? 1 / run : 0;
        return { key: `slope:${top.toFixed(4)}`, body: () => createSlopeGeom(top) };
      }
      case 'slope_inv': {
        const run = slopeRun(b);
        const top = run >= 2 && h >= 3 ? 1 / run : 0;
        return { key: `slope_inv:${top.toFixed(4)}`, body: () => createSlopeInvGeom(top) };
      }
      case 'slope_corner':
        return { key: 'slope_corner', body: createCornerGeom };
      case 'window':
      case 'door': {
        const L = longSide(b);
        const tx = Math.min(0.3, 2.4 / (L * S));
        const ty = Math.min(0.3, 2.4 / (h * P));
        const tz = b.shape === 'door' ? 0.3 : 0.2;
        return {
          key: `${b.shape}:${L}x${h}`,
          body: () => createFrameGeom(tx, ty),
          pane: () => createPaneGeom(tx, ty, tz),
          paneKind: b.shape === 'door' ? 'dark' : 'glass',
        };
      }
      case 'panel': {
        const baseY = Math.min(0.5, 1 / h);
        const wallZ = 1.6 / S;
        return { key: `panel:${h}`, body: () => createPanelGeom(baseY, wallZ) };
      }
      case 'arch': {
        const L = longSide(b);
        if (L < 3) return { key: 'box', body: () => new THREE.BoxGeometry(1, 1, 1) };
        return { key: `arch:${L}x${h}`, body: () => createArchGeom(1 / L, Math.min(0.5, 3 / h)) };
      }
      case 'cone':
        return { key: 'cone', body: () => new THREE.CylinderGeometry(0.25, 0.5, 1, 16) };
      default:
        return { key: 'box', body: () => new THREE.BoxGeometry(1, 1, 1) };
    }
  }
  const geomCache = new Map();
  function geomFor(spec) {
    if (!geomCache.has(spec.key)) {
      geomCache.set(spec.key, { body: spec.body(), pane: spec.pane ? spec.pane() : null, paneKind: spec.paneKind || null });
    }
    return geomCache.get(spec.key);
  }

  // Y rotation (radians) applied in unit space. Slopes: the unit low side is
  // +x; +Y rotation cycles it through -z, -x, +z. Everything else uses the
  // placement rotation so framed/panel parts lie along their long world axis.
  const LOW_DIR_Y_ROT = {
    '+x': 0, '-z': Math.PI * 0.5, '-x': Math.PI, '+z': Math.PI * 1.5,
    '+x+z': 0, '+x-z': Math.PI * 0.5, '-x-z': Math.PI, '-x+z': Math.PI * 1.5,
  };
  function pickRotY(b) {
    if (SLOPE_SHAPES.has(b.shape)) return b.low_dir ? (LOW_DIR_Y_ROT[b.low_dir] || 0) : 0;
    if (b.w === b.d) return 0;
    // Non-square parts: the unit geometry's long axis is x. If the world
    // footprint is longer along z the part has been turned a quarter turn.
    return b.d > b.w ? Math.PI * 0.5 : 0;
  }

  // Instance matrix = T · S_world · R_y (see note above).
  const _pos = new THREE.Vector3(), _scl = new THREE.Vector3();
  const _quat = new THREE.Quaternion(), _rot = new THREE.Matrix4();
  function bodyMatrix(b, out) {
    const sx = b.w * S, sy = b.h * P, sz = b.d * S;
    _pos.set(b.x * S + sx / 2, b.y * P + sy / 2, b.z * S + sz / 2);
    _scl.set(sx - 0.15, sy - 0.1, sz - 0.15);
    out.compose(_pos, _quat, _scl);
    const ry = pickRotY(b);
    if (ry) out.multiply(_rot.makeRotationY(ry));
    return out;
  }
  const tmp = new THREE.Object3D();
  function pushStudMatrices(b, into) {
    for (let dx = 0; dx < b.w; dx++) {
      for (let dz = 0; dz < b.d; dz++) {
        tmp.position.set((b.x + dx + 0.5) * S, (b.y + b.h) * P + 0.8, (b.z + dz + 0.5) * S);
        tmp.scale.set(1, 1, 1);
        tmp.rotation.set(0, 0, 0);
        tmp.updateMatrix();
        into.push(tmp.matrix.clone());
      }
    }
  }

  // Group bricks by RGB × geometry key so each instanced mesh shares one
  // BufferGeometry.
  const groups = new Map();
  for (const b of data.bricks) {
    const spec = geomSpec(b);
    const key = b.rgb.join(',') + '|' + spec.key;
    if (!groups.has(key)) {
      groups.set(key, { rgb: b.rgb, color: b.color, geom: geomFor(spec), bricks: [] });
    }
    groups.get(key).bricks.push(b);
  }
  const colorKeys = new Set(data.bricks.map(b => b.rgb.join(',')));

  const meshesByColor = [];
  const studGeom = new THREE.CylinderGeometry(2.4, 2.4, 1.6, 16);
  const glassMat = new THREE.MeshLambertMaterial({ color: 0xcfe6f2, transparent: true, opacity: 0.55, depthWrite: false });

  function fillInstances(g, bricks) {
    const bodyMats = [], studMats = [];
    const m = new THREE.Matrix4();
    for (const b of bricks) {
      if (b.h <= 0) continue;
      bodyMats.push(bodyMatrix(b, m).clone());
      if (hasStuds(b)) pushStudMatrices(b, studMats);
    }
    g.body.count = bodyMats.length;
    for (let i = 0; i < bodyMats.length; i++) g.body.setMatrixAt(i, bodyMats[i]);
    g.body.instanceMatrix.needsUpdate = true;
    g.body.computeBoundingSphere();
    if (g.pane) {
      g.pane.count = bodyMats.length;
      for (let i = 0; i < bodyMats.length; i++) g.pane.setMatrixAt(i, bodyMats[i]);
      g.pane.instanceMatrix.needsUpdate = true;
      g.pane.computeBoundingSphere();
    }
    if (g.studs) {
      g.studs.count = studMats.length;
      for (let i = 0; i < studMats.length; i++) g.studs.setMatrixAt(i, studMats[i]);
      g.studs.instanceMatrix.needsUpdate = true;
      g.studs.computeBoundingSphere();
    }
  }

  for (const g of groups.values()) {
    const color = new THREE.Color(`rgb(${g.rgb.join(',')})`);
    const mat = new THREE.MeshLambertMaterial({ color });

    const bodyInst = new THREE.InstancedMesh(g.geom.body, mat, g.bricks.length);
    bodyInst.castShadow = true;
    bodyInst.receiveShadow = true;

    let paneInst = null;
    if (g.geom.pane) {
      const paneMat = g.geom.paneKind === 'glass' ? glassMat
        : new THREE.MeshLambertMaterial({ color: color.clone().multiplyScalar(0.55) });
      paneInst = new THREE.InstancedMesh(g.geom.pane, paneMat, g.bricks.length);
    }

    const studCount = g.bricks.reduce((n, b) => n + (hasStuds(b) ? b.w * b.d : 0), 0);
    const studInst = studCount > 0 ? new THREE.InstancedMesh(studGeom, mat, studCount) : null;

    const entry = { rgb: g.rgb, color: g.color, bricks: g.bricks, body: bodyInst, pane: paneInst, studs: studInst };
    fillInstances(entry, g.bricks);

    scene.add(bodyInst);
    if (paneInst) scene.add(paneInst);
    if (studInst) scene.add(studInst);
    meshesByColor.push(entry);
  }

  // UI.
  document.getElementById('brick-count').textContent = data.bricks.length;
  document.getElementById('color-count').textContent = colorKeys.size;
  document.getElementById('dimensions').textContent = `${Math.round(maxX - minX)} x ${Math.round(maxZ - minZ)} x ${Math.round(modelHeight)} mm`;
  const report = data.report;
  document.getElementById('status').textContent = report.geometrically_connected === undefined ? '' :
    (report.geometrically_connected ? 'One connected assembly.' : `${report.unsupported_parts} unsupported parts. ${report.connected_components} assemblies.`) +
    (report.single_stud_overhangs ? ` ${report.single_stud_overhangs} single-stud overhangs.` : '');

  const legend = document.getElementById('legend');
  const legendCounts = new Map();
  for (const b of data.bricks) {
    const k = b.rgb.join(',');
    if (!legendCounts.has(k)) legendCounts.set(k, { rgb: b.rgb, color: b.color, n: 0 });
    legendCounts.get(k).n++;
  }
  for (const g of legendCounts.values()) {
    const row = document.createElement('div');
    const [r, gr, bl] = g.rgb;
    row.innerHTML = `<span class="sw" style="background:rgb(${r},${gr},${bl})"></span>${g.color} &times; ${g.n}`;
    legend.appendChild(row);
  }

  const slider = document.getElementById('layer');
  const layerVal = document.getElementById('layer-val');
  const layers = [...new Set(data.bricks.map(b => b.y))].sort((a, b) => a - b);
  slider.max = layers.length;
  slider.value = layers.length;
  layerVal.textContent = `${layers.length} / ${layers.length}`;

  function applyLayerCutoff(cutoff) {
    layerVal.textContent = `${cutoff} / ${layers.length}`;
    for (const g of meshesByColor) {
      // Rebuild bodies + studs with only bricks whose base (y) is within the cutoff.
      fillInstances(g, g.bricks.filter(b => cutoff > 0 && b.y <= layers[cutoff - 1]));
    }
  }
  slider.addEventListener('input', () => applyLayerCutoff(parseInt(slider.value, 10)));

  new ResizeObserver(() => {
    camera.aspect = canvas.clientWidth / canvas.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
    resetView();
  }).observe(document.getElementById('stage'));

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();
}

main().catch(err => {
  console.error(err);
  const error = document.getElementById('error');
  error.hidden = false;
  error.textContent = `Unable to display model: ${err.message}`;
});
