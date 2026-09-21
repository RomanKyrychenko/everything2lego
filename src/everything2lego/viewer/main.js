import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
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

  // Wedge geometries for slope-family parts. Each is unit-sized (1×1×1) and
  // centred on the origin, matching BoxGeometry conventions, so scaling and
  // positioning work the same way.
  //
  // Regular slope: flat bottom, slanted top. Low corner at (+x, -y).
  // Inverted slope: flat top, slanted bottom. Low corner at (+x, +y).
  // Corner slope: pyramid-shaped, one bottom corner is low, opposite top is high.
  function createSlopeGeom() {
    const shape = new THREE.Shape();
    shape.moveTo(-0.5, -0.5);   // bottom-left
    shape.lineTo(0.5, -0.5);    // bottom-right (LOW at +x)
    shape.lineTo(-0.5, 0.5);    // top-left (HIGH at -x)
    shape.lineTo(-0.5, -0.5);   // close
    const g = new THREE.ExtrudeGeometry(shape, { depth: 1, bevelEnabled: false });
    g.translate(0, 0, -0.5);
    g.computeVertexNormals();
    return g;
  }
  function createSlopeInvGeom() {
    // Full flat top (y=+0.5), vertical face at +x, slanted from (-0.5, 0.5)
    // down to (+0.5, -0.5). The "low corner" (where material droops down) is
    // at (+x, -y).
    const shape = new THREE.Shape();
    shape.moveTo(-0.5, 0.5);    // top-left
    shape.lineTo(0.5, 0.5);     // top-right
    shape.lineTo(0.5, -0.5);    // bottom-right (LOW at +x)
    shape.lineTo(-0.5, 0.5);    // close (slanted hypotenuse)
    const g = new THREE.ExtrudeGeometry(shape, { depth: 1, bevelEnabled: false });
    g.translate(0, 0, -0.5);
    g.computeVertexNormals();
    return g;
  }
  function createCornerGeom() {
    // Pyramid corner: full box minus the top-far quadrant. High corner at
    // (-x, +y, -z), low corner (bottom drop) at (+x, +z).
    const positions = new Float32Array([
      // Bottom face (y=-0.5)
      -0.5, -0.5, -0.5,   0.5, -0.5, -0.5,   0.5, -0.5, 0.5,   -0.5, -0.5, 0.5,
      // Top corner (single) at (-0.5, 0.5, -0.5)
      -0.5, 0.5, -0.5,
    ]);
    const indices = [
      // Bottom (looking down): 0-1-2-0-2-3
      0, 2, 1,   0, 3, 2,
      // Back (z=-0.5): 0, 1, 4
      0, 1, 4,
      // Left (x=-0.5): 0, 4, 3
      0, 4, 3,
      // Sloped face 1 (from top corner to +x edge of bottom-back): 4, 1, 2
      4, 1, 2,
      // Sloped face 2 (from top corner to +z edge of bottom-left): 4, 2, 3
      4, 2, 3,
    ];
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    g.setIndex(indices);
    g.computeVertexNormals();
    return g;
  }

  const geomBox = new THREE.BoxGeometry(1, 1, 1);
  const geomSlope = createSlopeGeom();
  const geomSlopeInv = createSlopeInvGeom();
  const geomCorner = createCornerGeom();

  // Map world low direction to Y-axis rotation (radians). +x is the base,
  // then +Y rotation cycles through -z, -x, +z.
  const LOW_DIR_Y_ROT = {
    '+x': 0,
    '-z': Math.PI * 0.5,
    '-x': Math.PI,
    '+z': Math.PI * 1.5,
    '+x+z': 0,
    '+x-z': Math.PI * 0.5,
    '-x-z': Math.PI,
    '-x+z': Math.PI * 1.5,
  };

  function pickGeom(brick) {
    if (brick.shape === 'slope') return geomSlope;
    if (brick.shape === 'slope_inv') return geomSlopeInv;
    if (brick.shape === 'slope_corner') return geomCorner;
    return geomBox;
  }
  function pickRotY(brick) {
    if (!brick.low_dir) return 0;
    return LOW_DIR_Y_ROT[brick.low_dir] || 0;
  }

  // Group bricks by RGB × geometry so each instanced mesh has consistent
  // geometry. Slopes need their own instance groups because BufferGeometry
  // must be shared across all instances.
  const groups = new Map();
  for (const b of data.bricks) {
    const geomKey = b.shape === 'slope' || b.shape === 'slope_inv' || b.shape === 'slope_corner'
      ? b.shape : 'box';
    const key = b.rgb.join(',') + '|' + geomKey;
    if (!groups.has(key)) {
      groups.set(key, { rgb: b.rgb, color: b.color, geomKey, bricks: [] });
    }
    groups.get(key).bricks.push(b);
  }

  const meshesByColor = [];
  const studGeom = new THREE.CylinderGeometry(2.4, 2.4, 1.6, 16);
  const tmp = new THREE.Object3D();

  const maxLayer = Math.max(...data.bricks.map(b => b.y + b.h));

  for (const g of groups.values()) {
    const mat = new THREE.MeshLambertMaterial({ color: new THREE.Color(`rgb(${g.rgb.join(',')})`) });

    const geom = g.geomKey === 'slope' ? geomSlope
                : g.geomKey === 'slope_inv' ? geomSlopeInv
                : g.geomKey === 'slope_corner' ? geomCorner
                : geomBox;

    // Bodies.
    const bodyInst = new THREE.InstancedMesh(geom, mat, g.bricks.length);
    bodyInst.castShadow = true;
    bodyInst.receiveShadow = true;
    // Studs (one per stud). Skip for tiles and for slope-family (sloped tops
    // don't have flat plate to stud out from — they still have some studs
    // in reality but rendering them everywhere looks wrong).
    const noStudShape = (sh) => sh === 'tile' || sh === 'slope' || sh === 'slope_inv' || sh === 'slope_corner';
    const studCount = g.bricks.reduce((n, b) => n + (noStudShape(b.shape) ? 0 : b.w * b.d), 0);
    const studInst = studCount > 0 ? new THREE.InstancedMesh(studGeom, mat, studCount) : null;

    let bi = 0, si = 0;
    for (const b of g.bricks) {
      const sx = b.w * S, sy = b.h * P, sz = b.d * S;
      const px = b.x * S + sx / 2;
      const py = b.y * P + sy / 2;
      const pz = b.z * S + sz / 2;
      tmp.position.set(px, py, pz);
      tmp.scale.set(sx - 0.15, sy - 0.1, sz - 0.15);
      tmp.rotation.set(0, pickRotY(b), 0);
      tmp.updateMatrix();
      bodyInst.setMatrixAt(bi++, tmp.matrix);

      if (!noStudShape(b.shape) && studInst) {
        for (let dx = 0; dx < b.w; dx++) {
          for (let dz = 0; dz < b.d; dz++) {
            tmp.position.set(
              (b.x + dx + 0.5) * S,
              (b.y + b.h) * P + 0.8,
              (b.z + dz + 0.5) * S
            );
            tmp.scale.set(1, 1, 1);
            tmp.rotation.set(0, 0, 0);
            tmp.updateMatrix();
            studInst.setMatrixAt(si++, tmp.matrix);
          }
        }
      }
    }
    bodyInst.instanceMatrix.needsUpdate = true;
    if (studInst) studInst.instanceMatrix.needsUpdate = true;

    scene.add(bodyInst);
    if (studInst) scene.add(studInst);
    meshesByColor.push({ rgb: g.rgb, color: g.color, bricks: g.bricks, body: bodyInst, studs: studInst, noStudShape });
  }

  // UI.
  document.getElementById('brick-count').textContent = data.bricks.length;
  document.getElementById('color-count').textContent = groups.size;
  document.getElementById('dimensions').textContent = `${Math.round(maxX - minX)} x ${Math.round(maxZ - minZ)} x ${Math.round(modelHeight)} mm`;
  const report = data.report;
  document.getElementById('status').textContent = report.geometrically_connected === undefined ? '' :
    (report.geometrically_connected ? 'One connected assembly.' : `${report.unsupported_parts} unsupported parts. ${report.connected_components} assemblies.`) +
    (report.single_stud_overhangs ? ` ${report.single_stud_overhangs} single-stud overhangs.` : '');

  const legend = document.getElementById('legend');
  for (const g of groups.values()) {
    const row = document.createElement('div');
    const [r, gr, bl] = g.rgb;
    row.innerHTML = `<span class="sw" style="background:rgb(${r},${gr},${bl})"></span>${g.color} &times; ${g.bricks.length}`;
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
      // Rebuild bodies + studs with only bricks whose base (y) < cutoff.
      const visible = g.bricks.filter(b => cutoff > 0 && b.y <= layers[cutoff - 1]);
      const bodyMats = [];
      const studMats = [];
      for (const b of visible) {
        const hVis = b.h;
        if (hVis <= 0) continue;
        const sx = b.w * S, sy = hVis * P, sz = b.d * S;
        tmp.position.set(b.x * S + sx / 2, b.y * P + sy / 2, b.z * S + sz / 2);
        tmp.scale.set(sx - 0.15, sy - 0.1, sz - 0.15);
        tmp.rotation.set(0, pickRotY(b), 0);
        tmp.updateMatrix();
        bodyMats.push(tmp.matrix.clone());
        if (!g.noStudShape(b.shape)) {
          for (let dx = 0; dx < b.w; dx++) {
            for (let dz = 0; dz < b.d; dz++) {
              tmp.position.set(
                (b.x + dx + 0.5) * S,
                (b.y + b.h) * P + 0.8,
                (b.z + dz + 0.5) * S
              );
              tmp.scale.set(1, 1, 1);
              tmp.rotation.set(0, 0, 0);
              tmp.updateMatrix();
              studMats.push(tmp.matrix.clone());
            }
          }
        }
      }
      g.body.count = bodyMats.length;
      for (let i = 0; i < bodyMats.length; i++) g.body.setMatrixAt(i, bodyMats[i]);
      g.body.instanceMatrix.needsUpdate = true;
      g.body.computeBoundingSphere();
      if (g.studs) {
        g.studs.count = studMats.length;
        for (let i = 0; i < studMats.length; i++) g.studs.setMatrixAt(i, studMats[i]);
        g.studs.instanceMatrix.needsUpdate = true;
        g.studs.computeBoundingSphere();
      }
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
