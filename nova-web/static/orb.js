// Nova's breathing particle orb: the same "mind" as the /about/ landing page,
// scaled down for a chat app.
//   - On the welcome screen it breathes large behind the headline.
//   - Once a conversation starts it dims and shrinks, so it never fights the text.
//   - While Nova is working (the message box has .busy) it breathes faster.
//   - It pauses when the tab is hidden, and respects "reduce motion".
import * as THREE from "three";

const canvas = document.getElementById("orb");
const welcome = document.getElementById("welcome");
const field = document.getElementById("field");
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const small = innerWidth < 640;
const COUNT = small ? 2200 : 4200;

let renderer;
try {
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
} catch {
  canvas.remove();                       // no WebGL: the page works fine without the orb
  throw new Error("WebGL unavailable");
}
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.1, 100);
camera.position.z = 6;

const mint = new THREE.Color(0x2dd4bf);
const violet = new THREE.Color(0xa855f7);
const pos = new Float32Array(COUNT * 3);
const col = new Float32Array(COUNT * 3);
const seed = new Float32Array(COUNT);
for (let i = 0; i < COUNT; i++) {
  const th = Math.random() * Math.PI * 2;
  const ph = Math.acos(Math.random() * 2 - 1);
  const r = 1.6 + (Math.random() - 0.5) * 0.4;
  pos[i * 3] = r * Math.sin(ph) * Math.cos(th);
  pos[i * 3 + 1] = r * Math.sin(ph) * Math.sin(th);
  pos[i * 3 + 2] = r * Math.cos(ph);
  const c = mint.clone().lerp(violet, Math.random());
  col.set([c.r, c.g, c.b], i * 3);
  seed[i] = Math.random();
}
const geo = new THREE.BufferGeometry();
geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
geo.setAttribute("aColor", new THREE.BufferAttribute(col, 3));
geo.setAttribute("aSeed", new THREE.BufferAttribute(seed, 1));

const mat = new THREE.ShaderMaterial({
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  uniforms: {
    uTime: { value: 0 }, uBreath: { value: 0 }, uSize: { value: small ? 44 : 42 },
    uPixel: { value: renderer.getPixelRatio() }, uFade: { value: 1 },
  },
  vertexShader: /* glsl */`
    attribute vec3 aColor; attribute float aSeed;
    uniform float uTime, uBreath, uSize, uPixel;
    varying vec3 vColor;
    void main() {
      vec3 p = position * (1.0 + uBreath)
             + normalize(position) * sin(uTime * 2.0 + aSeed * 40.0) * 0.03
             + vec3(sin(uTime * .7 + aSeed * 31.0), cos(uTime * .6 + aSeed * 17.0), sin(uTime * .5 + aSeed * 23.0)) * 0.012;
      vec4 mv = modelViewMatrix * vec4(p, 1.0);
      gl_Position = projectionMatrix * mv;
      gl_PointSize = uSize * uPixel * (0.55 + aSeed * 0.6) / -mv.z;
      vColor = aColor * (0.75 + 0.25 * sin(uTime * 2.4 + aSeed * 60.0));
    }`,
  fragmentShader: /* glsl */`
    uniform float uFade;
    varying vec3 vColor;
    void main() {
      float d = length(gl_PointCoord - 0.5);
      float core = smoothstep(0.18, 0.0, d);
      float halo = smoothstep(0.5, 0.0, d);
      gl_FragColor = vec4(vColor * (1.0 + core * 0.8), (halo * halo * 0.7 + core * 0.6) * 0.85 * uFade);
    }`,
});
const orb = new THREE.Points(geo, mat);
scene.add(orb);

// Where the orb sits: high and large on the welcome screen, small and dim in a chat.
let chatMix = 0;        // 0 = welcome, 1 = chatting
let busyMix = 0;        // 0 = idle, 1 = Nova is working
let phase = 0;          // breathing phase, advanced faster when busy
let mx = 0, my = 0;
addEventListener("pointermove", e => { mx = e.clientX / innerWidth - 0.5; my = e.clientY / innerHeight - 0.5; }, { passive: true });

function layout() {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
}
addEventListener("resize", layout);

const clock = new THREE.Clock();
let running = true;
document.addEventListener("visibilitychange", () => {
  running = !document.hidden;
  if (running) { clock.getDelta(); requestAnimationFrame(frame); }
});

function frame() {
  if (!running) return;
  const dt = Math.min(clock.getDelta(), 0.1);
  const t = clock.elapsedTime;

  const inChat = welcome.classList.contains("hidden") ? 1 : 0;
  const busy = field.classList.contains("busy") ? 1 : 0;
  chatMix += (inChat - chatMix) * Math.min(1, dt * 2.5);
  busyMix += (busy - busyMix) * Math.min(1, dt * 3);

  // Lung-like breathing: a slow fundamental plus a softer harmonic; faster while working
  phase += dt * (1.8 + busyMix * 2.6) * (reduceMotion ? 0 : 1);
  mat.uniforms.uBreath.value = (Math.sin(phase) * 0.09 + Math.sin(phase * 2 + 1.1) * 0.03) * (1 + busyMix * 0.5);
  mat.uniforms.uTime.value = reduceMotion ? 1.5 : t;

  // Welcome: centred high in the view. Chat: smaller, behind the top of the page, dimmed.
  const s = 0.88 - chatMix * 0.38;
  orb.scale.setScalar(s);
  orb.position.y = 1.7 - chatMix * 0.5;
  mat.uniforms.uFade.value = 1 - chatMix * 0.72 + busyMix * 0.25;
  orb.rotation.y = (reduceMotion ? 0 : t * 0.15) + mx * 0.3;
  orb.rotation.x = Math.sin(t * 0.1) * 0.1 + my * 0.15;

  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
