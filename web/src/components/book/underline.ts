import * as THREE from "three";
import { PAGE_D, PAGE_W, type PageLink } from "./book-scene";
import { DOT_PITCH, dotRule } from "./typeset";

const UNITS = 1200;
const LINES = 8;
const DRAW = 0.3;
const ERASE = 0.22;
const LINE_HEIGHT = 1.46;

const easeOut = (t: number) => 1 - Math.pow(1 - t, 3);
const same = (a: PageLink, b: PageLink) => a.href === b.href && a.side === b.side && a.box[1] === b.box[1];

const vertexShader = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

const fragmentShader = /* glsl */ `
uniform vec4 uLines[${LINES}];
uniform vec3 uColor;
uniform vec2 uPage;
uniform float uPitch;
varying vec2 vUv;
float span(float lo, float hi, float p, float w) {
  return clamp((p - lo) / w + 0.5, 0.0, 1.0) * clamp((hi - p) / w + 0.5, 0.0, 1.0);
}
void main() {
  vec2 p = vec2(vUv.x, 1.0 - vUv.y) * uPage;
  vec2 w = max(fwidth(p), vec2(0.0001));
  float alpha = 0.0;
  for (int i = 0; i < ${LINES}; i++) {
    vec4 l = uLines[i];
    if (l.w <= 0.0) continue;
    float pitch = l.w * uPitch;
    float local = mod(p.x - l.x, pitch);
    float dots = max(span(0.0, l.w, local, w.x), span(pitch, pitch + l.w, local, w.x));
    alpha = max(alpha, dots * span(l.x, l.z, p.x, w.x) * span(l.y, l.y + l.w, p.y, w.y));
  }
  if (alpha < 0.004) discard;
  gl_FragColor = vec4(uColor, alpha);
  #include <colorspace_fragment>
}
`;

type Side = "left" | "right";
type Mark = { link: PageLink; from: number; to: number; value: number; elapsed: number; duration: number };
type Layer = { mesh: THREE.Mesh<THREE.BufferGeometry, THREE.ShaderMaterial>; lines: THREE.Vector4[] };

export class LinkUnderline {
  private marks: Mark[] = [];
  private layers: Record<Side, Layer>;

  constructor(pages: Record<Side, [THREE.BufferGeometry, THREE.Object3D]>, color: string) {
    const make = ([geometry, parent]: [THREE.BufferGeometry, THREE.Object3D]): Layer => {
      const lines = Array.from({ length: LINES }, () => new THREE.Vector4());
      const material = new THREE.ShaderMaterial({ vertexShader, fragmentShader, transparent: true, depthWrite: false, uniforms: { uLines: { value: lines }, uColor: { value: new THREE.Color(color) }, uPage: { value: new THREE.Vector2(UNITS, (UNITS * PAGE_D) / PAGE_W) }, uPitch: { value: DOT_PITCH } } });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.y = 0.0016;
      mesh.renderOrder = 3;
      mesh.visible = false;
      parent.add(mesh);
      return { mesh, lines };
    };
    this.layers = { left: make(pages.left), right: make(pages.right) };
  }

  set(link: PageLink | null, instant: boolean): void {
    const target = link?.lines?.length ? link : null;
    let found = false;
    for (const mark of this.marks) {
      const on = target !== null && same(mark.link, target);
      if (on) {
        found = true;
        mark.link = target;
      }
      this.aim(mark, on ? 1 : 0);
    }
    if (target && !found) {
      const mark: Mark = { link: target, from: 0, to: 0, value: 0, elapsed: 0, duration: DRAW };
      this.marks.push(mark);
      this.aim(mark, 1);
    }
    if (instant) for (const mark of this.marks) mark.value = mark.from = mark.to;
    this.write();
  }

  clear(): void {
    this.marks = [];
    this.write();
  }

  lines(side: Side): number[][] {
    return this.layers[side].lines.filter((l) => l.w > 0).map((l) => [l.x / UNITS, l.y / UNITS, l.z / UNITS, l.w / UNITS]);
  }

  update(dt: number): boolean {
    if (!this.marks.some((m) => m.value !== m.to)) return false;
    for (const mark of this.marks) {
      if (mark.value === mark.to) continue;
      mark.elapsed += dt;
      mark.value = mark.from + (mark.to - mark.from) * easeOut(Math.min(1, mark.elapsed / mark.duration));
    }
    this.write();
    return true;
  }

  private aim(mark: Mark, to: number): void {
    if (mark.to === to) return;
    mark.from = mark.value;
    mark.to = to;
    mark.elapsed = 0;
    mark.duration = (to ? DRAW : ERASE) * Math.max(0.4, Math.abs(to - mark.value));
  }

  private write(): void {
    this.marks = this.marks.filter((m) => m.value > 0 || m.to > 0);
    for (const side of ["left", "right"] as const) {
      const layer = this.layers[side];
      let n = 0;
      for (const mark of this.marks.filter((m) => m.link.side === side).reverse()) {
        const lines = mark.link.lines ?? [];
        let left = mark.value * lines.reduce((sum, l) => sum + l.width, 0);
        for (const line of lines) {
          const shown = Math.min(line.width, left);
          left -= shown;
          if (shown <= 0 || n >= LINES) break;
          const { dot, drop } = dotRule(line.size * LINE_HEIGHT);
          layer.lines[n++].set(line.x, line.y + drop, line.x + shown, dot);
        }
      }
      layer.mesh.visible = n > 0;
      for (; n < LINES; n++) layer.lines[n].set(0, 0, 0, 0);
    }
  }
}
