import * as THREE from "three";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";

export type BookPalette = {
  desk: string;
  paper: string;
  ink: string;
  faint: string;
  blue: string;
  pink: string;
  cover: string;
  edge: string;
  ribbon: string;
};

export function readBookPalette(): BookPalette {
  const s = getComputedStyle(document.documentElement);
  const v = (name: string) => s.getPropertyValue(name).trim();
  return {
    desk: v("--book-desk"),
    paper: v("--book-paper"),
    ink: v("--book-ink"),
    faint: v("--book-faint"),
    blue: v("--book-blue"),
    pink: v("--book-pink"),
    cover: v("--book-cover"),
    edge: v("--book-edge"),
    ribbon: v("--book-ribbon"),
  };
}

export const inkShared = { resolution: { value: new THREE.Vector2(1, 1) } };

const hatchFragment = /* glsl */ `
float hatchTone = clamp(dot(outgoingLight, vec3(0.333)) / max(dot(diffuseColor.rgb, vec3(0.333)), 0.001), 0.0, 1.0);
vec3 hatchN = abs(normalize(vHatchNormal));
vec2 hatchP = hatchN.y > hatchN.x && hatchN.y > hatchN.z ? vHatchPos.xz : (hatchN.x > hatchN.z ? vHatchPos.zy : vHatchPos.xy);
float hatchA = hatchP.x * 0.71 + hatchP.y * 0.71;
float hatchB = hatchP.x * 0.71 - hatchP.y * 0.71;
float hatchWidthA = fwidth(hatchA * uHatch);
float hatchVisible = 1.0 - smoothstep(0.35, 0.8, hatchWidthA);
float hatchLineA = 1.0 - smoothstep(0.1, 0.1 + hatchWidthA * 1.2, abs(fract(hatchA * uHatch) - 0.5));
float hatchLineB = 1.0 - smoothstep(0.08, 0.08 + hatchWidthA * 1.2, abs(fract(hatchB * uHatch) - 0.5));
float hatchLineC = 1.0 - smoothstep(0.1, 0.1 + hatchWidthA * 2.4, abs(fract(hatchA * uHatch * 2.0) - 0.5));
float hatchAmount = hatchLineA * smoothstep(0.9, 0.72, hatchTone) * 0.55;
hatchAmount = max(hatchAmount, hatchLineB * smoothstep(0.66, 0.46, hatchTone) * 0.62);
hatchAmount = max(hatchAmount, hatchLineC * smoothstep(0.42, 0.28, hatchTone) * 0.7);
hatchAmount *= hatchVisible;
float hatchGrey = (1.0 - hatchVisible) * (1.0 - hatchTone) * 0.3;
outgoingLight = mix(diffuseColor.rgb * (0.9 + 0.1 * hatchTone), uInk, clamp(hatchAmount + hatchGrey, 0.0, 1.0));
`;

export function inkMaterial(palette: BookPalette, color?: string | THREE.Color): THREE.MeshLambertMaterial {
  const material = new THREE.MeshLambertMaterial({ color: color ?? palette.paper, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 });
  material.onBeforeCompile = (shader) => {
    shader.uniforms.uInk = { value: new THREE.Color(palette.ink) };
    shader.uniforms.uHatch = { value: 95 };
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", "#include <common>\nvarying vec3 vHatchPos;\nvarying vec3 vHatchNormal;")
      .replace("#include <begin_vertex>", "#include <begin_vertex>\nvHatchPos = position;\nvHatchNormal = normal;");
    shader.fragmentShader = shader.fragmentShader
      .replace("#include <common>", "#include <common>\nvarying vec3 vHatchPos;\nvarying vec3 vHatchNormal;\nuniform vec3 uInk;\nuniform float uHatch;")
      .replace("#include <opaque_fragment>", `${hatchFragment}\n#include <opaque_fragment>`);
  };
  material.customProgramCacheKey = () => "book-ink";
  return material;
}

export function liquidMaterial(color: string): THREE.MeshLambertMaterial {
  const c = new THREE.Color(color);
  return new THREE.MeshLambertMaterial({ color: c, emissive: c, emissiveIntensity: 0.45 });
}

export function glassMaterial(palette: BookPalette): THREE.MeshLambertMaterial {
  return new THREE.MeshLambertMaterial({ color: palette.paper, transparent: true, opacity: 0.16, depthWrite: false, side: THREE.DoubleSide });
}

export function inkLineMaterial(color: string, width: number, opacity = 1): LineMaterial {
  return new LineMaterial({ color: new THREE.Color(color).getHex(), linewidth: width, worldUnits: false, transparent: opacity < 1, opacity });
}

let seed = 1;
const jitter = (amount: number) => {
  seed = (seed * 16807) % 2147483647;
  return ((seed / 2147483647) * 2 - 1) * amount;
};

export function inkSegments(points: number[], material: LineMaterial, wobble = 0.0012): LineSegments2 {
  const out = new Float32Array(points.length);
  for (let i = 0; i < points.length; i += 6) {
    const dx = points[i + 3] - points[i];
    const dy = points[i + 4] - points[i + 1];
    const dz = points[i + 5] - points[i + 2];
    const over = Math.abs(jitter(0.012));
    out[i] = points[i] - dx * over + jitter(wobble);
    out[i + 1] = points[i + 1] - dy * over + jitter(wobble);
    out[i + 2] = points[i + 2] - dz * over + jitter(wobble);
    out[i + 3] = points[i + 3] + dx * over + jitter(wobble);
    out[i + 4] = points[i + 4] + dy * over + jitter(wobble);
    out[i + 5] = points[i + 5] + dz * over + jitter(wobble);
  }
  const geometry = new LineSegmentsGeometry();
  geometry.setPositions(out);
  return new LineSegments2(geometry, material);
}

export function inkEdges(geometry: THREE.BufferGeometry, material: LineMaterial, threshold = 28, wobble = 0.0012): LineSegments2 {
  const edges = new THREE.EdgesGeometry(geometry, threshold);
  const points = Array.from(edges.attributes.position.array as Float32Array);
  edges.dispose();
  return inkSegments(points, material, wobble);
}

export function hullMaterial(palette: BookPalette, width: number): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    side: THREE.BackSide,
    uniforms: { uColor: { value: new THREE.Color(palette.ink) }, uWidth: { value: width }, uResolution: inkShared.resolution },
    vertexShader: /* glsl */ `
      uniform float uWidth;
      uniform vec2 uResolution;
      void main() {
        vec4 clip = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
        vec3 n = normalMatrix * normal;
        vec2 offset = normalize(n.xy + vec2(1e-5));
        clip.xy += offset * uWidth * 2.0 / uResolution * clip.w;
        gl_Position = clip;
      }
    `,
    fragmentShader: /* glsl */ `
      uniform vec3 uColor;
      void main() {
        gl_FragColor = vec4(uColor, 1.0);
        #include <colorspace_fragment>
      }
    `,
  });
}

export type Kit = {
  palette: BookPalette;
  ink: THREE.MeshLambertMaterial;
  line: LineMaterial;
  thin: LineMaterial;
  hull: THREE.ShaderMaterial;
  glass: THREE.MeshLambertMaterial;
  blue: THREE.MeshLambertMaterial;
  pink: THREE.MeshLambertMaterial;
  solid: (geometry: THREE.BufferGeometry, options?: { material?: THREE.Material; edges?: boolean; hull?: boolean; threshold?: number; shadow?: boolean }) => THREE.Group;
};

export function createKit(palette: BookPalette): Kit {
  const ink = inkMaterial(palette);
  const line = inkLineMaterial(palette.ink, 1.3);
  const thin = inkLineMaterial(palette.ink, 0.8);
  const hull = hullMaterial(palette, 1.1);
  const kit: Kit = {
    palette,
    ink,
    line,
    thin,
    hull,
    glass: glassMaterial(palette),
    blue: liquidMaterial(palette.blue),
    pink: liquidMaterial(palette.pink),
    solid(geometry, options = {}) {
      const group = new THREE.Group();
      const mesh = new THREE.Mesh(geometry, options.material ?? ink);
      mesh.castShadow = options.shadow ?? true;
      mesh.receiveShadow = true;
      group.add(mesh);
      if (options.hull) group.add(new THREE.Mesh(geometry, hull));
      if (options.edges !== false) group.add(inkEdges(geometry, line, options.threshold));
      return group;
    },
  };
  return kit;
}

export type Card = { group: THREE.Group; redraw: () => void };

export function createCard(kit: Kit, width: number, height: number, draw: (ctx: CanvasRenderingContext2D, w: number, h: number) => void, tilt = -0.42): Card {
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(width * 1500);
  canvas.height = Math.round(height * 1500);
  const ctx = canvas.getContext("2d")!;
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 8;
  const redraw = () => {
    ctx.fillStyle = kit.palette.paper;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    draw(ctx, canvas.width, canvas.height);
    texture.needsUpdate = true;
  };
  redraw();
  const group = new THREE.Group();
  const board = kit.solid(new THREE.BoxGeometry(width, height, 0.004), { edges: true });
  const face = new THREE.Mesh(new THREE.PlaneGeometry(width, height), new THREE.MeshBasicMaterial({ map: texture }));
  face.position.z = 0.0026;
  const tilted = new THREE.Group();
  tilted.add(board, face);
  tilted.position.y = (height / 2) * Math.cos(tilt);
  tilted.position.z = (height / 2) * Math.sin(tilt) * -1;
  tilted.rotation.x = tilt;
  group.add(tilted);
  return { group, redraw };
}

export function washPlane(color: string, width: number, depth: number, opacity = 0.32): THREE.Mesh<THREE.PlaneGeometry, THREE.MeshBasicMaterial> {
  const canvas = document.createElement("canvas");
  canvas.width = canvas.height = 128;
  const ctx = canvas.getContext("2d")!;
  const g = ctx.createRadialGradient(64, 64, 4, 64, 64, 64);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(0.55, "rgba(255,255,255,0.45)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 128, 128);
  const texture = new THREE.CanvasTexture(canvas);
  const material = new THREE.MeshBasicMaterial({ color, alphaMap: texture, transparent: true, opacity, depthWrite: false });
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(width, depth), material);
  mesh.rotation.x = -Math.PI / 2;
  mesh.position.y = 0.002;
  return mesh;
}

export function ribbonTube(points: THREE.Vector3[], radius: number, material: THREE.Material): THREE.Mesh {
  const curve = new THREE.CatmullRomCurve3(points);
  const mesh = new THREE.Mesh(new THREE.TubeGeometry(curve, 28, radius, 6), material);
  mesh.castShadow = true;
  return mesh;
}
