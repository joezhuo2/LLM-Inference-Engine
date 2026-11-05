import * as THREE from "three";
import { type BookPalette, createKit, inkShared, type Kit, readBookPalette } from "./ink";
import { drawCaptionPage, drawTextPage, type Fonts, type PageText } from "./typeset";

export const PAGE_W = 1;
export const PAGE_D = 1.38;
export const BLOCK = 0.07;

export type Frame = { center: [number, number, number]; width: number; depth: number; az: number; el: number };
export type Beat = { caption: string; frame: Frame; narrow?: Frame };
export type OverlayKind = string | null;
export type PagePainter = (canvas: HTMLCanvasElement, palette: BookPalette, fonts: Fonts) => void;
export type PageLink = { side: "left" | "right"; box: [number, number, number, number]; href: string; label: string };
export type Stop = {
  exhibit: number | null;
  spread: number;
  page: PageText | null;
  figure: string;
  caption: string;
  instruction?: string;
  folio: string;
  overlay: OverlayKind;
  beats: Beat[];
  paint?: { left?: PagePainter; right?: PagePainter };
  links?: PageLink[];
};

export type BookModel = {
  exhibits: THREE.Group[];
  handles: THREE.Object3D[];
  dragStart?: (handle: THREE.Object3D) => void;
  drag?: (x: number) => void;
  dragEnd?: () => void;
  update: (dt: number, exhibit: number, stop: number, reduced: boolean) => void;
  enter?: (stop: number) => void;
};

export type Rect = { left: number; top: number; width: number; height: number };
export type OverlayRect = { kind: string; rect: Rect };

type SceneOptions = {
  canvas: HTMLCanvasElement;
  host: HTMLElement;
  stops: Stop[];
  coverTitle: string;
  coverSubtitle: string;
  fonts: Fonts;
  reduced: boolean;
  buildModel: (kit: Kit) => BookModel;
  onOverlay: (rect: OverlayRect | null) => void;
  onLens: (lens: { x: number; y: number; source: HTMLCanvasElement; u: number; v: number } | null) => void;
  onFrameTime?: (ms: number) => void;
  onPageClick: (side: "prev" | "next") => void;
  coverFrame?: Frame;
  onCoverClick?: () => void;
  onLink?: (link: PageLink) => void;
};

const easeInOut = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);
const easeOutBack = (t: number) => 1 + 2.1 * Math.pow(t - 1, 3) + 1.1 * Math.pow(t - 1, 2);
const clamp01 = (t: number) => Math.min(1, Math.max(0, t));
const rad = (deg: number) => (deg * Math.PI) / 180;
const defaultCoverFrame: Frame = { center: [0.45, 0, 0], width: 1.9, depth: 2, az: 12, el: 58 };

type Tween = { elapsed: number; duration: number; step: (t: number) => void; resolve: () => void };

function pageGeometry(side: 1 | -1, y: number): THREE.BufferGeometry {
  const geometry = new THREE.PlaneGeometry(PAGE_W, PAGE_D, 40, 1);
  geometry.rotateX(-Math.PI / 2);
  const position = geometry.attributes.position as THREE.BufferAttribute;
  for (let i = 0; i < position.count; i++) {
    const u = position.getX(i) + PAGE_W / 2;
    const fromSpine = side === 1 ? u : PAGE_W - u;
    position.setX(i, side === 1 ? u : u - PAGE_W);
    position.setY(i, y + pageHeight(fromSpine));
  }
  geometry.computeVertexNormals();
  return geometry;
}

function edgeTexture(paper: string, ink: string): THREE.CanvasTexture {
  const canvas = document.createElement("canvas");
  canvas.width = 4;
  canvas.height = 64;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = paper;
  ctx.fillRect(0, 0, 4, 64);
  ctx.globalAlpha = 0.18;
  ctx.fillStyle = ink;
  for (let y = 1; y < 64; y += 4) ctx.fillRect(0, y, 4, 1);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(1, 1 / BLOCK / 9);
  return texture;
}

const pageHeight = (fromSpine: number) => -0.032 * Math.exp(-fromSpine * 13) - 0.006 * fromSpine * fromSpine;

class TextSide {
  readonly canvases: [HTMLCanvasElement, HTMLCanvasElement];
  readonly textures: [THREE.CanvasTexture, THREE.CanvasTexture];
  readonly meshes: [THREE.Mesh<THREE.BufferGeometry, THREE.MeshBasicMaterial>, THREE.Mesh<THREE.BufferGeometry, THREE.MeshBasicMaterial>];
  active = 0;

  constructor(geometry: THREE.BufferGeometry, parent: THREE.Object3D, anisotropy: number, textureWidth: number) {
    const make = () => {
      const canvas = document.createElement("canvas");
      canvas.width = textureWidth;
      canvas.height = Math.round((textureWidth * PAGE_D) / PAGE_W);
      const texture = new THREE.CanvasTexture(canvas);
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.anisotropy = anisotropy;
      const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({ map: texture, transparent: true, opacity: 0, depthWrite: false }));
      mesh.position.y = 0.0012;
      mesh.renderOrder = 2;
      parent.add(mesh);
      return { canvas, texture, mesh };
    };
    const a = make();
    const b = make();
    this.canvases = [a.canvas, b.canvas];
    this.textures = [a.texture, b.texture];
    this.meshes = [a.mesh, b.mesh];
  }

  get current(): HTMLCanvasElement {
    return this.canvases[this.active];
  }

  prepare(draw: (canvas: HTMLCanvasElement) => void): number {
    const next = 1 - this.active;
    draw(this.canvases[next]);
    this.textures[next].needsUpdate = true;
    return next;
  }

  blend(next: number, t: number): void {
    this.meshes[next].material.opacity = t;
    if (next !== this.active) this.meshes[this.active].material.opacity = 1 - t;
  }

  commit(next: number): void {
    this.active = next;
  }
}

export class BookScene {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera = new THREE.PerspectiveCamera(26, 1, 0.05, 40);
  private timer = new THREE.Timer();
  private tweens: Tween[] = [];
  private raf = 0;
  private visible = true;
  private frozen = false;
  private options: SceneOptions;
  private kit: Kit;
  private model: BookModel;
  private leftHalf = new THREE.Group();
  private bookRoot = new THREE.Group();
  private modelRoot = new THREE.Group();
  private leftText: TextSide;
  private rightText: TextSide;
  private leaf: { front: THREE.Mesh; back: THREE.Mesh; geometry: THREE.BufferGeometry; frontMap: THREE.CanvasTexture; backMap: THREE.CanvasTexture; frontCanvas: HTMLCanvasElement; backCanvas: HTMLCanvasElement };
  private desk: THREE.Mesh<THREE.PlaneGeometry, THREE.MeshLambertMaterial>;
  private cam = { x: 0, y: 0, z: 0, az: 0, el: 60, dist: 6 };
  private camGoal = { x: 0, y: 0, z: 0, az: 0, el: 60, dist: 6 };
  private camStartTime = -1;
  private camFrom = { ...this.cam };
  private camDuration = 1.8;
  private applied = { stop: -1, beat: 0 };
  private target = { stop: 0, beat: 0 };
  private running = false;
  private opened = false;
  private closedView = false;
  private settling: Promise<void> = Promise.resolve();
  private narrow = false;
  private width = 1;
  private height = 1;
  private time = 0;
  private drag: { mode: "orbit" | "handle"; x: number; y: number; moved: number } | null = null;
  private orbit = { az: 0, el: 0 };
  private overlayKey = "";
  private raycaster = new THREE.Raycaster();
  private pointer = new THREE.Vector2();
  private leftPlane: THREE.Mesh;
  private rightPlane: THREE.Mesh;
  private disposers: (() => void)[] = [];

  constructor(options: SceneOptions) {
    this.options = options;
    const palette = readBookPalette();
    this.kit = createKit(palette);
    this.renderer = new THREE.WebGLRenderer({ canvas: options.canvas, antialias: true, alpha: false });
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.NoToneMapping;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.scene.background = new THREE.Color(palette.desk);

    const hemi = new THREE.HemisphereLight("#ffffff", "#e8e2d4", 1.45);
    const sun = new THREE.DirectionalLight("#fff6e8", 1.85);
    sun.position.set(-2.4, 4.4, 1.5);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.camera.left = -2.6;
    sun.shadow.camera.right = 2.6;
    sun.shadow.camera.top = 2.2;
    sun.shadow.camera.bottom = -2.2;
    sun.shadow.camera.near = 0.5;
    sun.shadow.camera.far = 9;
    sun.shadow.bias = -0.0004;
    sun.shadow.normalBias = 0.004;
    sun.shadow.radius = 6;
    sun.shadow.blurSamples = 16;
    sun.shadow.intensity = 0.62;
    this.scene.add(hemi, sun);


    this.desk = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new THREE.MeshLambertMaterial({ color: palette.desk }));
    this.desk.rotation.x = -Math.PI / 2;
    this.desk.position.y = -0.03;
    this.desk.receiveShadow = true;
    this.scene.add(this.desk);

    const anisotropy = this.renderer.capabilities.getMaxAnisotropy();
    const screenPage = (Math.max(options.host.clientWidth, options.host.clientHeight) * Math.min(window.devicePixelRatio || 1, 2)) / 2;
    const textureWidth = Math.min(this.renderer.capabilities.maxTextureSize, 2560, Math.max(1600, Math.round((screenPage * 1.6) / 64) * 64));
    const coverMaterial = new THREE.MeshLambertMaterial({ color: palette.cover });
    const paperMaterial = new THREE.MeshLambertMaterial({ color: palette.paper });
    const blockMaterial = new THREE.MeshLambertMaterial({ map: edgeTexture(palette.paper, palette.ink), side: THREE.DoubleSide });

    const halfBook = (side: 1 | -1, parent: THREE.Object3D) => {
      const board = new THREE.Mesh(new THREE.BoxGeometry(PAGE_W + 0.03, 0.022, PAGE_D + 0.05), coverMaterial);
      board.position.set((side * (PAGE_W + 0.03)) / 2, -BLOCK - 0.011, 0);
      board.castShadow = true;
      board.receiveShadow = true;
      const profile = new THREE.Shape();
      profile.moveTo(0, -BLOCK);
      for (let k = 0; k <= 40; k++) {
        const u = (k / 40) * PAGE_W;
        profile.lineTo(side * u, pageHeight(u) - 0.0015);
      }
      profile.lineTo(side * PAGE_W, -BLOCK);
      profile.closePath();
      const blockGeometry = new THREE.ExtrudeGeometry(profile, { depth: PAGE_D, bevelEnabled: false, curveSegments: 1 });
      blockGeometry.translate(0, 0, -PAGE_D / 2);
      const block = new THREE.Mesh(blockGeometry, blockMaterial);
      block.castShadow = true;
      block.receiveShadow = true;
      const page = new THREE.Mesh(pageGeometry(side, 0), paperMaterial);
      page.receiveShadow = true;
      page.castShadow = true;
      parent.add(board, block, page);
      return page;
    };

    this.bookRoot.position.y = BLOCK + 0.022 - 0.03;
    const rightHalf = new THREE.Group();
    this.bookRoot.add(rightHalf, this.leftHalf);
    this.scene.add(this.bookRoot);
    const rightPage = halfBook(1, rightHalf);
    this.rightPlane = rightPage;
    this.leftPlane = halfBook(-1, this.leftHalf);
    this.leftText = new TextSide(this.leftPlane.geometry, this.leftHalf, anisotropy, textureWidth);
    this.rightText = new TextSide(rightPage.geometry, rightHalf, anisotropy, textureWidth);

    const spine = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.03, PAGE_D + 0.05), coverMaterial);
    spine.position.set(0, -BLOCK - 0.012, 0);
    this.bookRoot.add(spine);

    const ribbon = new THREE.Mesh(new THREE.BoxGeometry(0.035, 0.003, 0.34), new THREE.MeshLambertMaterial({ color: palette.ribbon }));
    ribbon.position.set(0.13, -BLOCK - 0.026, PAGE_D / 2 + 0.14);
    ribbon.castShadow = true;
    this.bookRoot.add(ribbon);

    const coverCanvas = document.createElement("canvas");
    coverCanvas.width = 1000;
    coverCanvas.height = 1380;
    const cover = coverCanvas.getContext("2d")!;
    cover.fillStyle = palette.cover;
    cover.fillRect(0, 0, 1000, 1380);
    cover.strokeStyle = palette.edge;
    cover.lineWidth = 3;
    cover.strokeRect(60, 60, 880, 1260);
    cover.fillStyle = palette.paper;
    cover.font = `150px ${options.fonts.serif}`;
    cover.fillText(options.coverTitle, 110, 300);
    cover.fillStyle = palette.edge;
    cover.font = `38px ${options.fonts.sans}`;
    cover.fillText(options.coverSubtitle, 114, 370);
    const coverTexture = new THREE.CanvasTexture(coverCanvas);
    coverTexture.colorSpace = THREE.SRGBColorSpace;
    coverTexture.anisotropy = anisotropy;
    const coverFace = new THREE.Mesh(new THREE.PlaneGeometry(PAGE_W + 0.03, PAGE_D + 0.05), new THREE.MeshLambertMaterial({ map: coverTexture }));
    coverFace.rotation.x = Math.PI / 2;
    coverFace.rotation.z = Math.PI;
    coverFace.position.set(-(PAGE_W + 0.03) / 2, -BLOCK - 0.0225, 0);
    coverFace.material.side = THREE.DoubleSide;
    this.leftHalf.add(coverFace);

    const leafGeometry = new THREE.PlaneGeometry(PAGE_W, PAGE_D, 40, 1);
    leafGeometry.rotateX(-Math.PI / 2);
    const leafCanvas = () => {
      const canvas = document.createElement("canvas");
      canvas.width = textureWidth;
      canvas.height = Math.round((textureWidth * PAGE_D) / PAGE_W);
      const texture = new THREE.CanvasTexture(canvas);
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.anisotropy = anisotropy;
      return { canvas, texture };
    };
    const frontLeaf = leafCanvas();
    const backLeaf = leafCanvas();
    backLeaf.texture.wrapS = THREE.RepeatWrapping;
    backLeaf.texture.repeat.x = -1;
    backLeaf.texture.offset.x = 1;
    const front = new THREE.Mesh(leafGeometry, new THREE.MeshLambertMaterial({ map: frontLeaf.texture, side: THREE.FrontSide }));
    const back = new THREE.Mesh(leafGeometry, new THREE.MeshLambertMaterial({ map: backLeaf.texture, side: THREE.BackSide }));
    front.castShadow = true;
    front.visible = false;
    back.visible = false;
    this.bookRoot.add(front, back);
    this.leaf = { front, back, geometry: leafGeometry, frontMap: frontLeaf.texture, backMap: backLeaf.texture, frontCanvas: frontLeaf.canvas, backCanvas: backLeaf.canvas };

    this.bookRoot.add(this.modelRoot);
    this.model = options.buildModel(this.kit);
    for (const exhibit of this.model.exhibits) {
      exhibit.scale.y = 0.001;
      exhibit.visible = false;
      this.modelRoot.add(exhibit);
    }

    this.listen();
    this.resize();
  }

  private listen(): void {
    const { canvas, host } = this.options;
    const onDown = (event: PointerEvent) => {
      const handle = this.pickHandle(event);
      this.drag = { mode: handle ? "handle" : "orbit", x: event.clientX, y: event.clientY, moved: 0 };
      if (handle) this.model.dragStart?.(handle);
      canvas.setPointerCapture(event.pointerId);
    };
    const onMove = (event: PointerEvent) => {
      if (this.drag?.mode === "handle") {
        const x = this.pointOnPage(event);
        if (x !== null) this.model.drag?.(x);
        return;
      }
      if (this.drag) {
        this.drag.moved += Math.abs(event.clientX - this.drag.x) + Math.abs(event.clientY - this.drag.y);
        this.orbit.az = THREE.MathUtils.clamp(this.orbit.az - (event.clientX - this.drag.x) * 0.25, -55, 55);
        this.orbit.el = THREE.MathUtils.clamp(this.orbit.el + (event.clientY - this.drag.y) * 0.18, -30, 18);
        this.drag.x = event.clientX;
        this.drag.y = event.clientY;
        return;
      }
      canvas.style.cursor = this.pickHandle(event) ? "grab" : this.linkAt(event) || this.pageSide(event) || this.overCover(event) ? "pointer" : "default";
      this.updateLens(event);
    };
    const onUp = (event: PointerEvent) => {
      if (this.drag?.mode === "handle") this.model.dragEnd?.();
      const click = this.drag?.mode === "orbit" && this.drag.moved < 6 && event.type === "pointerup";
      this.drag = null;
      if (!click) return;
      if (this.overCover(event)) {
        this.options.onCoverClick?.();
        return;
      }
      const link = this.linkAt(event);
      if (link) {
        this.options.onLink?.(link);
        return;
      }
      const side = this.pageSide(event);
      if (side) this.options.onPageClick(side);
    };
    const onLeave = () => this.options.onLens(null);
    canvas.addEventListener("pointerdown", onDown);
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerup", onUp);
    canvas.addEventListener("pointercancel", onUp);
    canvas.addEventListener("pointerleave", onLeave);
    const observer = new ResizeObserver(() => this.resize());
    observer.observe(host);
    const intersection = new IntersectionObserver(([entry]) => this.setVisible(entry.isIntersecting));
    intersection.observe(host);
    const onVisibility = () => this.setVisible(!document.hidden);
    document.addEventListener("visibilitychange", onVisibility);
    const theme = new MutationObserver(() => {
      const palette = readBookPalette();
      this.desk.material.color.set(palette.desk);
      (this.scene.background as THREE.Color).set(palette.desk);
    });
    theme.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    this.disposers.push(() => {
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointercancel", onUp);
      canvas.removeEventListener("pointerleave", onLeave);
      observer.disconnect();
      intersection.disconnect();
      theme.disconnect();
      document.removeEventListener("visibilitychange", onVisibility);
    });
  }

  private setPointer(event: PointerEvent): void {
    const rect = this.options.canvas.getBoundingClientRect();
    this.pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
    this.raycaster.setFromCamera(this.pointer, this.camera);
  }

  private pickHandle(event: PointerEvent): THREE.Object3D | null {
    if (!this.model.handles.length) return null;
    this.setPointer(event);
    const live = this.model.handles.filter((h) => {
      let node: THREE.Object3D | null = h;
      while (node) {
        if (!node.visible) return false;
        node = node.parent;
      }
      return true;
    });
    return this.raycaster.intersectObjects(live, true)[0]?.object ?? null;
  }

  private pageSide(event: PointerEvent): "prev" | "next" | null {
    if (!this.opened) return null;
    this.setPointer(event);
    const hit = this.raycaster.intersectObjects([this.leftPlane, this.rightPlane], false)[0];
    if (!hit) return null;
    const stop = this.target.stop;
    if (hit.object === this.leftPlane) return stop > 0 ? "prev" : null;
    return stop < this.options.stops.length - 1 ? "next" : null;
  }

  private overCover(event: PointerEvent): boolean {
    if (!this.closedView || !this.options.onCoverClick) return false;
    this.setPointer(event);
    return this.raycaster.intersectObject(this.bookRoot, true).length > 0;
  }

  private linkAt(event: PointerEvent): PageLink | null {
    const links = this.options.stops[this.applied.stop]?.links;
    if (!links?.length || !this.opened || this.running || this.leaf.front.visible) return null;
    this.setPointer(event);
    const hit = this.raycaster.intersectObjects([this.leftPlane, this.rightPlane], false)[0];
    if (!hit?.uv) return null;
    const side = hit.object === this.leftPlane ? "left" : "right";
    const u = hit.uv.x;
    const v = 1 - hit.uv.y;
    return links.find((l) => l.side === side && u >= l.box[0] && u <= l.box[2] && v >= l.box[1] && v <= l.box[3]) ?? null;
  }

  private pointOnPage(event: PointerEvent): number | null {
    this.setPointer(event);
    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -this.bookRoot.position.y - 0.03);
    const hit = this.raycaster.ray.intersectPlane(plane, new THREE.Vector3());
    return hit ? hit.x : null;
  }

  private updateLens(event: PointerEvent): void {
    const stop = this.options.stops[this.applied.stop];
    if (!stop?.page || this.narrow || this.camStartTime >= 0 || this.leaf.front.visible) {
      this.options.onLens(null);
      return;
    }
    this.setPointer(event);
    const hit = this.raycaster.intersectObject(this.leftPlane)[0];
    if (!hit?.uv) {
      this.options.onLens(null);
      return;
    }
    const rect = this.options.canvas.getBoundingClientRect();
    this.options.onLens({ x: event.clientX - rect.left, y: event.clientY - rect.top, source: this.leftText.current, u: hit.uv.x, v: 1 - hit.uv.y });
  }

  private setVisible(visible: boolean): void {
    this.visible = visible;
    if (visible && !this.raf) {
      this.timer.update();
      this.raf = requestAnimationFrame(this.tick);
    }
  }

  private resize(): void {
    const { host } = this.options;
    this.width = Math.max(1, host.clientWidth);
    this.height = Math.max(1, host.clientHeight);
    this.narrow = this.width / this.height < 0.95;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.renderer.setPixelRatio(dpr);
    this.renderer.setSize(this.width, this.height, false);
    inkShared.resolution.value.set(this.width, this.height);
    this.camera.aspect = this.width / this.height;
    this.camera.updateProjectionMatrix();
    const beat = this.closedView ? { caption: "", frame: this.options.coverFrame ?? defaultCoverFrame } : this.beatOf(this.applied.stop, this.applied.beat);
    if (beat) {
      this.setCamGoal(beat, 0);
      this.cam = { ...this.camGoal };
      this.camStartTime = -1;
    }
    this.overlayKey = "";
    this.frozen = false;
    this.publishOverlay();
  }

  private beatOf(stop: number, beat: number): Beat | undefined {
    return this.options.stops[stop]?.beats[beat] ?? this.options.stops[stop]?.beats[0];
  }

  private solveFrame(frame: Frame) {
    const vFov = rad(this.camera.fov);
    const hFov = 2 * Math.atan(Math.tan(vFov / 2) * this.camera.aspect);
    const el = rad(frame.el);
    const projected = frame.depth * Math.sin(el) + 0.12 * Math.cos(el);
    const dist = Math.max(frame.width / (2 * Math.tan(hFov / 2)), projected / (2 * Math.tan(vFov / 2)));
    return { x: frame.center[0], y: frame.center[1], z: frame.center[2], az: frame.az, el: frame.el, dist };
  }

  private setCamGoal(beat: Beat, duration: number): void {
    const frame = this.narrow && beat.narrow ? beat.narrow : beat.frame;
    this.camGoal = this.solveFrame(frame);
    if (duration <= 0 || this.options.reduced) {
      this.cam = { ...this.camGoal };
      this.camStartTime = -1;
    } else {
      this.camFrom = { ...this.cam };
      this.camDuration = duration;
      this.camStartTime = this.time;
    }
  }

  private tween(duration: number, step: (t: number) => void): Promise<void> {
    if (duration <= 0 || this.options.reduced) {
      step(1);
      return Promise.resolve();
    }
    return new Promise((resolve) => {
      this.tweens.push({ elapsed: 0, duration, step, resolve });
    });
  }

  private setLeftAngle(t: number): void {
    this.leftHalf.rotation.z = -Math.PI * (1 - t);
  }

  showClosed(): void {
    this.setLeftAngle(0);
    this.closedView = true;
    this.camGoal = this.solveFrame(this.options.coverFrame ?? defaultCoverFrame);
    this.cam = { ...this.camGoal };
    this.camStartTime = -1;
    this.frozen = false;
    if (!this.raf) this.raf = requestAnimationFrame(this.tick);
  }

  get isOpen(): boolean {
    return this.opened;
  }

  async open(): Promise<void> {
    if (this.opened) return;
    const resting = this.closedView;
    if (!resting) this.showClosed();
    this.closedView = false;
    const first = this.options.stops[0];
    const leftNext = this.leftText.prepare((c) => this.drawLeft(c, first));
    this.leftText.blend(leftNext, 1);
    this.leftText.commit(leftNext);
    const rightNext = this.rightText.prepare((c) => this.drawRight(c, first));
    this.rightText.blend(rightNext, 1);
    this.rightText.commit(rightNext);
    const beat = this.beatOf(0, 0)!;
    if (!this.options.reduced && !resting) await this.wait(0.5);
    this.setCamGoal(beat, 2.2);
    await this.tween(2.2, (t) => this.setLeftAngle(easeInOut(t)));
    this.applied.stop = 0;
    this.applied.beat = 0;
    this.opened = true;
    if (first.exhibit !== null) await this.popUp(first.exhibit, 0);
    this.publishOverlay();
    this.kick();
  }

  async close(): Promise<void> {
    if (!this.opened) return;
    this.opened = false;
    await this.settling;
    this.overlayKey = "";
    this.frozen = false;
    this.options.onOverlay(null);
    this.options.onLens(null);
    const stop = this.options.stops[this.applied.stop];
    if (stop?.exhibit !== null && stop?.exhibit !== undefined) await this.foldDown(stop.exhibit);
    this.applied = { stop: -1, beat: 0 };
    this.target = { stop: 0, beat: 0 };
    this.setCamGoal({ caption: "", frame: this.options.coverFrame ?? defaultCoverFrame }, 2.2);
    await this.tween(2.2, (t) => this.setLeftAngle(1 - easeInOut(t)));
    this.closedView = true;
  }

  private wait(seconds: number): Promise<void> {
    return this.tween(seconds, () => undefined);
  }

  private drawLeft(canvas: HTMLCanvasElement, stop: Stop): void {
    if (stop.paint?.left) stop.paint.left(canvas, this.kit.palette, this.options.fonts);
    else drawTextPage(canvas, stop.page, this.kit.palette, this.options.fonts);
  }

  private drawRight(canvas: HTMLCanvasElement, stop: Stop): void {
    if (stop.paint?.right) {
      stop.paint.right(canvas, this.kit.palette, this.options.fonts);
      return;
    }
    if (stop.overlay) {
      canvas.getContext("2d")!.clearRect(0, 0, canvas.width, canvas.height);
      return;
    }
    drawCaptionPage(canvas, stop.figure, stop.caption, stop.instruction, stop.folio, this.kit.palette, this.options.fonts);
  }

  goTo(stop: number, beat: number): void {
    this.target = { stop, beat };
    this.kick();
  }

  private kick(): void {
    if (this.running || !this.opened) return;
    this.settling = this.settle();
  }

  private async settle(): Promise<void> {
    this.running = true;
    while (this.opened && (this.applied.stop !== this.target.stop || this.applied.beat !== this.target.beat)) {
      const { stop, beat } = this.target;
      if (stop === this.applied.stop) {
        this.applied.beat = beat;
        this.overlayKey = "";
        this.frozen = false;
        this.options.onOverlay(null);
        this.setCamGoal(this.beatOf(stop, beat)!, 1.9);
        await this.waitForCamera();
      } else {
        await this.moveToStop(stop, beat);
      }
      this.publishOverlay();
    }
    this.running = false;
  }

  private waitForCamera(): Promise<void> {
    if (this.camStartTime < 0) return Promise.resolve();
    return this.tween(this.camDuration, () => undefined);
  }

  private async moveToStop(stop: number, beat: number): Promise<void> {
    const from = this.options.stops[this.applied.stop];
    const to = this.options.stops[stop];
    this.overlayKey = "";
    this.frozen = false;
    this.options.onOverlay(null);
    const exhibitChanges = from.exhibit !== to.exhibit || from.spread !== to.spread;
    if (exhibitChanges && from.exhibit !== null) await this.foldDown(from.exhibit);
    this.setCamGoal(this.beatOf(stop, beat)!, 1.9);
    if (from.spread !== to.spread) {
      await this.turnLeaf(from, to);
    } else {
      const left = this.leftText.prepare((c) => this.drawLeft(c, to));
      const right = this.rightText.prepare((c) => this.drawRight(c, to));
      await this.tween(0.7, (t) => {
        this.leftText.blend(left, t);
        this.rightText.blend(right, t);
      });
      this.leftText.commit(left);
      this.rightText.commit(right);
    }
    this.applied.stop = stop;
    this.applied.beat = beat;
    this.model.enter?.(stop);
    if (exhibitChanges && to.exhibit !== null) await this.popUp(to.exhibit, 0.05);
    await this.waitForCamera();
  }

  private foldDown(index: number): Promise<void> {
    const exhibit = this.model.exhibits[index];
    const start = exhibit.scale.y;
    return this.tween(0.55, (t) => {
      exhibit.scale.y = Math.max(0.001, start * (1 - easeInOut(t)));
      if (t >= 1) exhibit.visible = false;
    });
  }

  private popUp(index: number, delay: number): Promise<void> {
    const exhibit = this.model.exhibits[index];
    exhibit.visible = true;
    return this.tween(0.9 + delay, (t) => {
      const local = clamp01((t * (0.9 + delay) - delay) / 0.9);
      exhibit.scale.y = Math.max(0.001, easeOutBack(local));
    });
  }

  private composeLeaf(canvas: HTMLCanvasElement, overlay: HTMLCanvasElement): void {
    const ctx = canvas.getContext("2d")!;
    ctx.fillStyle = this.kit.palette.paper;
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(overlay, 0, 0);
  }

  private async turnLeaf(from: Stop, to: Stop): Promise<void> {
    const forward = to.spread > from.spread;
    const scratch = document.createElement("canvas");
    scratch.width = this.leaf.frontCanvas.width;
    scratch.height = this.leaf.frontCanvas.height;
    const left = this.leftText.prepare((c) => this.drawLeft(c, to));
    const right = this.rightText.prepare((c) => this.drawRight(c, to));
    const zeroActive = () => {
      this.leftText.meshes[this.leftText.active].material.opacity = 0;
      this.rightText.meshes[this.rightText.active].material.opacity = 0;
    };
    if (forward) {
      this.composeLeaf(this.leaf.frontCanvas, this.rightText.current);
      this.composeLeaf(this.leaf.backCanvas, this.leftText.canvases[left]);
      this.rightText.blend(right, 1);
      this.rightText.commit(right);
    } else {
      this.composeLeaf(this.leaf.frontCanvas, this.rightText.canvases[right]);
      this.composeLeaf(this.leaf.backCanvas, this.leftText.current);
      this.leftText.blend(left, 1);
      this.leftText.commit(left);
    }
    this.leaf.frontMap.needsUpdate = true;
    this.leaf.backMap.needsUpdate = true;
    this.leaf.front.visible = true;
    this.leaf.back.visible = true;
    this.setLeaf(forward ? 0 : 1);
    if (!forward) zeroActive();
    this.leftText.meshes[this.leftText.active].material.opacity = forward ? this.leftText.meshes[this.leftText.active].material.opacity : 1;
    await this.tween(1.8, (t) => this.setLeaf(forward ? easeInOut(t) : 1 - easeInOut(t)));
    this.leaf.front.visible = false;
    this.leaf.back.visible = false;
    if (forward) {
      this.leftText.blend(left, 1);
      this.leftText.commit(left);
    } else {
      this.rightText.blend(right, 1);
      this.rightText.commit(right);
    }
  }

  private setLeaf(t: number): void {
    const position = this.leaf.geometry.attributes.position as THREE.BufferAttribute;
    const columns = 41;
    const ds = PAGE_W / 40;
    const xs: number[] = [];
    const ys: number[] = [];
    let x = 0;
    let y = 0.002;
    for (let j = 0; j < columns; j++) {
      xs.push(x);
      ys.push(y);
      const phi = Math.PI * clamp01(t * 1.4 - 0.4 * (j / 40));
      x += Math.cos(phi) * ds;
      y += Math.sin(phi) * ds;
    }
    for (let i = 0; i < position.count; i++) {
      const j = i % columns;
      position.setX(i, xs[j]);
      position.setY(i, ys[j] + 0.002);
    }
    position.needsUpdate = true;
    this.leaf.geometry.computeVertexNormals();
    this.leaf.geometry.computeBoundingSphere();
  }

  private publishOverlay(): void {
    const stop = this.options.stops[this.applied.stop];
    if (!stop?.overlay || this.camStartTime >= 0) return;
    const key = `${stop.overlay}-${this.width}-${this.height}`;
    if (key === this.overlayKey) return;
    this.overlayKey = key;
    this.applyCamera();
    const y = this.bookRoot.position.y;
    const rectOf = (x0: number, x1: number): Rect => {
      const corners = [
        new THREE.Vector3(x0, y, -PAGE_D / 2 + 0.07),
        new THREE.Vector3(x1, y, -PAGE_D / 2 + 0.07),
        new THREE.Vector3(x0, y, PAGE_D / 2 - 0.07),
        new THREE.Vector3(x1, y, PAGE_D / 2 - 0.07),
      ].map((v) => v.project(this.camera));
      const px = corners.map((v) => (v.x * 0.5 + 0.5) * this.width);
      const py = corners.map((v) => (-v.y * 0.5 + 0.5) * this.height);
      const left = Math.min(...px);
      const top = Math.min(...py);
      return { left, top, width: Math.max(...px) - left, height: Math.max(...py) - top };
    };
    this.options.onOverlay({ kind: stop.overlay, rect: rectOf(0.08, PAGE_W - 0.07) });
    this.frozenSoon = true;
  }

  private frozenSoon = false;

  private applyCamera(): void {
    const reduced = this.options.reduced;
    const sway = reduced || this.options.stops[this.applied.stop]?.overlay ? 0 : Math.sin(this.time * 0.22) * 0.7;
    const az = rad(this.cam.az + this.orbit.az + sway);
    const el = rad(Math.min(89.6, this.cam.el + this.orbit.el));
    const target = new THREE.Vector3(this.cam.x, this.cam.y + this.bookRoot.position.y, this.cam.z);
    this.camera.position.set(target.x + this.cam.dist * Math.cos(el) * Math.sin(az), target.y + this.cam.dist * Math.sin(el), target.z + this.cam.dist * Math.cos(el) * Math.cos(az));
    this.camera.up.set(0, 1, 0);
    this.camera.lookAt(target);
    this.camera.updateMatrixWorld();
  }

  private tick = (): void => {
    this.raf = 0;
    if (!this.visible) return;
    const started = performance.now();
    this.timer.update();
    const dt = Math.min(this.timer.getDelta(), 0.1);
    this.time += dt;

    for (const tween of [...this.tweens]) {
      tween.elapsed += dt;
      const t = clamp01(tween.elapsed / tween.duration);
      tween.step(t);
      if (t >= 1) {
        this.tweens.splice(this.tweens.indexOf(tween), 1);
        tween.resolve();
      }
    }

    if (this.camStartTime >= 0) {
      const t = clamp01((this.time - this.camStartTime) / this.camDuration);
      const e = easeInOut(t);
      const lerp = (a: number, b: number) => a + (b - a) * e;
      this.cam = {
        x: lerp(this.camFrom.x, this.camGoal.x),
        y: lerp(this.camFrom.y, this.camGoal.y),
        z: lerp(this.camFrom.z, this.camGoal.z),
        az: lerp(this.camFrom.az, this.camGoal.az),
        el: lerp(this.camFrom.el, this.camGoal.el),
        dist: Math.exp(lerp(Math.log(this.camFrom.dist), Math.log(this.camGoal.dist))),
      };
      if (t >= 1) {
        this.camStartTime = -1;
        this.publishOverlay();
      }
    }

    if (!this.drag) {
      const settle = 1 - Math.exp(-dt * 2.4);
      this.orbit.az += (0 - this.orbit.az) * settle;
      this.orbit.el += (0 - this.orbit.el) * settle;
    }

    const stop = this.options.stops[this.applied.stop];
    if (stop?.exhibit !== null && stop?.exhibit !== undefined && this.opened) this.model.update(dt, stop.exhibit, this.applied.stop, this.options.reduced);

    this.applyCamera();

    if (!this.frozen) {
      this.renderer.render(this.scene, this.camera);
      if (this.frozenSoon) {
        this.frozen = true;
        this.frozenSoon = false;
      }
    }
    this.options.onFrameTime?.(performance.now() - started);
    this.raf = requestAnimationFrame(this.tick);
  };

  dispose(): void {
    cancelAnimationFrame(this.raf);
    this.raf = 0;
    this.visible = false;
    for (const d of this.disposers) d();
    this.scene.traverse((object) => {
      const mesh = object as THREE.Mesh;
      mesh.geometry?.dispose?.();
      const materials = Array.isArray(mesh.material) ? mesh.material : mesh.material ? [mesh.material] : [];
      for (const material of materials) {
        const map = (material as THREE.MeshBasicMaterial).map;
        map?.dispose();
        material.dispose();
      }
    });
    this.renderer.dispose();
  }
}
