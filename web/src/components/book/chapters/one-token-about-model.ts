import * as THREE from "three";
import { createCard, inkSegments, type Kit, washPlane } from "../ink";
import type { Fonts } from "../typeset";
import { type Part, readoutCard, titleCard, W, X0 } from "./segments";
import { approach, lay, liquid, makeRail, pen, picker, unitBox } from "./where-the-time-went-model";

export type Station = { label: string; where: string; takes: string; gives: string; time: string };
export type StationNumbers = { token: string; tpotMs: string; stations: Station[] };

export function createStationsExhibit(kit: Kit, fonts: Fonts, n: StationNumbers): Part {
  const root = new THREE.Group();
  const count = n.stations.length;
  const pitch = W / count;
  const SIDE = pitch * 0.72;
  const H = 0.05;
  const Z = -0.3;
  const centre = (i: number) => X0 + pitch * (i + 0.5);
  titleCard(kit, fonts, root, `the path of the token "${n.token}": seven stations, two on the host`, `the blue ground is the GPU; one decoded token takes ${n.tpotMs} ms (measured)`);

  const gpu = n.stations.map((s, i) => (s.where === "GPU" ? i : -1)).filter((i) => i >= 0);
  const first = gpu[0] ?? 0;
  const last = gpu[gpu.length - 1] ?? 0;
  const wash = washPlane(kit.palette.blue, (last - first + 1) * pitch + 0.08, 0.36, 0.3);
  wash.position.set((centre(first) + centre(last)) / 2, 0.002, Z);
  root.add(wash);

  const boxes = n.stations.map((s, i) => {
    const plain = kit.solid(unitBox);
    const lit = liquid(kit.pink);
    root.add(plain, lit);
    const name = createCard(kit, pitch * 0.98, 0.044, (ctx, w, h) => {
      const write = pen(ctx, fonts, w, h);
      ctx.textAlign = "center";
      write(s.label, 0.5, 0.7, 0.56, s.where === "GPU" ? kit.palette.blue : kit.palette.ink);
      ctx.textAlign = "left";
    });
    name.group.position.set(centre(i), 0, Z + 0.1);
    root.add(name.group);
    return { plain, lit, height: H };
  });

  const path: number[] = [];
  for (let i = 0; i + 1 < count; i++) path.push(centre(i) + SIDE / 2, 0.02, Z, centre(i + 1) - SIDE / 2, 0.02, Z);
  root.add(inkSegments(path, kit.line));
  const token = new THREE.Mesh(new THREE.BoxGeometry(0.026, 0.026, 0.026), kit.pink);
  token.castShadow = true;
  token.position.set(centre(0), H * 2 + 0.04, Z);
  root.add(token);

  let chosen = 0;
  const readout = readoutCard(kit, fonts, root, 0.1, () => {
    const s = n.stations[chosen];
    return [`${chosen + 1}. ${s.label}, on the ${s.where}: ${s.time}`, `takes in: ${s.takes}`, `gives: ${s.gives}`];
  });
  const rail = makeRail(kit, fonts, root, 0.27, n.stations.map((_, i) => centre(i)), "which station", "tokenizer to text");
  const pick = picker(rail, 0, 0, 3, (i) => {
    chosen = i;
    readout.redraw();
  });

  return {
    root,
    pick,
    update: (dt, reduced) => {
      pick.update(dt, reduced);
      boxes.forEach((b, i) => {
        b.height = approach(b.height, i === chosen ? H * 2 : H, dt * 1.4, reduced);
        lay(b.plain, centre(i) - SIDE / 2, SIDE, b.height, Z, SIDE);
        lay(b.lit, centre(i) - SIDE / 2, SIDE, b.height, Z, SIDE);
        b.plain.visible = i !== chosen;
        b.lit.visible = i === chosen;
      });
      token.position.x = approach(token.position.x, centre(chosen), dt * 0.9, reduced);
      token.rotation.y += reduced ? 0 : dt * 0.6;
    },
  };
}
