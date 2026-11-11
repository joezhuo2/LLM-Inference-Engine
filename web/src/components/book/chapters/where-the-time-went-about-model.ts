import * as THREE from "three";
import type { BookModel } from "../book-scene";
import { createCard, type Kit, washPlane } from "../ink";
import type { Fonts } from "../typeset";
import { approach, fmt, lay, liquid, makeRail, pen, picker, unitBox } from "./where-the-time-went-model";

export type TimeAboutNumbers = { tpotMs: number; span: number; busy: number; kernels: number; idlePct: number; groups: { label: string; us: number; ops: string[] }[] };

export function createTimeAboutModel(kit: Kit, fonts: Fonts, n: TimeAboutNumbers): BookModel {
  const root = new THREE.Group();
  const X0 = 0.08;
  const W = 0.84;
  const tpotUs = n.tpotMs * 1e3;
  const scale = W / tpotUs;
  const D = 0.05;
  const H = 0.035;
  const GAP = 0.003;
  const rows = [-0.42, -0.31, -0.2, -0.09];
  const sum = n.groups.reduce((a, g) => a + g.us, 0);
  const wash = washPlane(kit.palette.blue, 0.9, 0.5, 0.1);
  wash.position.set(0.5, 0.002, -0.25);
  root.add(wash);

  const title = createCard(kit, 0.86, 0.1, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    write("a step profile: when each kernel of one decode step ran on the GPU", 0.03, 0.4, 0.28, kit.palette.ink, true);
    write(`${n.kernels} kernels launched one by one, batch 1; every bar is to the same scale (measured)`, 0.03, 0.82, 0.2, kit.palette.ink);
  });
  title.group.position.set(0.5, 0, -0.59);
  root.add(title.group);

  const tag = (row: number, text: string, value: string, color: string) => {
    const card = createCard(kit, 0.84, 0.034, (ctx, w, h) => {
      const write = pen(ctx, fonts, w, h);
      write(text, 0.02, 0.72, 0.58, color);
      ctx.textAlign = "right";
      write(value, 0.98, 0.72, 0.58, color);
      ctx.textAlign = "left";
    });
    card.group.position.set(0.5, 0, rows[row] + 0.042);
    root.add(card.group);
  };
  tag(0, "time per token: what a user waits, host work between steps included", `${fmt(n.tpotMs, 2)} ms`, kit.palette.ink);
  tag(1, "span: from the start of the first kernel to the end of the last", `${fmt(n.span)} µs`, kit.palette.ink);
  tag(2, "busy: the part of the span in which a kernel is running", `${fmt(n.busy)} µs`, kit.palette.blue);
  tag(3, "time per op: the kernel time, by the operation that launched it", `${fmt(sum)} µs`, kit.palette.pink);

  const tokenBar = kit.solid(unitBox, { material: kit.glass, shadow: false });
  lay(tokenBar, X0, tpotUs * scale, H, rows[0], D);
  const spanBar = kit.solid(unitBox);
  lay(spanBar, X0, n.span * scale, H, rows[1], D);
  const busyBar = liquid(kit.blue);
  lay(busyBar, X0, n.busy * scale, H, rows[2], D);
  root.add(tokenBar, spanBar, busyBar);

  const blocks = n.groups.map(() => {
    const plain = kit.solid(unitBox);
    const lit = liquid(kit.pink);
    root.add(plain, lit);
    return { plain, lit, height: H };
  });
  const starts: number[] = [];
  n.groups.reduce((x, g) => {
    starts.push(x);
    return x + g.us * scale + GAP;
  }, X0);

  let g = 0;
  const readout = createCard(kit, 0.86, 0.12, (ctx, w, h) => {
    const write = pen(ctx, fonts, w, h);
    const group = n.groups[g];
    write(`${group.label}: ${fmt(group.us, group.us < 100 ? 1 : 0)} µs of a step, ${fmt((group.us / sum) * 100, group.us / sum < 0.02 ? 1 : 0)}% of the kernel time`, 0.03, 0.34, 0.23, kit.palette.pink, true);
    write(`ops in the profile: ${group.ops.join(", ")}`, 0.03, 0.62, 0.17, kit.palette.ink);
    write(`token minus span is host work between steps; span minus busy is launch gaps, ${n.idlePct}% of the span`, 0.03, 0.88, 0.16, kit.palette.ink);
  });
  readout.group.position.set(0.5, 0, 0.1);
  root.add(readout.group);

  const rail = makeRail(kit, fonts, root, 0.27, n.groups.map((_, i) => 0.2 + (0.6 * i) / (n.groups.length - 1)), "which operation", n.groups.map((x) => x.label).slice(0, 2).join(", ") + " …");
  const pick = picker(rail, 0, 0, 3, (i) => {
    g = i;
    readout.redraw();
  });

  return {
    exhibits: [root],
    handles: [pick.knob],
    dragStart: () => pick.start(),
    drag: (x) => pick.drag(x),
    dragEnd: () => pick.end(),
    enter: () => pick.enter(),
    update: (dt, _exhibit, _stop, reduced) => {
      pick.update(dt, reduced);
      blocks.forEach((b, i) => {
        b.height = approach(b.height, i === g ? H * 2.6 : H, dt * 1.4, reduced);
        const w = n.groups[i].us * scale;
        lay(b.plain, starts[i], w, b.height, rows[3], D);
        lay(b.lit, starts[i], w, b.height, rows[3], D);
        b.plain.visible = i !== g;
        b.lit.visible = i === g;
      });
    },
  };
}
