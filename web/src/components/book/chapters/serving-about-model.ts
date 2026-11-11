import type { BookModel } from "../book-scene";
import type { Kit } from "../ink";
import type { Fonts } from "../typeset";
import { queueExhibit } from "./serving-model";

export function createServingAboutModel(kit: Kit, fonts: Fonts): BookModel {
  const queue = queueExhibit(kit, fonts, true);
  return {
    exhibits: [queue.root],
    handles: [queue.knob],
    dragStart: () => queue.setHeld(true),
    drag: (x) => queue.drag(x),
    dragEnd: () => queue.setHeld(false),
    enter: () => queue.setMode("lead"),
    update: (dt, _exhibit, _stop, reduced) => queue.update(dt, reduced),
  };
}
