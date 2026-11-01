"use client";

import clsx from "clsx";
import { useEffect, useRef, useState } from "react";
import type { Seq } from "@/lib/dashboard/model";
import { type Feed, SLOTS, slotColor } from "./feed";

const TAIL = 120;

type LaneRefs = { chip: HTMLSpanElement | null; meta: HTMLSpanElement | null; text: HTMLElement | null };

function status(seq: Seq): string {
  if (seq.running) return `running, ${seq.outputs.length} tokens`;
  if (seq.finished >= 0) return `finished at step ${seq.finished}, ${seq.outputs.length} tokens`;
  return `waiting, ${seq.outputs.length} tokens`;
}

function tail(text: string): string {
  const flat = text.replaceAll("\n", " ⏎ ");
  return flat.length > TAIL ? flat.slice(-TAIL) : flat;
}

export function Requests({ feed, total }: { feed: Feed; total: number }) {
  const [selected, setSelected] = useState<number | null>(null);
  const [lanes, setLanes] = useState<(number | null)[]>(() => Array.from({ length: SLOTS }, () => null));
  const laneIds = useRef<(number | null)[]>(Array.from({ length: SLOTS }, () => null));
  const refs = useRef<LaneRefs[]>(Array.from({ length: SLOTS }, () => ({ chip: null, meta: null, text: null })));
  const footer = useRef<HTMLParagraphElement>(null);
  const full = useRef<HTMLDivElement>(null);
  const fullText = useRef("");
  const pickedRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let raf = 0;
    const paint = () => {
      raf = 0;
      const model = feed.model;
      let changed = false;
      for (let slot = 0; slot < SLOTS; slot++) {
        const id = model.lastOccupant(slot);
        if (id !== laneIds.current[slot]) {
          laneIds.current[slot] = id;
          changed = true;
        }
        const seq = id === null ? undefined : model.seqs.get(id);
        const r = refs.current[slot];
        if (r.meta) {
          r.meta.textContent = seq ? String(seq.outputs.length) : "";
          r.meta.title = seq ? status(seq) : "";
        }
        if (r.chip) r.chip.style.background = seq?.running ? slotColor(slot) : "transparent";
        if (r.text) r.text.textContent = seq ? tail(seq.text) : "";
      }
      if (changed) setLanes([...laneIds.current]);
      const done = [...model.seqs.values()].filter((s) => s.finished >= 0).length;
      const last = model.last;
      if (footer.current) footer.current.textContent = `${done} of ${total} requests finished · ${last?.waiting ?? 0} waiting`;
      const pick = feed.selected === null ? undefined : model.seqs.get(feed.selected);
      const text = pick ? pick.text : "";
      if (pickedRef.current) pickedRef.current.textContent = pick ? `#${pick.id}, ${status(pick)}` : "pick a request to read its whole answer; it is highlighted in the timeline and the pool";
      if (full.current && text !== fullText.current) {
        const box = full.current;
        const atEnd = box.scrollTop + box.clientHeight >= box.scrollHeight - 4;
        box.textContent = text;
        fullText.current = text;
        if (atEnd) box.scrollTop = box.scrollHeight;
      }
    };
    const kick = () => {
      if (!raf) raf = requestAnimationFrame(paint);
    };
    const unsubscribe = feed.subscribe(kick);
    kick();
    return () => {
      unsubscribe();
      cancelAnimationFrame(raf);
    };
  }, [feed, total]);

  const choose = (id: number | null) => {
    if (id === null) return;
    feed.select(id);
    setSelected(feed.selected);
  };

  return (
    <div className="flex h-full min-h-0 flex-col gap-2">
      <ul className="space-y-0.5">
        {lanes.map((id, slot) => (
          <li key={slot}>
            <button
              type="button"
              disabled={id === null}
              aria-pressed={id !== null && id === selected}
              onClick={() => choose(id)}
              className={clsx(
                "pressable flex w-full items-center gap-2 rounded border px-1.5 py-1 text-left",
                id !== null && id === selected ? "border-text/60 bg-accent-dim" : "border-transparent",
                id === null && "opacity-50",
              )}
            >
              <span
                aria-hidden
                ref={(el) => {
                  refs.current[slot].chip = el;
                }}
                className="size-2 shrink-0 rounded-sm border"
                style={{ borderColor: slotColor(slot) }}
              />
              <span className="w-7 shrink-0 text-text">{id === null ? "–" : `#${id}`}</span>
              <span
                ref={(el) => {
                  refs.current[slot].meta = el;
                }}
                className="w-12 shrink-0 text-right tabular-nums text-muted"
              />
              <span className="min-w-0 flex-1 overflow-hidden text-ellipsis whitespace-nowrap text-[11px] text-text/85 [direction:rtl]">
                <bdi
                  dir="ltr"
                  ref={(el) => {
                    refs.current[slot].text = el;
                  }}
                />
              </span>
            </button>
          </li>
        ))}
      </ul>
      <p ref={footer} className="text-[11px] text-muted" />
      <div className="min-h-0 flex-1 rounded border border-border p-2">
        <div ref={pickedRef} className="mb-1 text-[11px] text-muted" />
        <div ref={full} className="h-28 overflow-y-auto lg:h-[calc(100%-1.25rem)] whitespace-pre-wrap font-sans text-[13px] leading-relaxed text-text" />
      </div>
    </div>
  );
}
