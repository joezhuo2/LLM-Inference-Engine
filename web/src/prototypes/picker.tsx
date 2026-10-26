"use client";

import { Fragment, type ReactNode, useEffect, useLayoutEffect, useRef, useState, useSyncExternalStore } from "react";
import "./picker.css";

export type Variant = { name: string; render: () => ReactNode };

const urlEvent = "proto-picker-url";

function subscribe(onChange: () => void) {
  window.addEventListener(urlEvent, onChange);
  window.addEventListener("popstate", onChange);
  return () => {
    window.removeEventListener(urlEvent, onChange);
    window.removeEventListener("popstate", onChange);
  };
}

function readIndex(count: number) {
  const v = parseInt(new URLSearchParams(window.location.search).get("v") ?? "", 10);
  return v >= 1 && v <= count ? v - 1 : 0;
}

export function Picker({
  variants,
  replay = false,
  position,
}: {
  variants: Variant[];
  replay?: boolean;
  position?: "top";
}) {
  const current = useSyncExternalStore(
    subscribe,
    () => readIndex(variants.length),
    () => 0,
  );
  const [mountKey, setMountKey] = useState(0);
  const pickerRef = useRef<HTMLElement>(null);
  const highlightRef = useRef<HTMLSpanElement>(null);
  const itemRefs = useRef<(HTMLButtonElement | null)[]>([]);

  const setActive = (i: number) => {
    if (i < 0 || i >= variants.length) return;
    const url = new URL(window.location.href);
    url.searchParams.set("v", String(i + 1));
    window.history.replaceState(null, "", url);
    window.dispatchEvent(new Event(urlEvent));
    setMountKey((k) => k + 1);
  };
  const remount = () => setMountKey((k) => k + 1);

  useLayoutEffect(() => {
    const move = () => {
      const el = itemRefs.current[current];
      const highlight = highlightRef.current;
      if (!el || !highlight) return;
      highlight.style.width = `${el.offsetWidth}px`;
      highlight.style.transform = `translateX(${el.offsetLeft}px)`;
    };
    move();
    window.addEventListener("resize", move);
    return () => window.removeEventListener("resize", move);
  }, [current]);

  useEffect(() => {
    const id = requestAnimationFrame(() =>
      requestAnimationFrame(() => pickerRef.current?.setAttribute("data-ready", "")),
    );
    return () => cancelAnimationFrame(id);
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && (/^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName) || target.isContentEditable)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const num = parseInt(e.key, 10);
      const now = readIndex(variants.length);
      if (num >= 1 && num <= variants.length) setActive(num - 1);
      else if (e.key === "ArrowRight") setActive((now + 1) % variants.length);
      else if (e.key === "ArrowLeft") setActive((now - 1 + variants.length) % variants.length);
      else if (e.key === "r" || e.key === "R") remount();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  });

  return (
    <>
      <Fragment key={`${current}-${mountKey}`}>{variants[current].render()}</Fragment>
      <nav className="proto-picker" aria-label="Prototype variants" ref={pickerRef} data-position={position}>
        <span className="proto-picker-highlight" aria-hidden="true" ref={highlightRef} />
        {variants.map((v, i) => (
          <button
            key={v.name}
            type="button"
            ref={(el) => {
              itemRefs.current[i] = el;
            }}
            className="proto-picker-item"
            data-active={i === current ? "" : undefined}
            aria-current={i === current ? "true" : undefined}
            onClick={() => setActive(i)}
          >
            {v.name}
          </button>
        ))}
        {replay && (
          <>
            <span className="proto-picker-divider" aria-hidden="true" />
            <button
              type="button"
              className="proto-picker-item proto-picker-replay"
              aria-label="Replay animation (R)"
              onClick={remount}
            >
              ↻
            </button>
          </>
        )}
      </nav>
    </>
  );
}
