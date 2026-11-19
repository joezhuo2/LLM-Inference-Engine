"use client";

import Link from "next/link";
import { useEffect, useId, useRef } from "react";
import { Glossed } from "@/components/glossed";
import { KernelSearch } from "@/components/kernels/kernel-search";
import { atlasIntro, atlasStories } from "@/content/kernels-atlas";
import { shortPathName } from "@/content/short-path";
import { readPalette } from "@/lib/canvas";
import { usePrefersReducedMotion } from "@/lib/reduced-motion";
import s from "./atlas.module.css";
import { drawMachine, machineAge, machineStages, setMachineInk } from "./machine";

const padNumber = (n: number) => String(n).padStart(2, "0");
const total = padNumber(atlasStories.length);
const stages = machineStages(atlasStories.map((story) => story.slug));
const ramp = (v: number, a: number, b: number) => Math.min(1, Math.max(0, (v - a) / (b - a)));
const noScript = `.${s.text}{position:static;transform:none;opacity:1;visibility:visible;width:auto;text-align:left}.${s.atlas}{max-width:64rem;margin:auto;padding:2.5rem 16px}.${s.chapter}{height:auto;margin-bottom:3rem}.${s.chapter}:nth-child(n) .${s.text}{text-align:left}.${s.chapter}:nth-child(n) p{margin-left:0}.${s.scene},.${s.meter},.${s.hint},.${s.openIndex}{display:none}@layer base{div[hidden][id^="S:"]{display:block!important}}main [role=status][aria-label=loading]{display:none}footer{order:1}`;

function Extras() {
  return (
    <div className={s.extras}>
      <div className={s.extraLinks}>
        <Link href="/every-byte/start">read them in Every Byte, the 3D book</Link>
        <Link href="/kernels/short-path">{`if you have an hour: the ${shortPathName}`}</Link>
        <Link href="/kernels/glossary" transitionTypes={["nav-forward"]}>glossary: every term the chapters explain</Link>
      </div>
      <KernelSearch />
    </div>
  );
}

export function Atlas() {
  const reducedMotion = usePrefersReducedMotion();
  const titleId = useId();
  const chaptersRef = useRef<HTMLDivElement>(null);
  const dialogRef = useRef<HTMLDialogElement>(null);
  const machineRef = useRef<HTMLCanvasElement>(null);
  const meterRef = useRef<HTMLSpanElement>(null);
  const hintRef = useRef<HTMLDivElement>(null);
  const openRef = useRef<HTMLButtonElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const entriesRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const chapters = chaptersRef.current!, dialog = dialogRef.current!, machine = machineRef.current!, meter = meterRef.current!, hint = hintRef.current!;
    const openIndex = openRef.current!, closeIndex = closeRef.current!, indexEntries = entriesRef.current!;
    const reduced = { matches: reducedMotion };
    const sections = [...chapters.children] as HTMLElement[];
    const captions = sections.map((section) => section.firstElementChild as HTMLElement);
    for (let n = 0; n < sections.length; n++) {
      captions[n].style.visibility = "hidden";
      sections[n].inert = true;
    }
    setMachineInk(readPalette());
    let closing = false;
    function showIndex() {
      stopGlide();
      if (dialog.open) return;
      dialog.showModal();
      document.body.classList.add(s.indexOpen);
    }
    function hideIndex(onClosed?: () => void) {
      if (!dialog.open || closing) return;
      stopGlide();
      closing = true;
      dialog.classList.add(s.closing);
      const finish = () => {
        dialog.removeEventListener("animationend", ended);
        clearTimeout(timer);
        dialog.close();
        dialog.classList.remove(s.closing);
        document.body.classList.remove(s.indexOpen);
        closing = false;
        if (onClosed) onClosed();
      };
      const ended = (event: AnimationEvent) => {
        if (event.target === dialog && !event.pseudoElement) finish();
      };
      const timer = setTimeout(finish, reduced.matches ? 0 : 240);
      dialog.addEventListener("animationend", ended);
    }
    const onOpen = () => showIndex();
    const onClose = () => hideIndex();
    const onCancel = (event: Event) => {
      event.preventDefault();
      hideIndex();
    };
    const onEntry = (event: MouseEvent) => {
      const entry = (event.target as Element).closest("a");
      if (!entry || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      hideIndex(() => travelTo(entry.hash));
    };
    const onDialogClick = (e: MouseEvent) => {
      const r = dialog.getBoundingClientRect();
      if (e.target === dialog && (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom)) hideIndex();
    };
    openIndex.addEventListener("click", onOpen);
    closeIndex.addEventListener("click", onClose);
    dialog.addEventListener("cancel", onCancel);
    indexEntries.addEventListener("click", onEntry);
    dialog.addEventListener("click", onDialogClick);

    let glideFrame = 0, targetY = 0, lastTime = 0, expectedY = 0, wheelPosition = 0;
    let trip: { side: number } | null = null;
    let scrollPane: HTMLDialogElement | null = null;
    function stopGlide() {
      cancelAnimationFrame(glideFrame);
      glideFrame = 0;
      trip = null;
      scrollPane = null;
    }
    function travelTo(hash: string) {
      stopGlide();
      const section = document.getElementById(hash.slice(1));
      if (!section) return;
      const from = scrollY;
      const to = Math.min(document.documentElement.scrollHeight - innerHeight, from + section.getBoundingClientRect().top);
      const distance = Math.abs(to - from);
      if (location.hash !== hash) history.pushState(history.state, "", hash);
      function arrive() {
        stopGlide();
        scrollTo(0, to);
        render();
        const destination = section!.querySelector<HTMLElement>("h2 a, h1")!;
        if (!destination.hasAttribute("href")) destination.tabIndex = -1;
        destination.focus({ preventScroll: true });
      }
      if (reduced.matches || distance <= 1) {
        arrive();
        return;
      }
      const duration = Math.min(4200, 600 + 700 * Math.sqrt(distance / step));
      const startSide = Math.cos((Math.PI * (from - origin)) / step);
      const endSide = Math.cos((Math.PI * (to - origin)) / step);
      trip = { side: startSide };
      const started = performance.now();
      expectedY = from;
      function advance(now: number) {
        const t = Math.min(1, Math.max(0, (now - started) / duration));
        if (t === 1) {
          arrive();
          return;
        }
        const eased = t * t * (3 - 2 * t);
        trip!.side = startSide + (endSide - startSide) * eased;
        scrollTo(0, from + (to - from) * eased);
        expectedY = scrollY;
        glideFrame = requestAnimationFrame(advance);
      }
      glideFrame = requestAnimationFrame(advance);
    }
    function glide(time: number) {
      const dt = Math.min(64, Math.max(0, time - lastTime));
      lastTime = time;
      const remaining = targetY - wheelPosition;
      if (Math.abs(remaining) < 0.6) {
        (scrollPane || window).scrollTo(0, targetY);
        stopGlide();
        return;
      }
      wheelPosition += remaining * (1 - Math.exp(-dt / (scrollPane ? 155 : 290)));
      (scrollPane || window).scrollTo(0, wheelPosition);
      expectedY = scrollPane ? scrollPane.scrollTop : scrollY;
      glideFrame = requestAnimationFrame(glide);
    }
    const onWheel = (event: WheelEvent) => {
      if (reduced.matches || event.ctrlKey || event.shiftKey || !event.cancelable || Math.abs(event.deltaX) > Math.abs(event.deltaY)) return;
      const pane = dialog.open ? dialog : null;
      if (closing) return event.preventDefault();
      if (pane) {
        const bounds = pane.getBoundingClientRect();
        if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) return event.preventDefault();
      }
      event.preventDefault();
      if (trip || pane !== scrollPane) stopGlide();
      scrollPane = pane;
      const height = pane ? pane.clientHeight : innerHeight;
      const position = pane ? pane.scrollTop : scrollY;
      const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? height : 1;
      const delta = event.deltaY * unit;
      const maxY = (pane || document.documentElement).scrollHeight - height;
      if (!glideFrame || Math.sign(delta) !== Math.sign(targetY - position)) {
        wheelPosition = position;
        targetY = position;
      }
      targetY = Math.max(0, Math.min(maxY, targetY + delta * (pane ? 1 : 1.45)));
      if (!glideFrame && Math.abs(targetY - position) >= 1) {
        lastTime = performance.now();
        expectedY = position;
        glideFrame = requestAnimationFrame(glide);
      }
    };
    const onWindowScroll = () => {
      if (!scrollPane && glideFrame && Math.abs(scrollY - expectedY) > 1) stopGlide();
    };
    const onPaneScroll = () => {
      if (scrollPane && glideFrame && Math.abs(dialog.scrollTop - expectedY) > 1) stopGlide();
    };
    addEventListener("wheel", onWheel, { passive: false, capture: true });
    const stoppers = ["pointerdown", "touchstart", "keydown", "hashchange", "pagehide"];
    for (const name of stoppers) addEventListener(name, stopGlide, { passive: true });
    addEventListener("scroll", onWindowScroll, { passive: true });
    dialog.addEventListener("scroll", onPaneScroll, { passive: true });

    let frame = 0, lastSample = -1, active = -1, step = 1, origin = 0;
    function render() {
      frame = 0;
      const last = captions.length - 1;
      const progress = Math.min(last, Math.max(0, (scrollY - origin) / step));
      const index = Math.round(progress), caption = captions[index];
      if (index !== active) {
        if (active >= 0) {
          captions[active].style.visibility = "hidden";
          sections[active].inert = true;
        }
        active = index;
        caption.style.visibility = "visible";
        sections[index].inert = false;
      }
      caption.style.opacity = String(reduced.matches ? 1 : 1 - ramp(Math.abs(progress - index), 0.3, 0.5) * 0.96);
      hint.hidden = progress > 0.3;
      meter.style.transform = "scaleX(" + progress / last + ")";
      const sample = reduced.matches ? index : progress;
      if (sample !== lastSample) {
        drawMachine(machine, machineAge(stages, sample - 1), reduced.matches, trip?.side ?? Math.cos(sample * Math.PI));
        lastSample = sample;
      }
    }
    function queue() {
      if (!frame) frame = requestAnimationFrame(render);
    }
    function measure() {
      stopGlide();
      step = sections[0].getBoundingClientRect().height;
      origin = sections[0].getBoundingClientRect().top + scrollY;
      lastSample = -1;
      queue();
    }
    const onTheme = new MutationObserver(() => {
      setMachineInk(readPalette());
      lastSample = -1;
      queue();
    });
    onTheme.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    addEventListener("scroll", queue, { passive: true });
    const resizer = new ResizeObserver(measure);
    resizer.observe(machine);
    addEventListener("resize", measure);
    let live = true;
    document.fonts.ready.then(() => live && measure());
    measure();
    return () => {
      live = false;
      stopGlide();
      cancelAnimationFrame(frame);
      onTheme.disconnect();
      resizer.disconnect();
      openIndex.removeEventListener("click", onOpen);
      closeIndex.removeEventListener("click", onClose);
      dialog.removeEventListener("cancel", onCancel);
      indexEntries.removeEventListener("click", onEntry);
      dialog.removeEventListener("click", onDialogClick);
      dialog.removeEventListener("scroll", onPaneScroll);
      removeEventListener("wheel", onWheel, { capture: true });
      for (const name of stoppers) removeEventListener(name, stopGlide);
      removeEventListener("scroll", onWindowScroll);
      removeEventListener("scroll", queue);
      removeEventListener("resize", measure);
      document.body.classList.remove(s.indexOpen);
    };
  }, [reducedMotion]);

  return (
    <div className={s.atlas}>
      <noscript>
        <style>{noScript}</style>
      </noscript>
      <div className={s.scene} aria-hidden="true">
        <canvas ref={machineRef} className={s.machine} />
      </div>
      <button ref={openRef} type="button" className={s.openIndex} aria-haspopup="dialog">
        chapter index <span aria-hidden="true">☰</span>
      </button>
      <div ref={chaptersRef} className={s.chapters} role="region" aria-label="Inference stories from transistor to server">
        <section id="kernels" className={s.chapter}>
          <div className={s.text}>
            <h1>kernels</h1>
            <p>{atlasIntro}</p>
            <Extras />
          </div>
        </section>
        {atlasStories.map((story, n) => {
          const num = padNumber(n + 1);
          const words = story.title.split(" "), lastWord = words.pop();
          return (
            <article key={story.slug} id={`chapter-${num}`} className={s.chapter}>
              <div className={s.text}>
                <div className={s.chapterMeta}>
                  {`${num}/${total}`}
                  {story.dates ? <span className={s.dates}>{story.dates}</span> : null}
                </div>
                <h2>
                  <Link href={story.href} transitionTypes={["nav-forward"]}>
                    {words.join(" ") + (words.length ? " " : "")}
                    <span className={s.lastWord}>
                      {lastWord}
                      <span className={s.out} aria-hidden="true">↗</span>
                    </span>
                  </Link>
                </h2>
                <Glossed>
                  <p>{story.line}</p>
                </Glossed>
                {story.abandoned ? <small className={s.pathKey}>abandoned experiment</small> : null}
              </div>
            </article>
          );
        })}
      </div>
      <div className={s.meter} aria-hidden="true">
        <span ref={meterRef} />
      </div>
      <div ref={hintRef} className={s.hint} aria-hidden="true">
        ↓ Scroll to assemble
      </div>
      <dialog ref={dialogRef} className={s.index} aria-labelledby={titleId}>
        <div className={s.dialogTop}>
          <h2 id={titleId}>chapter index</h2>
          <button ref={closeRef} type="button" aria-label="Close chapter index">
            ×
          </button>
        </div>
        <Extras />
        <div ref={entriesRef} className={s.entries}>
          {atlasStories.map((story, n) => (
            <a key={story.slug} className={s.indexEntry} href={`#chapter-${padNumber(n + 1)}`}>
              <span>{padNumber(n + 1)}</span>
              <span className={s.indexTitle}>{story.title}</span>
              <i aria-hidden="true">↓</i>
            </a>
          ))}
        </div>
      </dialog>
    </div>
  );
}
