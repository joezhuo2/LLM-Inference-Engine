export type SmoothScrollOptions = { ease?: number; target?: HTMLElement };

export function smoothScroll({ ease = 0.08, target: element }: SmoothScrollOptions = {}) {
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return () => {};

  const read = () => (element ? element.scrollTop : scrollY);
  const write = (y: number) => (element ? (element.scrollTop = y) : scrollTo({ top: y, behavior: "instant" }));
  const limit = () => (element ? element.scrollHeight - element.clientHeight : document.documentElement.scrollHeight - innerHeight);
  const viewport = () => (element ? element.clientHeight : innerHeight);

  let target = read();
  let current = target;
  let frame = 0;

  const tick = () => {
    if (Math.abs(read() - current) > 2) {
      target = current = read();
      frame = 0;
      return;
    }
    current += (target - current) * ease;
    if (Math.abs(target - current) < 0.5) {
      current = target;
      frame = 0;
    } else {
      frame = requestAnimationFrame(tick);
    }
    write(current);
  };

  const inner = (from: EventTarget | null) => {
    for (let el = from instanceof Element ? from : null; el && el !== element; el = el.parentElement) {
      if (el.scrollHeight > el.clientHeight && /auto|scroll/.test(getComputedStyle(el).overflowY)) return true;
    }
    return false;
  };

  const onWheel = (event: WheelEvent) => {
    if (event.ctrlKey || Math.abs(event.deltaX) > Math.abs(event.deltaY)) return;
    if (event.defaultPrevented || inner(event.target)) return;
    event.preventDefault();
    const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? viewport() : 1;
    if (!frame) target = current = read();
    target = Math.max(0, Math.min(limit(), target + event.deltaY * unit));
    if (!frame) frame = requestAnimationFrame(tick);
  };

  const host: EventTarget = element ?? window;
  host.addEventListener("wheel", onWheel as EventListener, { passive: false });
  return () => {
    host.removeEventListener("wheel", onWheel as EventListener);
    cancelAnimationFrame(frame);
  };
}
