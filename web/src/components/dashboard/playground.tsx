"use client";

import clsx from "clsx";
import { useEffect, useRef, useState } from "react";
import { readSse } from "@/lib/dashboard/sse";
import type { Feed } from "./feed";

type Message = { role: "user" | "assistant"; content: string };

type Run = { state: "idle" } | { state: "streaming"; seq: number | null } | { state: "done"; seq: number | null; reason: string } | { state: "error"; message: string };

type Chunk = { choices?: { delta?: { content?: string }; finish_reason?: string | null }[]; engine?: { seq_id: number; step: number } };

const reasons: Record<string, string> = { stop: "it reached the end of its answer", length: "it hit the token limit" };

export function Playground({ feed, url, connected }: { feed: Feed; url: string; connected: boolean }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [draft, setDraft] = useState("");
  const [maxTokens, setMaxTokens] = useState(256);
  const [temperature, setTemperature] = useState(0.7);
  const [run, setRun] = useState<Run>({ state: "idle" });
  const abort = useRef<AbortController | null>(null);
  const log = useRef<HTMLDivElement>(null);

  useEffect(() => () => abort.current?.abort(), []);

  useEffect(() => {
    const box = log.current;
    if (box) box.scrollTop = box.scrollHeight;
  }, [messages]);

  const follow = (seq: number | null) => feed.follow(seq);

  const send = async () => {
    const text = draft.trim();
    if (!text || run.state === "streaming") return;
    const history: Message[] = [...messages, { role: "user", content: text }];
    setMessages([...history, { role: "assistant", content: "" }]);
    setDraft("");
    setRun({ state: "streaming", seq: null });
    follow(null);
    const controller = new AbortController();
    abort.current = controller;
    let tokens = 0;
    let seq: number | null = null;
    let reason = "";
    const append = (piece: string) =>
      setMessages((m) => {
        const last = m[m.length - 1];
        return [...m.slice(0, -1), { ...last, content: last.content + piece }];
      });
    let res: Response;
    try {
      res = await fetch(`${url}/v1/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: "tinyllama", messages: history, max_tokens: maxTokens, temperature, stream: true }),
        signal: controller.signal,
      });
    } catch {
      if (!controller.signal.aborted) setRun({ state: "error", message: "the request did not reach the engine (it stopped, or the browser blocked it)" });
      return;
    }
    if (!res.ok || !res.body) {
      const body = await res.text().catch(() => "");
      setRun({ state: "error", message: `the engine answered ${res.status}${body ? `: ${body.slice(0, 160)}` : ""}` });
      return;
    }
    let finished = false;
    try {
      await readSse(
        res.body,
        (e) => {
          if (e.data === "[DONE]") {
            finished = true;
            return;
          }
          const chunk = JSON.parse(e.data) as Chunk;
          if (chunk.engine && chunk.engine.seq_id !== seq) {
            seq = chunk.engine.seq_id;
            setRun({ state: "streaming", seq });
            follow(seq);
          }
          const choice = chunk.choices?.[0];
          if (choice?.delta?.content) {
            tokens++;
            append(choice.delta.content);
          }
          if (choice?.finish_reason) reason = choice.finish_reason;
        },
        controller.signal,
      );
    } catch {
      if (!controller.signal.aborted) {
        follow(null);
        setRun({ state: "error", message: `the engine went away after ${tokens} tokens` });
        return;
      }
    }
    follow(null);
    if (controller.signal.aborted) setRun({ state: "done", seq, reason: "you stopped it; closing the connection cancels the request and frees its blocks" });
    else if (!finished) setRun({ state: "error", message: `the stream ended after ${tokens} tokens without [DONE]; the engine probably stopped` });
    else setRun({ state: "done", seq, reason: reasons[reason] ?? (reason || "it finished") });
  };

  const stop = () => abort.current?.abort();

  const clear = () => {
    setMessages([]);
    setRun({ state: "idle" });
    follow(null);
  };

  const streaming = run.state === "streaming";

  return (
    <div className="flex h-full min-h-0 flex-col gap-2">
      <div ref={log} aria-live="polite" className="min-h-28 flex-1 space-y-2 overflow-y-auto rounded border border-border p-2 font-sans text-[13px] leading-relaxed">
        {messages.length === 0 && (
          <p className="font-mono text-[11px] text-muted">
            {connected ? "Ask the engine something. Your sequence is outlined in the timeline and the pool while it runs." : "Connect to an engine to chat with it."}
          </p>
        )}
        {messages.map((m, i) => (
          <p key={i} className={clsx("whitespace-pre-wrap", m.role === "user" ? "text-muted" : "text-text")}>
            <span className="font-mono text-[11px] text-muted">{m.role === "user" ? "you " : "engine "}</span>
            {m.content || (streaming && i === messages.length - 1 ? "…" : "")}
          </p>
        ))}
      </div>
      <p role="status" className={clsx("min-h-4 text-[11px]", run.state === "error" ? "text-warn" : "text-muted")}>
        {run.state === "streaming" && (run.seq === null ? "waiting for the first token" : `streaming as sequence #${run.seq}, outlined in the timeline and the pool`)}
        {run.state === "done" && `${run.seq === null ? "done" : `sequence #${run.seq} done`}: ${run.reason}`}
        {run.state === "error" && run.message}
      </p>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          send();
        }}
        className="space-y-2"
      >
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
          disabled={!connected}
          rows={2}
          placeholder="Explain the KV cache in two sentences."
          aria-label="Message to the engine"
          className="block w-full resize-none rounded border border-border bg-bg px-2 py-1.5 font-sans text-[13px] text-text placeholder:text-muted disabled:opacity-50"
        />
        <div className="flex flex-wrap items-center gap-2">
          {streaming ? (
            <button type="button" onClick={stop} className="pressable rounded border border-border px-2.5 py-1 text-warn">
              stop
            </button>
          ) : (
            <button type="submit" disabled={!connected || !draft.trim()} className="pressable rounded border border-border px-2.5 py-1 text-accent disabled:opacity-40">
              send
            </button>
          )}
          <button type="button" onClick={clear} disabled={streaming || messages.length === 0} className="pressable rounded border border-border px-2.5 py-1 text-text disabled:opacity-40">
            clear
          </button>
          <label className="flex items-center gap-1 text-muted">
            max tokens
            <input type="number" min={1} max={2048} value={maxTokens} onChange={(e) => setMaxTokens(Math.max(1, Number(e.target.value) || 1))} className="w-14 rounded border border-border bg-bg px-1 py-0.5 text-text" />
          </label>
          <label className="flex items-center gap-1 text-muted">
            temperature
            <input type="number" min={0} max={2} step={0.1} value={temperature} onChange={(e) => setTemperature(Math.max(0, Number(e.target.value) || 0))} className="w-12 rounded border border-border bg-bg px-1 py-0.5 text-text" />
          </label>
        </div>
      </form>
    </div>
  );
}
