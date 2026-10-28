import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { ImageResponse } from "next/og";
import { environment } from "@/lib/data";

export const ogSize = { width: 1200, height: 630 };

export function ogImage(name: string, alt: string) {
  return { url: `/og/${name}.png`, ...ogSize, alt, type: "image/png" };
}

const fontDir = join(process.cwd(), "node_modules/@fontsource/jetbrains-mono/files");

const glyphs = "01{}<>=+*#/;:%";

function rainColumn(seed: number): string[] {
  let x = seed * 2654435761;
  return Array.from({ length: 14 }, () => {
    x = (x * 1103515245 + 12345) % 2147483648;
    return glyphs[x % glyphs.length];
  });
}

export async function ogCard({ eyebrow, title, text, stats }: { eyebrow: string; title: string; text: string; stats: { label: string; value: string }[] }) {
  const [regular, bold] = await Promise.all([
    readFile(join(fontDir, "jetbrains-mono-latin-400-normal.woff")),
    readFile(join(fontDir, "jetbrains-mono-latin-700-normal.woff")),
  ]);
  const footer = `measured on ${String(environment.gpu ?? "unknown GPU")} · commit ${String(environment.commit ?? "unknown")}`;
  return new ImageResponse(
    (
      <div style={{ width: "100%", height: "100%", display: "flex", background: "#0a0b0a", color: "#e6e8e6", fontFamily: "JetBrains Mono", position: "relative" }}>
        <div style={{ position: "absolute", top: 0, right: 48, display: "flex", gap: 28 }}>
          {[3, 7, 1, 5, 9].map((seed, i) => (
            <div key={seed} style={{ display: "flex", flexDirection: "column", fontSize: 26, lineHeight: 1.6, marginTop: (i * 97) % 160 - 60 }}>
              {rainColumn(seed).map((g, j, col) => (
                <span key={j} style={{ color: j === col.length - 1 ? "#4ade80" : `rgba(74, 222, 128, ${0.04 + (j / col.length) * 0.22})` }}>
                  {g}
                </span>
              ))}
            </div>
          ))}
        </div>
        <div style={{ position: "absolute", inset: 0, display: "flex", background: "linear-gradient(90deg, #0a0b0a 55%, rgba(10, 11, 10, 0.6) 100%)" }} />
        <div style={{ position: "relative", display: "flex", flexDirection: "column", justifyContent: "space-between", padding: "64px 72px", width: "100%" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            <div style={{ display: "flex", fontSize: 24, color: "#8b918b" }}>
              <span style={{ color: "#4ade80" }}>llmie</span>
              <span style={{ marginLeft: 16 }}>/ {eyebrow}</span>
            </div>
            <div style={{ display: "flex", fontSize: Math.min(76, Math.floor(1400 / title.length)), fontWeight: 700, lineHeight: 1.1, maxWidth: 900 }}>{title}</div>
            <div style={{ display: "block", lineClamp: 4, fontSize: 26, lineHeight: 1.45, color: "#8b918b", maxWidth: 820 }}>{text}</div>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 28 }}>
            <div style={{ display: "flex", gap: 56 }}>
              {stats
                .filter((s) => s.value !== "–")
                .map((s) => (
                  <div key={s.label} style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                    <span style={{ fontSize: 44, color: "#4ade80" }}>{s.value}</span>
                    <span style={{ fontSize: 18, color: "#8b918b" }}>{s.label}</span>
                  </div>
                ))}
            </div>
            <div style={{ display: "flex", fontSize: 16, color: "#6b7280" }}>{footer}</div>
          </div>
        </div>
      </div>
    ),
    {
      ...ogSize,
      fonts: [
        { name: "JetBrains Mono", data: regular, weight: 400, style: "normal" },
        { name: "JetBrains Mono", data: bold, weight: 700, style: "normal" },
      ],
    },
  );
}
