import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { ThemeProvider } from "next-themes";
import { TopBar } from "@/components/top-bar";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const jetbrains = JetBrains_Mono({ subsets: ["latin"], variable: "--font-jetbrains" });

export const metadata: Metadata = {
  title: { default: "llm-inference-engine", template: "%s · llm-inference-engine" },
  description: "A TinyLlama inference engine written from scratch in C++ and CUDA: paged KV cache, continuous batching, CUDA Graphs.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning className={`${inter.variable} ${jetbrains.variable}`}>
      <body className="min-h-dvh bg-bg text-text">
        <ThemeProvider attribute="data-theme" defaultTheme="dark" enableSystem={false} disableTransitionOnChange>
          <TopBar />
          <main className="mx-auto w-full max-w-5xl px-4 py-10 sm:px-6">{children}</main>
        </ThemeProvider>
      </body>
    </html>
  );
}
