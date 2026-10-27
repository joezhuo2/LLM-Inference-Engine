import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { ThemeProvider } from "next-themes";
import { Footer } from "@/components/footer";
import { IntroLoader } from "@/components/intro-loader";
import { ScrollToTop } from "@/components/scroll-to-top";
import { TopBar } from "@/components/top-bar";
import { ogImage } from "@/lib/og";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const jetbrains = JetBrains_Mono({ subsets: ["latin"], variable: "--font-jetbrains" });

const siteUrl = process.env.VERCEL_PROJECT_PRODUCTION_URL ? `https://${process.env.VERCEL_PROJECT_PRODUCTION_URL}` : "http://localhost:3000";

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: { default: "llm-inference-engine", template: "%s · llm-inference-engine" },
  description: "A TinyLlama inference engine written from scratch in C++ and CUDA: paged KV cache, continuous batching, CUDA Graphs.",
  openGraph: { siteName: "llm-inference-engine", type: "website", images: [ogImage("overview", "llm-inference-engine: a TinyLlama inference engine in C++ and CUDA")] },
  twitter: { card: "summary_large_image" },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning className={`${inter.variable} ${jetbrains.variable}`}>
      <body suppressHydrationWarning className="min-h-dvh bg-bg text-text">
        <ThemeProvider attribute="data-theme" defaultTheme="dark" enableSystem={false} disableTransitionOnChange>
          <noscript>
            <style>{".intro-loader { display: none; }"}</style>
          </noscript>
          <a
            href="#content"
            className="sr-only z-50 rounded border border-border bg-surface px-3 py-2 font-mono text-sm text-accent focus:not-sr-only focus:fixed focus:left-4 focus:top-2"
          >
            skip to content
          </a>
          <IntroLoader />
          <ScrollToTop />
          <TopBar />
          <main id="content" tabIndex={-1} className="scroll-mt-12">
            {children}
          </main>
          <Footer />
        </ThemeProvider>
      </body>
    </html>
  );
}
