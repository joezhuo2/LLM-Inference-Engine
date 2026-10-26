import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { ThemeProvider } from "next-themes";
import { Footer } from "@/components/footer";
import { IntroLoader } from "@/components/intro-loader";
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
      <body suppressHydrationWarning className="min-h-dvh bg-bg text-text">
        <ThemeProvider attribute="data-theme" defaultTheme="dark" enableSystem={false} disableTransitionOnChange>
          <noscript>
            <style>{".intro-loader { display: none; }"}</style>
          </noscript>
          <IntroLoader />
          <TopBar />
          <main>{children}</main>
          <Footer />
        </ThemeProvider>
      </body>
    </html>
  );
}
