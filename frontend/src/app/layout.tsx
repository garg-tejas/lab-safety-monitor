import type { Metadata } from "next";
import { IBM_Plex_Sans, IBM_Plex_Mono } from "next/font/google";
import "./globals.css";
import { Sidebar } from "@/components/layout/sidebar";
import { WebSocketProvider } from "@/providers/websocket-provider";
import { QueryProvider } from "@/providers/query-provider";
import { DemoModeProvider } from "@/providers/demo-context";
import { Toaster } from "sonner";

const sans = IBM_Plex_Sans({ weight: ["400", "500", "600", "700"], variable: "--font-console-sans", subsets: ["latin"] });
const mono = IBM_Plex_Mono({ weight: ["400", "500", "600"], variable: "--font-jetbrains-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "SentinelVision · Safety observation",
  description: "Review laboratory safety events, monitor a camera feed, and inspect recorded video.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className={`${sans.variable} ${mono.variable} antialiased`}>
        <QueryProvider><DemoModeProvider><WebSocketProvider>
          <a href="#main-content" className="skip-link">Skip to workspace</a>
          <div className="console-shell">
            <Sidebar />
            <main id="main-content" className="console-main" tabIndex={-1}>{children}</main>
          </div>
          <Toaster richColors position="top-right" closeButton duration={4000} />
        </WebSocketProvider></DemoModeProvider></QueryProvider>
      </body>
    </html>
  );
}
