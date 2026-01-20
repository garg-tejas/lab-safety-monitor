import type { Metadata } from "next";
import { Rajdhani, Orbitron, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { Sidebar } from "@/components/layout/sidebar";
import { WebSocketProvider } from "@/providers/websocket-provider";
import { QueryProvider } from "@/providers/query-provider";
import { DemoModeProvider } from "@/providers/demo-context";
import { Toaster } from "sonner";

const rajdhani = Rajdhani({
  weight: ["300", "400", "500", "600", "700"],
  variable: "--font-rajdhani",
  subsets: ["latin"],
});

const orbitron = Orbitron({
  weight: ["400", "500", "600", "700", "800", "900"],
  variable: "--font-orbitron",
  subsets: ["latin"],
});

const jetbrainsMono = JetBrains_Mono({
  weight: ["400", "500", "600", "700"],
  variable: "--font-jetbrains-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "SentinelVision - Lab Safety",
  description: "AI-powered laboratory safety compliance monitoring system",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${rajdhani.variable} ${orbitron.variable} ${jetbrainsMono.variable} antialiased`}
      >
        <QueryProvider>
          <DemoModeProvider>
            <WebSocketProvider>
              <div className="flex h-screen overflow-hidden bg-background">
                <Sidebar />
                <div className="flex-1 flex flex-col h-screen overflow-hidden relative">
                  {/* Industrial tech grid overlay */}
                  <div className="absolute inset-0 -z-10 grid-overlay opacity-[0.03] pointer-events-none" />
                  {/* Diagonal accent stripe */}
                  <div className="absolute top-0 right-0 w-64 h-64 -z-10 bg-gradient-to-br from-primary/5 to-transparent blur-3xl" />
                  <main className="flex-1 overflow-y-auto p-4 pt-20 lg:p-8 lg:pt-8">
                    {children}
                  </main>
                </div>
              </div>
              <Toaster
                richColors
                position="top-right"
                expand={true}
                closeButton
                duration={4000}
                toastOptions={{
                  classNames: {
                    toast: "glass corner-cut border-2 border-border shadow-lg font-mono",
                    title: "font-black uppercase tracking-wide text-sm",
                    description: "text-muted-foreground text-xs uppercase tracking-wide",
                    actionButton: "bg-primary text-primary-foreground corner-cut font-bold uppercase tracking-wider",
                    cancelButton: "bg-muted text-muted-foreground corner-cut",
                    closeButton: "bg-muted/80 border-border/50 hover:bg-muted corner-cut",
                  },
                }}
              />
            </WebSocketProvider>
          </DemoModeProvider>
        </QueryProvider>
      </body>
    </html>
  );
}
