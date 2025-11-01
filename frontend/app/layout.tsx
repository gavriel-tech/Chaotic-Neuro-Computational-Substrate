import type { Metadata } from "next";
import "./globals.css";
import { ColorSchemeProvider } from "@/components/providers/ColorSchemeProvider";
import { WebSocketBridge } from "@/components/providers/WebSocketBridge";
import { NotificationContainer } from "@/components/UI/Notification";
import { ConfirmContainer } from "@/components/UI/ConfirmDialog";

export const metadata: Metadata = {
  title: "GMCS - Universal Chaotic-Neuro Computational Substrate Platform v0.1",
  description: "Modular node-based system for chaotic oscillators, THRML energy-based models, and signal processing",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" data-scheme="cyan">
      <body className="bg-bg-primary text-cyber-cyan-glow min-h-screen antialiased">
        <ColorSchemeProvider>
          <WebSocketBridge />
          <NotificationContainer />
          <ConfirmContainer />
          {children}
        </ColorSchemeProvider>
      </body>
    </html>
  );
}
