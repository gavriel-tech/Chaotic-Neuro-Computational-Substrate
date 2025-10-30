"use client";

import { useEffect } from "react";
import { restoreSavedScheme, useColorSchemeStore } from "@/lib/stores/colorScheme";

interface Props {
  children: React.ReactNode;
}

export function ColorSchemeProvider({ children }: Props) {
  const cycleScheme = useColorSchemeStore((state) => state.cycleScheme);
  const scheme = useColorSchemeStore((state) => state.scheme);

  // Restore saved theme on mount
  useEffect(() => {
    restoreSavedScheme();
  }, []);

  // Update CSS custom properties whenever scheme changes
  useEffect(() => {
    const root = document.documentElement;
    const body = document.body;
    root.dataset.scheme = scheme;
    body.dataset.scheme = scheme;
  }, [scheme]);

  // Keyboard shortcut: cycle color scheme with "C"
  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.key.toLowerCase() === "c") {
        cycleScheme();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [cycleScheme]);

  return <>{children}</>;
}
