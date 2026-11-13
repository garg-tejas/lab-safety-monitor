"use client";

import { createContext, useContext, useEffect, useState, useSyncExternalStore } from "react";
import { useMounted } from "@/hooks/use-mounted";
import { getStorageItem, setStorageItem } from "@/lib/storage";

type Theme = "light" | "dark" | "system";
interface ThemeContextType { theme: Theme; resolvedTheme: "light" | "dark"; setTheme: (theme: Theme) => void; toggleTheme: () => void }
const ThemeContext = createContext<ThemeContextType | undefined>(undefined);
const THEME_STORAGE_KEY = "marketwise-theme";
const subscribeToSystem = (callback: () => void) => {
  const media = window.matchMedia("(prefers-color-scheme: dark)");
  media.addEventListener("change", callback);
  return () => media.removeEventListener("change", callback);
};
const systemSnapshot = () => window.matchMedia("(prefers-color-scheme: dark)").matches;
const serverSnapshot = () => true;

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const mounted = useMounted();
  const [theme, setThemeState] = useState<Theme>(() => {
    const saved = getStorageItem(THEME_STORAGE_KEY);
    return saved === "light" || saved === "dark" ? saved : "system";
  });
  const prefersDark = useSyncExternalStore(subscribeToSystem, systemSnapshot, serverSnapshot);
  const resolvedTheme = theme === "system" ? prefersDark ? "dark" : "light" : theme;
  useEffect(() => {
    document.documentElement.classList.toggle("dark", resolvedTheme === "dark");
  }, [resolvedTheme]);
  const setTheme = (nextTheme: Theme) => {
    setThemeState(nextTheme);
    setStorageItem(THEME_STORAGE_KEY, nextTheme);
  };
  return <ThemeContext.Provider value={{ theme, resolvedTheme, setTheme, toggleTheme: () => setTheme(resolvedTheme === "dark" ? "light" : "dark") }}><div style={mounted ? undefined : { visibility: "hidden" }}>{children}</div></ThemeContext.Provider>;
}

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) throw new Error("useTheme must be used within a ThemeProvider");
  return context;
}
