"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { Camera, ChevronLeft, ChevronRight, History, LayoutDashboard, Menu, Moon, Settings, Shield, Sun, Users, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { STORAGE_KEYS, getStorageItem, migrateStorageKeys, setStorageItem } from "@/lib/storage";
import { useDemoMode } from "@/providers/demo-context";
import { useMounted } from "@/hooks/use-mounted";

const items = [
  { title: "Overview", href: "/dashboard", icon: LayoutDashboard, number: "01" },
  { title: "Monitor", href: "/monitor", icon: Camera, number: "02" },
  { title: "Event ledger", href: "/events", icon: History, number: "03" },
  { title: "People", href: "/persons", icon: Users, number: "04" },
  { title: "Settings", href: "/settings", icon: Settings, number: "05" },
];

export function Sidebar() {
  const pathname = usePathname();
  const mounted = useMounted();
  const [storedCollapsed, setCollapsed] = useState(() => getStorageItem(STORAGE_KEYS.SIDEBAR_COLLAPSED) === "true");
  const collapsed = mounted && storedCollapsed;
  const [mobileOpen, setMobileOpen] = useState(false);
  const [storedDark, setIsDark] = useState(() => getStorageItem(STORAGE_KEYS.THEME) !== "light");
  const isDark = !mounted || storedDark;
  const menuRef = useRef<HTMLButtonElement>(null);
  const { isDemoMode, setDemoMode, isBackendAvailable, checkBackendStatus } = useDemoMode();

  useEffect(() => {
    migrateStorageKeys();
    document.documentElement.classList.toggle("dark", isDark);
  }, [isDark]);

  useEffect(() => {
    if (!mobileOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setMobileOpen(false);
        menuRef.current?.focus();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [mobileOpen]);

  const toggleTheme = () => {
    const dark = !isDark;
    setIsDark(dark);
    document.documentElement.classList.toggle("dark", dark);
    setStorageItem(STORAGE_KEYS.THEME, dark ? "dark" : "light");
  };

  const toggleMode = async () => {
    if (!isDemoMode) setDemoMode(true);
    else if (await checkBackendStatus()) setDemoMode(false);
  };

  return (
    <aside className={cn("console-sidebar", collapsed && "is-collapsed")}>
      <div className="console-brand">
        <Link href="/dashboard" className="console-wordmark" aria-label="SentinelVision overview">
          <span className="console-mark"><Shield size={21} strokeWidth={1.5} /></span>
          <span className="brand-copy"><strong>Sentinel<span>Vision</span></strong><small>Safety observation console</small></span>
        </Link>
        <button ref={menuRef} type="button" className="console-mobile-toggle" aria-label={mobileOpen ? "Close navigation" : "Open navigation"} aria-expanded={mobileOpen} aria-controls="console-navigation" onClick={() => setMobileOpen(!mobileOpen)}>
          {mobileOpen ? <X size={20} /> : <Menu size={20} />}
        </button>
      </div>
      <div id="console-navigation" className={cn("console-navigation", mobileOpen && "is-open")}>
        <p className="console-nav-label">Workspace / safety</p>
        <nav aria-label="Main navigation">
          {items.map(({ title, href, icon: Icon, number }) => (
            <Link key={href} href={href} title={collapsed ? title : undefined} aria-label={title} aria-current={pathname === href ? "page" : undefined} className={cn("console-nav-link", pathname === href && "is-active")} onClick={() => setMobileOpen(false)}>
              <Icon size={18} strokeWidth={1.6} /><span>{title}</span><small>{number}</small>
            </Link>
          ))}
        </nav>
        <div className="console-sidebar-bottom">
          <div className="console-mode-panel">
            <span className="console-eyebrow">Data source</span>
            <strong><i className={cn("console-dot", isDemoMode ? "demo" : isBackendAvailable ? "connected" : "offline")} />{isDemoMode ? "Demonstration" : isBackendAvailable ? "Backend reachable" : "Backend unavailable"}</strong>
            <p>{isDemoMode ? "Simulated records. Prerecorded video." : "Camera and model readiness are checked when a feed starts."}</p>
            <button type="button" onClick={toggleMode}>{isDemoMode ? "Check live connection →" : "Use demonstration →"}</button>
          </div>
          <button type="button" className="console-utility" onClick={toggleTheme} aria-label={isDark ? "Use light theme" : "Use dark theme"}>
            {isDark ? <Sun size={17} /> : <Moon size={17} />}<span>{isDark ? "Light workspace" : "Dark workspace"}</span>
          </button>
          <button type="button" className="console-utility console-collapse" onClick={() => {
            setCollapsed(!collapsed);
            setStorageItem(STORAGE_KEYS.SIDEBAR_COLLAPSED, String(!collapsed));
          }} aria-label={collapsed ? "Expand navigation" : "Collapse navigation"}>
            {collapsed ? <ChevronRight size={17} /> : <ChevronLeft size={17} />}<span>Collapse navigation</span>
          </button>
          <div className="console-signature"><span>SV / OBSERVE</span><span>01</span></div>
        </div>
      </div>
    </aside>
  );
}
