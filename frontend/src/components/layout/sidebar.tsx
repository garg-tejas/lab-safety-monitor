"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import {
  LayoutDashboard,
  Camera,
  History,
  Users,
  Shield,
  Settings,
  ChevronLeft,
  ChevronRight,
  Moon,
  Sun,
  Menu,
  X,
  Zap,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  STORAGE_KEYS,
  getStorageItem,
  setStorageItem,
  migrateStorageKeys,
} from "@/lib/storage";

const sidebarItems = [
  {
    title: "DASHBOARD",
    href: "/dashboard",
    icon: LayoutDashboard,
  },
  {
    title: "LIVE MONITOR",
    href: "/monitor",
    icon: Camera,
  },
  {
    title: "EVENTS LOG",
    href: "/events",
    icon: History,
  },
  {
    title: "PERSONS",
    href: "/persons",
    icon: Users,
  },
];

// Mobile sidebar overlay
function MobileSidebar({
  isOpen,
  onClose,
  pathname,
  isDark,
  onToggleTheme,
}: {
  isOpen: boolean;
  onClose: () => void;
  pathname: string;
  isDark: boolean;
  onToggleTheme: () => void;
}) {
  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 z-40 bg-black/80 backdrop-blur-sm lg:hidden"
            onClick={onClose}
          />
          {/* Sidebar */}
          <motion.div
            initial={{ x: "-100%" }}
            animate={{ x: 0 }}
            exit={{ x: "-100%" }}
            transition={{ type: "spring", damping: 30, stiffness: 300 }}
            className="fixed left-0 top-0 bottom-0 z-50 w-72 bg-sidebar border-r-4 border-primary lg:hidden shadow-2xl"
          >
            <div className="flex flex-col h-full scan-lines">
              {/* Header */}
              <div className="p-6 flex items-center justify-between border-b-2 border-sidebar-border bg-gradient-to-b from-sidebar-accent/30 to-transparent">
                <div className="flex items-center gap-3">
                  <div className="relative">
                    <div className="absolute inset-0 bg-primary/30 corner-cut blur-md" />
                    <div className="relative flex items-center justify-center w-12 h-12 corner-cut bg-primary text-primary-foreground shadow-lg border-2 border-primary glow-primary">
                      <Shield className="w-6 h-6" />
                    </div>
                  </div>
                  <div>
                    <h1 className="text-xl font-black gradient-text tracking-wider">SentinelVision</h1>
                    <p className="text-xs text-muted-foreground font-mono uppercase tracking-wide">
                      LAB SAFETY AI
                    </p>
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={onClose}
                  className="h-10 w-10 corner-cut border border-border hover:border-primary hover:bg-primary/10"
                >
                  <X className="h-5 w-5" />
                </Button>
              </div>

              {/* Navigation */}
              <nav className="flex-1 p-4 space-y-2">
                {sidebarItems.map((item, index) => {
                  const isActive = pathname === item.href;
                  return (
                    <motion.div
                      key={item.href}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.05 }}
                    >
                      <Link href={item.href} onClick={onClose}>
                        <Button
                          variant="ghost"
                          className={cn(
                            "w-full justify-start gap-3 h-12 relative group font-bold uppercase tracking-wide corner-cut text-xs",
                            isActive
                              ? "bg-sidebar-accent text-sidebar-primary border-l-4 border-primary glow-primary"
                              : "text-sidebar-foreground hover:bg-sidebar-accent/50 hover:border-l-4 hover:border-primary/50"
                          )}
                        >
                          <item.icon
                            className={cn(
                              "w-5 h-5 shrink-0",
                              isActive ? "text-primary" : "text-muted-foreground group-hover:text-primary"
                            )}
                          />
                          <span>{item.title}</span>
                          {isActive && (
                            <motion.div
                              layoutId="mobile-active-glow"
                              className="absolute inset-0 bg-primary/5 pointer-events-none corner-cut"
                            />
                          )}
                        </Button>
                      </Link>
                    </motion.div>
                  );
                })}
              </nav>

              {/* Footer */}
              <div className="p-4 border-t-2 border-sidebar-border space-y-2 bg-gradient-to-t from-sidebar-accent/30 to-transparent">
                <Button
                  variant="ghost"
                  onClick={onToggleTheme}
                  className="w-full justify-start gap-3 h-12 font-bold uppercase text-xs tracking-wide corner-cut hover:border-l-4 hover:border-primary/50"
                >
                  {isDark ? (
                    <Sun className="w-5 h-5 text-amber-400" />
                  ) : (
                    <Moon className="w-5 h-5 text-slate-400" />
                  )}
                  <span>{isDark ? "LIGHT MODE" : "DARK MODE"}</span>
                </Button>
                <Link href="/settings" onClick={onClose}>
                  <Button
                    variant="ghost"
                    className={cn(
                      "w-full justify-start gap-3 h-12 font-bold uppercase text-xs tracking-wide corner-cut",
                      pathname === "/settings"
                        ? "bg-sidebar-accent text-sidebar-primary border-l-4 border-primary"
                        : "text-muted-foreground hover:text-foreground hover:border-l-4 hover:border-primary/50"
                    )}
                  >
                    <Settings className="w-5 h-5 shrink-0" />
                    <span>SETTINGS</span>
                  </Button>
                </Link>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}

// Mobile header with hamburger menu
export function MobileHeader({
  onOpenSidebar,
}: {
  onOpenSidebar: () => void;
}) {
  return (
    <div className="lg:hidden fixed top-0 left-0 right-0 z-30 h-16 bg-sidebar/95 backdrop-blur-lg border-b-2 border-primary flex items-center justify-between px-4 scan-lines">
      <div className="flex items-center gap-3">
        <Button
          variant="ghost"
          size="icon"
          onClick={onOpenSidebar}
          className="h-10 w-10 corner-cut border border-border hover:border-primary hover:glow-primary"
        >
          <Menu className="h-5 w-5" />
        </Button>
        <div className="flex items-center gap-2">
          <div className="flex items-center justify-center w-10 h-10 corner-cut bg-primary text-primary-foreground border border-primary/50">
            <Shield className="w-5 h-5" />
          </div>
          <span className="font-black gradient-text text-lg tracking-wider">SentinelVision</span>
        </div>
      </div>
    </div>
  );
}

export function Sidebar() {
  const pathname = usePathname();

  // Lazy state initialization - runs only once on mount
  const [collapsed, setCollapsed] = useState(() => {
    if (typeof window !== "undefined") {
      migrateStorageKeys(); // Migrate old keys to versioned keys
      const stored = getStorageItem(STORAGE_KEYS.SIDEBAR_COLLAPSED);
      return stored === "true";
    }
    return false;
  });

  const [mounted, setMounted] = useState(false);

  const [isDark, setIsDark] = useState(() => {
    if (typeof window !== "undefined") {
      return document.documentElement.classList.contains("dark");
    }
    return false;
  });

  const [mobileOpen, setMobileOpen] = useState(false);

  // Mark component as mounted (for theme toggle hydration)
  useEffect(() => {
    setMounted(true);
  }, []);

  // Close mobile sidebar on route change
  useEffect(() => {
    setMobileOpen(false);
  }, [pathname]);

  // Save collapsed state to localStorage
  const handleToggleCollapse = () => {
    const newState = !collapsed;
    setCollapsed(newState);
    setStorageItem(STORAGE_KEYS.SIDEBAR_COLLAPSED, String(newState));
  };

  // Toggle theme
  const handleToggleTheme = () => {
    const newIsDark = !isDark;
    setIsDark(newIsDark);
    const newTheme = newIsDark ? "dark" : "light";
    document.documentElement.classList.remove("light", "dark");
    document.documentElement.classList.add(newTheme);
    setStorageItem(STORAGE_KEYS.THEME, newTheme);
  };

  return (
    <>
      {/* Mobile sidebar and header */}
      <MobileHeader onOpenSidebar={() => setMobileOpen(true)} />
      <MobileSidebar
        isOpen={mobileOpen}
        onClose={() => setMobileOpen(false)}
        pathname={pathname}
        isDark={isDark}
        onToggleTheme={handleToggleTheme}
      />

      {/* Desktop sidebar */}
      <motion.div
        initial={false}
        animate={{ width: collapsed ? 95 : 295 }}
        transition={{ duration: 0.3, ease: [0.25, 0.46, 0.45, 0.94] }}
        className={cn(
          "hidden lg:flex flex-col h-screen bg-sidebar border-r-4 border-primary relative scan-lines"
        )}
      >
        {/* Collapse toggle button */}
        <motion.button
          whileHover={{ scale: 1.15 }}
          whileTap={{ scale: 0.9 }}
          onClick={handleToggleCollapse}
          className="absolute -right-4 top-8 z-10 flex h-8 w-8 items-center justify-center corner-cut border-2 bg-primary text-primary-foreground shadow-lg hover:glow-primary transition-all"
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          <motion.div
            animate={{ rotate: collapsed ? 0 : 180 }}
            transition={{ duration: 0.3 }}
          >
            <ChevronRight className="h-4 w-4 font-bold" />
          </motion.div>
        </motion.button>

        {/* Header/Logo */}
        <div className={cn("p-4 border-b-2 border-sidebar-border bg-gradient-to-b from-sidebar-accent/20 to-transparent", collapsed ? "px-3" : "p-6")}>
          <div
            className={cn(
              "flex items-center gap-3 transition-all duration-300",
              collapsed && "justify-center flex-col gap-2"
            )}
          >
            <motion.div
              whileHover={{ scale: 1.1, rotate: 5 }}
              className="relative"
            >
              <div className="absolute inset-0 bg-primary/30 corner-cut blur-lg" />
              <div className="relative flex items-center justify-center w-12 h-12 corner-cut bg-primary text-primary-foreground shadow-lg border-2 border-primary/50 glow-primary">
                <Shield className="w-6 h-6" />
              </div>
            </motion.div>
            <AnimatePresence>
              {!collapsed && (
                <motion.div
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  transition={{ duration: 0.2 }}
                  className="overflow-hidden"
                >
                  <h1 className="text-xl font-black leading-none gradient-text tracking-wider">
                    SentinelVision
                  </h1>
                  <p className="text-[10px] text-muted-foreground mt-1 font-mono uppercase tracking-widest">
                    LAB SAFETY AI
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Navigation */}
          <nav className={cn("space-y-1.5", collapsed ? "mt-6" : "mt-8")}>
            {sidebarItems.map((item) => {
              const isActive = pathname === item.href;
              return (
                <Link key={item.href} href={item.href}>
                  <motion.div whileHover={{ x: 2 }} whileTap={{ scale: 0.98 }}>
                    <Button
                      variant="ghost"
                      className={cn(
                        "w-full justify-start gap-3 transition-all duration-200 relative group h-11 font-bold uppercase tracking-wide corner-cut",
                        collapsed && "justify-center px-2",
                        collapsed ? "text-[0px]" : "text-xs",
                        isActive
                          ? "bg-sidebar-accent text-sidebar-primary border-l-4 border-primary glow-primary"
                          : "text-sidebar-foreground hover:bg-sidebar-accent/50 hover:text-sidebar-foreground hover:border-l-4 hover:border-primary/50"
                      )}
                    >
                      <item.icon
                        className={cn(
                          "w-5 h-5 shrink-0 transition-colors",
                          isActive ? "text-primary" : "text-muted-foreground group-hover:text-primary"
                        )}
                      />
                      <AnimatePresence>
                        {!collapsed && (
                          <motion.span
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            transition={{ duration: 0.15 }}
                          >
                            {item.title}
                          </motion.span>
                        )}
                      </AnimatePresence>
                      {isActive && (
                        <motion.div
                          layoutId="desktop-active-glow"
                          className="absolute inset-0 bg-primary/5 pointer-events-none corner-cut"
                          transition={{ type: "spring", stiffness: 500, damping: 35 }}
                        />
                      )}
                    </Button>
                  </motion.div>
                </Link>
              );
            })}
          </nav>
        </div>

        {/* Footer with theme toggle and settings */}
        <div
          className={cn(
            "mt-auto border-t-2 border-sidebar-border bg-gradient-to-t from-sidebar-accent/20 to-transparent",
            collapsed ? "p-2" : "p-4"
          )}
        >
          {/* Theme toggle */}
          {mounted && (
            <motion.div whileHover={{ x: 2 }} whileTap={{ scale: 0.98 }}>
              <Button
                variant="ghost"
                onClick={handleToggleTheme}
                className={cn(
                  "w-full justify-start gap-3 text-muted-foreground hover:text-foreground hover:bg-sidebar-accent/50 transition-all duration-200 mb-1.5 h-11 font-bold uppercase tracking-wide corner-cut hover:border-l-4 hover:border-primary/50",
                  collapsed && "justify-center px-2",
                  collapsed ? "text-[0px]" : "text-xs"
                )}
              >
                <motion.div
                  animate={{ rotate: isDark ? 360 : 0 }}
                  transition={{ duration: 0.5 }}
                >
                  {isDark ? (
                    <Sun className="w-5 h-5 shrink-0 text-amber-400" />
                  ) : (
                    <Moon className="w-5 h-5 shrink-0 text-slate-400" />
                  )}
                </motion.div>
                <AnimatePresence>
                  {!collapsed && (
                    <motion.span
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                    >
                      {isDark ? "LIGHT MODE" : "DARK MODE"}
                    </motion.span>
                  )}
                </AnimatePresence>
              </Button>
            </motion.div>
          )}

          {/* Settings Link */}
          <Link href="/settings">
            <motion.div whileHover={{ x: 2 }} whileTap={{ scale: 0.98 }}>
              <Button
                variant="ghost"
                className={cn(
                  "w-full justify-start gap-3 transition-all duration-200 relative h-11 font-bold uppercase tracking-wide corner-cut",
                  collapsed && "justify-center px-2",
                  collapsed ? "text-[0px]" : "text-xs",
                  pathname === "/settings"
                    ? "bg-sidebar-accent text-sidebar-primary border-l-4 border-primary glow-primary"
                    : "text-muted-foreground hover:text-foreground hover:bg-sidebar-accent/50 hover:border-l-4 hover:border-primary/50"
                )}
              >
                <Settings className={cn(
                  "w-5 h-5 shrink-0",
                  pathname === "/settings" ? "text-primary" : ""
                )} />
                <AnimatePresence>
                  {!collapsed && (
                    <motion.span
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                    >
                      SETTINGS
                    </motion.span>
                  )}
                </AnimatePresence>
                {pathname === "/settings" && (
                  <motion.div
                    layoutId="desktop-active-glow"
                    className="absolute inset-0 bg-primary/5 pointer-events-none corner-cut"
                    transition={{ type: "spring", stiffness: 500, damping: 35 }}
                  />
                )}
              </Button>
            </motion.div>
          </Link>

          {/* System indicator */}
          {!collapsed && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-4 p-3 corner-cut bg-success/10 border border-success/30"
            >
              <div className="flex items-center gap-2">
                <div className="relative">
                  <motion.div
                    animate={{ scale: [1, 1.3, 1], opacity: [1, 0.6, 1] }}
                    transition={{ duration: 2, repeat: Infinity }}
                    className="w-2 h-2 bg-success rounded-full"
                  />
                </div>
                <span className="text-[10px] font-black text-success uppercase tracking-widest font-mono">
                  SYSTEM ACTIVE
                </span>
              </div>
            </motion.div>
          )}
        </div>
      </motion.div>
    </>
  );
}
