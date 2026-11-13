"use client";

import { cn } from "@/lib/utils";
import type { ReactNode } from "react";

interface PageHeaderProps { title: string; description?: string; children?: ReactNode; className?: string; badge?: ReactNode; action?: ReactNode }
export function PageHeader({ title, description, children, className, badge, action }: PageHeaderProps) {
  return <header className={cn("console-header", className)}><div><span className="console-eyebrow">SentinelVision / Workspace</span><h1>{title}</h1>{description && <p>{description}</p>}{children}</div><div className="console-header-aside">{badge}{action}</div></header>;
}

interface StatusBadgeProps { icon: ReactNode; label: string; value?: string | number; variant?: "default" | "success" | "warning" | "danger" | "info"; pulse?: boolean }
export function StatusBadge({ icon, label, value, variant = "default" }: StatusBadgeProps) {
  return <div className={cn("console-source-tag", variant !== "default" && `text-${variant}`)}>{icon}<span>{label}{value !== undefined ? `: ${value}` : ""}</span></div>;
}
export function LiveIndicator({ label = "Live" }: { label?: string }) {
  return <span className="console-source-tag"><i className="console-dot connected" />{label}</span>;
}
export function LastUpdated({ timestamp, isRefreshing }: { timestamp?: string | Date; isRefreshing?: boolean }) {
  return <span className="console-source-tag">{isRefreshing ? "Refreshing…" : "Record updated"} {timestamp ? new Date(timestamp).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" }) : "—"}</span>;
}
