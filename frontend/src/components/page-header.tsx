"use client";

import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { ReactNode } from "react";

interface PageHeaderProps {
  title: string;
  description?: string;
  children?: ReactNode;
  className?: string;
  badge?: ReactNode;
  action?: ReactNode;
}

/**
 * Animated page header component with gradient background effect.
 */
export function PageHeader({
  title,
  description,
  children,
  className,
  badge,
  action,
}: PageHeaderProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.25, 0.46, 0.45, 0.94] }}
      className={cn("relative", className)}
    >
      {/* Subtle gradient background */}
      <div className="absolute inset-0 -z-10 overflow-hidden rounded-2xl opacity-50">
        <motion.div
          animate={{
            backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"],
          }}
          transition={{
            duration: 15,
            ease: "linear",
            repeat: Infinity,
          }}
          className="absolute inset-0 bg-gradient-to-r from-primary/5 via-info/5 to-primary/5 bg-[length:200%_100%]"
        />
      </div>

      <div className="flex items-center justify-between">
        <div className="flex items-start gap-4">
          <div>
            <motion.h2
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="text-3xl font-bold tracking-tight gradient-text"
            >
              {title}
            </motion.h2>
            {description && (
              <motion.p
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.5, delay: 0.2 }}
                className="text-muted-foreground mt-1"
              >
                {description}
              </motion.p>
            )}
          </div>
          {badge && (
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4, delay: 0.3 }}
            >
              {badge}
            </motion.div>
          )}
        </div>
        {action && (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            {action}
          </motion.div>
        )}
      </div>
      {children}
    </motion.div>
  );
}

/**
 * Status badge component for page headers.
 */
interface StatusBadgeProps {
  icon: ReactNode;
  label: string;
  value?: string | number;
  variant?: "default" | "success" | "warning" | "danger" | "info";
  pulse?: boolean;
}

export function StatusBadge({
  icon,
  label,
  value,
  variant = "default",
  pulse = false,
}: StatusBadgeProps) {
  const variantStyles = {
    default: "bg-muted/50 border-border/50 text-muted-foreground",
    success: "bg-success/10 border-success/20 text-success",
    warning: "bg-warning/10 border-warning/20 text-warning",
    danger: "bg-danger/10 border-danger/20 text-danger",
    info: "bg-info/10 border-info/20 text-info",
  };

  const iconVariantStyles = {
    default: "text-primary",
    success: "text-success",
    warning: "text-warning",
    danger: "text-danger",
    info: "text-info",
  };

  return (
    <div
      className={cn(
        "flex items-center gap-2 px-4 py-2 rounded-full border transition-all duration-200",
        variantStyles[variant]
      )}
    >
      <motion.div
        animate={pulse ? { scale: [1, 1.2, 1] } : {}}
        transition={{ duration: 2, repeat: Infinity }}
        className={cn("w-4 h-4", iconVariantStyles[variant])}
      >
        {icon}
      </motion.div>
      <span className="text-sm">
        {label}
        {value !== undefined && (
          <>
            :{" "}
            <span className="text-foreground font-medium">{value}</span>
          </>
        )}
      </span>
    </div>
  );
}

/**
 * Live indicator component.
 */
export function LiveIndicator({ label = "Live" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-danger/10 border border-danger/20">
      <span className="relative flex h-2.5 w-2.5">
        <motion.span
          animate={{ scale: [1, 1.5, 1], opacity: [0.75, 0, 0.75] }}
          transition={{ duration: 1.5, repeat: Infinity }}
          className="absolute inline-flex h-full w-full rounded-full bg-danger"
        />
        <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-danger" />
      </span>
      <span className="text-sm font-medium text-danger">{label}</span>
    </div>
  );
}

/**
 * Last updated indicator.
 */
interface LastUpdatedProps {
  timestamp?: string | Date;
  isRefreshing?: boolean;
}

export function LastUpdated({ timestamp, isRefreshing }: LastUpdatedProps) {
  const formattedTime = timestamp
    ? new Date(timestamp).toLocaleTimeString()
    : "--";

  return (
    <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-muted/50 border border-border/50">
      <motion.div
        animate={isRefreshing ? { rotate: 360 } : {}}
        transition={{ duration: 1, repeat: isRefreshing ? Infinity : 0 }}
      >
        <svg
          className={cn(
            "w-4 h-4",
            isRefreshing ? "text-primary" : "text-muted-foreground"
          )}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
          />
        </svg>
      </motion.div>
      <span className="text-sm text-muted-foreground">
        Last updated:{" "}
        <span className="text-foreground font-medium">{formattedTime}</span>
      </span>
    </div>
  );
}
