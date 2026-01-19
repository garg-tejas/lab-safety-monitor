"use client";

import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { AnimatedCounter } from "@/components/ui/animated-counter";
import { cn } from "@/lib/utils";

type CardVariant = "default" | "success" | "warning" | "danger" | "info";

interface StatsCardProps {
  title: string;
  value: string | number;
  description?: string;
  icon?: React.ReactNode;
  trend?: {
    value: number;
    isPositive: boolean;
  };
  loading?: boolean;
  variant?: CardVariant;
  animate?: boolean;
}

const variantStyles: Record<
  CardVariant,
  { iconBg: string; iconColor: string; accentBg: string; glowColor: string; borderColor: string; }
> = {
  default: {
    iconBg: "bg-primary/15 border-2 border-primary/40",
    iconColor: "text-primary",
    accentBg: "from-primary/10 via-primary/5 to-transparent",
    glowColor: "group-hover:glow-primary",
    borderColor: "border-primary/30",
  },
  success: {
    iconBg: "bg-success/15 border-2 border-success/40",
    iconColor: "text-success",
    accentBg: "from-success/10 via-success/5 to-transparent",
    glowColor: "group-hover:glow-success",
    borderColor: "border-success/30",
  },
  warning: {
    iconBg: "bg-warning/15 border-2 border-warning/40",
    iconColor: "text-warning",
    accentBg: "from-warning/10 via-warning/5 to-transparent",
    glowColor: "group-hover:glow-warning",
    borderColor: "border-warning/30",
  },
  danger: {
    iconBg: "bg-danger/15 border-2 border-danger/40",
    iconColor: "text-danger",
    accentBg: "from-danger/10 via-danger/5 to-transparent",
    glowColor: "group-hover:glow-danger",
    borderColor: "border-danger/30",
  },
  info: {
    iconBg: "bg-info/15 border-2 border-info/40",
    iconColor: "text-info",
    accentBg: "from-info/10 via-info/5 to-transparent",
    glowColor: "group-hover:glow-info",
    borderColor: "border-info/30",
  },
};

export function StatsCard({
  title,
  value,
  description,
  icon,
  trend,
  loading = false,
  variant = "default",
  animate = true,
}: StatsCardProps) {
  const styles = variantStyles[variant];

  if (loading) {
    return (
      <Card variant="glass" className="relative overflow-hidden corner-cut border-2">
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <Skeleton className="h-4 w-24 shimmer corner-cut" />
          <Skeleton className="h-12 w-12 corner-cut shimmer" />
        </CardHeader>
        <CardContent>
          <Skeleton className="h-10 w-20 mb-1 shimmer corner-cut" />
          <Skeleton className="h-3 w-32 shimmer corner-cut" />
        </CardContent>
      </Card>
    );
  }

  // Parse numeric value for animation
  const numericValue =
    typeof value === "number"
      ? value
      : parseFloat(value.toString().replace(/[^0-9.-]/g, ""));
  const isPercentage =
    typeof value === "string" && value.toString().includes("%");
  const suffix = isPercentage ? "%" : "";
  const canAnimate = animate && !isNaN(numericValue);

  return (
    <motion.div
      whileHover={{ y: -6, scale: 1.02 }}
      transition={{ duration: 0.2, type: "spring", stiffness: 400, damping: 25 }}
    >
      <Card
        variant="glass"
        className={cn(
          "relative overflow-hidden group transition-all duration-300 corner-cut border-2",
          "hover:shadow-xl scan-lines",
          styles.borderColor,
          styles.glowColor
        )}
      >
        {/* Diagonal stripe pattern on hover */}
        <div className="absolute inset-0 diagonal-stripes opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none" />

        {/* Animated gradient background */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className={cn(
            "absolute inset-0 bg-gradient-to-br opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none",
            styles.accentBg
          )}
        />

        {/* Tech border corners */}
        <div className="absolute top-0 left-0 w-3 h-3 border-t-2 border-l-2 border-primary opacity-60" />
        <div className="absolute top-0 right-0 w-3 h-3 border-t-2 border-r-2 border-primary opacity-60" />
        <div className="absolute bottom-0 left-0 w-3 h-3 border-b-2 border-l-2 border-primary opacity-60" />
        <div className="absolute bottom-0 right-0 w-3 h-3 border-b-2 border-r-2 border-primary opacity-60" />

        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3 relative">
          <CardTitle className="text-xs font-black text-muted-foreground uppercase tracking-widest font-mono">
            {title}
          </CardTitle>
          {icon && (
            <motion.div
              whileHover={{ scale: 1.15, rotate: [0, -5, 5, 0] }}
              whileTap={{ scale: 0.9 }}
              transition={{ duration: 0.3 }}
              className={cn(
                "flex items-center justify-center w-12 h-12 corner-cut transition-all duration-300 shadow-lg",
                styles.iconBg
              )}
            >
              <div className={cn("w-5 h-5", styles.iconColor)}>{icon}</div>
            </motion.div>
          )}
        </CardHeader>
        <CardContent className="relative">
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.1 }}
            className="text-4xl font-black tracking-tight data-mono"
          >
            {canAnimate ? (
              <AnimatedCounter
                value={numericValue}
                suffix={suffix}
                decimals={isPercentage ? 1 : 0}
                duration={1200}
              />
            ) : (
              value
            )}
          </motion.div>
          {description && (
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.4, delay: 0.2 }}
              className="text-xs text-muted-foreground mt-2 uppercase font-semibold tracking-wide"
            >
              {description}
            </motion.p>
          )}
          {trend && (
            <motion.div
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.4, delay: 0.3 }}
              className={cn(
                "inline-flex items-center gap-1.5 text-xs font-black mt-3 px-3 py-1 corner-cut border-2 uppercase tracking-wider",
                trend.isPositive
                  ? "text-success bg-success/10 border-success/40"
                  : "text-danger bg-danger/10 border-danger/40"
              )}
            >
              <motion.span
                animate={{
                  y: trend.isPositive ? [0, -3, 0] : [0, 3, 0],
                  scale: [1, 1.2, 1]
                }}
                transition={{ duration: 1.5, repeat: Infinity }}
                className="text-base"
              >
                {trend.isPositive ? "▲" : "▼"}
              </motion.span>
              <span>{Math.abs(trend.value)}%</span>
            </motion.div>
          )}

          {/* Status bar at bottom */}
          <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-transparent via-primary to-transparent opacity-50" />
        </CardContent>
      </Card>
    </motion.div>
  );
}
