"use client";

import { motion } from "framer-motion";
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Area,
  AreaChart,
  Cell,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AlertTriangle, BarChart3 } from "lucide-react";

interface TimelineData {
  date: string;
  violations: number;
}

interface PPEBreakdown {
  ppe_type: string;
  count: number;
}

interface ViolationChartProps {
  data: TimelineData[];
  title?: string;
}

// Custom tooltip component with glassmorphism
interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{ value: number; name?: string; payload?: { name?: string } }>;
  label?: string | number;
  formatter?: (value: number) => string;
  labelFormatter?: (label: string | number) => string;
}

function CustomTooltip({
  active,
  payload,
  label,
  formatter,
  labelFormatter,
}: CustomTooltipProps) {
  if (!active || !payload || !payload.length) return null;

  const value = payload[0].value;
  const name = payload[0].payload?.name || payload[0].name || "violations";
  
  const formattedLabel = labelFormatter
    ? labelFormatter(label || "")
    : typeof label === "string"
    ? new Date(label).toLocaleDateString("en-US", {
        weekday: "short",
        month: "short",
        day: "numeric",
      })
    : label;

  return (
    <motion.div
      initial={{ opacity: 0, y: 5 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass rounded-xl px-4 py-3 shadow-lg border border-border/50"
    >
      <p className="text-xs text-muted-foreground mb-1">{formattedLabel}</p>
      <div className="flex items-center gap-2">
        <div className="w-2 h-2 rounded-full bg-danger" />
        <p className="text-sm font-semibold text-foreground">
          {formatter ? formatter(value) : value}
          <span className="text-muted-foreground font-normal ml-1">
            {name}
          </span>
        </p>
      </div>
    </motion.div>
  );
}

export function ViolationTimelineChart({
  data,
  title = "Violations Over Time",
}: ViolationChartProps) {
  const gradientId = "violationGradient";
  const lineGradientId = "violationLineGradient";

  // Calculate max value for domain
  const maxValue = Math.max(...data.map((d) => d.violations), 1);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card variant="glass" hover glow="danger" className="overflow-hidden">
        <CardHeader className="pb-2">
          <CardTitle className="text-lg flex items-center gap-3">
            <motion.div
              whileHover={{ scale: 1.1, rotate: -5 }}
              className="flex items-center justify-center w-10 h-10 rounded-xl bg-danger/10"
            >
              <AlertTriangle className="w-5 h-5 text-danger" />
            </motion.div>
            <div>
              <span>{title}</span>
              <p className="text-xs font-normal text-muted-foreground mt-0.5">
                Last 7 days of safety violations
              </p>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent className="pt-4">
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#ef4444" stopOpacity={0.4} />
                  <stop offset="50%" stopColor="#ef4444" stopOpacity={0.15} />
                  <stop offset="100%" stopColor="#ef4444" stopOpacity={0} />
                </linearGradient>
                <linearGradient id={lineGradientId} x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#f87171" />
                  <stop offset="50%" stopColor="#ef4444" />
                  <stop offset="100%" stopColor="#dc2626" />
                </linearGradient>
              </defs>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="currentColor"
                strokeOpacity={0.08}
                vertical={false}
              />
              <XAxis
                dataKey="date"
                tickFormatter={(value) =>
                  new Date(value).toLocaleDateString("en-US", {
                    month: "short",
                    day: "numeric",
                  })
                }
                stroke="currentColor"
                strokeOpacity={0.2}
                tick={{ fill: "currentColor", opacity: 0.5, fontSize: 11 }}
                tickLine={false}
                axisLine={false}
                dy={10}
              />
              <YAxis
                domain={[0, maxValue + Math.ceil(maxValue * 0.2)]}
                stroke="currentColor"
                strokeOpacity={0.2}
                tick={{ fill: "currentColor", opacity: 0.5, fontSize: 11 }}
                tickLine={false}
                axisLine={false}
                width={35}
                allowDecimals={false}
              />
              <Tooltip
                content={<CustomTooltip formatter={(v) => `${v}`} />}
                cursor={{ stroke: "currentColor", strokeOpacity: 0.1, strokeDasharray: "4 4" }}
              />
              <Area
                type="monotone"
                dataKey="violations"
                stroke={`url(#${lineGradientId})`}
                strokeWidth={3}
                fill={`url(#${gradientId})`}
                dot={false}
                activeDot={{
                  r: 6,
                  fill: "#ef4444",
                  stroke: "var(--background)",
                  strokeWidth: 3,
                  className: "drop-shadow-lg",
                }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </motion.div>
  );
}

interface PPEBreakdownChartProps {
  data: PPEBreakdown[];
  title?: string;
}

// Color palette for bars
const barColors = [
  { main: "#14b8a6", light: "#5eead4" }, // teal
  { main: "#f59e0b", light: "#fcd34d" }, // amber
  { main: "#ef4444", light: "#fca5a5" }, // red
  { main: "#8b5cf6", light: "#c4b5fd" }, // purple
  { main: "#22c55e", light: "#86efac" }, // green
  { main: "#3b82f6", light: "#93c5fd" }, // blue
];

export function PPEBreakdownChart({
  data,
  title = "Violations by PPE Type",
}: PPEBreakdownChartProps) {
  const formattedData = data.map((item, index) => ({
    ...item,
    name: formatPPE(item.ppe_type),
    color: barColors[index % barColors.length],
  }));

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.1 }}
    >
      <Card variant="glass" hover glow="warning" className="overflow-hidden">
        <CardHeader className="pb-2">
          <CardTitle className="text-lg flex items-center gap-3">
            <motion.div
              whileHover={{ scale: 1.1, rotate: 5 }}
              className="flex items-center justify-center w-10 h-10 rounded-xl bg-warning/10"
            >
              <BarChart3 className="w-5 h-5 text-warning" />
            </motion.div>
            <div>
              <span>{title}</span>
              <p className="text-xs font-normal text-muted-foreground mt-0.5">
                Distribution of missing PPE items
              </p>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent className="pt-4">
          <ResponsiveContainer width="100%" height={280}>
            <BarChart
              data={formattedData}
              layout="vertical"
              barCategoryGap="25%"
              margin={{ top: 5, right: 30, left: 10, bottom: 5 }}
            >
              <defs>
                {formattedData.map((entry, index) => (
                  <linearGradient
                    key={`gradient-${index}`}
                    id={`barGradient${index}`}
                    x1="0"
                    y1="0"
                    x2="1"
                    y2="0"
                  >
                    <stop offset="0%" stopColor={entry.color.main} stopOpacity={1} />
                    <stop offset="100%" stopColor={entry.color.light} stopOpacity={0.8} />
                  </linearGradient>
                ))}
              </defs>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="currentColor"
                strokeOpacity={0.08}
                horizontal={false}
              />
              <XAxis
                type="number"
                stroke="currentColor"
                strokeOpacity={0.2}
                tick={{ fill: "currentColor", opacity: 0.5, fontSize: 11 }}
                tickLine={false}
                axisLine={false}
                allowDecimals={false}
              />
              <YAxis
                dataKey="name"
                type="category"
                width={90}
                stroke="currentColor"
                strokeOpacity={0.2}
                tick={{ fill: "currentColor", opacity: 0.7, fontSize: 12 }}
                tickLine={false}
                axisLine={false}
              />
              <Tooltip
                content={
                  <CustomTooltip
                    formatter={(v) => `${v}`}
                    labelFormatter={(label) => String(label)}
                  />
                }
                cursor={{ fill: "currentColor", opacity: 0.05 }}
              />
              <Bar
                dataKey="count"
                radius={[0, 8, 8, 0]}
                maxBarSize={40}
              >
                {formattedData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={`url(#barGradient${index})`}
                    className="transition-all duration-200 hover:opacity-80"
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </motion.div>
  );
}

function formatPPE(ppe: string): string {
  return ppe
    .replace("safety ", "")
    .replace("protective ", "")
    .replace(/_/g, " ")
    .split(" ")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}
