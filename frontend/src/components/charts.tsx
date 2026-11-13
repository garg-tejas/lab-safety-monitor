"use client";

import { AreaChart, Area, BarChart, Bar, CartesianGrid, XAxis, YAxis, ResponsiveContainer, Tooltip } from "recharts";

interface TimelineData { date: string; violations: number }
interface PPEBreakdown { ppe_type: string; count: number }
const tooltipStyle = { background: "var(--popover)", color: "var(--popover-foreground)", border: "1px solid var(--border)", borderRadius: 4, fontSize: 12 };
const axisStyle = { fill: "var(--muted-foreground)", fontSize: 10 };

export function ViolationTimelineChart({ data, title = "Violation timeline" }: { data: TimelineData[]; title?: string }) {
  return <section className="console-chart" aria-label={title}>
    <h3>{title}</h3><p>Daily event counts · latest seven-day query</p>
    {data.length ? <><div aria-hidden="true"><ResponsiveContainer width="100%" height={230} minWidth={0}>
      <AreaChart data={data} margin={{ top: 8, right: 8, left: -24, bottom: 0 }}>
        <CartesianGrid vertical={false} stroke="var(--border)" strokeDasharray="2 4" />
        <XAxis dataKey="date" tickFormatter={(value) => new Date(value).toLocaleDateString("en-GB", { day: "numeric", month: "short" })} tick={axisStyle} tickLine={false} axisLine={false} dy={9} minTickGap={22} />
        <YAxis allowDecimals={false} tick={axisStyle} axisLine={false} tickLine={false} />
        <Tooltip contentStyle={tooltipStyle} labelFormatter={(value) => String(value)} />
        <Area dataKey="violations" type="linear" stroke="var(--primary)" fill="var(--primary)" fillOpacity={.08} strokeWidth={2} isAnimationActive={false} />
      </AreaChart>
    </ResponsiveContainer></div><table className="sr-only"><caption>Daily violation counts</caption><thead><tr><th>Date</th><th>Violations</th></tr></thead><tbody>{data.map((row) => <tr key={row.date}><td>{row.date}</td><td>{row.violations}</td></tr>)}</tbody></table></> : <div className="console-chart-empty">No timeline records available.</div>}
  </section>;
}

export function PPEBreakdownChart({ data, title = "Missing protective equipment" }: { data: PPEBreakdown[]; title?: string }) {
  return <section className="console-chart" aria-label={title}>
    <h3>{title}</h3><p>Reported missing-item counts · an event can contain multiple items</p>
    {data.length ? <><div aria-hidden="true"><ResponsiveContainer width="100%" height={230} minWidth={0}>
      <BarChart data={data} layout="vertical" margin={{ left: 0, right: 25, top: 0, bottom: 0 }}>
        <CartesianGrid horizontal={false} stroke="var(--border)" strokeDasharray="2 4" />
        <XAxis type="number" allowDecimals={false} tick={axisStyle} axisLine={false} tickLine={false} />
        <YAxis type="category" dataKey="ppe_type" width={80} tick={axisStyle} tickFormatter={(value) => String(value).replaceAll("_", " ")} axisLine={false} tickLine={false} />
        <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "var(--muted)" }} />
        <Bar dataKey="count" fill="var(--warning)" barSize={10} radius={[0, 2, 2, 0]} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer></div><table className="sr-only"><caption>Missing protective equipment counts</caption><thead><tr><th>Equipment</th><th>Count</th></tr></thead><tbody>{data.map((row) => <tr key={row.ppe_type}><td>{row.ppe_type}</td><td>{row.count}</td></tr>)}</tbody></table></> : <div className="console-chart-empty">No equipment records available.</div>}
  </section>;
}
