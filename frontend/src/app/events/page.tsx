"use client";

import { EventsTable } from "@/components/events-table";
import { DemoBanner } from "@/components/demo-banner";
import { PageHeader, StatusBadge } from "@/components/page-header";
import { PageLoader } from "@/components/page-loader";
import { useRecentViolations } from "@/lib/queries";
import { Clock } from "lucide-react";

export default function EventsPage() {
  const { data: events, isLoading, error, refetch } = useRecentViolations(50);
  if (isLoading) return <PageLoader />;
  return <div>
    <DemoBanner />
    <PageHeader title="The event ledger." description="Review recent reported violations, their duration, and the protective equipment recorded with each event." action={<StatusBadge icon={<Clock size={13} />} label="Recent records" value={events?.length ?? "—"} />} />
    {error && <div className="console-query-error" role="alert">Event records could not be loaded.<button type="button" onClick={() => refetch()}>Retry</button></div>}
    <section className="console-panel console-paper" aria-label="Violation records">
      <div className="console-panel-heading"><div><h2>Violation records</h2><p>LATEST 50 / REPORTED EVENT STATE</p></div><span className="console-source-tag">Chronological record</span></div>
      <div className="p-4 sm:p-5">{events?.length ? <EventsTable events={events} /> : <p className="py-14 text-center text-muted-foreground">{error ? "Records unavailable." : "No recent violation records were returned."}</p>}</div>
    </section>
    <p className="console-info-line">An ongoing flag describes the stored event state. A record count does not establish current safety or model accuracy.</p>
  </div>;
}
