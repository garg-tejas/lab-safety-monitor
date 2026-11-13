"use client";

import Link from "next/link";
import { ArrowUpRight, Clock } from "lucide-react";

export interface LedgerEvent {
  id: string;
  timestamp: string;
  person: string;
  description?: string;
  missingPpe: string[];
  ongoing?: boolean;
}

export function EventLedger({ events, demonstration, title = "Recent events", emptyMessage = "No events in this view. Start a feed to receive safety alerts.", footer }: { events: LedgerEvent[]; demonstration: boolean; title?: string; emptyMessage?: string; footer?: string }) {
  return (
    <section className="console-panel console-paper console-ledger" aria-label={title}>
      <div className="console-panel-heading">
        <div><h2>{title}</h2><p>{demonstration ? "SIMULATED RECORDS" : "REPORTED EVENTS"}</p></div>
        <span className="console-source-tag">{String(events.length).padStart(2, "0")} shown</span>
      </div>
      {events.length ? <ol className="console-event-list">
        {events.map((event) => <li key={event.id} className="console-event">
          <time dateTime={event.timestamp}>{new Date(event.timestamp).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit" })}{event.ongoing ? " / ONGOING" : ""}</time>
          <strong>{event.person}</strong>
          {event.description && <p>{event.description}</p>}
          <div className="console-event-tags">{event.missingPpe.map((ppe) => <span key={ppe}>{ppe.replaceAll("_", " ")}</span>)}</div>
        </li>)}
      </ol> : <div className="console-empty-events"><Clock size={25} strokeWidth={1.3} /><p>{emptyMessage}</p></div>}
      <div className="console-ledger-footer">
        {footer || "Review the event record before drawing a safety conclusion."}
        <Link href="/events">Open event ledger <ArrowUpRight size={12} className="inline" /></Link>
      </div>
    </section>
  );
}
