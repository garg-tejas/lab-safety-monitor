"use client";

import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ComplianceEvent } from "@/lib/api";

interface EventsTableProps {
  events: ComplianceEvent[];
  loading?: boolean;
}

export function EventsTable({ events, loading = false }: EventsTableProps) {
  if (loading) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        Loading events...
      </div>
    );
  }

  if (events.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        No events found
      </div>
    );
  }

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Time</TableHead>
          <TableHead>Person</TableHead>
          <TableHead>Status</TableHead>
          <TableHead>Duration</TableHead>
          <TableHead>Detected PPE</TableHead>
          <TableHead>Missing PPE</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {events.map((event) => (
          <TableRow key={event.id}>
            <TableCell className="font-medium">
              {new Date(event.timestamp).toLocaleString()}
            </TableCell>
            <TableCell>{event.person_id || "Unknown"}</TableCell>
            <TableCell>
              <div className="flex flex-col gap-1">
                <Badge variant={event.is_violation ? "destructive" : "default"}>
                  {event.is_violation ? "Violation" : "Compliant"}
                </Badge>
                {event.is_ongoing !== undefined && event.is_violation && (
                  <Badge variant={event.is_ongoing ? "secondary" : "outline"} className="text-xs">
                    {event.is_ongoing ? "Ongoing" : "Resolved"}
                  </Badge>
                )}
              </div>
            </TableCell>
            <TableCell>
              {event.duration_frames && event.duration_frames > 1 ? (
                <span className="text-sm text-muted-foreground">
                  {event.duration_frames} frames
                  {event.start_frame && event.end_frame && (
                    <span className="block text-xs">
                      ({event.start_frame} - {event.end_frame})
                    </span>
                  )}
                </span>
              ) : (
                <span className="text-sm text-muted-foreground">-</span>
              )}
            </TableCell>
            <TableCell>
              <div className="flex flex-wrap gap-1">
                {event.detected_ppe.length > 0 ? (
                  event.detected_ppe.map((ppe) => (
                    <Badge key={ppe} variant="outline" className="text-xs">
                      {formatPPE(ppe)}
                    </Badge>
                  ))
                ) : (
                  <span className="text-muted-foreground text-sm">None</span>
                )}
              </div>
            </TableCell>
            <TableCell>
              <div className="flex flex-wrap gap-1">
                {event.missing_ppe.length > 0 ? (
                  event.missing_ppe.map((ppe) => (
                    <Badge key={ppe} variant="destructive" className="text-xs">
                      {formatPPE(ppe)}
                    </Badge>
                  ))
                ) : (
                  <span className="text-green-600 text-sm">All present</span>
                )}
              </div>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

function formatPPE(ppe: string): string {
  return ppe
    .replace("safety ", "")
    .replace("protective ", "")
    .split(" ")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}
