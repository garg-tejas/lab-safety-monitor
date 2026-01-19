"use client";

import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ComplianceEvent } from "@/lib/api";
import { Clock, User, AlertTriangle, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface EventsTableProps {
  events: ComplianceEvent[];
  loading?: boolean;
}

export function EventsTable({ events, loading = false }: EventsTableProps) {
  if (loading) {
    return (
      <div className="space-y-3">
        {[...Array(5)].map((_, i) => (
          <div
            key={i}
            className="flex items-center gap-4 p-4 rounded-xl border border-border/50"
          >
            <Skeleton className="h-10 w-10 rounded-xl shimmer" />
            <div className="flex-1 space-y-2">
              <Skeleton className="h-4 w-48 shimmer" />
              <Skeleton className="h-3 w-32 shimmer" />
            </div>
            <Skeleton className="h-6 w-20 rounded-full shimmer" />
          </div>
        ))}
      </div>
    );
  }

  if (events.length === 0) {
    return null;
  }

  return (
    <div className="rounded-xl border border-border/50 overflow-hidden">
      <Table>
        <TableHeader>
          <TableRow className="bg-muted/30 hover:bg-muted/30">
            <TableHead className="font-semibold">Time</TableHead>
            <TableHead className="font-semibold">Person</TableHead>
            <TableHead className="font-semibold">Status</TableHead>
            <TableHead className="font-semibold">Duration</TableHead>
            <TableHead className="font-semibold">Detected PPE</TableHead>
            <TableHead className="font-semibold">Violations</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {events.map((event) => (
            <TableRow
              key={event.id}
              className={cn(
                "transition-colors hover:bg-muted/50",
                event.is_violation && "hover:bg-danger/5"
              )}
            >
              <TableCell>
                <div className="flex items-center gap-2">
                  <div
                    className={cn(
                      "flex items-center justify-center w-8 h-8 rounded-lg",
                      event.is_violation ? "bg-danger/10" : "bg-success/10"
                    )}
                  >
                    <Clock
                      className={cn(
                        "w-4 h-4",
                        event.is_violation ? "text-danger" : "text-success"
                      )}
                    />
                  </div>
                  <div>
                    <p className="font-medium text-sm">
                      {new Date(event.timestamp).toLocaleTimeString()}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {new Date(event.timestamp).toLocaleDateString()}
                    </p>
                  </div>
                </div>
              </TableCell>
              <TableCell>
                <div className="flex items-center gap-2">
                  <div className="flex items-center justify-center w-7 h-7 rounded-full bg-muted">
                    <User className="w-3.5 h-3.5 text-muted-foreground" />
                  </div>
                  <span className="font-medium text-sm">
                    {event.person_id || "Unknown"}
                  </span>
                </div>
              </TableCell>
              <TableCell>
                <div className="flex flex-col gap-1.5">
                  <Badge
                    variant={event.is_violation ? "danger" : "success"}
                    className="w-fit"
                  >
                    {event.is_violation ? (
                      <span className="flex items-center gap-1">
                        <AlertTriangle className="w-3 h-3" />
                        Violation
                      </span>
                    ) : (
                      <span className="flex items-center gap-1">
                        <CheckCircle2 className="w-3 h-3" />
                        Compliant
                      </span>
                    )}
                  </Badge>
                  {event.is_ongoing !== undefined && event.is_violation && (
                    <Badge
                      variant={event.is_ongoing ? "warning-soft" : "secondary"}
                      className="text-[10px] w-fit"
                    >
                      {event.is_ongoing ? "Ongoing" : "Resolved"}
                    </Badge>
                  )}
                </div>
              </TableCell>
              <TableCell>
                {event.duration_frames && event.duration_frames > 1 ? (
                  <div className="text-sm">
                    <span className="font-medium">{event.duration_frames}</span>
                    <span className="text-muted-foreground ml-1">frames</span>
                    {event.start_frame && event.end_frame && (
                      <p className="text-xs text-muted-foreground mt-0.5">
                        {event.start_frame} - {event.end_frame}
                      </p>
                    )}
                  </div>
                ) : (
                  <span className="text-sm text-muted-foreground">-</span>
                )}
              </TableCell>
              <TableCell>
                <div className="flex flex-wrap gap-1.5 max-w-[200px]">
                  {event.detected_ppe.length > 0 ? (
                    event.detected_ppe.map((ppe) => (
                      <Badge
                        key={ppe}
                        variant="success-soft"
                        className="text-[10px]"
                      >
                        {formatPPE(ppe)}
                      </Badge>
                    ))
                  ) : (
                    <span className="text-muted-foreground text-sm">None</span>
                  )}
                </div>
              </TableCell>
              <TableCell>
                <div className="flex flex-wrap gap-1.5 max-w-[200px]">
                  {/* Missing PPE violations */}
                  {event.missing_ppe.length > 0 &&
                    event.missing_ppe.map((ppe) => (
                      <Badge
                        key={ppe}
                        variant="danger-soft"
                        className="text-[10px]"
                      >
                        No {formatPPE(ppe)}
                      </Badge>
                    ))}
                  {/* Action violations (Drinking/Eating) */}
                  {event.action_violations &&
                    event.action_violations.length > 0 &&
                    event.action_violations.map((action) => (
                      <Badge
                        key={action}
                        variant="warning-soft"
                        className="text-[10px]"
                      >
                        {formatAction(action)}
                      </Badge>
                    ))}
                  {/* Show compliant if no violations */}
                  {event.missing_ppe.length === 0 &&
                    (!event.action_violations ||
                      event.action_violations.length === 0) && (
                      <Badge variant="success-soft" className="text-[10px]">
                        All Clear
                      </Badge>
                    )}
                </div>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

function formatPPE(ppe: string): string {
  return ppe
    .replace("safety ", "")
    .replace("protective ", "")
    .replace("_", " ")
    .split(" ")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function formatAction(action: string): string {
  // Capitalize first letter of action (e.g., "drinking" -> "Drinking")
  return action.charAt(0).toUpperCase() + action.slice(1);
}
