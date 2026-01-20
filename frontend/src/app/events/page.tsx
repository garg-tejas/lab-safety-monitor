"use client";

import { motion } from "framer-motion";
import { EventsTable } from "@/components/events-table";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PageHeader, StatusBadge } from "@/components/page-header";
import { MotionWrapper } from "@/components/motion-wrapper";
import { PageLoader } from "@/components/page-loader";
import { DemoBanner } from "@/components/demo-banner";
import { useRecentViolations } from "@/lib/queries";
import { History, FileSearch, Clock } from "lucide-react";

export default function EventsPage() {
  const { data: events, isLoading } = useRecentViolations(50);

  if (isLoading) {
    return <PageLoader />;
  }

  return (
    <MotionWrapper className="space-y-8">
      {/* Demo Mode Banner */}
      <DemoBanner />
      {/* Header */}
      <PageHeader
        title="Events Log"
        description="Complete compliance history and violation records"
        action={
          <StatusBadge
            icon={<Clock className="w-4 h-4" />}
            label="Showing"
            value={`${events?.length ?? 0} events`}
          />
        }
      />

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <Card variant="glass" hover>
          <CardHeader>
            <CardTitle className="flex items-center gap-3">
              <motion.div
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ delay: 0.3 }}
                className="flex items-center justify-center w-10 h-10 rounded-xl bg-primary/10"
              >
                <History className="w-5 h-5 text-primary" />
              </motion.div>
              <div>
                <span>Compliance History</span>
                <p className="text-xs font-normal text-muted-foreground mt-0.5">
                  Detailed log of all safety events and violations
                </p>
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {!events || events.length === 0 ? (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.3 }}
                className="flex flex-col items-center justify-center py-16 text-center"
              >
                <motion.div
                  animate={{ y: [0, -5, 0] }}
                  transition={{ duration: 2, repeat: Infinity }}
                  className="flex items-center justify-center w-16 h-16 rounded-2xl bg-muted mb-4"
                >
                  <FileSearch className="w-8 h-8 text-muted-foreground" />
                </motion.div>
                <p className="font-medium text-foreground">No events recorded</p>
                <p className="text-sm text-muted-foreground mt-1">
                  Compliance events will appear here once detected
                </p>
              </motion.div>
            ) : (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.4 }}
              >
                <EventsTable events={events} loading={isLoading} />
              </motion.div>
            )}
          </CardContent>
        </Card>
      </motion.div>
    </MotionWrapper>
  );
}
