"use client";

import { motion } from "framer-motion";
import { PersonsTable } from "@/components/persons-table";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PageHeader, StatusBadge } from "@/components/page-header";
import { MotionWrapper } from "@/components/motion-wrapper";
import { PageLoader } from "@/components/page-loader";
import { usePersons } from "@/lib/queries";
import { useQueryClient } from "@tanstack/react-query";
import { queryKeys } from "@/lib/queries";
import { Users, UserSearch } from "lucide-react";

export default function PersonsPage() {
  const { data, isLoading } = usePersons(1, 100);
  const queryClient = useQueryClient();

  const handleUpdate = () => {
    queryClient.invalidateQueries({ queryKey: queryKeys.persons.all });
  };

  if (isLoading) {
    return <PageLoader />;
  }

  const persons = data?.persons ?? [];

  return (
    <MotionWrapper className="space-y-8">
      {/* Header */}
      <PageHeader
        title="Persons Directory"
        description="Track and manage detected individuals"
        action={
          <StatusBadge
            icon={<Users className="w-4 h-4" />}
            label="Tracked"
            value={`${persons.length} individuals`}
            variant="info"
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
                className="flex items-center justify-center w-10 h-10 rounded-xl bg-info/10"
              >
                <Users className="w-5 h-5 text-info" />
              </motion.div>
              <div>
                <span>Tracked Individuals</span>
                <p className="text-xs font-normal text-muted-foreground mt-0.5">
                  Click on a name to edit, view compliance history
                </p>
              </div>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {persons.length === 0 ? (
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
                  <UserSearch className="w-8 h-8 text-muted-foreground" />
                </motion.div>
                <p className="font-medium text-foreground">No persons tracked yet</p>
                <p className="text-sm text-muted-foreground mt-1">
                  Individuals will appear here once detected by the system
                </p>
              </motion.div>
            ) : (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.4 }}
              >
                <PersonsTable
                  persons={persons}
                  loading={isLoading}
                  onUpdate={handleUpdate}
                />
              </motion.div>
            )}
          </CardContent>
        </Card>
      </motion.div>
    </MotionWrapper>
  );
}
