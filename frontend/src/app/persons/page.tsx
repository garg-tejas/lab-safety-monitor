"use client";

import { PersonsTable } from "@/components/persons-table";
import { DemoBanner } from "@/components/demo-banner";
import { PageHeader, StatusBadge } from "@/components/page-header";
import { PageLoader } from "@/components/page-loader";
import { usePersons, queryKeys } from "@/lib/queries";
import { useDemoMode } from "@/providers/demo-context";
import { useQueryClient } from "@tanstack/react-query";
import { Users } from "lucide-react";

export default function PersonsPage() {
  const { data, isLoading, error, refetch } = usePersons(1, 100);
  const { isDemoMode } = useDemoMode();
  const queryClient = useQueryClient();
  if (isLoading) return <PageLoader />;
  const persons = data?.persons || [];
  return <div>
    <DemoBanner />
    <PageHeader title="People in the record." description="Review tracked identities and their associated event history. A tracking identity may require human confirmation." action={<StatusBadge icon={<Users size={13} />} label="Shown" value={error ? "—" : persons.length} />} />
    {error && <div className="console-query-error" role="alert">People records could not be loaded.<button type="button" onClick={() => refetch()}>Retry</button></div>}
    <section className="console-panel console-paper" aria-label="People directory">
      <div className="console-panel-heading"><div><h2>Identity directory</h2><p>FIRST 100 / EVENT ASSOCIATIONS</p></div><span className="console-source-tag">{isDemoMode ? "Read-only samples" : "Names can be edited"}</span></div>
      <div className="p-4 sm:p-5">{persons.length ? <PersonsTable persons={persons} onUpdate={() => { queryClient.invalidateQueries({ queryKey: queryKeys.persons.all }); }} /> : <p className="py-14 text-center text-muted-foreground">{error ? "Records unavailable." : "No identities have been returned."}</p>}</div>
    </section>
    <p className="console-info-line">{isDemoMode ? "DEMONSTRATION / Sample names are read only. No person updates are sent to the backend." : "Names are labels for review; they do not establish verified identity."}</p>
  </div>;
}
