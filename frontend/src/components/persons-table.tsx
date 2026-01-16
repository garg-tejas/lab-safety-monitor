"use client";

import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Person } from "@/lib/api";
import api from "@/lib/api";
import { Edit2, Check, X } from "lucide-react";

interface PersonsTableProps {
  persons: Person[];
  loading?: boolean;
  onUpdate?: () => void;
}

export function PersonsTable({ persons, loading = false, onUpdate }: PersonsTableProps) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState<string>("");
  const [saving, setSaving] = useState(false);

  if (loading) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        Loading persons...
      </div>
    );
  }

  if (persons.length === 0) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        No persons tracked yet
      </div>
    );
  }

  const handleStartEdit = (person: Person) => {
    setEditingId(person.id);
    setEditName(person.name || "");
  };

  const handleCancelEdit = () => {
    setEditingId(null);
    setEditName("");
  };

  const handleSaveEdit = async (personId: string) => {
    setSaving(true);
    try {
      await api.updatePerson(personId, editName.trim() || null);
      setEditingId(null);
      setEditName("");
      if (onUpdate) {
        onUpdate();
      }
    } catch (error) {
      console.error("Failed to update person:", error);
      alert("Failed to update person name");
    } finally {
      setSaving(false);
    }
  };

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Name</TableHead>
          <TableHead>ID</TableHead>
          <TableHead>First Seen</TableHead>
          <TableHead>Last Seen</TableHead>
          <TableHead>Events</TableHead>
          <TableHead>Violations</TableHead>
          <TableHead>Compliance Rate</TableHead>
          <TableHead>Actions</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {persons.map((person) => (
          <TableRow key={person.id}>
            <TableCell className="font-medium">
              {editingId === person.id ? (
                <input
                  type="text"
                  value={editName}
                  onChange={(e) => setEditName(e.target.value)}
                  placeholder="Enter name"
                  className="w-32 px-2 py-1 border rounded text-sm"
                  disabled={saving}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      handleSaveEdit(person.id);
                    } else if (e.key === "Escape") {
                      handleCancelEdit();
                    }
                  }}
                />
              ) : (
                person.name || <span className="text-muted-foreground">Unnamed</span>
              )}
            </TableCell>
            <TableCell className="font-mono text-xs">
              {person.id.substring(0, 8)}...
            </TableCell>
            <TableCell className="text-sm">
              {new Date(person.first_seen).toLocaleString()}
            </TableCell>
            <TableCell className="text-sm">
              {new Date(person.last_seen).toLocaleString()}
            </TableCell>
            <TableCell>{person.total_events}</TableCell>
            <TableCell>
              <Badge variant={person.violation_count > 0 ? "destructive" : "default"}>
                {person.violation_count}
              </Badge>
            </TableCell>
            <TableCell>
              <Badge
                variant={
                  person.compliance_rate >= 90
                    ? "default"
                    : person.compliance_rate >= 70
                      ? "secondary"
                      : "destructive"
                }
              >
                {person.compliance_rate.toFixed(1)}%
              </Badge>
            </TableCell>
            <TableCell>
              {editingId === person.id ? (
                <div className="flex gap-1">
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => handleSaveEdit(person.id)}
                    disabled={saving}
                  >
                    <Check className="w-4 h-4" />
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={handleCancelEdit}
                    disabled={saving}
                  >
                    <X className="w-4 h-4" />
                  </Button>
                </div>
              ) : (
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => handleStartEdit(person)}
                >
                  <Edit2 className="w-4 h-4" />
                </Button>
              )}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
