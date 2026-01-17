import React, { useState, useEffect } from 'react';
import { getStatistics } from '../services/api';
import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { format } from 'date-fns';

const COLORS = ['#10b981', '#ef4444', '#f59e0b', '#3b82f6'];

function Dashboard() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadStatistics();
    const interval = setInterval(loadStatistics, 5000); // Refresh every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const loadStatistics = async () => {
    try {
      const data = await getStatistics();
      setStats(data);
      setLoading(false);
    } catch (err) {
      setError('Failed to load statistics');
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">
        {error}
      </div>
    );
  }

  const violationData = [
    { name: 'Helmet', count: stats?.helmet_violations || 0 },
    { name: 'Shoes', count: stats?.shoes_violations || 0 },
    { name: 'Goggles', count: stats?.goggles_violations || 0 },
    { name: 'Mask', count: stats?.mask_violations || 0 },
  ];

  const complianceData = [
    { name: 'Compliant', value: stats?.total_compliances || 0 },
    { name: 'Violations', value: stats?.total_violations || 0 },
  ];

  return (
    <div className="space-y-6">
      <h2 className="text-3xl font-bold text-gray-800">Safety Compliance Dashboard</h2>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard
          title="Total Events"
          value={stats?.total_events || 0}
          icon="📊"
          color="bg-blue-500"
        />
        <StatsCard
          title="Compliance Rate"
          value={`${stats?.compliance_rate || 0}%`}
          icon="✅"
          color="bg-green-500"
        />
        <StatsCard
          title="Violations"
          value={stats?.total_violations || 0}
          icon="⚠️"
          color="bg-red-500"
        />
        <StatsCard
          title="Unique Persons"
          value={stats?.unique_persons || 0}
          icon="👥"
          color="bg-purple-500"
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Violation Types Bar Chart */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-xl font-semibold mb-4">PPE Violations by Type</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={violationData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="count" fill="#ef4444" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Compliance Pie Chart */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-xl font-semibold mb-4">Compliance Overview</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={complianceData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {complianceData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={index === 0 ? '#10b981' : '#ef4444'} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Last Updated */}
      <div className="bg-white rounded-lg shadow-md p-4">
        <p className="text-sm text-gray-600">
          Last updated: {stats?.last_updated ? format(new Date(stats.last_updated), 'PPP p') : 'Never'}
        </p>
      </div>
    </div>
  );
}

function StatsCard({ title, value, icon, color }) {
  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-gray-600 text-sm font-medium">{title}</p>
          <p className="text-3xl font-bold mt-2">{value}</p>
        </div>
        <div className={`${color} text-white text-4xl rounded-full w-16 h-16 flex items-center justify-center`}>
          {icon}
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
