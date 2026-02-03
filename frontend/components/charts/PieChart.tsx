"use client";
import {
  PieChart as RePieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import type { ChartSpec } from "@/types/chart";

const COLORS = [
  "#60a5fa", "#f472b6", "#34d399", "#fbbf24", "#a78bfa",
  "#fb923c", "#22d3ee", "#e879f9", "#4ade80", "#f87171",
  "#818cf8", "#2dd4bf", "#f97316", "#06b6d4", "#d946ef",
];

export default function PieChartView({ spec }: { spec: ChartSpec }) {
  const data = spec.data_inline || [];
  if (!data.length) return <div className="text-sm text-slate-500">No data</div>;

  // Determine name and value fields
  const colorField = spec.encoding.color?.field;
  const thetaField = spec.encoding.theta?.field;

  const keys = Object.keys(data[0]);
  const nameField = colorField || keys.find((k) => typeof data[0][k] === "string") || keys[0];
  const valueField =
    thetaField ||
    keys.find((k) => k !== nameField && typeof data[0][k] === "number") ||
    keys[1];

  return (
    <ResponsiveContainer width="100%" height="100%">
      <RePieChart>
        <Pie
          data={data}
          dataKey={valueField}
          nameKey={nameField}
          cx="50%"
          cy="50%"
          outerRadius="70%"
          innerRadius="35%"
          paddingAngle={2}
          label={({ name, percent }) =>
            `${String(name).slice(0, 15)} ${(percent * 100).toFixed(0)}%`
          }
          labelLine={{ stroke: "#64748b" }}
        >
          {data.map((_entry, index) => (
            <Cell key={index} fill={COLORS[index % COLORS.length]} />
          ))}
        </Pie>
        <Tooltip
          contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8 }}
          labelStyle={{ color: "#e2e8f0" }}
          itemStyle={{ color: "#94a3b8" }}
        />
        <Legend
          wrapperStyle={{ color: "#94a3b8", fontSize: 12 }}
          layout="horizontal"
          align="center"
        />
      </RePieChart>
    </ResponsiveContainer>
  );
}
