"use client";
import {
  ScatterChart as ReScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  ZAxis,
} from "recharts";
import type { ChartSpec } from "@/types/chart";

const COLORS = [
  "#60a5fa", "#f472b6", "#34d399", "#fbbf24", "#a78bfa",
  "#fb923c", "#22d3ee", "#e879f9", "#4ade80", "#f87171",
];

export default function ScatterChartView({ spec }: { spec: ChartSpec }) {
  const data = spec.data_inline || [];
  if (!data.length) return <div className="text-sm text-slate-500">No data</div>;

  const xField = spec.encoding.x?.field || Object.keys(data[0])[0];
  const yField = spec.encoding.y?.field || Object.keys(data[0])[1];
  const colorField = spec.encoding.color?.field;

  // Group data by color field if present
  const groups: Map<string, Record<string, any>[]> = new Map();
  if (colorField) {
    data.forEach((d) => {
      const key = String(d[colorField] ?? "other");
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key)!.push(d);
    });
  } else {
    groups.set("all", data);
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ReScatterChart margin={{ top: 5, right: 20, left: 20, bottom: 40 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
        <XAxis
          dataKey={xField}
          type="number"
          name={xField}
          tick={{ fill: "#94a3b8", fontSize: 12 }}
          label={{ value: xField, position: "insideBottom", offset: -10, fill: "#94a3b8" }}
        />
        <YAxis
          dataKey={yField}
          type="number"
          name={yField}
          tick={{ fill: "#94a3b8", fontSize: 12 }}
          label={{ value: yField, angle: -90, position: "insideLeft", fill: "#94a3b8" }}
        />
        <ZAxis range={[30, 120]} />
        <Tooltip
          contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8 }}
          labelStyle={{ color: "#e2e8f0" }}
          cursor={{ strokeDasharray: "3 3" }}
        />
        {colorField && groups.size > 1 && (
          <Legend wrapperStyle={{ color: "#94a3b8" }} />
        )}
        {[...groups.entries()].map(([name, gData], i) => (
          <Scatter
            key={name}
            name={name === "all" ? undefined : name}
            data={gData}
            fill={COLORS[i % COLORS.length]}
            opacity={0.7}
          />
        ))}
      </ReScatterChart>
    </ResponsiveContainer>
  );
}
