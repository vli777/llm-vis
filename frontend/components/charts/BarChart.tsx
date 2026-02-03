"use client";
import {
  BarChart as ReBarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import type { ChartSpec } from "@/types/chart";

const COLORS = [
  "#60a5fa", "#f472b6", "#34d399", "#fbbf24", "#a78bfa",
  "#fb923c", "#22d3ee", "#e879f9", "#4ade80", "#f87171",
];

export default function BarChartView({ spec }: { spec: ChartSpec }) {
  const data = spec.data_inline || [];
  if (!data.length) return <div className="text-sm text-slate-500">No data</div>;

  const xField = spec.encoding.x?.field || Object.keys(data[0])[0];
  const yField = spec.encoding.y?.field || Object.keys(data[0])[1];
  const colorField = spec.encoding.color?.field;

  const isHorizontal = spec.options.orientation === "horizontal";

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ReBarChart
        data={data}
        layout={isHorizontal ? "vertical" : "horizontal"}
        margin={{ top: 5, right: 20, left: 20, bottom: 40 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
        {isHorizontal ? (
          <>
            <YAxis
              dataKey={xField}
              type="category"
              tick={{ fill: "#94a3b8", fontSize: 12 }}
              width={120}
            />
            <XAxis
              type="number"
              tick={{ fill: "#94a3b8", fontSize: 12 }}
            />
          </>
        ) : (
          <>
            <XAxis
              dataKey={xField}
              tick={{ fill: "#94a3b8", fontSize: 12 }}
              angle={-35}
              textAnchor="end"
              height={60}
              interval={data.length > 20 ? Math.floor(data.length / 15) : 0}
            />
            <YAxis tick={{ fill: "#94a3b8", fontSize: 12 }} />
          </>
        )}
        <Tooltip
          contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8 }}
          labelStyle={{ color: "#e2e8f0" }}
          itemStyle={{ color: "#94a3b8" }}
        />
        <Bar dataKey={yField} radius={[4, 4, 0, 0]}>
          {data.map((_entry, index) => (
            <Cell
              key={index}
              fill={colorField ? COLORS[index % COLORS.length] : COLORS[0]}
            />
          ))}
        </Bar>
      </ReBarChart>
    </ResponsiveContainer>
  );
}
