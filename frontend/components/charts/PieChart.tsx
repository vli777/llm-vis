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
  "#636EFA",
  "#EF553B",
  "#00CC96",
  "#AB63FA",
  "#FFA15A",
  "#19D3F3",
  "#FF6692",
  "#B6E880",
  "#FF97FF",
  "#FECB52",
];

export default function PieChartView({ spec }: { spec: ChartSpec }) {
  const data = spec.data_inline || [];
  if (!data.length) return <div className="text-sm theme-muted">No data</div>;

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
    <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={220}>
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
          contentStyle={{
            background: "var(--color-surface)",
            border: "1px solid var(--color-border)",
            borderRadius: 8,
          }}
          labelStyle={{ color: "var(--color-text)" }}
          itemStyle={{ color: "var(--color-text)" }}
        />
        <Legend
          wrapperStyle={{ color: "var(--color-muted)", fontSize: 10, paddingTop: 16, opacity: 0.7 }}
          layout="horizontal"
          align="center"
          verticalAlign="bottom"
          height={36}
        />
      </RePieChart>
    </ResponsiveContainer>
  );
}
