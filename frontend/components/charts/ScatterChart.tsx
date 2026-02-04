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

export default function ScatterChartView({ spec }: { spec: ChartSpec }) {
  const data = spec.data_inline || [];
  if (!data.length) return <div className="text-sm theme-muted">No data</div>;

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
    <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={220}>
      <ReScatterChart margin={{ top: 15, right: 20, left: 40, bottom: 110 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
        <XAxis
          dataKey={xField}
          type="number"
          name={xField}
          tick={{ fill: "var(--color-muted)", fontSize: 12 }}
          label={{ value: xField, position: "insideBottom", offset: -10, fill: "var(--color-muted)" }}
        />
        <YAxis
          dataKey={yField}
          type="number"
          name={yField}
          tick={{ fill: "var(--color-muted)", fontSize: 12 }}
          label={{ value: yField, angle: -90, position: "insideLeft", fill: "var(--color-muted)" }}
        />
        <ZAxis range={[30, 120]} />
        <Tooltip
          contentStyle={{
            background: "var(--color-surface)",
            border: "1px solid var(--color-border)",
            borderRadius: 8,
          }}
          labelStyle={{ color: "var(--color-text)" }}
          itemStyle={{ color: "var(--color-text)" }}
          cursor={{ strokeDasharray: "3 3" }}
        />
        {colorField && groups.size > 1 && (
          <Legend
            wrapperStyle={{ color: "var(--color-muted)", paddingTop: 16, fontSize: 10, opacity: 0.7 }}
            verticalAlign="bottom"
            align="center"
            height={36}
          />
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
