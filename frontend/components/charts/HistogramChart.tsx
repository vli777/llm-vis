"use client";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { ChartSpec } from "@/types/chart";

export default function HistogramChart({ spec }: { spec: ChartSpec }) {
  const data = spec.data_inline || [];
  if (!data.length) return <div className="text-sm theme-muted">No data</div>;

  return (
    <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={220}>
      <BarChart data={data} margin={{ top: 15, right: 20, left: 40, bottom: 90 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
        <XAxis
          dataKey="bin_label"
          tick={{ fill: "var(--color-muted)", fontSize: 11 }}
          angle={-35}
          textAnchor="end"
          height={60}
          interval={data.length > 15 ? Math.floor(data.length / 10) : 0}
        />
        <YAxis tick={{ fill: "var(--color-muted)", fontSize: 12 }} />
        <Tooltip
          contentStyle={{
            background: "var(--color-surface)",
            border: "1px solid var(--color-border)",
            borderRadius: 8,
          }}
          labelStyle={{ color: "var(--color-text)" }}
          itemStyle={{ color: "var(--color-text)" }}
          formatter={(value: any) => [value, "Count"]}
        />
        <Bar dataKey="count" fill="#636EFA" radius={[2, 2, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}
