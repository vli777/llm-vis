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

export default function BarChartView({ spec }: { spec: ChartSpec }) {
  const data = spec.data_inline || [];
  if (!data.length) return <div className="text-sm theme-muted">No data</div>;

  const xField = spec.encoding.x?.field || Object.keys(data[0])[0];
  const yField = spec.encoding.y?.field || Object.keys(data[0])[1];
  const colorField = spec.encoding.color?.field;

  const isHorizontal = spec.options.orientation === "horizontal";

  return (
    <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={220}>
      <ReBarChart
        data={data}
        layout={isHorizontal ? "vertical" : "horizontal"}
        margin={{ top: 15, right: 20, left: 40, bottom: 90 }}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
        {isHorizontal ? (
          <>
            <YAxis
              dataKey={xField}
              type="category"
              tick={{ fill: "var(--color-muted)", fontSize: 12 }}
              width={120}
            />
            <XAxis
              type="number"
              tick={{ fill: "var(--color-muted)", fontSize: 12 }}
            />
          </>
        ) : (
          <>
            <XAxis
              dataKey={xField}
              tick={{ fill: "var(--color-muted)", fontSize: 12 }}
              angle={-35}
              textAnchor="end"
              height={60}
              interval={data.length > 20 ? Math.floor(data.length / 15) : 0}
            />
            <YAxis tick={{ fill: "var(--color-muted)", fontSize: 12 }} />
          </>
        )}
        <Tooltip
          contentStyle={{
            background: "var(--color-surface)",
            border: "1px solid var(--color-border)",
            borderRadius: 8,
          }}
          labelStyle={{ color: "var(--color-text)" }}
          itemStyle={{ color: "var(--color-text)" }}
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
