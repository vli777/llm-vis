"use client";
import {
  LineChart as ReLineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
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

export default function LineChartView({ spec }: { spec: ChartSpec }) {
  const data = spec.data_inline || [];
  if (!data.length) return <div className="text-sm theme-muted">No data</div>;

  const xField = spec.encoding.x?.field || Object.keys(data[0])[0];
  const yField = spec.encoding.y?.field || Object.keys(data[0])[1];
  const colorField = spec.encoding.color?.field;

  // If color field is present, we need multiple lines
  const groups = new Set<string>();
  if (colorField) {
    data.forEach((d) => {
      if (d[colorField] != null) groups.add(String(d[colorField]));
    });
  }

  return (
    <ResponsiveContainer width="100%" height="100%" minWidth={0} minHeight={220}>
      <ReLineChart data={data} margin={{ top: 15, right: 20, left: 40, bottom: 110 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
        <XAxis
          dataKey={xField}
          tick={{ fill: "var(--color-muted)", fontSize: 12 }}
          angle={-35}
          textAnchor="end"
          height={60}
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
        />
        {groups.size > 0 ? (
          <>
            <Legend
              wrapperStyle={{ color: "var(--color-muted)", paddingTop: 16, fontSize: 10, opacity: 0.7 }}
              verticalAlign="bottom"
              align="center"
              height={36}
            />
            {[...groups].map((group, i) => (
              <Line
                key={group}
                dataKey={yField}
                data={data.filter((d) => String(d[colorField!]) === group)}
                name={group}
                stroke={COLORS[i % COLORS.length]}
                dot={false}
                strokeWidth={2}
              />
            ))}
          </>
        ) : (
          <Line
            dataKey={yField}
            stroke={COLORS[0]}
            dot={data.length <= 50}
            strokeWidth={2}
          />
        )}
      </ReLineChart>
    </ResponsiveContainer>
  );
}
