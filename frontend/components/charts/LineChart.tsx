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
  "#60a5fa", "#f472b6", "#34d399", "#fbbf24", "#a78bfa",
];

export default function LineChartView({ spec }: { spec: ChartSpec }) {
  const data = spec.data_inline || [];
  if (!data.length) return <div className="text-sm text-slate-500">No data</div>;

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
    <ResponsiveContainer width="100%" height="100%">
      <ReLineChart data={data} margin={{ top: 5, right: 20, left: 20, bottom: 40 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
        <XAxis
          dataKey={xField}
          tick={{ fill: "#94a3b8", fontSize: 12 }}
          angle={-35}
          textAnchor="end"
          height={60}
        />
        <YAxis tick={{ fill: "#94a3b8", fontSize: 12 }} />
        <Tooltip
          contentStyle={{ background: "#1e293b", border: "1px solid #334155", borderRadius: 8 }}
          labelStyle={{ color: "#e2e8f0" }}
          itemStyle={{ color: "#94a3b8" }}
        />
        {groups.size > 0 ? (
          <>
            <Legend wrapperStyle={{ color: "#94a3b8" }} />
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
