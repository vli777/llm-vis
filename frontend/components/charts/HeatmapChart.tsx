"use client";
import type { ChartSpec } from "@/types/chart";

/**
 * Heatmap using SVG (Recharts doesn't have a built-in heatmap).
 * Each record has: x-field, y-field, count.
 */
export default function HeatmapChart({ spec }: { spec: ChartSpec }) {
  const data = spec.data_inline || [];
  if (!data.length) return <div className="text-sm text-slate-500">No data</div>;

  const xField = spec.encoding.x?.field || Object.keys(data[0])[0];
  const yField = spec.encoding.y?.field || Object.keys(data[0])[1];
  const valField = "count";

  const xLabels = [...new Set(data.map((d) => String(d[xField])))];
  const yLabels = [...new Set(data.map((d) => String(d[yField])))];

  const maxVal = Math.max(...data.map((d) => Number(d[valField]) || 0), 1);

  const cellW = Math.max(30, Math.min(60, 500 / xLabels.length));
  const cellH = Math.max(25, Math.min(40, 300 / yLabels.length));
  const leftPad = 100;
  const topPad = 30;
  const svgWidth = leftPad + xLabels.length * cellW + 20;
  const svgHeight = topPad + yLabels.length * cellH + 50;

  const getColor = (val: number) => {
    const t = val / maxVal;
    // Blue gradient: from slate-800 to blue-500
    const r = Math.round(30 + t * 66);
    const g = Math.round(41 + t * 124);
    const b = Math.round(59 + t * 181);
    return `rgb(${r}, ${g}, ${b})`;
  };

  const lookup: Record<string, number> = {};
  for (const d of data) {
    lookup[`${d[xField]}|${d[yField]}`] = Number(d[valField]) || 0;
  }

  return (
    <div style={{ width: "100%", height: "100%", overflow: "auto" }}>
      <svg width={svgWidth} height={svgHeight} style={{ display: "block", margin: "0 auto" }}>
        {/* X axis labels */}
        {xLabels.map((label, i) => (
          <text
            key={`xl-${i}`}
            x={leftPad + i * cellW + cellW / 2}
            y={topPad - 8}
            fill="#94a3b8"
            fontSize={10}
            textAnchor="middle"
          >
            {label.slice(0, 10)}
          </text>
        ))}

        {/* Y axis labels + cells */}
        {yLabels.map((yLabel, yi) => (
          <g key={`row-${yi}`}>
            <text
              x={leftPad - 6}
              y={topPad + yi * cellH + cellH / 2 + 4}
              fill="#94a3b8"
              fontSize={10}
              textAnchor="end"
            >
              {yLabel.slice(0, 12)}
            </text>
            {xLabels.map((xLabel, xi) => {
              const val = lookup[`${xLabel}|${yLabel}`] || 0;
              return (
                <g key={`cell-${xi}-${yi}`}>
                  <rect
                    x={leftPad + xi * cellW}
                    y={topPad + yi * cellH}
                    width={cellW - 2}
                    height={cellH - 2}
                    fill={getColor(val)}
                    rx={3}
                  >
                    <title>{`${xLabel} × ${yLabel}: ${val}`}</title>
                  </rect>
                  {cellW > 35 && cellH > 20 && (
                    <text
                      x={leftPad + xi * cellW + cellW / 2 - 1}
                      y={topPad + yi * cellH + cellH / 2 + 3}
                      fill="#e2e8f0"
                      fontSize={10}
                      textAnchor="middle"
                    >
                      {val}
                    </text>
                  )}
                </g>
              );
            })}
          </g>
        ))}
      </svg>
    </div>
  );
}
