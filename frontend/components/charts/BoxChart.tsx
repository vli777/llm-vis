"use client";
import type { ChartSpec } from "@/types/chart";

/**
 * Box plot using SVG (Recharts doesn't have a built-in box plot).
 * Each record has: group, min, q1, median, q3, max, outliers[].
 */
export default function BoxChart({ spec }: { spec: ChartSpec }) {
  const data = spec.data_inline || [];
  if (!data.length) return <div className="text-sm text-slate-500">No data</div>;

  const padding = 40;
  const boxWidth = 50;
  const gap = 20;
  const svgWidth = Math.max(300, data.length * (boxWidth + gap) + padding * 2);
  const svgHeight = 280;
  const plotHeight = svgHeight - padding * 2;

  // Global min/max across all groups
  let globalMin = Infinity;
  let globalMax = -Infinity;
  for (const d of data) {
    const vals = [d.min, d.q1, d.median, d.q3, d.max, ...(d.outliers || [])];
    for (const v of vals) {
      if (typeof v === "number" && isFinite(v)) {
        if (v < globalMin) globalMin = v;
        if (v > globalMax) globalMax = v;
      }
    }
  }
  if (!isFinite(globalMin)) globalMin = 0;
  if (!isFinite(globalMax)) globalMax = 1;
  const range = globalMax - globalMin || 1;
  const buffer = range * 0.05;
  const yMin = globalMin - buffer;
  const yMax = globalMax + buffer;

  const scaleY = (v: number) => padding + plotHeight - ((v - yMin) / (yMax - yMin)) * plotHeight;

  // Y axis ticks
  const ticks = 5;
  const yTicks = Array.from({ length: ticks + 1 }, (_, i) => yMin + (i / ticks) * (yMax - yMin));

  return (
    <div style={{ width: "100%", height: "100%", overflow: "auto" }}>
      <svg
        width={svgWidth}
        height={svgHeight}
        style={{ minWidth: svgWidth, display: "block", margin: "0 auto" }}
      >
        {/* Y axis */}
        <line x1={padding} y1={padding} x2={padding} y2={svgHeight - padding} stroke="#475569" />
        {yTicks.map((t, i) => (
          <g key={i}>
            <line x1={padding - 4} y1={scaleY(t)} x2={padding} y2={scaleY(t)} stroke="#475569" />
            <text x={padding - 8} y={scaleY(t) + 4} fill="#94a3b8" fontSize={10} textAnchor="end">
              {t.toFixed(1)}
            </text>
          </g>
        ))}

        {data.map((d, i) => {
          const cx = padding + gap + i * (boxWidth + gap) + boxWidth / 2;
          const x1 = cx - boxWidth / 2;
          const x2 = cx + boxWidth / 2;

          return (
            <g key={i}>
              {/* Whisker line */}
              <line x1={cx} y1={scaleY(d.max)} x2={cx} y2={scaleY(d.min)} stroke="#94a3b8" strokeWidth={1} />
              {/* Whisker caps */}
              <line x1={x1 + 10} y1={scaleY(d.max)} x2={x2 - 10} y2={scaleY(d.max)} stroke="#94a3b8" strokeWidth={1} />
              <line x1={x1 + 10} y1={scaleY(d.min)} x2={x2 - 10} y2={scaleY(d.min)} stroke="#94a3b8" strokeWidth={1} />
              {/* Box */}
              <rect
                x={x1}
                y={scaleY(d.q3)}
                width={boxWidth}
                height={scaleY(d.q1) - scaleY(d.q3)}
                fill="#1e40af"
                fillOpacity={0.5}
                stroke="#60a5fa"
                strokeWidth={1.5}
                rx={3}
              />
              {/* Median line */}
              <line x1={x1} y1={scaleY(d.median)} x2={x2} y2={scaleY(d.median)} stroke="#fbbf24" strokeWidth={2} />
              {/* Outliers */}
              {(d.outliers || []).slice(0, 20).map((o: number, j: number) => (
                <circle key={j} cx={cx} cy={scaleY(o)} r={3} fill="#f87171" opacity={0.7} />
              ))}
              {/* Label */}
              <text
                x={cx}
                y={svgHeight - padding + 16}
                fill="#94a3b8"
                fontSize={11}
                textAnchor="middle"
              >
                {String(d.group).slice(0, 12)}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
