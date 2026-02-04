"use client";
import { useMemo } from "react";
import type { ChartSpec, ChartType } from "@/types/chart";
import BarChartView from "./BarChart";
import LineChartView from "./LineChart";
import ScatterChartView from "./ScatterChart";
import HistogramChart from "./HistogramChart";
import BoxChart from "./BoxChart";
import PieChartView from "./PieChart";
import HeatmapChart from "./HeatmapChart";
import DataTable from "./DataTable";

const RENDERERS: Record<ChartType, React.ComponentType<{ spec: ChartSpec }>> = {
  bar: BarChartView,
  line: LineChartView,
  scatter: ScatterChartView,
  hist: HistogramChart,
  box: BoxChart,
  pie: PieChartView,
  heatmap: HeatmapChart,
  table: DataTable,
  area: LineChartView, // reuse line renderer for area
};

type Props = {
  spec: ChartSpec;
  explanation?: string;
};

export default function RechartsCard({ spec, explanation }: Props) {
  const Renderer = useMemo(() => RENDERERS[spec.chart_type] || DataTable, [spec.chart_type]);

  return (
    <div className="theme-panel p-4 flex flex-col gap-2">
      {spec.title && (
        <h3 className="text-sm font-semibold theme-muted m-0">{spec.title}</h3>
      )}
      {spec.subtitle && (
        <p className="text-xs theme-muted m-0">{spec.subtitle}</p>
      )}
      <div className="flex-1 min-h-[240px]">
        <Renderer spec={spec} />
      </div>
      {explanation && (
        <p className="text-xs theme-muted mt-1 m-0">{explanation}</p>
      )}
    </div>
  );
}
