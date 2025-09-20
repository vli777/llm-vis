"use client";
import { VegaLite } from "react-vega";

export function VegaLiteChart({ spec }: { spec: any }) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950 p-3">
      <VegaLite spec={spec} actions={false} />
    </div>
  );
}
