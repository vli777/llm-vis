"use client";
import { VegaLite } from "react-vega";

export function VegaLiteChart({ spec }: { spec: any }) {
  return (
    <div
      className="rounded-lg border border-slate-800 bg-slate-950 p-3"
      style={{ width: "100%", maxWidth: "100%", overflow: "hidden" }}
    >
      <div style={{ width: "100%", maxWidth: "100%", overflow: "hidden" }}>
        <VegaLite spec={spec} actions={false} />
      </div>
    </div>
  );
}
