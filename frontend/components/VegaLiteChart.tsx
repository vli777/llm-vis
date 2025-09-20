"use client";
import { VegaLite } from "react-vega";

export function VegaLiteChart({ spec }: { spec: any }) {
  const data = spec.data?.values ? {} : { data: spec.data };
  // We pass inline data if present; otherwise, spec.data should already be structured
  return (
    <div style={{ background: "#0b1220", borderRadius: 10, padding: 12 }}>
      <VegaLite spec={spec} actions={false} />
    </div>
  );
}
