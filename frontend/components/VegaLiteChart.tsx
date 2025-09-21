"use client";
import { VegaLite } from "react-vega";
import { useMemo, useEffect, useState } from "react";

export function VegaLiteChart({ spec }: { spec: any }) {
  const [error, setError] = useState<string | null>(null);
  const cleanSpec = useMemo(() => {
    setError(null);
    return spec;
  }, [spec]);
  useEffect(() => {
    if (!cleanSpec) return;
    const { data, ...rest } = cleanSpec;
    console.log("Vega-Lite spec:", rest);
  }, [cleanSpec]);

  return (
    <div
      className="rounded-lg border border-slate-800 bg-slate-950 p-3"
      style={{ width: "100%", maxWidth: "100%", overflow: "hidden" }}
    >
      <div style={{ width: "100%", maxWidth: "100%", overflow: "hidden" }}>
        {error ? (
          <div className="text-sm text-rose-400">Chart error: {error}</div>
        ) : (
          <VegaLite
            spec={cleanSpec}
            actions={false}
            onError={(err) => setError(err.message || String(err))}
          />
        )}
      </div>
    </div>
  );
}
