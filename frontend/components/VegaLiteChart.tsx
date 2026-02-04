"use client";
import { VegaLite } from "react-vega";
import { useMemo, useEffect, useState } from "react";

export function VegaLiteChart({ spec }: { spec: any }) {
  const [error, setError] = useState<string | null>(null);
  const cleanSpec = useMemo(() => {
    setError(null);
    return spec;
  }, [spec]);

  const sizedSpec = useMemo(() => {
    if (!cleanSpec || typeof cleanSpec !== "object") return cleanSpec;
    const next: Record<string, any> = { ...cleanSpec };
    if (next.width === undefined) next.width = "container";
    if (next.height === undefined) next.height = "container";
    return next;
  }, [cleanSpec]);

  useEffect(() => {
    if (!cleanSpec) return;
    const { data, ...rest } = cleanSpec;
    console.log("Vega-Lite spec:", rest);
  }, [cleanSpec]);

  return (
    <div
      className="theme-panel p-3"
      style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column" }}
    >
      <div style={{ flex: 1, width: "100%", maxWidth: "100%", overflow: "hidden" }}>
        {error ? (
          <div className="text-sm theme-accent">Chart error: {error}</div>
        ) : (
          <VegaLite
            spec={sizedSpec}
            actions={false}
            style={{ height: "100%", width: "100%" }}
            onError={(err) => setError(err.message || String(err))}
          />
        )}
      </div>
    </div>
  );
}
