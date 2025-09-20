"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { UploadZone } from "../components/UploadZone";
import { PromptBar } from "../components/PromptBar";
import { VegaLiteChart } from "../components/VegaLiteChart";
import { apiFetch, apiGetJSON, apiPostJSON } from "../lib/api";

type Viz = {
  id: string;
  type: "chart" | "table";
  title?: string;
  spec?: any;
  table?: { columns: string[]; rows: any[] };
};

export default function Page() {
  const [tables, setTables] = useState<any[]>([]);
  const [viz, setViz] = useState<Viz[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const controllerRef = useRef<AbortController | null>(null);
  const latestReq = useRef<number>(0);

  const refreshTables = async () => {
    setError(null);
    try {
      const r = await apiGetJSON<{ tables: any[] }>("/tables");
      setTables(r.tables || []);
    } catch (e: any) {
      setError(e.message);
    }
  };

  useEffect(() => {
    refreshTables();
  }, []);

  const onUploaded = () => refreshTables();

  const onPrompt = async (prompt: string) => {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;

    const myReqId = Date.now();
    latestReq.current = myReqId;
    setPending(true);

    try {
      const res = await apiFetch("/nlq", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
        signal: controller.signal,
      }).then((r) => r.json());

      if (latestReq.current !== myReqId) return;

      if (res.type === "chart") {
        setViz((v) => [
          {
            id: crypto.randomUUID(),
            type: "chart",
            title: res.title,
            spec: res.spec,
          },
          ...v,
        ]);
      } else if (res.type === "table") {
        setViz((v) => [
          {
            id: crypto.randomUUID(),
            type: "table",
            title: res.title,
            table: res.table,
          },
          ...v,
        ]);
      } else {
        console.error("Unexpected response", res);
      }
    } catch (e) {
      if ((e as any).name !== "AbortError") console.error(e);
    } finally {
      if (latestReq.current === myReqId) setPending(false);
    }
  };

  return (
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: 24 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <h1 style={{ fontSize: 28, marginBottom: 8, marginTop: 0, flex: 1 }}>
          AI Data Vis
        </h1>        
      </div>

      {error && (
        <div
          style={{
            background: "#7f1d1d",
            padding: 8,
            borderRadius: 8,
            marginBottom: 8,
          }}
        >
          {error}
        </div>
      )}

      <UploadZone onUploaded={onUploaded} />

      <div
        style={{
          marginTop: 16,
          padding: 12,
          background: "#111827",
          borderRadius: 12,
        }}
      >
        <PromptBar
          onSubmit={onPrompt}
          placeholder="e.g., Scatter ARR vs Valuation (log), color by industry"
          disabled={pending}
        />
        <div
          style={{ display: "flex", gap: 16, marginTop: 12, flexWrap: "wrap" }}
        >
          <div style={{ flex: 1, minWidth: 260 }}>
            <h3>Tables in session</h3>
            <ul>
              {tables.map((t: any) => (
                <li key={t.name} style={{ opacity: 0.9 }}>
                  {t.name} <span style={{ opacity: 0.6 }}>({t.rows} rows)</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr",
          gap: 16,
          marginTop: 24,
        }}
      >
        {viz.map((v) => (
          <div
            key={v.id}
            style={{
              background: "#111827",
              padding: 16,
              borderRadius: 12,
              position: "relative",
            }}
          >
            {v.title && <h3 style={{ marginTop: 0 }}>{v.title}</h3>}
            {v.type === "chart" && v.spec && <VegaLiteChart spec={v.spec} />}
            {v.type === "table" && v.table && (
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <thead>
                    <tr>
                      {v.table.columns.map((c) => (
                        <th
                          key={c}
                          style={{
                            textAlign: "left",
                            borderBottom: "1px solid #374151",
                            padding: 8,
                          }}
                        >
                          {c}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {v.table.rows.map((row, i) => (
                      <tr key={i}>
                        {v.table.columns.map((c) => (
                          <td
                            key={c}
                            style={{
                              padding: 8,
                              borderBottom: "1px dashed #1f2937",
                            }}
                          >
                            {row[c]}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
