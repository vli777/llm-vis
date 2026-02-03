"use client";
import { useEffect, useRef, useState } from "react";
import { UploadZone } from "../components/UploadZone";
import { PromptBar } from "../components/PromptBar";
import { apiGetJSON, apiPostJSON, getSessionId } from "../lib/api";
import TablesPanel from "@/components/TablesPanel";
import RechartsCard from "@/components/charts/RechartsCard";
import EDATimeline from "@/components/EDATimeline";
import { useSSE } from "@/lib/useSSE";
import type { EDAReport, ViewResult } from "@/types/chart";
import { Loader2 } from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export default function Page() {
  const [tables, setTables] = useState<any[]>([]);
  const [report, setReport] = useState<EDAReport | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [useStreaming, setUseStreaming] = useState(true);
  const tablesReq = useRef<number>(0);

  const { state: sseState, connect: sseConnect, close: sseClose } = useSSE();

  const refreshTables = async () => {
    const reqId = Date.now();
    tablesReq.current = reqId;
    try {
      const r = await apiGetJSON<{ tables: any[] }>("/tables");
      const next = r.tables || [];
      if (tablesReq.current !== reqId) return;
      setTables(next);
    } catch (e) {
      console.error(e);
      if (tablesReq.current !== reqId) return;
      setTables([]);
    }
  };

  useEffect(() => {
    refreshTables();
  }, []);

  const onUploaded = () => {
    refreshTables();
    setReport(null);
    setError(null);
    sseClose();
  };

  const runEDA = async (query?: string) => {
    if (tables.length === 0) {
      setError("Upload a CSV file first.");
      return;
    }

    setPending(true);
    setError(null);
    setReport(null);
    sseClose();

    let connectedSSE = false;
    try {
      const body: any = {};
      if (query) body.query = query;

      if (useStreaming) {
        // Streaming mode: POST with ?stream=1, then connect SSE
        const res = await apiPostJSON<{ run_id: string; streaming: boolean }>(
          "/api/runs?stream=1",
          body
        );
        if (res.streaming && res.run_id) {
          sseConnect(res.run_id, getSessionId(), API_BASE);
          connectedSSE = true;
          // Keep pending until SSE completes
          return;
        }
      }

      // Sync fallback
      const res = await apiPostJSON<EDAReport>("/api/runs", body);
      setReport(res);
    } catch (e: any) {
      console.error("EDA run failed:", e);
      setError(e?.message || "EDA run failed.");
    } finally {
      if (!connectedSSE) {
        setPending(false);
      }
    }
  };

  // Track SSE completion to clear pending state
  useEffect(() => {
    if (sseState.status === "complete" || sseState.status === "error") {
      setPending(false);
    }
    if (sseState.status === "error" && sseState.error) {
      setError(sseState.error);
    }
  }, [sseState.status, sseState.error]);

  const onPrompt = async (prompt: string) => {
    await runEDA(prompt);
  };

  // Determine what to render: SSE streaming views or sync report
  const isStreaming = sseState.status === "streaming" || sseState.status === "connecting";
  const hasStreamedViews = sseState.views.size > 0;
  const showStreamingUI = hasStreamedViews || isStreaming || sseState.status === "complete";

  // Build views map for sync mode
  const syncViewsById = new Map<string, ViewResult>();
  if (report) {
    for (const v of report.views) {
      syncViewsById.set(v.id, v);
    }
  }

  return (
    <div style={{ maxWidth: 1200, margin: "0 auto", padding: 24 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <h1 style={{ fontSize: 28, marginBottom: 8, marginTop: 0, flex: 1 }}>
          AI Data Vis
        </h1>
      </div>

      <UploadZone onUploaded={onUploaded} />

      <div
        style={{
          marginTop: 16,
          padding: 12,
          background: "#111827",
          borderRadius: 12,
        }}
      >
        <div className="flex items-center gap-3">
          <div className="flex-1">
            <PromptBar
              onSubmit={onPrompt}
              placeholder="Ask a question about the data, or click Run EDA"
              disabled={pending}
            />
          </div>
          <button
            onClick={() => runEDA()}
            disabled={pending || tables.length === 0}
            className="rounded-lg border border-blue-600 bg-blue-700 px-4 py-2.5 text-sm font-medium text-white hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 shrink-0"
          >
            {pending && <Loader2 className="h-4 w-4 animate-spin" />}
            {pending ? "Running..." : "Run EDA"}
          </button>
        </div>

        <div
          style={{ display: "flex", gap: 16, marginTop: 12, flexWrap: "wrap" }}
        >
          <div style={{ flex: 1, minWidth: 260 }}>
            <h3>Tables in session</h3>
            <TablesPanel tables={tables} />
          </div>
        </div>
      </div>

      {/* Error display */}
      {error && (
        <div className="mt-4 p-3 rounded-lg bg-red-950/50 border border-red-800 text-red-300 text-sm">
          {error}
        </div>
      )}

      {/* SSE Streaming UI */}
      {showStreamingUI && (
        <div className="mt-6">
          {/* Timeline sidebar + views */}
          <div className="flex gap-4">
            {/* Timeline */}
            <div className="w-64 shrink-0">
              <EDATimeline
                steps={sseState.steps}
                progress={sseState.progress}
                status={sseState.status}
              />
            </div>

            {/* Streamed views */}
            <div className="flex-1">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {Array.from(sseState.views.values()).map((view) => (
                  <div key={view.id} className="min-h-[320px]">
                    <RechartsCard
                      spec={view.spec}
                      explanation={view.explanation}
                    />
                  </div>
                ))}
              </div>

              {isStreaming && sseState.views.size === 0 && (
                <div className="flex items-center justify-center py-12 text-slate-500">
                  <Loader2 className="h-6 w-6 animate-spin mr-2" />
                  <span>Generating charts...</span>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Synchronous Report (fallback when not streaming) */}
      {report && !showStreamingUI && (
        <div className="mt-6">
          {/* Profile summary */}
          {report.profile && (
            <div className="mb-4 p-3 rounded-lg bg-slate-900 border border-slate-800">
              <h2 className="text-lg font-semibold text-slate-200 mb-1">
                {report.table_name}
              </h2>
              <p className="text-sm text-slate-400">
                {report.profile.row_count.toLocaleString()} rows,{" "}
                {report.profile.columns.length} columns
                {report.profile.visualization_hints?.summary && (
                  <>
                    {" "}&mdash;{" "}
                    {report.profile.visualization_hints.summary.numeric_columns} numeric,{" "}
                    {report.profile.visualization_hints.summary.categorical_columns} categorical
                    {report.profile.visualization_hints.summary.has_temporal_data && ", has temporal data"}
                  </>
                )}
              </p>
            </div>
          )}

          {/* Steps with their views */}
          {report.steps.map((step, si) => (
            <div key={si} className="mb-6">
              <div className="flex items-center gap-2 mb-3">
                <div className="h-6 w-6 rounded-full bg-blue-700 flex items-center justify-center text-xs font-bold text-white">
                  {si + 1}
                </div>
                <h2 className="text-base font-semibold text-slate-200">
                  {step.headline}
                </h2>
              </div>

              {step.warnings.length > 0 && (
                <div className="mb-2 text-xs text-amber-400">
                  {step.warnings.map((w, wi) => (
                    <div key={wi}>&#9888; {w}</div>
                  ))}
                </div>
              )}

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {step.views.map((viewId) => {
                  const view = syncViewsById.get(viewId);
                  if (!view) return null;
                  return (
                    <div key={view.id} className="min-h-[320px]">
                      <RechartsCard
                        spec={view.spec}
                        explanation={view.explanation}
                      />
                    </div>
                  );
                })}
              </div>

              {step.findings.length > 0 && (
                <div className="mt-2 text-xs text-slate-400 space-y-1">
                  {step.findings.map((f, fi) => (
                    <div key={fi}>{f}</div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
