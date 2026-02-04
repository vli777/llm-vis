"use client";
import { useEffect, useRef, useState } from "react";
import { UploadZone } from "../components/UploadZone";
import { PromptBar } from "../components/PromptBar";
import { apiGetJSON, apiPostJSON, getSessionId } from "../lib/api";
import RechartsCard from "@/components/charts/RechartsCard";
import DataTable from "@/components/charts/DataTable";
import { useSSE } from "@/lib/useSSE";
import type { EDAReport, ViewResult } from "@/types/chart";
import { Loader2, ChevronDown, ChevronRight } from "lucide-react";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

const STEP_LABELS: Record<string, string> = {
  summary_stats: "Summary Statistics",
  analysis_intents: "Analysis Intents",
  intent_selection: "Intent Selection",
  intent_views: "Intent-Driven Views",
  quality_overview: "Quality Overview",
  relationships: "Relationships",
  outliers_segments: "Outliers & Segments",
  query_driven: "Query Analysis",
};

export default function Page() {
  const [tables, setTables] = useState<any[]>([]);
  const [report, setReport] = useState<EDAReport | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [useStreaming, setUseStreaming] = useState(true);
  const [showTables, setShowTables] = useState(false);
  const tablesReq = useRef<number>(0);
  const autoRunKey = useRef<string | null>(null);

  const { state: sseState, connect: sseConnect, close: sseClose } = useSSE();

  const refreshTables = async () => {
    const reqId = Date.now();
    tablesReq.current = reqId;
    try {
      const r = await apiGetJSON<{ tables: any[] }>("/tables");
      const next = r.tables || [];
      if (tablesReq.current !== reqId) return;
      setTables(next);
      return next;
    } catch (e) {
      console.error(e);
      if (tablesReq.current !== reqId) return;
      setTables([]);
      return [];
    }
  };

  useEffect(() => {
    refreshTables();
  }, []);

  const onUploaded = async () => {
    await refreshTables();
    setReport(null);
    setError(null);
    sseClose();
  };

  const runEDA = async (query?: string, tablesOverride?: any[]) => {
    const availableTables = tablesOverride ?? tables;
    if (availableTables.length === 0) {
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
        const res = await apiPostJSON<{ run_id: string; streaming: boolean }>(
          "/api/runs?stream=1",
          body
        );
        if (res.streaming && res.run_id) {
          sseConnect(res.run_id, getSessionId(), API_BASE);
          connectedSSE = true;
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

  useEffect(() => {
    if (pending) return;
    if (tables.length === 0) {
      autoRunKey.current = null;
      return;
    }
    if (report || sseState.segments.length > 0) return;
    if (sseState.status === "connecting" || sseState.status === "streaming") return;

    const key = tables
      .map((t: any) => `${t.name || t.table_name || ""}:${t.rows ?? ""}`)
      .join("|");
    if (autoRunKey.current === key) return;
    autoRunKey.current = key;
    void runEDA(undefined, tables);
  }, [tables, pending, report, sseState.status, sseState.segments.length]);

  const onPrompt = async (prompt: string) => {
    await runEDA(prompt);
  };

  const isStreaming =
    sseState.status === "streaming" || sseState.status === "connecting";
  const hasSegments = sseState.segments.length > 0;
  const showStreamingUI =
    hasSegments || isStreaming || sseState.status === "complete";

  // Build views map for sync mode
  const syncViewsById = new Map<string, ViewResult>();
  if (report) {
    for (const v of report.views) {
      syncViewsById.set(v.id, v);
    }
  }

  return (
    <div className="mx-auto max-w-[1200px] p-6 pb-28">
      <h1 className="text-2xl font-semibold theme-muted mb-2 mt-0">
        AI Data Vis
      </h1>

      <UploadZone onUploaded={onUploaded} />

      {/* Collapsible tables indicator */}
      {tables.length > 0 && (
        <button
          onClick={() => setShowTables(!showTables)}
          className="mt-2 flex items-center gap-1 text-xs theme-muted hover:text-black"
        >
          {showTables ? (
            <ChevronDown className="h-3 w-3" />
          ) : (
            <ChevronRight className="h-3 w-3" />
          )}
          {tables.length} table{tables.length !== 1 ? "s" : ""} in session
        </button>
      )}
      {showTables && (
        <div className="mt-1 text-xs theme-muted pl-4">
          {tables.map((t: any, i: number) => (
            <div key={i}>
              {t.name || t.table_name || `Table ${i + 1}`}
              {t.rows != null && ` (${t.rows} rows)`}
            </div>
          ))}
        </div>
      )}

      {/* Error display */}
      {error && (
        <div className="mt-4 p-3 rounded-lg border text-sm theme-accent" style={{ borderColor: "var(--color-accent)" }}>
          {error}
        </div>
      )}

      <div className="fixed bottom-4 left-0 right-0 px-6 z-50 pointer-events-none">
        <div className="mx-auto max-w-[1200px] pointer-events-auto">
          <PromptBar
            onSubmit={onPrompt}
            placeholder="Ask a question about the data"
            disabled={pending}
          />
        </div>
      </div>

      {/* ============================================================
          SSE Streaming UI — Notebook-style sequential layout
          ============================================================ */}
      {showStreamingUI && (
        <div className="mt-6 space-y-6">
          {/* Progress indicator */}
          {isStreaming && (
            <div className="flex items-center gap-2 text-sm theme-primary">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>{sseState.progress}</span>
            </div>
          )}
          {sseState.status === "complete" && (
            <div className="text-sm theme-primary">{sseState.progress}</div>
          )}

          {/* Step segments — rendered sequentially like a notebook */}
          {sseState.segments
            .slice()
            .sort((a, b) => (a.step_index ?? 0) - (b.step_index ?? 0))
            .map((segment, si) => {
            const tableViews = segment.views.filter(
              (v) => v.spec.chart_type === "table"
            );
            const chartViews = segment.views.filter(
              (v) => v.spec.chart_type !== "table"
            );

            return (
              <div key={si}>
                {/* Step header */}
                <div className="flex items-center gap-2 mb-3">
                  <div className="h-6 w-6 rounded-full theme-chip flex items-center justify-center text-xs font-bold shrink-0">
                    {si + 1}
                  </div>
                  <h2 className="text-base font-semibold theme-muted">
                    {segment.complete && segment.headline
                      ? segment.headline
                      : STEP_LABELS[segment.step_type] || segment.step_type}
                  </h2>
                  {!segment.complete && (
                    <Loader2 className="h-4 w-4 animate-spin theme-primary" />
                  )}
                </div>
                <div className="trace-slot text-xs theme-muted">
                  {segment.complete && segment.decision_trace && (
                    <span className="trace-fade">{segment.decision_trace}</span>
                  )}
                </div>

                {/* Summary tables — full width */}
                {tableViews.map((view) => (
                  <div
                    key={view.id}
                    className="mb-4 theme-panel p-4 flex flex-col gap-2"
                  >
                    {view.spec.title && (
                      <h3 className="text-sm font-semibold theme-muted m-0">
                        {view.spec.title}
                      </h3>
                    )}
                    <div className="overflow-auto max-h-[600px]">
                      <DataTable spec={view.spec} />
                    </div>
                  </div>
                ))}

                {/* Chart views — 2-col grid */}
                {chartViews.length > 0 && (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    {chartViews.map((view) => (
                      <div key={view.id} className="min-h-[320px]">
                        <RechartsCard
                          spec={view.spec}
                          explanation={view.explanation}
                        />
                      </div>
                    ))}
                  </div>
                )}

                {/* Findings / Insights */}
                {segment.complete && segment.findings.length > 0 && (
                  <div className="mt-3 p-3 theme-panel">
                    <ul className="text-sm theme-muted list-disc list-inside space-y-1">
                      {segment.findings.map((f, fi) => (
                        <li key={fi}>{f}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* ============================================================
          Synchronous Report (fallback when not streaming)
          ============================================================ */}
      {report && !showStreamingUI && (
        <div className="mt-6">
          {/* Profile summary */}
          {report.profile && (
            <div className="mb-4 theme-panel p-3">
              <h2 className="text-lg font-semibold theme-muted mb-1">
                {report.table_name}
              </h2>
              <p className="text-sm theme-muted">
                {report.profile.row_count.toLocaleString()} rows,{" "}
                {report.profile.columns.length} columns
              </p>
            </div>
          )}

          {/* Steps with their views — same notebook layout */}
          {report.steps.map((step, si) => {
            const tableViewIds = step.views.filter((id) => {
              const v = syncViewsById.get(id);
              return v && v.spec.chart_type === "table";
            });
            const chartViewIds = step.views.filter((id) => {
              const v = syncViewsById.get(id);
              return v && v.spec.chart_type !== "table";
            });

            return (
              <div key={si} className="mb-6">
                <div className="flex items-center gap-2 mb-3">
                  <div className="h-6 w-6 rounded-full theme-chip flex items-center justify-center text-xs font-bold">
                    {si + 1}
                  </div>
                  <h2 className="text-base font-semibold theme-muted">
                    {step.headline}
                  </h2>
                </div>
                <div className="trace-slot text-xs theme-muted">
                  {step.decision_trace && (
                    <span className="trace-fade">{step.decision_trace}</span>
                  )}
                </div>

                {step.warnings.length > 0 && (
                  <div className="mb-2 text-xs theme-accent">
                    {step.warnings.map((w, wi) => (
                      <div key={wi}>&#9888; {w}</div>
                    ))}
                  </div>
                )}

                {/* Summary tables — full width */}
                {tableViewIds.map((viewId) => {
                  const view = syncViewsById.get(viewId);
                  if (!view) return null;
                  return (
                    <div
                      key={view.id}
                      className="mb-4 theme-panel p-4 flex flex-col gap-2"
                    >
                      {view.spec.title && (
                        <h3 className="text-sm font-semibold theme-muted m-0">
                          {view.spec.title}
                        </h3>
                      )}
                      <div className="overflow-auto max-h-[600px]">
                        <DataTable spec={view.spec} />
                      </div>
                    </div>
                  );
                })}

                {/* Chart views — 2-col grid */}
                {chartViewIds.length > 0 && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {chartViewIds.map((viewId) => {
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
                )}

                {/* Findings */}
                {step.findings.length > 0 && (
                  <div className="mt-3 p-3 theme-panel">
                    <ul className="text-sm theme-muted list-disc list-inside space-y-1">
                      {step.findings.map((f, fi) => (
                        <li key={fi}>{f}</li>
                      ))}
                    </ul>
                  </div>
                )}

              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
