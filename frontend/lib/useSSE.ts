import { useCallback, useEffect, useRef, useState } from "react";
import type { AnalysisInsights, ViewResult } from "@/types/chart";

export type StepSegment = {
  step_type: string;
  views: ViewResult[];
  headline: string;
  findings: string[];
  decision_trace?: string | null;
  step_index?: number | null;
  complete: boolean;
};

export type SSEState = {
  status: "idle" | "connecting" | "streaming" | "complete" | "error";
  segments: StepSegment[];
  progress: string;
  error: string | null;
  runId: string | null;
  analysisInsights: AnalysisInsights | null;
};

type SSEEventData = Record<string, any>;

const INITIAL_STATE: SSEState = {
  status: "idle",
  segments: [],
  progress: "",
  error: null,
  runId: null,
  analysisInsights: null,
};

const FOUNDATION_STEP_TYPES = new Set([
  "summary_stats",
  "analysis_intents",
  "intent_views",
  "quality_overview",
  "relationships",
  "outliers_segments",
]);

/**
 * Custom hook for consuming SSE events from the EDA streaming endpoint.
 *
 * Views are grouped into step segments for notebook-style sequential rendering.
 */
export function useSSE() {
  const [state, setState] = useState<SSEState>(INITIAL_STATE);

  const eventSourceRef = useRef<EventSource | null>(null);
  const appendModeRef = useRef<boolean>(false);
  const suppressSegmentRef = useRef<boolean>(false);

  const close = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
  }, []);

  const reset = useCallback(() => {
    close();
    setState(INITIAL_STATE);
  }, [close]);

  const connect = useCallback(
    (
      runId: string,
      sessionId: string,
      baseUrl: string,
      opts?: { append?: boolean }
    ) => {
      close();
      appendModeRef.current = Boolean(opts?.append);
      suppressSegmentRef.current = false;

      setState((prev) => ({
        status: "connecting",
        segments: opts?.append ? prev.segments : [],
        progress: "Connecting...",
        error: null,
        runId,
        analysisInsights: opts?.append ? prev.analysisInsights : null,
      }));

      const url = `${baseUrl}/api/runs/${runId}/events?session_id=${encodeURIComponent(sessionId)}`;
      const es = new EventSource(url);
      eventSourceRef.current = es;

      es.addEventListener("run_started", (e: MessageEvent) => {
        const data: SSEEventData = JSON.parse(e.data);
        setState((prev) => ({
          ...prev,
          status: "streaming",
          progress: `Analyzing ${data.table_name} (${data.row_count} rows)...`,
        }));
      });

      es.addEventListener("progress", (e: MessageEvent) => {
        const data: SSEEventData = JSON.parse(e.data);
        setState((prev) => ({
          ...prev,
          progress:
            data.stage === "profile_complete"
              ? `Profiled ${data.columns} columns, ${data.row_count} rows`
              : data.stage || "Processing...",
        }));
      });

      es.addEventListener("analysis_intents", (e: MessageEvent) => {
        if (appendModeRef.current) return;
        const data: AnalysisInsights = JSON.parse(e.data);
        setState((prev) => ({
          ...prev,
          analysisInsights: data,
        }));
      });

      es.addEventListener("step_started", (e: MessageEvent) => {
        const data: SSEEventData = JSON.parse(e.data);
        if (appendModeRef.current && FOUNDATION_STEP_TYPES.has(data.step_type)) {
          suppressSegmentRef.current = true;
          return;
        }
        suppressSegmentRef.current = false;
        setState((prev) => ({
          ...prev,
          progress: `Running ${data.step_type?.replace(/_/g, " ")}...`,
          segments: [
            ...prev.segments,
            {
              step_type: data.step_type,
              views: [],
              headline: "",
              findings: [],
              decision_trace: null,
              step_index: data.step_index ?? null,
              complete: false,
            },
          ],
        }));
      });

      es.addEventListener("view_planned", (e: MessageEvent) => {
        const data: SSEEventData = JSON.parse(e.data);
        setState((prev) => ({
          ...prev,
          progress: `Building ${data.chart_type}: ${data.intent}`,
        }));
      });

      es.addEventListener("view_ready", (e: MessageEvent) => {
        const view: ViewResult = JSON.parse(e.data);
        if (appendModeRef.current && suppressSegmentRef.current) {
          return;
        }
        setState((prev) => {
          const segments = [...prev.segments];
          if (segments.length === 0) {
            // Fallback: no step_started yet — create an implicit segment
            segments.push({
              step_type: "analysis",
              views: [view],
              headline: "",
              findings: [],
              complete: false,
            });
          } else {
            const last = { ...segments[segments.length - 1] };
            last.views = [...last.views, view];
            segments[segments.length - 1] = last;
          }
          return { ...prev, segments };
        });
      });

      es.addEventListener("step_summary", (e: MessageEvent) => {
        const data: SSEEventData = JSON.parse(e.data);
        if (appendModeRef.current && suppressSegmentRef.current) {
          return;
        }
        setState((prev) => {
          const segments = [...prev.segments];
          if (segments.length > 0) {
            const last = { ...segments[segments.length - 1] };
            last.headline = data.headline || "";
            last.findings = data.findings || [];
            last.decision_trace = data.decision_trace || null;
            if (data.step_index !== undefined) {
              last.step_index = data.step_index;
            }
            last.complete = true;
            segments[segments.length - 1] = last;
          }
          return { ...prev, segments };
        });
      });

      es.addEventListener("warning", (e: MessageEvent) => {
        const data: SSEEventData = JSON.parse(e.data);
        console.warn("EDA warning:", data.message);
      });

      es.addEventListener("run_complete", (e: MessageEvent) => {
        const data: SSEEventData = JSON.parse(e.data);
        setState((prev) => ({
          ...prev,
          status: "complete",
          progress: `Complete: ${data.total_views} views across ${data.total_steps} steps`,
        }));
        es.close();
      });

      es.addEventListener("error", (e: MessageEvent) => {
        if (e.data) {
          const data: SSEEventData = JSON.parse(e.data);
          setState((prev) => ({
            ...prev,
            status: "error",
            error: data.message || "Unknown error",
          }));
        }
      });

      es.onerror = () => {
        setState((prev) => {
          if (prev.status === "complete") return prev;
          return {
            ...prev,
            status: "error",
            error: "Connection lost",
          };
        });
        es.close();
      };
    },
    [close]
  );

  // Clean up on unmount
  useEffect(() => {
    return () => close();
  }, [close]);

  return { state, connect, close, reset };
}
