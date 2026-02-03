import { useCallback, useEffect, useRef, useState } from "react";
import type { ViewResult, StepResult } from "@/types/chart";

export type SSEState = {
  status: "idle" | "connecting" | "streaming" | "complete" | "error";
  views: Map<string, ViewResult>;
  steps: StepResult[];
  progress: string;
  error: string | null;
  runId: string | null;
};

type SSEEventData = Record<string, any>;

/**
 * Custom hook for consuming SSE events from the EDA streaming endpoint.
 */
export function useSSE() {
  const [state, setState] = useState<SSEState>({
    status: "idle",
    views: new Map(),
    steps: [],
    progress: "",
    error: null,
    runId: null,
  });

  const eventSourceRef = useRef<EventSource | null>(null);

  const close = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
  }, []);

  const connect = useCallback(
    (runId: string, sessionId: string, baseUrl: string) => {
      close();

      setState({
        status: "connecting",
        views: new Map(),
        steps: [],
        progress: "Connecting...",
        error: null,
        runId,
      });

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
          progress: data.stage === "profile_complete"
            ? `Profiled ${data.columns} columns, ${data.row_count} rows`
            : data.stage || "Processing...",
        }));
      });

      es.addEventListener("step_started", (e: MessageEvent) => {
        const data: SSEEventData = JSON.parse(e.data);
        setState((prev) => ({
          ...prev,
          progress: `Running ${data.step_type?.replace(/_/g, " ")}...`,
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
        setState((prev) => {
          const next = new Map(prev.views);
          next.set(view.id, view);
          return { ...prev, views: next };
        });
      });

      es.addEventListener("step_summary", (e: MessageEvent) => {
        const data: SSEEventData = JSON.parse(e.data);
        const step: StepResult = {
          step_type: data.step_type,
          headline: data.headline,
          views: [], // We'll associate views separately
          findings: data.findings || [],
          warnings: [],
        };
        setState((prev) => ({
          ...prev,
          steps: [...prev.steps, step],
        }));
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

  return { state, connect, close };
}
