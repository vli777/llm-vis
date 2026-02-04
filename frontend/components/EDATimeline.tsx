"use client";
import type { StepResult } from "@/types/chart";
import { Loader2 } from "lucide-react";

type Props = {
  steps: StepResult[];
  progress: string;
  status: "idle" | "connecting" | "streaming" | "complete" | "error";
};

const STEP_LABELS: Record<string, string> = {
  summary_stats: "Summary Statistics",
  analysis_intents: "Analysis Intents",
  intent_views: "Intent-Driven Views",
  quality_overview: "Quality Overview",
  relationships: "Relationships",
  outliers_segments: "Outliers & Segments",
  query_driven: "Query Analysis",
};

export default function EDATimeline({ steps, progress, status }: Props) {
  const isActive = status === "streaming" || status === "connecting";

  return (
    <div className="theme-panel p-3">
      <h3 className="text-sm font-semibold theme-muted mb-2">Analysis Progress</h3>

      {/* Progress indicator */}
      {isActive && (
        <div className="flex items-center gap-2 text-xs theme-primary mb-3">
          <Loader2 className="h-3 w-3 animate-spin" />
          <span>{progress}</span>
        </div>
      )}

      {status === "complete" && (
        <div className="text-xs theme-primary mb-3">{progress}</div>
      )}

      {status === "error" && (
        <div className="text-xs theme-accent mb-3">{progress}</div>
      )}

      {/* Step list */}
      <div className="space-y-3">
        {steps.map((step, i) => (
          <div key={i} className="flex items-start gap-2">
            <div className="mt-0.5 h-4 w-4 rounded-full theme-chip flex items-center justify-center text-[10px] font-bold shrink-0">
              {i + 1}
            </div>
            <div className="min-w-0">
              <div className="text-sm font-medium theme-muted">
                {STEP_LABELS[step.step_type] || step.step_type}
              </div>
              <div className="text-xs theme-muted mt-0.5">{step.headline}</div>
              {step.findings.length > 0 && (
                <ul className="mt-1 space-y-0.5 text-xs theme-muted list-disc list-inside">
                  {step.findings.map((f, fi) => (
                    <li key={fi} className="leading-tight">{f}</li>
                  ))}
                </ul>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
