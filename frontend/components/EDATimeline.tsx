"use client";
import type { StepResult } from "@/types/chart";
import { Loader2 } from "lucide-react";

type Props = {
  steps: StepResult[];
  progress: string;
  status: "idle" | "connecting" | "streaming" | "complete" | "error";
};

const STEP_LABELS: Record<string, string> = {
  quality_overview: "Quality Overview",
  relationships: "Relationships",
  outliers_segments: "Outliers & Segments",
  query_driven: "Query Analysis",
};

export default function EDATimeline({ steps, progress, status }: Props) {
  const isActive = status === "streaming" || status === "connecting";

  return (
    <div className="rounded-lg border border-slate-800 bg-slate-950 p-3">
      <h3 className="text-sm font-semibold text-slate-300 mb-2">Analysis Progress</h3>

      {/* Progress indicator */}
      {isActive && (
        <div className="flex items-center gap-2 text-xs text-blue-400 mb-3">
          <Loader2 className="h-3 w-3 animate-spin" />
          <span>{progress}</span>
        </div>
      )}

      {status === "complete" && (
        <div className="text-xs text-green-400 mb-3">{progress}</div>
      )}

      {status === "error" && (
        <div className="text-xs text-red-400 mb-3">{progress}</div>
      )}

      {/* Step list */}
      <div className="space-y-2">
        {steps.map((step, i) => (
          <div key={i} className="flex items-start gap-2">
            <div className="mt-0.5 h-4 w-4 rounded-full bg-blue-700 flex items-center justify-center text-[10px] font-bold text-white shrink-0">
              {i + 1}
            </div>
            <div>
              <div className="text-xs font-medium text-slate-300">
                {STEP_LABELS[step.step_type] || step.step_type}
              </div>
              <div className="text-[10px] text-slate-500">{step.headline}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
