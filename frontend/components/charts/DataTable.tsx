"use client";
import type { ChartSpec } from "@/types/chart";

export default function DataTable({ spec }: { spec: ChartSpec }) {
  const data = spec.data_inline || [];
  if (!data.length) return <div className="text-sm theme-muted">No data</div>;

  const columns = Object.keys(data[0]);

  return (
    <div className="overflow-auto max-h-full w-full border rounded" style={{ borderColor: "var(--color-border)" }}>
      <table className="min-w-full text-xs">
        <thead className="sticky top-0 z-10" style={{ background: "var(--color-surface)" }}>
          <tr>
            {columns.map((col) => (
              <th
                key={col}
                className="px-3 py-2 text-left font-medium theme-muted border-b-2"
                style={{ borderColor: "var(--color-border)" }}
              >
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, idx) => (
            <tr key={idx} className="border-b last:border-b-0" style={{ borderColor: "var(--color-border)" }}>
              {columns.map((col) => (
                <td
                  key={col}
                  className="px-3 py-1.5 theme-muted max-w-xs truncate"
                  title={String(row[col] ?? "")}
                >
                  {row[col] === null || row[col] === undefined ? (
                    <span className="theme-muted italic">null</span>
                  ) : (
                    String(row[col])
                  )}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
