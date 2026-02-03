"use client";
import type { ChartSpec } from "@/types/chart";

export default function DataTable({ spec }: { spec: ChartSpec }) {
  const data = spec.data_inline || [];
  if (!data.length) return <div className="text-sm text-slate-500">No data</div>;

  const columns = Object.keys(data[0]);

  return (
    <div className="overflow-auto max-h-full w-full border border-slate-700 rounded">
      <table className="min-w-full text-xs">
        <thead className="bg-slate-800 sticky top-0 z-10">
          <tr>
            {columns.map((col) => (
              <th
                key={col}
                className="px-3 py-2 text-left font-medium text-slate-200 border-b-2 border-slate-700 bg-slate-800"
              >
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="bg-slate-900">
          {data.map((row, idx) => (
            <tr key={idx} className="border-b border-slate-800 last:border-b-0">
              {columns.map((col) => (
                <td
                  key={col}
                  className="px-3 py-1.5 text-slate-300 max-w-xs truncate"
                  title={String(row[col] ?? "")}
                >
                  {row[col] === null || row[col] === undefined ? (
                    <span className="text-slate-500 italic">null</span>
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
