import React from "react";
import { humanBytes } from "@/lib/utils"; 

export type TableInfo = {
  name: string;
  file_name?: string | null;
  file_ext?: string | null;
  file_size?: number | null; // bytes
  created_at?: string | null; // ISO
  n_rows?: number;
  n_cols?: number;
  columns?: string[];
  dtypes?: Record<string, string>;
};

function TableListItem({ t }: { t: TableInfo }) {
  const sizeTxt = humanBytes(t.file_size);
  const metaBits = [
    t.file_ext ? `.${t.file_ext}` : null,
    sizeTxt || null,
    typeof t.n_rows === "number" ? `${t.n_rows} rows` : null,
    typeof t.n_cols === "number" ? `${t.n_cols} cols` : null,
  ]
    .filter(Boolean)
    .join(" • ");

  const colsTxt = (t.columns || []).join(", ");
  const dtypeTxt = t.dtypes
    ? Object.entries(t.dtypes)
        .map(([k, v]) => `${k}: ${v}`)
        .join(", ")
    : "";

  return (
    <li className="mb-2">
      <div className="font-semibold text-slate-200">{t.name}</div>

      {metaBits && (
        <div className="text-xs text-slate-400">{metaBits}</div>
      )}

      {colsTxt && (
        <div className="text-xs text-slate-300 mt-1 break-words whitespace-normal">
          <span className="text-slate-200">columns:</span>{" "}
          <span className="break-words">{colsTxt}</span>
        </div>
      )}

      {dtypeTxt && (
        <div className="text-xs text-slate-300 mt-1 break-words whitespace-normal">
          <span className="text-slate-200">types:</span>{" "}
          <span className="break-words">{dtypeTxt}</span>
        </div>
      )}
    </li>
  );
}

export default function TablesPanel({ tables }: { tables: TableInfo[] }) {
  return (
    <ul className="list-none p-0 m-0">
      {tables.map((t) => (
        <TableListItem key={t.name} t={t} />
      ))}
    </ul>
  );
}
