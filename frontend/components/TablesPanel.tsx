import React, { useState } from "react";
import { humanBytes } from "@/lib/utils";
import { apiGetJSON } from "@/lib/api";
import { ChevronDown, ChevronRight, Loader2 } from "lucide-react";

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

type PreviewData = {
  columns: string[];
  rows: Record<string, any>[];
  total_rows: number;
  offset: number;
  limit: number;
  returned_rows: number;
  has_more: boolean;
  next_offset: number | null;
};

function TableListItem({ t }: { t: TableInfo }) {
  const [isExpanded, setIsExpanded] = useState(true);
  const [preview, setPreview] = useState<PreviewData | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollContainerRef = React.useRef<HTMLDivElement>(null);

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

  // Auto-fetch initial preview on mount since we start expanded
  React.useEffect(() => {
    if (isExpanded && !preview && !loading) {
      setLoading(true);
      setError(null);
      apiGetJSON<PreviewData>(`/table/${t.name}/preview?offset=0&limit=50`)
        .then(setPreview)
        .catch((e: any) => setError(e.message || "Failed to load preview"))
        .finally(() => setLoading(false));
    }
  }, [isExpanded, preview, loading, t.name]);

  // Load more data when scrolling to bottom
  const loadMore = React.useCallback(async () => {
    if (!preview || !preview.has_more || loadingMore) return;

    setLoadingMore(true);
    try {
      const nextData = await apiGetJSON<PreviewData>(
        `/table/${t.name}/preview?offset=${preview.next_offset}&limit=50`
      );
      setPreview((prev) => ({
        ...nextData,
        rows: [...(prev?.rows || []), ...nextData.rows],
      }));
    } catch (e: any) {
      setError(e.message || "Failed to load more");
    } finally {
      setLoadingMore(false);
    }
  }, [preview, loadingMore, t.name]);

  // Scroll event handler for infinite scroll
  const handleScroll = React.useCallback(
    (e: React.UIEvent<HTMLDivElement>) => {
      const target = e.currentTarget;
      const scrollPercentage =
        (target.scrollTop + target.clientHeight) / target.scrollHeight;

      // Load more when scrolled 80% down
      if (scrollPercentage > 0.8 && preview?.has_more && !loadingMore) {
        loadMore();
      }
    },
    [preview, loadingMore, loadMore]
  );

  const togglePreview = async () => {
    setIsExpanded(!isExpanded);
  };

  return (
    <li className="mb-3">
      <div
        className="theme-hover cursor-pointer p-2 -mx-2 rounded transition-colors"
        onClick={togglePreview}
      >
        <div className="flex items-start gap-2">
          <div className="mt-0.5">
            {isExpanded ? (
              <ChevronDown className="h-4 w-4 theme-muted" />
            ) : (
              <ChevronRight className="h-4 w-4 theme-muted" />
            )}
          </div>
          <div className="flex-1 min-w-0">
            <div className="font-semibold theme-muted">
              {t.name}{" "}
              {metaBits && (
                <span className="text-xs theme-muted">{metaBits}</span>
              )}
            </div>

            {dtypeTxt && (
              <div className="text-xs theme-muted break-words whitespace-normal">
                <span className="theme-muted">types:</span>{" "}
                <span className="break-words">{dtypeTxt}</span>
              </div>
            )}

            {colsTxt && (
              <div className="text-xs theme-muted mt-1 break-words whitespace-normal truncate">
                <span className="theme-muted">columns:</span>{" "}
                <span className="break-words">{colsTxt}</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {isExpanded && (
        <div className="mt-4 ml-6">
          {loading && (
            <div className="flex items-center gap-2 text-sm theme-muted py-2">
              <Loader2 className="h-4 w-4 animate-spin" />
              Loading preview...
            </div>
          )}

          {error && <div className="text-sm theme-accent py-2">{error}</div>}

          {preview && (
            <div>
              <div className="text-xs theme-muted mb-2">
                Showing {preview.rows.length} of {preview.total_rows} rows
                {preview.has_more && " (scroll for more)"}
              </div>
              <div
                ref={scrollContainerRef}
                className="overflow-auto max-h-96 [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none] border rounded"
                style={{ borderColor: "var(--color-border)" }}
                onScroll={handleScroll}
              >
                <table className="min-w-full text-xs">
                  <thead className="sticky top-0 z-10 shadow-md" style={{ background: "var(--color-surface)" }}>
                    <tr>
                      {preview.columns.map((col) => (
                        <th
                          key={col}
                          className="px-2 py-2 text-left font-medium theme-muted border-b-2"
                          style={{ borderColor: "var(--color-border)" }}
                        >
                          {col}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {preview.rows.map((row, idx) => (
                      <tr
                        key={idx}
                        className="border-b last:border-b-0"
                        style={{ borderColor: "var(--color-border)" }}
                      >
                        {preview.columns.map((col) => (
                          <td
                            key={col}
                            className="px-2 py-1 theme-muted max-w-xs truncate"
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
                {loadingMore && (
                  <div className="flex items-center justify-center gap-2 py-3 text-sm theme-muted">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Loading more...
                  </div>
                )}
              </div>
            </div>
          )}
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
