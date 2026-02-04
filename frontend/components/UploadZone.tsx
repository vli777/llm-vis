import { useRef, useState } from "react";
import { apiPostFormRaw } from "@/lib/api";
import { Loader2 } from "lucide-react";

export function UploadZone({
  onUploaded,
}: {
  onUploaded: () => void | Promise<void>;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const upload = async (file: File) => {
    const form = new FormData();
    form.append("file", file);
    setBusy(true);
    setMsg(null);
    try {
      const { res, data } = await apiPostFormRaw("/upload", form);

      if (res.status === 409 && data?.duplicate) {
        // Backend matched byte-identical file; reuse existing table
        setMsg(`Duplicate detected. Reusing table “${data.table}”.`);
        const maybePromise = onUploaded?.();
        if (
          maybePromise &&
          typeof (maybePromise as PromiseLike<void>).then === "function"
        ) {
          maybePromise.catch(console.error);
        }
        return;
      }

      if (!res.ok) {
        // Show server-provided detail if available
        const detail = data?.detail || (await res.text());
        throw new Error(detail || `Upload failed (${res.status})`);
      }

      setMsg(
        data?.table
          ? `Uploaded “${data.table}” (${data.rows ?? "?"} rows).`
          : "Upload complete."
      );
      const maybePromise = onUploaded?.();
      if (
        maybePromise &&
        typeof (maybePromise as PromiseLike<void>).then === "function"
      ) {
        maybePromise.catch(console.error);
      }
    } catch (e: any) {
      console.error("Upload error:", e);
      const errorMsg = e?.message || "Upload failed.";
      setMsg(errorMsg);
      // Show more helpful error for network issues
      if (errorMsg.includes("fetch")) {
        setMsg("Cannot connect to backend. Is it running on http://localhost:8000?");
      }
    } finally {
      setBusy(false);
      // Allow picking the same file again
      if (inputRef.current) inputRef.current.value = "";
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDragEnter = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    // Only set to false if we're leaving the container itself, not child elements
    if (e.currentTarget === e.target) {
      setIsDragging(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      const file = files[0];
      // Check if it's a CSV file
      if (file.type === "text/csv" || file.name.endsWith(".csv")) {
        upload(file);
      } else {
        setMsg("Please upload a CSV file.");
      }
    }
  };

  return (
    <div
      className={`theme-panel flex items-center justify-between gap-4 border-2 border-dashed p-6 transition-all ${
        isDragging ? "scale-[1.02]" : ""
      }`}
      style={{
        borderColor: isDragging ? "var(--color-primary)" : "var(--color-border)",
        background: isDragging ? "var(--color-surface)" : "var(--color-surface-strong)",
      }}
      onDragOver={handleDragOver}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <div>
        <div className="font-medium theme-muted">Upload CSV</div>
        <div className="text-sm theme-muted">
          Drag & drop or choose a file. Stored in memory for this session.
        </div>
        {msg && <div className="mt-1 text-xs theme-muted">{msg}</div>}
      </div>

      <div>
        <input
          ref={inputRef}
          type="file"
          accept=".csv,text/csv"
          className="hidden"
          onChange={(e) => e.target.files && upload(e.target.files[0])}
        />

        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={busy}
          className="theme-button flex items-center gap-2 rounded-lg px-3 py-2 text-sm"
        >
          {busy && <Loader2 className="h-4 w-4 animate-spin theme-muted" />}
          <span>{busy ? "Uploading…" : "Choose file"}</span>
        </button>
      </div>
    </div>
  );
}
