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
      console.error(e);
      setMsg(e?.message || "Upload failed.");
    } finally {
      setBusy(false);
      // Allow picking the same file again
      if (inputRef.current) inputRef.current.value = "";
    }
  };

  return (
    <div className="flex items-center justify-between gap-3 rounded-xl border border-slate-800 bg-slate-950 p-4">
      <div>
        <div className="font-medium">Upload CSV</div>
        <div className="text-sm text-slate-400">
          Drag & drop or choose a file. Stored in memory for this session.
        </div>
        {msg && <div className="mt-1 text-xs text-slate-300">{msg}</div>}
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
          className="flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-sm text-slate-200 hover:bg-slate-800 disabled:opacity-60"
        >
          {busy && <Loader2 className="h-4 w-4 animate-spin text-slate-400" />}
          <span>{busy ? "Uploading…" : "Choose file"}</span>
        </button>
      </div>
    </div>
  );
}
