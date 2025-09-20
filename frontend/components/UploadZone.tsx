"use client";
import { useRef, useState } from "react";
import { apiPostForm } from "../lib/api";

export function UploadZone({ onUploaded }: { onUploaded: () => void }) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);

  const upload = async (file: File) => {
    const form = new FormData();
    form.append("file", file);
    setBusy(true);
    try {
      await apiPostForm("/upload", form);
      onUploaded();
    } catch (e) {
      console.error(e);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      style={{
        padding: 16,
        border: "1px dashed #374151",
        borderRadius: 12,
        background: "#0f172a",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        gap: 12,
      }}
    >
      <div>
        <div style={{ fontWeight: 600 }}>Upload CSV</div>
        <div style={{ opacity: 0.8 }}>
          Drag & drop or choose a file. Stored in memory for this session.
        </div>
      </div>
      <div style={{ display: "flex", gap: 8 }}>
        <input
          ref={inputRef}
          type="file"
          accept=".csv,text/csv"
          style={{ display: "none" }}
          onChange={(e) => e.target.files && upload(e.target.files[0])}
        />
        <button onClick={() => inputRef.current?.click()} disabled={busy}>
          {busy ? "Uploading..." : "Choose file"}
        </button>
      </div>
    </div>
  );
}
