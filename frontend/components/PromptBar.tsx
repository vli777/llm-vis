import { useEffect, useRef, useState } from "react";
import { Loader2 } from "lucide-react";

type Props = {
  onSubmit: (prompt: string) => void | Promise<void>;
  placeholder?: string;
  disabled?: boolean; // parent controls in-flight state
};

export function PromptBar({ onSubmit, placeholder, disabled = false }: Props) {
  const [text, setText] = useState("");
  const composingRef = useRef(false);
  const formRef = useRef<HTMLFormElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Focus the input when re-enabled
  useEffect(() => {
    if (!disabled) inputRef.current?.focus();
  }, [disabled]);

  const submit = async () => {
    const value = text.trim();
    if (!value || disabled) return;
    try {
      await onSubmit(value);
      setText(""); // clear only on success
    } catch (err) {
      // hide errors from users; log for devs
      // eslint-disable-next-line no-console
      console.error(err);
    }
  };

  return (
    <form
      ref={formRef}
      onSubmit={(e) => {
        e.preventDefault();
        if (!composingRef.current) submit();
      }}
      aria-busy={disabled || undefined}
      role="search" // semantic enough for a command bar; change if you prefer
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        padding: 8,
        borderRadius: 10,
        border: "1px solid #374151",
        background: "#0b1220",
      }}
    >
      {/* Left-aligned spinner */}
      {disabled ? (
        <Loader2
          aria-hidden="true"
          className="spinner"
          style={{ width: 18, height: 18, marginLeft: 4, marginRight: 4, animation: "spin 1s linear infinite", color: "#00c43bff" }}
        />
      ) : (
        // keep layout stable when spinner hidden
        <span style={{ width: 18, height: 18, marginLeft: 4, marginRight: 4 }} />
      )}

      <input
        ref={inputRef}
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder={placeholder || "Describe a chart"}
        disabled={disabled}
        onCompositionStart={() => { composingRef.current = true; }}
        onCompositionEnd={() => { composingRef.current = false; }}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            // prevent newline in single-line input (just in case)
            e.preventDefault();
          }
        }}
        aria-label="Prompt"
        autoComplete="off"
        spellCheck={false}
        style={{
          flex: 1,
          padding: "10px 12px",
          border: "none",
          outline: "none",
          background: "transparent",
          color: "#e5e7eb",
          opacity: disabled ? 0.6 : 1,
        }}
      />

      <button
        type="submit"
        disabled={disabled || !text.trim()}
        aria-disabled={disabled || !text.trim()}
        style={{
          padding: "10px 14px",
          borderRadius: 8,
          border: "1px solid #374151",
          background: "#0b1220",
          color: "#e5e7eb",
          opacity: disabled || !text.trim() ? 0.6 : 1,
          cursor: disabled ? "not-allowed" : "pointer",
        }}
      >
        Generate
      </button>

      {/* tiny keyframes for inline spinner without a global CSS file */}
      <style>{`
        @keyframes spin { to { transform: rotate(360deg); } }
      `}</style>
    </form>
  );
}