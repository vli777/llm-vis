import { applyPatch } from "fast-json-patch"; 

export function normalizeMark(spec: any) {
  if (typeof spec.mark === "string") {
    spec.mark = { type: spec.mark };
  }
  return spec;
}

export function applyVizPatch(card: any, patch: any[]) {
  const spec = normalizeMark(card.spec);
  const patched = applyPatch(spec, patch, /*validate*/ false).newDocument;
  return { ...card, spec: patched };
}
