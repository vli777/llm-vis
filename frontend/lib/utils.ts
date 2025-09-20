export function humanBytes(n?: number | null) {
  if (!n && n !== 0) return "";
  const units = ["B","KB","MB","GB","TB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v % 1 === 0 ? v : v.toFixed(1)} ${units[i]}`;
}