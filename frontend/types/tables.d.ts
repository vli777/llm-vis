export type TableMeta = {
  file_name: string | null;
  file_ext: string | null;
  file_size: number | null; // bytes
  created_at: string | null; // ISO
  n_rows: number;
  n_cols: number;
  columns: string[];
  dtypes: Record<string, string>;
};

export type TableInfo = { name: string } & TableMeta;