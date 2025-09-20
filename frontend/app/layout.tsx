import "./globals.css";

export const metadata = { title: "AI Data Vis", description: "Turn prompts into charts" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-slate-950 text-slate-200 antialiased">{children}</body>
    </html>
  );
}