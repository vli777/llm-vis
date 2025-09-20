export const metadata = { title: "AI Data Vis", description: "Turn prompts into charts" };
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{ fontFamily: "system-ui, sans-serif", margin: 0, background: "#0b0c10", color: "#e5e7eb"}}>
        {children}
      </body>
    </html>
  );
}
