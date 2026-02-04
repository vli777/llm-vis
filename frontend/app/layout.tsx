import "./globals.css";

export const metadata = {
  title: "AI Data Vis",
  description: "Turn prompts into charts",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="theme-body antialiased">
        {children}
      </body>
    </html>
  );
}
