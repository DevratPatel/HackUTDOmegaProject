import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "HackUTD Omega Project",
  description: "HackUTD Omega Project is a project for HackUTD.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body></body>
    </html>
  );
}
