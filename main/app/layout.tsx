import type { Metadata } from "next";
import "./globals.css";
import Navbar from "../components/ui/Navbar";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { Package } from "lucide-react"; // Import the package icon from lucide-react

export const metadata: Metadata = {
  title: "HackUTD Omega Project",
  description: "HackUTD Omega Project is a project for HackUTD.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-background text-foreground">
        <Navbar />
        <main>{children}</main>
      </body>
    </html>
  );
}
