"use client";
import { UsageStats } from "@/components/ui/usage-stats";
import { RecommendedUpgrades } from "@/components/ui/recommended-upgrades";
import Navbar from "../../components/ui/Navbar";
import { ChatBot } from "@/app/chat/chat";
export default function DashboardPage() {
  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <ChatBot />
      <main className="container mx-auto p-6 space-y-10">
        <div>
          <UsageStats />
        </div>
        <div>
          <RecommendedUpgrades />
        </div>
      </main>
    </div>
  );
}
