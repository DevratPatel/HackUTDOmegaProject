import { PersonalizationForm } from "@/components/ui/form";
import { UsageStats } from "@/components/ui/usage-stats";
import { RecommendedUpgrades } from "@/components/ui/recommended-upgrades";
import Navbar from "../../components/ui/Navbar";
export default function DashboardPage() {
  return (
    <div className="min-h-screen bg-background">
      <Navbar />
      <main className="container mx-auto p-6 space-y-10">
        <div>
          <PersonalizationForm />
        </div>
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