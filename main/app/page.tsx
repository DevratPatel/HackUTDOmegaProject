import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Button } from "@/components/ui/button";
import { Package } from "lucide-react";

export default function DashboardPage() {
  return (
    <div className="min-h-screen bg-background">
      <main className="container mx-auto p-6 space-y-10">
        {/* Usage Section */}
        <div>
          <h2 className="text-2xl font-semibold mb-6">Current Usage</h2>

          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {/* Storage Card */}
            <Card>
              <CardHeader>
                <CardTitle>Storage</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xl text-muted-foreground">
                      75 / 100 GB
                    </span>
                    <span className="text-sm text-muted-foreground">75%</span>
                  </div>
                  <Progress value={75} className="h-2" />
                </div>
              </CardContent>
            </Card>

            {/* Bandwidth Card */}
            <Card>
              <CardHeader>
                <CardTitle>Bandwidth</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xl text-muted-foreground">
                      50 / 100 TB
                    </span>
                    <span className="text-sm text-muted-foreground">50%</span>
                  </div>
                  <Progress value={50} className="h-2" />
                </div>
              </CardContent>
            </Card>

            {/* Users Card */}
            <Card>
              <CardHeader>
                <CardTitle>Users</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xl text-muted-foreground">
                      8 / 10 seats
                    </span>
                    <span className="text-sm text-muted-foreground">80%</span>
                  </div>
                  <Progress value={80} className="h-2" />
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Recommended Upgrades Section */}
        <div>
          <h2 className="text-2xl font-semibold mb-6">Recommended Upgrades</h2>

          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {/* Storage Upgrade */}
            <Card className="relative">
              <CardHeader>
                <Package className="w-10 h-10 mb-4" />
                <CardTitle>Storage Plus</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <p className="text-muted-foreground">
                    Increase your storage to 250GB
                  </p>
                  <p className="text-2xl font-bold">$19.99/mo</p>
                </div>
                <Button className="w-full" variant="default">
                  Upgrade <span className="ml-2">→</span>
                </Button>
              </CardContent>
            </Card>

            {/* Bandwidth Upgrade */}
            <Card className="relative">
              <CardHeader>
                <Package className="w-10 h-10 mb-4" />
                <CardTitle>Bandwidth Pro</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <p className="text-muted-foreground">
                    Upgrade to 250TB bandwidth
                  </p>
                  <p className="text-2xl font-bold">$29.99/mo</p>
                </div>
                <Button className="w-full" variant="default">
                  Upgrade <span className="ml-2">→</span>
                </Button>
              </CardContent>
            </Card>

            {/* Team Upgrade */}
            <Card className="relative">
              <CardHeader>
                <Package className="w-10 h-10 mb-4" />
                <CardTitle>Team Scale</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <p className="text-muted-foreground">
                    Expand to 25 team seats
                  </p>
                  <p className="text-2xl font-bold">$49.99/mo</p>
                </div>
                <Button className="w-full" variant="default">
                  Upgrade <span className="ml-2">→</span>
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
    </div>
  );
}
