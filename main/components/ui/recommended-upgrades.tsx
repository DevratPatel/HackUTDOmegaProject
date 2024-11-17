import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Package } from "lucide-react";

export function RecommendedUpgrades() {
  return (
    <div>
      <h2 className="text-2xl font-semibold mb-6">Recommended Upgrades</h2>
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
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

        <Card className="relative">
          <CardHeader>
            <Package className="w-10 h-10 mb-4" />
            <CardTitle>Team Scale</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <p className="text-muted-foreground">Expand to 25 team seats</p>
              <p className="text-2xl font-bold">$49.99/mo</p>
            </div>
            <Button className="w-full" variant="default">
              Upgrade <span className="ml-2">→</span>
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
