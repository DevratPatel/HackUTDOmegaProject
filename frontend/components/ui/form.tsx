import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";

export function PersonalizationForm() {
  return (
    <Card className="w-full max-w-2xl mx-auto">
      <CardHeader>
        <CardTitle className="text-2xl">Personalize Your Experience</CardTitle>
        <p className="text-muted-foreground">
          Answer a few questions to get tailored recommendations
        </p>
      </CardHeader>
      <CardContent className="space-y-8">
        <div className="space-y-4">
          <h3 className="text-lg font-semibold">What's your business size?</h3>
          <RadioGroup defaultValue="small">
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="small" id="small" />
              <Label htmlFor="small">Small Business</Label>
            </div>
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="enterprise" id="enterprise" />
              <Label htmlFor="enterprise">Enterprise</Label>
            </div>
          </RadioGroup>
        </div>

        <div className="space-y-4">
          <h3 className="text-lg font-semibold">
            How would you describe your data usage?
          </h3>
          <RadioGroup defaultValue="low">
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="low" id="low" />
              <Label htmlFor="low">Low</Label>
            </div>
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="high" id="high" />
              <Label htmlFor="high">High</Label>
            </div>
          </RadioGroup>
        </div>

        <div className="space-y-4">
          <h3 className="text-lg font-semibold">What's your team size?</h3>
          <RadioGroup defaultValue="small-team">
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="small-team" id="small-team" />
              <Label htmlFor="small-team">Small (1-10)</Label>
            </div>
            <div className="flex items-center space-x-2">
              <RadioGroupItem value="large-team" id="large-team" />
              <Label htmlFor="large-team">Large (11+)</Label>
            </div>
          </RadioGroup>
        </div>

        <Button className="w-full">Submit Answers</Button>
      </CardContent>
    </Card>
  );
}
