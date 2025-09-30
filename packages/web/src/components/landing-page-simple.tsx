"use client";

import { Button } from "@/components/ui/button";
import { MessageSquare } from "lucide-react";

interface LandingPageSimpleProps {
  onStartChat: () => void;
}

export function LandingPageSimple({ onStartChat }: LandingPageSimpleProps) {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-background via-background to-muted/30">
      <div className="text-center max-w-2xl mx-auto px-4">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-4 text-foreground">
            ML Documentation Copilot
          </h1>
          <p className="text-xl text-muted-foreground mb-8">
            Your intelligent AI assistant for ML infrastructure
          </p>
        </div>
        
        <Button
          size="lg"
          className="h-14 px-8 text-lg font-semibold"
          onClick={onStartChat}
        >
          <MessageSquare className="h-5 w-5 mr-2" />
          Start Chatting
        </Button>
      </div>
    </div>
  );
}
