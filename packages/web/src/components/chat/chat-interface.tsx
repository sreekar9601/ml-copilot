"use client";

import { useState } from "react";
import { Message, TutorialResponse, ApiResponse, MultiSourceResponse } from "@/types/chat";
import { ChatMessages } from "./chat-messages";
import { ChatInput } from "./chat-input";
import { useChatApi } from "@/hooks/use-chat-api";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ArrowLeft, GraduationCap } from "lucide-react";
import { cn } from "@/lib/utils";
import { getActiveFeatures } from "@/lib/query-analyzer";

interface ChatInterfaceProps {
  className?: string;
  onBackToLanding?: () => void;
}

export function ChatInterface({ className, onBackToLanding }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [tutorialMode, setTutorialMode] = useState(false);
  const chatApi = useChatApi();

  const handleSendMessage = async (content: string) => {
    // Create user message
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      role: "user",
      content,
      timestamp: new Date(),
    };

    // Add user message immediately
    setMessages(prev => [...prev, userMessage]);

    try {
      // Call smart API with tutorial mode flag
      const response = await chatApi.mutateAsync({
        q: content,
        tutorialMode,
      });

      const { data, queryAnalysis } = response;

      // Determine content based on response type
      let messageContent = "";
      let tutorial: TutorialResponse | undefined;
      let sources: any[] = [];

      if ('tutorial' in data) {
        // Tutorial response
        const tutorialData = data as TutorialResponse;
        tutorial = tutorialData;
        messageContent = `# ${tutorialData.tutorial.title}\n\nTutorial with ${tutorialData.tutorial.steps.length} steps generated successfully.`;
      } else {
        // Standard or multi-source response
        const apiData = data as ApiResponse | MultiSourceResponse;
        messageContent = apiData.answer;
        sources = apiData.sources;
      }

      // Create assistant message with enhanced metadata
      const assistantMessage: Message = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: messageContent,
        timestamp: new Date(),
        sources: sources,
        tutorial: tutorial,
        queryMode: queryAnalysis.mode,
        queryAnalysis: {
          reasoning: queryAnalysis.reasoning,
          detectedVendors: queryAnalysis.detectedVendors,
          activeFeatures: getActiveFeatures(queryAnalysis.mode),
        },
      };

      // Add assistant response
      setMessages(prev => [...prev, assistantMessage]);

    } catch (error) {
      // Create error message
      const errorMessage: Message = {
        id: `error-${Date.now()}`,
        role: "assistant",
        content: `I encountered an error while processing your question: ${
          error instanceof Error ? error.message : "Unknown error"
        }. Please try again or check if the backend service is running.`,
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, errorMessage]);
    }
  };

  return (
    <Card className={cn("flex flex-col h-full", className)}>
      <div className="p-4 border-b bg-muted/30 space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold">Chat with ML Documentation Copilot</h2>
            <p className="text-sm text-muted-foreground">
              Ask questions about ML frameworks, deployment, and best practices
            </p>
          </div>
          {onBackToLanding && (
            <Button
              variant="ghost"
              size="sm"
              onClick={onBackToLanding}
              className="flex items-center gap-2"
            >
              <ArrowLeft className="h-4 w-4" />
              Back to Home
            </Button>
          )}
        </div>

        {/* Tutorial Mode Toggle */}
        <div className="flex items-center gap-3 pt-2 border-t">
          <Button
            variant={tutorialMode ? "default" : "outline"}
            size="sm"
            onClick={() => setTutorialMode(!tutorialMode)}
            className="flex items-center gap-2"
          >
            <GraduationCap className="h-4 w-4" />
            Tutorial Mode
          </Button>
          
          {tutorialMode ? (
            <div className="flex-1">
              <p className="text-xs text-muted-foreground">
                📖 Queries will generate step-by-step tutorials with commands and code examples
              </p>
            </div>
          ) : (
            <div className="flex-1">
              <p className="text-xs text-muted-foreground">
                🎯 Smart routing: Multi-vendor detection, query expansion, and adaptive search
              </p>
            </div>
          )}
        </div>
      </div>
      <ChatMessages messages={messages} className="flex-1" />
      <ChatInput
        onSubmit={handleSendMessage}
        isLoading={chatApi.isPending}
        className="border-0 border-t rounded-none bg-muted/20"
      />
    </Card>
  );
}
