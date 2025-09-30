"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { 
  Bot, 
  Zap, 
  BookOpen, 
  Code, 
  Database, 
  Cloud, 
  ArrowRight, 
  Sparkles,
  MessageSquare,
  CheckCircle,
  Star
} from "lucide-react";
import { cn } from "@/lib/utils";

interface LandingPageProps {
  onStartChat: () => void;
}

export function LandingPage({ onStartChat }: LandingPageProps) {
  const [isHovered, setIsHovered] = useState(false);

  const features = [
    {
      icon: <Database className="h-5 w-5" />,
      title: "Multi-Source Knowledge",
      description: "Access PyTorch, MLflow, AWS SageMaker, Kubernetes, and Docker documentation"
    },
    {
      icon: <Code className="h-5 w-5" />,
      title: "Code Examples",
      description: "Get practical code snippets and implementation examples"
    },
    {
      icon: <BookOpen className="h-5 w-5" />,
      title: "Best Practices",
      description: "Learn production-grade ML infrastructure patterns"
    },
    {
      icon: <Cloud className="h-5 w-5" />,
      title: "Production Ready",
      description: "Enterprise-grade answers with source citations"
    }
  ];

  const exampleQuestions = [
    "How to deploy models with AWS SageMaker?",
    "Kubernetes service networking best practices",
    "MLflow model registry and tracking",
    "PyTorch DataLoader with MLflow integration",
    "Docker multi-stage builds for ML applications",
    "Kubernetes resource limits for ML workloads"
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20 flex flex-col">
      {/* Hero Section */}
      <div className="container mx-auto px-4 py-16">
        <div className="text-center max-w-4xl mx-auto">
          {/* Header */}
          <div className="flex items-center justify-center gap-3 mb-8">
            <div className="h-12 w-12 rounded-xl bg-gradient-to-br from-primary to-primary/80 flex items-center justify-center shadow-lg">
              <Bot className="h-6 w-6 text-primary-foreground" />
            </div>
            <div className="text-left">
              <h1 className="text-3xl font-bold bg-gradient-to-r from-foreground to-foreground/80 bg-clip-text text-transparent">
                ML Documentation Copilot
              </h1>
              <p className="text-muted-foreground">
                Your intelligent AI assistant for ML infrastructure
              </p>
            </div>
          </div>

          {/* Main CTA */}
          <div className="mb-12">
            <h2 className="text-5xl font-bold mb-6 bg-gradient-to-r from-foreground via-foreground to-foreground/80 bg-clip-text text-transparent">
              Ask Questions, Get
              <span className="bg-gradient-to-r from-primary to-primary/80 bg-clip-text text-transparent"> Expert Answers</span>
            </h2>
            <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto leading-relaxed">
              Get comprehensive answers with detailed citations from official documentation. 
              Perfect for ML engineers, DevOps teams, and data scientists.
            </p>
            
            <Button
              size="lg"
              className="h-14 px-8 text-lg font-semibold bg-gradient-to-r from-primary to-primary/90 hover:from-primary/90 hover:to-primary shadow-lg hover:shadow-xl transition-all duration-300"
              onClick={onStartChat}
              onMouseEnter={() => setIsHovered(true)}
              onMouseLeave={() => setIsHovered(false)}
            >
              <MessageSquare className="h-5 w-5 mr-2" />
              Start Chatting
              <ArrowRight className={cn("h-5 w-5 ml-2 transition-transform duration-300", isHovered && "translate-x-1")} />
            </Button>
          </div>

          {/* Status Badge */}
          <div className="flex items-center justify-center gap-2 mb-12">
            <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse"></div>
            <Badge variant="secondary" className="text-sm">
              <Zap className="h-3 w-3 mr-1" />
              Powered by Google Gemini
            </Badge>
            <Badge variant="outline" className="text-sm">
              <CheckCircle className="h-3 w-3 mr-1" />
              Production Ready
            </Badge>
          </div>
        </div>

        {/* Features Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-16">
          {features.map((feature, index) => (
            <Card 
              key={index}
              className="group hover:shadow-lg transition-all duration-300 hover:-translate-y-1 border-0 bg-card/50 backdrop-blur-sm"
            >
              <CardContent className="p-6 text-center">
                <div className="h-12 w-12 mx-auto mb-4 rounded-lg bg-primary/10 flex items-center justify-center text-primary group-hover:bg-primary/20 transition-colors duration-300">
                  {feature.icon}
                </div>
                <h3 className="font-semibold mb-2 text-foreground">{feature.title}</h3>
                <p className="text-sm text-muted-foreground leading-relaxed">{feature.description}</p>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Example Questions */}
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-8">
            <h3 className="text-2xl font-bold mb-2 text-foreground">Try These Questions</h3>
            <p className="text-muted-foreground">Click any question to start chatting</p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {exampleQuestions.map((question, index) => (
              <Card 
                key={index}
                className="group cursor-pointer hover:shadow-md transition-all duration-300 hover:-translate-y-1 border-0 bg-card/30 backdrop-blur-sm hover:bg-card/50"
                onClick={onStartChat}
              >
                <CardContent className="p-4">
                  <div className="flex items-start gap-3">
                    <div className="h-8 w-8 rounded-lg bg-primary/10 flex items-center justify-center text-primary group-hover:bg-primary/20 transition-colors duration-300 flex-shrink-0">
                      <MessageSquare className="h-4 w-4" />
                    </div>
                    <p className="text-sm text-foreground group-hover:text-primary transition-colors duration-300 leading-relaxed">
                      {question}
                    </p>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>

        {/* Footer Stats */}
        <div className="mt-16 text-center">
          <div className="flex items-center justify-center gap-8 text-sm text-muted-foreground">
            <div className="flex items-center gap-2">
              <Star className="h-4 w-4 text-yellow-500" />
              <span>Enhanced with multi-source retrieval</span>
            </div>
            <div className="flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-primary" />
              <span>Production-grade quality</span>
            </div>
            <div className="flex items-center gap-2">
              <CheckCircle className="h-4 w-4 text-green-500" />
              <span>Built with Next.js & FastAPI</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
