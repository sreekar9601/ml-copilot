"use client";

import { useEffect, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Sparkles, Zap, Shield, Cpu, ArrowRight } from "lucide-react";
import gsap from "gsap";

interface LandingPageEnhancedProps {
  onStartChat: () => void;
}

export function LandingPageEnhanced({ onStartChat }: LandingPageEnhancedProps) {
  const heroRef = useRef<HTMLDivElement>(null);
  const featuresRef = useRef<HTMLDivElement>(null);
  const ctaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const ctx = gsap.context(() => {
      // Hero animation
      gsap.fromTo(heroRef.current, 
        {
          opacity: 0,
          y: 30,
        },
        {
          opacity: 1,
          y: 0,
          duration: 1,
          ease: "power3.out",
        }
      );

      // Features stagger animation
      gsap.fromTo(".feature-card", 
        {
          opacity: 0,
          y: 50,
        },
        {
          opacity: 1,
          y: 0,
          stagger: 0.2,
          duration: 0.8,
          ease: "power3.out",
          delay: 0.3,
        }
      );

      // CTA animation
      gsap.fromTo(ctaRef.current, 
        {
          opacity: 0,
          scale: 0.95,
        },
        {
          opacity: 1,
          scale: 1,
          duration: 0.6,
          ease: "back.out(1.7)",
          delay: 1,
        }
      );

      // Floating animation for sparkles
      gsap.to(".sparkle-icon", {
        y: -10,
        duration: 2,
        ease: "power1.inOut",
        yoyo: true,
        repeat: -1,
      });
    });

    return () => ctx.revert();
  }, []);

  const features = [
    {
      icon: Zap,
      title: "Lightning Fast",
      description: "Get instant answers powered by advanced AI models and semantic search",
    },
    {
      icon: Shield,
      title: "Accurate & Reliable",
      description: "Responses backed by official documentation from PyTorch, MLflow, and more",
    },
    {
      icon: Cpu,
      title: "Multi-Source",
      description: "Access knowledge from multiple ML frameworks in one unified interface",
    },
  ];

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-background via-background to-primary/5 px-4 py-16">
      {/* Hero Section */}
      <div ref={heroRef} className="text-center max-w-4xl mx-auto mb-16">

        <h1 className="text-5xl md:text-7xl font-bold mb-6 bg-gradient-to-r from-foreground via-foreground to-foreground/70 bg-clip-text text-transparent">
          ML Documentation Copilot
        </h1>

        <p className="text-xl md:text-2xl text-muted-foreground mb-8 max-w-2xl mx-auto leading-relaxed">
          Your intelligent AI assistant for{" "}
          <span className="text-primary font-semibold">PyTorch</span>,{" "}
          <span className="text-primary font-semibold">MLflow</span>,{" "}
          <span className="text-primary font-semibold">AWS SageMaker</span>,{" "}
          and more
        </p>

        <div className="flex flex-wrap items-center justify-center gap-4 text-sm text-muted-foreground">
          <div className="flex items-center gap-2">
            <div className="h-2 w-2 rounded-full bg-green-500 animate-pulse"></div>
            <span>Real-time Responses</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="h-2 w-2 rounded-full bg-blue-500 animate-pulse"></div>
            <span>Source Citations</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="h-2 w-2 rounded-full bg-purple-500 animate-pulse"></div>
            <span>Code Examples</span>
          </div>
        </div>
      </div>

      {/* Features Grid */}
      <div ref={featuresRef} className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-6xl mx-auto mb-16">
        {features.map((feature, index) => (
          <Card
            key={index}
            className="feature-card p-6 bg-card backdrop-blur-sm border-2 border-primary/20 hover:border-primary/40 transition-all duration-300 hover:shadow-lg hover:scale-105"
          >
            <div className="flex flex-col items-center text-center">
              <div className="h-12 w-12 rounded-lg bg-primary/20 flex items-center justify-center mb-4">
                <feature.icon className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-lg font-semibold mb-2 text-foreground">{feature.title}</h3>
              <p className="text-sm text-muted-foreground">{feature.description}</p>
            </div>
          </Card>
        ))}
      </div>

      {/* CTA Section */}
      <div ref={ctaRef} className="text-center">
        <Button
          size="lg"
          className="text-lg px-8 py-6 h-auto rounded-full shadow-lg hover:shadow-xl transition-all duration-300 group"
          onClick={onStartChat}
        >
          <Sparkles className="h-5 w-5 mr-2 group-hover:rotate-12 transition-transform" />
          Start Chatting
          <ArrowRight className="h-5 w-5 ml-2 group-hover:translate-x-1 transition-transform" />
        </Button>

        <p className="text-sm text-muted-foreground mt-4">
          No login required • Free to use • Instant answers
        </p>
      </div>

      {/* Background decorations */}
      <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none">
        <div className="absolute top-20 left-10 w-72 h-72 bg-primary/5 rounded-full blur-3xl"></div>
        <div className="absolute bottom-20 right-10 w-96 h-96 bg-primary/5 rounded-full blur-3xl"></div>
      </div>
    </div>
  );
}
