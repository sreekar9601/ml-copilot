"use client";

import { useMutation } from "@tanstack/react-query";
import { ApiResponse, MultiSourceResponse, TutorialResponse } from "@/types/chat";
import { analyzeQuery, QueryAnalysis } from "@/lib/query-analyzer";

interface ChatRequest {
  q: string;
  top_k?: number;
  include_sources?: boolean;
  endpoint?: string;
  max_steps?: number;
  vendor_balance?: boolean;
  integration_focus?: boolean;
}

type ChatResponse = ApiResponse | MultiSourceResponse | TutorialResponse;

export interface EnhancedChatRequest extends ChatRequest {
  tutorialMode?: boolean;
}

export interface EnhancedChatResponse {
  data: ChatResponse;
  queryAnalysis: QueryAnalysis;
}

// Use environment variable for API URL - required for security
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL;

// Only check for API_BASE_URL on the client side to avoid build-time errors
if (typeof window !== 'undefined' && !API_BASE_URL) {
  throw new Error(
    "NEXT_PUBLIC_API_URL environment variable is required. " +
    "Please set it in your Vercel environment variables or .env.local file."
  );
}

// Debug logging (only in development)
if (process.env.NODE_ENV === 'development') {
  console.log("🔍 API_BASE_URL:", API_BASE_URL);
  console.log("🔍 NEXT_PUBLIC_API_URL:", process.env.NEXT_PUBLIC_API_URL);
}

/**
 * Smart API call that automatically routes to the optimal endpoint
 * based on query analysis
 */
async function askQuestionSmart(request: EnhancedChatRequest): Promise<EnhancedChatResponse> {
  // Analyze query to determine optimal endpoint
  const analysis = analyzeQuery(request.q, request.tutorialMode);
  
  // Log routing decision in development
  if (process.env.NODE_ENV === 'development') {
    console.log("🎯 Query Analysis:", {
      query: request.q,
      mode: analysis.mode,
      endpoint: analysis.endpoint,
      reasoning: analysis.reasoning,
      vendors: analysis.detectedVendors,
      confidence: analysis.confidence,
    });
  }
  
  // Prepare request based on endpoint type
  let requestBody: any = {
    q: request.q,
    top_k: request.top_k ?? (analysis.mode === 'tutorial' ? 8 : 10),
    include_sources: request.include_sources ?? true,
  };
  
  // Add endpoint-specific parameters
  if (analysis.mode === 'tutorial') {
    requestBody.max_steps = request.max_steps ?? 8;
    // Tutorial endpoint uses 'query' instead of 'q'
    requestBody = {
      query: requestBody.q,
      max_steps: requestBody.max_steps,
    };
  } else if (analysis.mode === 'multi-source') {
    requestBody.vendor_balance = request.vendor_balance ?? true;
    requestBody.integration_focus = request.integration_focus ?? analysis.detectedVendors.length >= 2;
  }
  
  // Make API call to selected endpoint
  const response = await fetch(`${API_BASE_URL}${analysis.endpoint}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(requestBody),
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => null);
    throw new Error(
      errorData?.detail || `HTTP ${response.status}: ${response.statusText}`
    );
  }

  const data = await response.json();
  
  return {
    data,
    queryAnalysis: analysis,
  };
}

/**
 * Hook for smart chat API with automatic endpoint routing
 * Uses query analysis to maximize advanced RAG features
 */
export function useChatApi() {
  return useMutation({
    mutationFn: askQuestionSmart,
    onError: (error) => {
      console.error("Chat API error:", error);
    },
  });
}

/**
 * Legacy hook for direct endpoint calls (if needed for debugging)
 */
export function useChatApiDirect(endpoint: string = '/ask') {
  return useMutation({
    mutationFn: async (request: ChatRequest): Promise<ApiResponse> => {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(request),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(
          errorData?.detail || `HTTP ${response.status}: ${response.statusText}`
        );
      }

      return response.json();
    },
    onError: (error) => {
      console.error("Chat API error:", error);
    },
  });
}
