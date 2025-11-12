import { QueryMode } from "@/lib/query-analyzer";

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  sources?: Source[];
  tutorial?: TutorialResponse;
  queryMode?: QueryMode;
  queryAnalysis?: {
    reasoning: string;
    detectedVendors: string[];
    activeFeatures: string[];
  };
}

export interface Source {
  chunk_id: string;
  title: string;
  url: string;
  heading_path: string;
  anchor_link: string;
  relevance_score: number;
  vendor?: string;
  doc_type?: string;
  topics?: string[];
  quality_score?: number;
  has_code_examples?: boolean;
  technical_depth?: string;
}

// Standard Q&A response
export interface ApiResponse {
  answer: string;
  sources: Source[];
  query: string;
  retrieval_time_ms: number;
  generation_time_ms: number;
  total_time_ms: number;
  chunks_retrieved: number;
}

// Multi-source response (extends ApiResponse)
export interface MultiSourceResponse extends ApiResponse {
  query_analysis: {
    detected_vendors: Array<{
      vendor: string;
      confidence: number;
      entities: string[];
      use_cases: string[];
    }>;
    integration_intent: boolean;
    comparison_intent: boolean;
    complexity_level: string;
    cross_vendor_entities: string[];
    required_use_cases: string[];
  };
  vendor_distribution: Record<string, number>;
  integration_suggestions: string[];
}

// Tutorial/How-to response
export interface TutorialResponse {
  tutorial: {
    title: string;
    intent: string;
    prereqs: string[];
    steps: TutorialStep[];
  };
  processing_time: number;
  plan_steps: number;
  completed_steps: number;
  total_citations: number;
}

export interface TutorialStep {
  title: string;
  explanation: string;
  commands?: string[];
  code?: string;
  notes?: string[];
  citations: Citation[];
}

export interface Citation {
  title: string;
  url: string;
  quote: string;
  source_vendor?: string;
}
