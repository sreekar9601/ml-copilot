export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

export interface Source {
  chunk_id: string;
  title: string;
  url: string;
  heading_path: string;
  anchor_link: string;
  relevance_score: number;
}

export interface ApiResponse {
  answer: string;
  sources: Source[];
  query: string;
  retrieval_time_ms: number;
  generation_time_ms: number;
  total_time_ms: number;
  chunks_retrieved: number;
}
