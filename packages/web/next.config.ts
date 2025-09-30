import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL,
  },
  // Remove standalone output for Vercel compatibility
  // Vercel handles the build output automatically
};

export default nextConfig;
