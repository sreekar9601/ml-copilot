#!/bin/bash
set -e

echo "--- VERCEL BUILD ENV DEBUG ---"
echo
echo "--- 1. Current Directory ---"
pwd
echo
echo "--- 2. Node & PNPM Versions ---"
node -v || true
pnpm -v || true
echo
echo "--- 3. Repo Root Listing ---"
ls -la || true
echo
echo "--- 4. packages/web listing ---"
ls -la packages/web || true
echo
echo "--- 5. packages/web/src listing ---"
ls -la packages/web/src || true
echo
echo "--- 6. packages/web/tsconfig.json ---"
cat packages/web/tsconfig.json || true
echo
echo "--- 7. packages/web/next.config.ts ---"
cat packages/web/next.config.ts || true
echo
echo "--- 8. pnpm-workspace.yaml ---"
cat pnpm-workspace.yaml || true
echo
echo "--- End of Debug ---"

# Fail intentionally so logs are preserved
exit 1


