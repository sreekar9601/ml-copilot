#!/usr/bin/env python3
"""
Focused Re-ingestion for Critical Sources
Targets PyTorch (failed) and missing Hugging Face PEFT/TRL docs
"""

import os
import sys
import asyncio

# Setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from complete_end_to_end_ingestion import CompleteMLIngestion
from dotenv import load_dotenv

load_dotenv()


async def main():
    """Run focused re-ingestion."""
    print("\n" + "="*60)
    print("FOCUSED RE-INGESTION")
    print("Targeting: PyTorch + Hugging Face PEFT/TRL")
    print("="*60 + "\n")
    
    # Get credentials
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')
    
    if not qdrant_url or not qdrant_api_key:
        print("ERROR: QDRANT_URL and QDRANT_API_KEY must be set")
        return
    
    # Initialize ingestion system
    ingestion = CompleteMLIngestion(
        sources_file='focused_reingestion.yaml',
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key
    )
    
    # Run ingestion - no source limit, get all critical sources
    await ingestion.run_complete_ingestion(
        tier_filter=None,
        max_sources=10  # Should get all 10 critical sources
    )
    
    print("\n✅ Focused re-ingestion complete!")
    print("\nYou should now have:")
    print("  - PyTorch documentation")
    print("  - Hugging Face Transformers")
    print("  - PEFT (LoRA/QLoRA)")
    print("  - TRL (training)")
    print("  - Quantization guides")
    print("\nTest your LLM finetuning question again!")


if __name__ == "__main__":
    asyncio.run(main())


