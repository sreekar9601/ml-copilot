"""Agent evaluation metrics and runner."""

import time
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
from langchain_core.messages import HumanMessage
import json

from ..orchestrator import orchestrator
from .test_scenarios import (
    get_all_scenarios,
    get_multi_turn_scenarios,
    get_performance_scenarios
)


class AgentEvaluator:
    """Evaluate agent performance against test scenarios."""
    
    def __init__(self):
        self.results = []
    
    async def run_scenario(self, scenario: dict) -> dict:
        """Run a single test scenario and evaluate."""
        
        print(f"\n{'='*60}")
        print(f"Running: {scenario['id']} - {scenario.get('category', 'unknown')}")
        print(f"Query: {scenario['query'][:80]}{'...' if len(scenario['query']) > 80 else ''}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            initial_state = {
                "messages": [HumanMessage(content=scenario["query"])],
                "conversation_id": f"test_{scenario['id']}",
                "iteration_count": 0,
                "total_cost": 0.0,
                "current_tool": None,
                "tool_results": [],
                "reflection": None,
                "needs_improvement": False,
                "detected_frameworks": [],
                "query_intent": "unknown"
            }
            
            result = await orchestrator.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": f"test_{scenario['id']}"}}
            )
            
            latency_ms = (time.time() - start_time) * 1000
            response = result["messages"][-1].content
            cost = result["total_cost"]
            tools_used = list(set(
                tr.get("tool_name", "unknown") 
                for tr in result.get("tool_results", [])
                if "tool_name" in tr
            ))
            frameworks_detected = result.get("detected_frameworks", [])
            
            # Evaluate success criteria
            success = scenario["success_criteria"](response)
            
            # Check latency
            latency_ok = latency_ms <= scenario["max_latency_ms"]
            
            # Check cost
            cost_ok = cost <= scenario["max_cost_usd"]
            
            # Check tools (if specified)
            tools_ok = True
            if "expected_tools" in scenario:
                expected = set(scenario["expected_tools"])
                actual = set(tools_used)
                tools_ok = len(expected & actual) > 0  # At least one tool matches
            
            # Check frameworks (if specified)
            frameworks_ok = True
            if "expected_frameworks" in scenario:
                expected_fw = set(fw.lower() for fw in scenario["expected_frameworks"])
                actual_fw = set(fw.lower() for fw in frameworks_detected)
                frameworks_ok = len(expected_fw & actual_fw) > 0  # At least one matches
            
            overall_success = success and latency_ok and cost_ok and tools_ok and frameworks_ok
            
            # Print results
            print(f"\n✅ PASS" if overall_success else f"\n❌ FAIL")
            print(f"  Criteria Met: {'✓' if success else '✗'}")
            print(f"  Latency: {latency_ms:.0f}ms {'✓' if latency_ok else '✗ (>' + str(scenario['max_latency_ms']) + 'ms)'}")
            print(f"  Cost: ${cost:.4f} {'✓' if cost_ok else '✗ (>$' + str(scenario['max_cost_usd']) + ')'}")
            print(f"  Tools: {', '.join(tools_used)} {'✓' if tools_ok else '✗'}")
            if frameworks_detected:
                print(f"  Frameworks: {', '.join(frameworks_detected)} {'✓' if frameworks_ok else '✗'}")
            print(f"  Response length: {len(response)} chars")
            
            return {
                "scenario_id": scenario["id"],
                "category": scenario.get("category", "unknown"),
                "query": scenario["query"],
                "success": overall_success,
                "response": response,
                "response_length": len(response),
                "latency_ms": latency_ms,
                "latency_ok": latency_ok,
                "cost_usd": cost,
                "cost_ok": cost_ok,
                "tools_used": tools_used,
                "tools_ok": tools_ok,
                "frameworks_detected": frameworks_detected,
                "frameworks_ok": frameworks_ok,
                "criteria_met": success,
                "error": None,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            return {
                "scenario_id": scenario["id"],
                "category": scenario.get("category", "unknown"),
                "query": scenario["query"],
                "success": False,
                "error": str(e),
                "latency_ms": (time.time() - start_time) * 1000,
                "cost_usd": 0.0,
                "tools_used": [],
                "response": None,
                "timestamp": datetime.now().isoformat()
            }
    
    async def run_multi_turn_scenario(self, scenario: dict) -> dict:
        """Run a multi-turn conversation scenario."""
        
        print(f"\n{'='*60}")
        print(f"Running Multi-Turn: {scenario['id']}")
        print(f"{'='*60}")
        
        conversation_id = f"test_multi_{scenario['id']}"
        total_cost = 0.0
        turn_results = []
        overall_success = True
        
        for i, turn in enumerate(scenario["conversation"], 1):
            print(f"\nTurn {i}/{len(scenario['conversation'])}: {turn['query'][:60]}...")
            
            start_time = time.time()
            
            try:
                # For multi-turn, we need to get previous messages from memory
                from ..memory.chat_history import EnhancedChatHistory
                memory = EnhancedChatHistory(conversation_id)
                history = memory.get_messages()
                
                initial_state = {
                    "messages": [*history, HumanMessage(content=turn["query"])],
                    "conversation_id": conversation_id,
                    "iteration_count": 0,
                    "total_cost": 0.0,
                    "current_tool": None,
                    "tool_results": [],
                    "reflection": None,
                    "needs_improvement": False,
                    "detected_frameworks": [],
                    "query_intent": "unknown"
                }
                
                result = await orchestrator.ainvoke(
                    initial_state,
                    config={"configurable": {"thread_id": conversation_id}}
                )
                
                response = result["messages"][-1].content
                cost = result["total_cost"]
                total_cost += cost
                
                # Save to memory for next turn
                memory.add_messages(
                    [HumanMessage(content=turn["query"]), result["messages"][-1]],
                    cost=cost
                )
                
                # Check success criteria
                success = turn["success_criteria"](response)
                overall_success = overall_success and success
                
                turn_results.append({
                    "turn": i,
                    "query": turn["query"],
                    "response_length": len(response),
                    "latency_ms": (time.time() - start_time) * 1000,
                    "cost_usd": cost,
                    "success": success
                })
                
                print(f"  {'✓' if success else '✗'} Response: {len(response)} chars, ${cost:.4f}")
                
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
                overall_success = False
                turn_results.append({
                    "turn": i,
                    "query": turn["query"],
                    "error": str(e),
                    "success": False
                })
        
        cost_ok = total_cost <= scenario["max_total_cost_usd"]
        final_success = overall_success and cost_ok
        
        print(f"\n{'✅ PASS' if final_success else '❌ FAIL'} Multi-Turn Complete")
        print(f"  Total Cost: ${total_cost:.4f} {'✓' if cost_ok else '✗'}")
        
        return {
            "scenario_id": scenario["id"],
            "category": "multi_turn",
            "success": final_success,
            "total_cost_usd": total_cost,
            "cost_ok": cost_ok,
            "turns": turn_results,
            "timestamp": datetime.now().isoformat()
        }
    
    async def run_all(self, include_multi_turn: bool = True) -> Dict:
        """Run all test scenarios and generate report."""
        
        print("\n" + "="*60)
        print("🧪 AGENT EVALUATION SUITE")
        print("="*60)
        
        start_time = datetime.now()
        
        # Run single-turn scenarios
        scenarios = get_all_scenarios()
        print(f"\n📋 Running {len(scenarios)} single-turn scenarios...")
        
        results = []
        for scenario in scenarios:
            result = await self.run_scenario(scenario)
            results.append(result)
            await asyncio.sleep(0.5)  # Small delay between tests
        
        # Run multi-turn scenarios
        multi_turn_results = []
        if include_multi_turn:
            multi_scenarios = get_multi_turn_scenarios()
            print(f"\n📋 Running {len(multi_scenarios)} multi-turn scenarios...")
            
            for scenario in multi_scenarios:
                result = await self.run_multi_turn_scenario(scenario)
                multi_turn_results.append(result)
                await asyncio.sleep(0.5)
        
        # Calculate metrics
        total = len(results)
        passed = sum(1 for r in results if r["success"])
        avg_latency = sum(r["latency_ms"] for r in results) / total if total > 0 else 0
        total_cost = sum(r["cost_usd"] for r in results)
        
        # Calculate by category
        categories = {}
        for result in results:
            cat = result.get("category", "unknown")
            if cat not in categories:
                categories[cat] = {"total": 0, "passed": 0}
            categories[cat]["total"] += 1
            if result["success"]:
                categories[cat]["passed"] += 1
        
        # Multi-turn metrics
        multi_turn_passed = sum(1 for r in multi_turn_results if r["success"])
        multi_turn_total = len(multi_turn_results)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        report = {
            "summary": {
                "total_scenarios": total,
                "passed": passed,
                "failed": total - passed,
                "success_rate": passed / total if total > 0 else 0,
                "avg_latency_ms": avg_latency,
                "total_cost_usd": total_cost,
                "avg_cost_usd": total_cost / total if total > 0 else 0,
                "duration_seconds": duration,
                "timestamp": start_time.isoformat()
            },
            "by_category": {
                cat: {
                    "total": stats["total"],
                    "passed": stats["passed"],
                    "success_rate": stats["passed"] / stats["total"] if stats["total"] > 0 else 0
                }
                for cat, stats in categories.items()
            },
            "multi_turn": {
                "total": multi_turn_total,
                "passed": multi_turn_passed,
                "success_rate": multi_turn_passed / multi_turn_total if multi_turn_total > 0 else 0
            },
            "results": results,
            "multi_turn_results": multi_turn_results
        }
        
        # Print summary
        print("\n" + "="*60)
        print("📊 EVALUATION SUMMARY")
        print("="*60)
        print(f"\n✅ Success Rate: {report['summary']['success_rate']*100:.1f}%")
        print(f"   Passed: {passed}/{total} scenarios")
        print(f"\n⏱️  Average Latency: {avg_latency:.0f}ms")
        print(f"💰 Total Cost: ${total_cost:.4f}")
        print(f"💰 Average Cost: ${report['summary']['avg_cost_usd']:.4f} per query")
        print(f"⏰ Duration: {duration:.1f}s")
        
        print(f"\n📂 By Category:")
        for cat, stats in report['by_category'].items():
            print(f"   {cat}: {stats['passed']}/{stats['total']} ({stats['success_rate']*100:.0f}%)")
        
        if include_multi_turn:
            print(f"\n🔄 Multi-Turn Conversations: {multi_turn_passed}/{multi_turn_total} ({report['multi_turn']['success_rate']*100:.0f}%)")
        
        print("\n" + "="*60)
        
        return report


async def run_evaluation():
    """Main evaluation runner."""
    evaluator = AgentEvaluator()
    report = await evaluator.run_all(include_multi_turn=True)
    
    # Save report
    output_file = f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Report saved to: {output_file}")
    
    return report


if __name__ == "__main__":
    print("Starting agent evaluation...")
    report = asyncio.run(run_evaluation())
    
    # Exit with appropriate code
    success_rate = report["summary"]["success_rate"]
    exit(0 if success_rate >= 0.8 else 1)  # Fail if < 80% success rate

