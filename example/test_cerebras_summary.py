import asyncio
import sys
import os

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.openai_service import generate_summaries


async def test_cerebras_summary():
    """Test the Cerebras summary API implementation."""
    # Test content
    contents = [
        "고령 근로자의 장기 고용은 젊은 근로자들의 취업 기회를 제한할 수 있으며, 이는 전체 노동 시장의 역동성과 경제 성장에 부정적 영향을 미칠 수 있습니다.",
        "대한민국은 급속한 고령화 사회로 진입하면서 노동 시장에서의 고령자 고용 문제가 중요한 사회적 과제로 부상하고 있습니다. 기업들은 인력 부족 문제를 해결하기 위해 고령 근로자들을 채용하거나 정년을 연장하는 정책을 검토하고 있습니다."
    ]
    
    # Generate summaries
    summaries = await generate_summaries(contents)
    
    # Print results
    print("\n--- Test Cerebras Summary API ---")
    for i, (content, summary) in enumerate(zip(contents, summaries)):
        print(f"\nContent {i+1}:\n{content}")
        print(f"\nSummary {i+1}:\n{summary}")
        print("\n" + "-" * 50)


if __name__ == "__main__":
    asyncio.run(test_cerebras_summary())