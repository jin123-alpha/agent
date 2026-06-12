import argparse
import json

from agent import (
    AGENT_GITHUB_SEARCH,
    AGENT_REPO_ANALYSIS,
    AGENT_REPORT,
    AGENT_REQUIREMENT,
    AGENT_SCORING,
    STAGE_CHECKLISTS,
    create_all_agents,
    create_critic_agent,
)
from result import print_debug_result
from runner import run_single_agent_debug


def mock_input_for_agent(agent_name: str):
    if agent_name == AGENT_REQUIREMENT:
        return {
            "user_request": (
                "我想做一个本地部署的 RAG 知识库系统，要求支持 PDF 上传、"
                "向量检索、Ollama、本地模型、Web UI，并且方便二次开发。"
            )
        }

    if agent_name == AGENT_GITHUB_SEARCH:
        return {
            "project_type": "本地部署 RAG 知识库系统",
            "keywords": ["RAG", "PDF", "Ollama", "Web UI", "local deployment"],
            "language": "不限",
            "constraints": ["开源", "支持本地模型", "方便二次开发"],
            "preferences": ["文档完善", "社区活跃"],
            "top_n": 5,
        }

    if agent_name == AGENT_REPO_ANALYSIS:
        return [
            {
                "full_name": "example/local-rag",
                "html_url": "https://github.com/example/local-rag",
                "description": "Local RAG knowledge base with PDF upload and Ollama support",
                "stars": 1234,
                "language": "Python",
                "updated_at": "2026-01-01T00:00:00Z",
            }
        ]

    if agent_name == AGENT_SCORING:
        return {
            "RequirementAgent": mock_input_for_agent(AGENT_GITHUB_SEARCH),
            "RepoAnalysisAgent": mock_output_for_agent(AGENT_REPO_ANALYSIS),
        }

    if agent_name == AGENT_REPORT:
        return {
            "RequirementAgent": mock_input_for_agent(AGENT_GITHUB_SEARCH),
            "GitHubSearchAgent": mock_output_for_agent(AGENT_GITHUB_SEARCH),
            "RepoAnalysisAgent": mock_output_for_agent(AGENT_REPO_ANALYSIS),
            "ScoringAgent": mock_output_for_agent(AGENT_SCORING),
        }

    return {"message": f"no mock input for {agent_name}"}


def mock_output_for_agent(agent_name: str) -> str:
    if agent_name == AGENT_REQUIREMENT:
        return json.dumps(mock_input_for_agent(AGENT_GITHUB_SEARCH), ensure_ascii=False)

    if agent_name == AGENT_GITHUB_SEARCH:
        return json.dumps(
            [
                {
                    "full_name": "example/local-rag",
                    "html_url": "https://github.com/example/local-rag",
                    "description": "Local RAG knowledge base with PDF upload and Ollama support",
                    "stars": 1234,
                    "language": "Python",
                    "updated_at": "2026-01-01T00:00:00Z",
                }
            ],
            ensure_ascii=False,
        )

    if agent_name == AGENT_REPO_ANALYSIS:
        return json.dumps(
            [
                {
                    "full_name": "example/local-rag",
                    "readme_summary": "本地 RAG 知识库，支持 PDF 上传、Ollama 和 Web UI。",
                    "features": ["PDF upload", "vector search", "Ollama", "Web UI"],
                    "has_tests": True,
                    "has_docs": True,
                    "has_ci": True,
                    "has_docker": True,
                    "license": "MIT",
                    "dependency_files": ["requirements.txt", "package.json"],
                    "project_structure_quality": "良好",
                }
            ],
            ensure_ascii=False,
        )

    if agent_name == AGENT_SCORING:
        return json.dumps(
            [
                {
                    "full_name": "example/local-rag",
                    "scores": {
                        "requirement_match": 9,
                        "community_activity": 6,
                        "doc_quality": 8,
                        "engineering": 8,
                        "deployment": 9,
                    },
                    "weighted_total": 8.05,
                    "rank": 1,
                }
            ],
            ensure_ascii=False,
        )

    if agent_name == AGENT_REPORT:
        return (
            "# 开源技术选型报告\n\n"
            "## 1. 需求概述\n本地部署 RAG 知识库系统。\n\n"
            "## 2. 候选项目概览\n| 排名 | 项目 | 加权总分 |\n|---|---|---|\n| 1 | example/local-rag | 8.05 |\n\n"
            "## 3. 详细分析\n支持 PDF、向量检索、Ollama 和 Web UI。\n\n"
            "## 4. 推荐结论\n推荐 example/local-rag。\n\n"
            "## 5. 注意事项\nmock 数据仅用于调试。"
        )

    return "{}"


def select_output_guardrail(agent):
    if agent.output_guardrails:
        return agent.output_guardrails[0]

    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Debug a single agent with mock JSON data.")
    parser.add_argument(
        "--agent",
        default=AGENT_GITHUB_SEARCH,
        choices=[
            AGENT_REQUIREMENT,
            AGENT_GITHUB_SEARCH,
            AGENT_REPO_ANALYSIS,
            AGENT_SCORING,
            AGENT_REPORT,
        ],
        help="Agent name to debug.",
    )
    parser.add_argument(
        "--real-agent",
        action="store_true",
        help="Call the real LLM graph instead of using mock output.",
    )
    parser.add_argument(
        "--guardrail",
        action="store_true",
        help="Run the selected agent's first OutputGuardrail after output.",
    )
    parser.add_argument(
        "--critic",
        action="store_true",
        help="Run the corresponding CriticAgent after output. This calls the real LLM.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    agents = create_all_agents()
    agent = agents[args.agent]
    output_guardrail = select_output_guardrail(agent) if args.guardrail else None
    critic_agent = None
    if args.critic:
        if args.agent not in STAGE_CHECKLISTS:
            raise ValueError(f"{args.agent} has no CriticAgent checklist.")
        critic_agent = create_critic_agent(args.agent)

    result = run_single_agent_debug(
        agent,
        mock_input_factory=lambda: mock_input_for_agent(args.agent),
        output_guardrail=output_guardrail,
        critic_agent=critic_agent,
        mock_output_factory=None if args.real_agent else lambda _prompt, _ctx: mock_output_for_agent(args.agent),
    )

    print_debug_result(result)


if __name__ == "__main__":
    main()
