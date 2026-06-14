import json
import logging
import re

logger = logging.getLogger(__name__)

def extract_features_from_readme(readme_content: str) -> str:
    """
    extract_features_from_readme(readme_content: str) -> str
    从 README 文本中提取项目功能特性及相应证据。
    返回结构化的 JSON 字符串，包含 purpose, main_features, supports_* 等字段以及 evidence。
    """
    if not readme_content or not isinstance(readme_content, str):
        return json.dumps({"error": "未提供 README 内容"}, ensure_ascii=False)

    text_lower = readme_content.lower()

    # 基于正则与关键词的启发式特性提取
    supports_web_ui = any(kw in text_lower for kw in ["web ui", "webui", "dashboard", "frontend", "gradio", "streamlit", "react"])
    supports_docker = any(kw in text_lower for kw in ["docker", "docker-compose", "container"])
    supports_local_deploy = any(kw in text_lower for kw in ["local deploy", "localhost", "self-hosted", "self host", "run locally", "docker"])
    supports_local_model = any(kw in text_lower for kw in ["local model", "local llm", "ollama", "lmstudio", "vllm", "llama.cpp", "llama-cpp", "huggingface"])
    supports_openai_compatible_api = any(kw in text_lower for kw in ["openai compatible", "openai api", "openai endpoints"])
    supports_ollama = "ollama" in text_lower
    supports_rag = any(kw in text_lower for kw in ["rag", "retrieval augmented generation", "vector", "knowledge base"])
    supports_multi_user = any(kw in text_lower for kw in ["multi-user", "rbac", "role-based access", "multiple users", "authentication"])
    developer_friendly = any(kw in text_lower for kw in ["api", "sdk", "developer", "plugin", "extension", "webhook", "rest api"])

    evidence = []
    
    def add_evidence(flag: bool, feature_name: str, keywords: list[str]):
        if flag:
            # 找到关键词所在的大致句子或段落提取为 snippet
            for kw in keywords:
                if kw in text_lower:
                    idx = text_lower.find(kw)
                    start = max(0, idx - 30)
                    end = min(len(readme_content), idx + 30)
                    snippet = readme_content[start:end].replace("\n", " ").strip()
                    evidence.append({
                        "source": "README.md",
                        "quote": f"...{snippet}...",
                        "feature": feature_name
                    })
                    break

    add_evidence(supports_web_ui, "supports_web_ui", ["web ui", "dashboard", "frontend", "gradio", "streamlit"])
    add_evidence(supports_docker, "supports_docker", ["docker", "docker-compose"])
    add_evidence(supports_local_deploy, "supports_local_deploy", ["local deploy", "self-hosted"])
    add_evidence(supports_local_model, "supports_local_model", ["local model", "local llm", "llama.cpp"])
    add_evidence(supports_openai_compatible_api, "supports_openai_compatible_api", ["openai compatible", "openai api"])
    add_evidence(supports_ollama, "supports_ollama", ["ollama"])
    add_evidence(supports_rag, "supports_rag", ["rag", "retrieval augmented generation", "vector db", "knowledge base"])
    add_evidence(supports_multi_user, "supports_multi_user", ["multi-user", "rbac"])
    add_evidence(developer_friendly, "developer_friendly", ["api", "sdk", "plugin", "webhook"])

    # 尝试提取前几句话做 purpose
    paragraphs = [p.strip() for p in readme_content.split("\n\n") if p.strip()]
    purpose = " 未提取到有效描述 "
    for p in paragraphs:
        if not p.startswith("#") and len(p) > 20:
            purpose = p[:200] + "..." if len(p) > 200 else p
            break

    result = {
        "purpose": purpose,
        "main_features": ["(需补充或确认的功能)"],
        "supports_web_ui": supports_web_ui,
        "supports_local_deploy": supports_local_deploy,
        "supports_docker": supports_docker,
        "supports_local_model": supports_local_model,
        "supports_openai_compatible_api": supports_openai_compatible_api,
        "supports_ollama": supports_ollama,
        "supports_rag": supports_rag,
        "supports_multi_user": supports_multi_user,
        "developer_friendly": developer_friendly,
        "evidence": evidence
    }

    return json.dumps(result, ensure_ascii=False, indent=2)
