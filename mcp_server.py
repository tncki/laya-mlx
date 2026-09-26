"""laya-mlx MCP 服务器 — AI 自动发现并调用 laya 决策引擎。

安装方式：
  1. NewMax 设置 → 连接 → MCP → 添加服务器
  2. 名称：laya-decisions
  3. 类型：本地命令 (stdio)
  4. 命令：/Users/jack/workspace/.venvs/laya-mlx/bin/python
  5. 参数：/Users/jack/workspace/laya-mlx/mcp_server.py
  6. 环境变量：HF_HOME=/Users/jack/workspace/.hf-cache
  7. 启用保存即可

首次调用加载约 800MB 检查点（本地模型约 2 秒，远端首次约 2 分钟），之后复用缓存。

        环境变量：
          LAYA_MODEL   模型路径或 HuggingFace 名称（默认本机 models/multilingual）
          LAYA_DEVICE  gpu / cpu（默认 gpu）
"""

import json
import os
import sys
from pathlib import Path

from mcp.server.mcpserver import MCPServer

# ── 环境 ──────────────────────────────────────────────────────────
os.environ.setdefault("HF_HOME", str(Path.home() / "workspace" / ".hf-cache"))

# 模型路径硬编码来自 TO-NEWMAX.md 建议，可按需要修改
# Add the repo to sys.path so it always resolves, even when run as module
_repo_root = Path(__file__).resolve().parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from laya_mlx import presets  # noqa: E402
from laya_mlx.agent import Agent  # noqa: E402

# ── 模型路径（优先环境变量，默认本地 multilingual） ────────────
_MODEL = os.environ.get("LAYA_MODEL", str(_repo_root / "models" / "multilingual"))
_DEVICE = os.environ.get("LAYA_DEVICE", "gpu")


# ── 引擎（模块级缓存，首次调用后常驻） ──────────────────────────
_agent: Agent | None = None


def _get_agent() -> Agent:
    global _agent
    if _agent is None:
        model = _MODEL
        # 本地路径不存在时回退到 HuggingFace（新克隆或清缓存后生效）
        if not os.path.isdir(model) and model.startswith("aac6fef/"):
            pass  # HuggingFace 模型 ID，直接传入
        elif not os.path.isdir(model):
            fallback = "aac6fef/laya-multilingual-mlx"
            print(f"[laya-mlx] 本地模型路径不存在: {model}，回落到 {fallback}", file=sys.stderr)
            model = fallback
        _agent = Agent(
            model,
            device=_DEVICE,
            dtype="float16",
            batch_size=16,
        )
    return _agent


def _decide(text: str, preset_name: str) -> dict:
    questions_fn = {
        "triage": presets.triage_questions,
        "email": presets.email_questions,
        "guard": presets.guard_questions,
        "moderation": presets.moderation_questions,
        "router": presets.router_questions,
    }[preset_name]
    return _get_agent().predict(text, questions_fn())


# ── MCP 服务器 ────────────────────────────────────────────────────
mcp = MCPServer(
    "laya-decisions",
    description="laya 决策引擎 — 文本分类、安全检测、意图识别、内容审核",
)


@mcp.tool(
    description="【工单分流】分析用户消息的意图与情绪，返回 intent、is_urgent、frustration、"
    "refund_requested、churn_risk。适合客服消息、用户反馈、工单等需要判断意图与紧急度的场景。"
)
def laya_triage(text: str) -> str:
    """对用户消息进行通用意图分类和属性分析"""
    return json.dumps(_decide(text, "triage"), ensure_ascii=False)


@mcp.tool(
    description="【邮件分类】对邮件内容进行分类和风险检测，判断是正常邮件、催款函、促销广告、"
    "钓鱼邮件还是垃圾邮件。"
)
def laya_email(text: str) -> str:
    """分类和分析邮件内容"""
    return json.dumps(_decide(text, "email"), ensure_ascii=False)


@mcp.tool(
    description="【安全防护】检测提示注入、越狱攻击、越权请求等安全威胁。"
    "适合对用户输入做安全过滤和风险评估。"
)
def laya_guard(text: str) -> str:
    """检测输入内容是否存在安全威胁"""
    return json.dumps(_decide(text, "guard"), ensure_ascii=False)


@mcp.tool(
    description="【内容审核】检测不当内容、敏感话题、违规信息。"
    "适合对用户生成内容进行内容审核和过滤。"
)
def laya_moderation(text: str) -> str:
    """检测内容是否包含违规信息"""
    return json.dumps(_decide(text, "moderation"), ensure_ascii=False)


@mcp.tool(
    description="【模型路由】判断一个请求本身的难度与属性，返回 difficulty、domain、needs_tools、"
    "is_sensitive。适合决定该交给强模型还是便宜模型、是否需要工具或人工审核。"
)
def laya_router(text: str) -> str:
    """评估请求的难度与属性以决定路由策略"""
    return json.dumps(_decide(text, "router"), ensure_ascii=False)


# ── 入口 ──────────────────────────────────────────────────────────
def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
