#!/usr/bin/env python3
"""
竞赛一键演示脚本：调用 FastAPI 产生流量并打印监控汇总。

用法（在项目根目录）：
  1) 终端 A：python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
  2) 终端 B：python scripts/demo_competition.py

可选参数：
  --base     API 根地址，默认 http://127.0.0.1:8000
  --learner  学习者 ID
  --knowledge 知识点 ID（与图谱中节点 id 一致）
  --rounds   循环轮数（每轮：submit + question）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import quote


def _get_json(url: str, timeout: int = 30) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_json(url: str, payload: dict, timeout: int = 120) -> dict:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="竞赛演示：API 压测 + 监控汇总")
    parser.add_argument("--base", default="http://127.0.0.1:8000", help="API 根地址")
    parser.add_argument("--learner", default="demo_competition_user", help="学习者 ID")
    parser.add_argument("--knowledge", default="arithmetic", help="知识点 ID")
    parser.add_argument("--rounds", type=int, default=5, help="循环轮数")
    args = parser.parse_args()

    base = args.base.rstrip("/")

    try:
        health = _get_json(f"{base}/api/v1/health", timeout=10)
    except urllib.error.URLError as e:
        print("无法连接 API，请先启动：python -m uvicorn api.main:app --port 8000", file=sys.stderr)
        raise SystemExit(1) from e

    print("health:", json.dumps(health, ensure_ascii=False))

    for i in range(args.rounds):
        _post_json(
            f"{base}/api/v1/submit",
            {
                "learner_id": args.learner,
                "knowledge_id": args.knowledge,
                "is_correct": (i % 2 == 0),
                "time_spent_seconds": float(20 + i),
            },
        )
        _post_json(
            f"{base}/api/v1/question",
            {
                "learner_id": args.learner,
                "knowledge_id": args.knowledge,
                "question": f"请用苏格拉底式提问引导我理解本题（演示轮次 {i + 1}/{args.rounds}）",
            },
        )
        time.sleep(0.15)

    q = quote(args.learner, safe="")
    summary = _get_json(f"{base}/api/v1/monitor/summary?learner_id={q}", timeout=30)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
