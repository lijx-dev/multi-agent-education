# 竞赛一键演示（Windows）：启动 API 后产生流量并输出监控 JSON。
# 用法：在项目根目录 PowerShell 执行：
#   .\scripts\demo_competition.ps1
# 可选：先手动启动 API：
#   python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "== 检查 API 健康 ==" -ForegroundColor Cyan
try {
  Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/v1/health" -Method Get | ConvertTo-Json -Compress
} catch {
  Write-Host "API 未就绪：请先运行 python -m uvicorn api.main:app --host 0.0.0.0 --port 8000" -ForegroundColor Yellow
  exit 1
}

Write-Host "== 运行 Python 演示脚本（submit + question + monitor/summary） ==" -ForegroundColor Cyan
python scripts/demo_competition.py --base http://127.0.0.1:8000 --learner demo_competition_user --knowledge arithmetic --rounds 5

Write-Host "== 完成。可在 Streamlit「系统监控」页点击刷新查看图表。 ==" -ForegroundColor Green
