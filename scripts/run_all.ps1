# Entrega final — executa o projeto inteiro (rodar no PowerShell, na raiz do projeto).
# Pré-requisitos: scripts/setup_env.ps1 já executado e data/processed/ gerado (ver README, seção "Dados").
# Tempo esperado na RX 9060 XT: ~1–2 min por época → ~1,5–2,5 h para os 3 experimentos de 25 épocas.
# "Continue": no PowerShell 5, 2>&1 transforma qualquer aviso do Python em erro; checamos o código de saída à mão.
$ErrorActionPreference = "Continue"
function Check($etapa) { if ($LASTEXITCODE -ne 0) { Write-Host "FALHOU: $etapa" -ForegroundColor Red; exit 1 } }
& "$env:USERPROFILE\venvs\skinlesion\Scripts\Activate.ps1"
$env:PYTHONUTF8 = "1"
$OutputEncoding = [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false   # acentos corretos no terminal e nos logs
New-Item -ItemType Directory -Force reports\final | Out-Null

# 0) teste rápido do fluxo (1 época, 64 imagens) — falha aqui = problema de ambiente, não de modelo
python scripts/smoke_test_data.py; Check "smoke_test_data"
python scripts/train.py --config configs/experiments/baseline.yaml --epochs 1 --limit 64; Check "teste rápido"

# 1) baseline + 2 experimentos controlados (log completo em reports/final/treino_log.txt)
foreach ($e in "baseline", "exp1_focal", "exp2_scratch") {
    python scripts/train.py --config "configs/experiments/$e.yaml" 2>&1 | Tee-Object -Append reports\final\treino_log.txt
    Check "treino $e"
}

# 2) comparação na validação → escolha do modelo final → avaliação no teste e no teste oficial → análise de erros
python scripts/final_report.py 2>&1 | Tee-Object reports\final\final_report_log.txt; Check "final_report"

# 3) demonstração com imagens nunca usadas no treino (teste oficial ISIC 2018, imagens originais)
python scripts/demo.py 2>&1 | Tee-Object reports\final\demo_log.txt
