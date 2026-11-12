"""O modelo sabe quando está errando? Analisa a confiança do modelo final no teste interno e no oficial.

Uso: python scripts/analise_confianca.py     (também é chamado por scripts/final_report.py)
Lê reports/final/preds_test.csv e preds_test_official.csv; grava reports/final/analise_confianca.json
e reports/figures/30_confianca_cobertura.png.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from skinlesion.confianca import analise_confianca

if __name__ == "__main__":
    print(json.dumps(analise_confianca(ROOT), indent=2, ensure_ascii=False))
