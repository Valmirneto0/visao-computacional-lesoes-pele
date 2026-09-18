# Análise de lesões de pele com incerteza

Projeto final de Visão Computacional e Redes Neurais — IBMEC-BH, 2º semestre de 2026. Tema 3.

**Problema em uma frase:** dada uma imagem dermatoscópica, delimitar a lesão, classificá-la em uma de 7 categorias, mostrar quais regiões sustentaram a decisão e sinalizar quando o modelo não deve ser confiado (baixa confiança ou imagem fora da distribuição).

> Sistema educacional. **Não substitui avaliação clínica.** Limitações, vieses e incerteza são parte do relatório.

## Entrada e saída

| | |
|---|---|
| Entrada | imagem RGB dermatoscópica (JPG) |
| Saída 1 | máscara binária da lesão (segmentação) |
| Saída 2 | distribuição de probabilidade sobre 7 classes (MEL, NV, BCC, AKIEC, BKL, DF, VASC) |
| Saída 3 | mapa de atenção (Grad-CAM) sobre a imagem |
| Saída 4 | sinal de incerteza: confiança calibrada + flag de abstenção / OOD |

## Dataset

ISIC 2018 Challenge (licença CC BY-NC 4.0):

- **Task 1** — 2.594 imagens com máscara de segmentação.
- **Task 3** (= HAM10000) — 10.015 imagens, 7 classes, metadados de idade, sexo, localização anatômica e `lesion_id`.

Os dados **não** são versionados. Baixe em https://challenge.isic-archive.com/data/#2018 e coloque em `data/raw/` (ver `scripts/prepare_raw.sh`). Os metadados `HAM10000_metadata.csv` vêm do Harvard Dataverse (doi:10.7910/DVN/DBW86T) ou do Kaggle.

## Estrutura

```
configs/        constantes do dataset e do split (data.yaml)
data/raw/       zips e pastas originais (ignorado pelo git)
data/splits/    split_v1.csv — único split usado em todos os experimentos
notebooks/      01_exploracao_inicial.ipynb (M1)
reports/        figuras, evidências e documentos das entregas
scripts/        setup_env.ps1 (ambiente Windows), prepare_raw.sh
src/skinlesion/ código do pipeline (M2+)
models/         pesos (ignorado pelo git)
```

## Ambiente

Windows + AMD RX 9060 XT: `scripts/setup_env.ps1` (venv Python 3.12 + PyTorch ROCm; fallback CPU). Dependências em `requirements.txt`.

## Marcos

| Marco | Data | Conteúdo |
|---|---|---|
| M1 | 18/09 | proposta + exploração inicial |
| M2 | 09/10 | pipeline de dados, split, auditoria |
| M3 | 23/10 | baseline (transfer learning) + análise de erros |
| M4 | 06/11 | segmentação + experimento comparativo |
| Final | 12/11 | demo, relatório, galeria de falhas, vídeo |

## Rastreabilidade

Cada resultado aponta para: `configs/*.yaml` usado, hash do commit, `data/splits/split_v1.csv` e o arquivo de pesos. Seed global: 42.
