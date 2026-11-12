# Classificação de lesões de pele em imagens dermatoscópicas

Projeto final de Visão Computacional e Redes Neurais — IBMEC-BH, 2º semestre de 2026. Tema 3.

**Pergunta (definida na M1):** quando o modelo acerta, quando erra e se ele sabe que está errando?

Para respondê-la, um baseline e dois experimentos controlados escolhem o melhor classificador; a análise de erros (`reports/final/analise_erros.json`) e a análise de confiança (`reports/final/analise_confianca.json`) do modelo final respondem às três partes.

**Tarefa:** classificação multiclasse (7 classes). Escopo da entrega final = exatamente o pedido no enunciado (`Entrega_final_do_Projeto.pdf`); segmentação e Grad-CAM previstos na M1 ficaram como evolução; a parte de incerteza é respondida pela análise de confiança do classificador.

> Sistema educacional. **Não substitui avaliação clínica.**

## Entrada e saída

| | |
|---|---|
| Entrada | imagem RGB dermatoscópica (JPG, qualquer tamanho) → pré-processada para [3, 224, 224] |
| Saída | 7 logits → softmax → classe prevista e probabilidades (MEL, NV, BCC, AKIEC, BKL, DF, VASC) |

## Dataset

ISIC 2018 Challenge (licença CC BY-NC 4.0):

- **Task 3** (= HAM10000) — 10.015 imagens, 7 classes, metadados e `lesion_id` → treino/validação/teste interno (70/15/15, agrupado por lesão).
- **Task 3 — teste oficial** — 1.512 imagens rotuladas → holdout externo, usado só na avaliação final.
- Task 1 (segmentação) foi auditada na M2, mas não é usada na entrega final.

Os dados **não** são versionados. Baixe em https://challenge.isic-archive.com/data/#2018 e coloque em `data/raw/` (ver `scripts/prepare_raw.sh`). Os metadados `HAM10000_metadata.csv` vêm do Harvard Dataverse (doi:10.7910/DVN/DBW86T) ou do Kaggle.

## Estrutura

```
configs/        constantes do dataset e do split (data.yaml)
data/raw/       zips e pastas originais (ignorado pelo git)
data/splits/    split_v2.csv (split oficial a partir da M2), manifestos train/validation/test/test_official.csv, class_weights_v1.json
configs/experiments/  baseline.yaml, exp1_focal.yaml, exp2_scratch.yaml (entrega final)
notebooks/      01_exploracao_inicial.ipynb (M1), 02_auditoria.ipynb (M2)
reports/        figuras, evidências e documentos das entregas; reports/final/ = resultados da entrega final
scripts/        M2: setup_env.ps1, prepare_raw.sh, preprocess.py, make_split_v2.py, m2_evidencias.py, smoke_test_data.py, audit_sample.py
                final: run_all.ps1, train.py, final_report.py, analise_confianca.py, demo.py
src/skinlesion/ data.py, artifacts.py (M2); models.py, losses.py, metrics.py, engine.py, confianca.py (final)
models/         pesos; só models/baseline.pt (modelo final, 43 MB) é versionado
```

## Ambiente

Windows + AMD RX 9060 XT: `scripts/setup_env.ps1` (venv Python 3.12 em `%USERPROFILE%\venvs\skinlesion` — fora do projeto porque o SDK ROCm não aceita espaço no caminho — + PyTorch 2.9.1 ROCm 7.2.1; fallback CPU). Dependências em `requirements.txt`.

## Entrega final — como executar (12/11)

Com o venv ativo, `data/processed/` gerado (seção M2 abaixo) e na raiz do projeto:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1     # tudo: teste rápido → 3 treinos → relatório → demo
```

Ou passo a passo:

```powershell
python scripts/train.py --config configs/experiments/baseline.yaml      # baseline neural
python scripts/train.py --config configs/experiments/exp1_focal.yaml    # experimento 1: focal loss
python scripts/train.py --config configs/experiments/exp2_scratch.yaml  # experimento 2: treino do zero
python scripts/final_report.py   # compara na validação, escolhe o final, avalia no teste e no teste oficial, análise de erros e de confiança
python scripts/demo.py           # imagem original → pré-processamento → modelo → resultado (imagens do teste oficial)
python scripts/demo.py --images caminho\para\imagem.jpg
```

**Sem o dataset** (o repositório não inclui `data/`): a demonstração roda só com o modelo versionado — crie o venv (`scripts/setup_env.ps1`) e use `python scripts/demo.py --images <qualquer imagem dermatoscópica>.jpg`. Os resultados de treino, validação e teste já estão em `reports/final/` e `reports/figures/`; refazê-los exige baixar o ISIC 2018 (seção Dataset) e rodar `scripts\run_all.ps1`.

| Experimento | O que muda em relação ao baseline | Hipótese |
|---|---|---|
| `baseline` | — (ResNet-18 ImageNet, fine-tuning total, CE ponderada, AdamW 3e-4, batch 32, 25 épocas, cosine, dropout 0,2, wd 1e-4) | referência |
| `exp1_focal` | perda: CE ponderada → focal loss (γ = 2, mesmos pesos) | foca nos exemplos difíceis → melhor macro-F1 e revocação das classes raras |
| `exp2_scratch` | inicialização: ImageNet → aleatória | sem transferência, ~7 mil imagens não bastam → desempenho bem menor |

Regras: a escolha da época e do modelo final usa só a validação (maior macro-F1); o teste interno e o teste oficial são avaliados uma única vez, com o modelo final.
Saídas: `models/*.pt`, `reports/final/` (comparação, métricas, previsões, análise de erros) e `reports/figures/20_*`–`29_*`.

## Rastreabilidade

Cada resultado aponta para: `configs/*.yaml` usado, hash do commit, `data/splits/split_v2.csv` e o arquivo de pesos. Seed global: 42.

## M2 — Dados (como reproduzir)

Resumo e decisões: [`reports/m2_dados/M2_Dados.md`](reports/m2_dados/M2_Dados.md). Na raiz do projeto, com o venv ativo e os dados em `data/raw/`:

```powershell
python scripts/preprocess.py               # redimensiona uma vez → data/processed/ + manifest_v1.csv
python scripts/m2_evidencias.py hash       # MD5 dos brutos + pHash (cache em reports/m2_dados/hashes_v1.csv)
python scripts/make_split_v2.py            # split_v2: Task 3 = split_v1; Task 1 agrupada por quase-duplicata
python scripts/preprocess.py               # regrava o manifest com o split_v2 (pula imagens já geradas)
python scripts/m2_evidencias.py scores     # scores de artefato
python scripts/m2_evidencias.py analyze    # auditoria + manifestos train/validation/test/test_official.csv
python scripts/m2_evidencias.py figures    # figuras 08b, 09b, 10
python scripts/smoke_test_data.py          # lotes válidos na GPU
```

O split oficial é `data/splits/split_v2.csv` (definido em `configs/data.yaml → split_file`). O teste interno e o teste oficial (holdout) ficam bloqueados até a avaliação final.
