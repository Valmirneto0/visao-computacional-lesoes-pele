# M2 — Dados: pipeline, split e auditoria

Projeto Tema 3 (lesões de pele com incerteza) · Valmir Carlos Cerutti Neto · revisão 18/10/2026 (prazo original 09/10) · seed 42.
Números deste documento vêm de `reports/m2_dados/evidencias_m2_script.json` (gerado por `scripts/m2_evidencias.py`); o notebook 02 grava `evidencias_m2.json`, com os mesmos valores nas chaves em comum.
Convenção: **medido** (contado nos dados) · **derivado** (calculado a partir do medido) · **inferido** (interpretação).

## 1. Origem, licença e versão

| Base | Conteúdo | Licença | Uso |
|---|---|---|---|
| ISIC 2018 Task 3 = HAM10000 (treino) | 10.015 imagens, 7 classes, `lesion_id`, metadados | CC BY-NC 4.0 | treino/val/teste interno |
| ISIC 2018 Task 3 (teste oficial) | 1.512 imagens rotuladas | CC BY-NC 4.0 | **holdout externo**, só na avaliação final |
| ISIC 2018 Task 1 (treino) | 2.594 imagens + máscaras | CC0 1.0 | segmentação |

Download: https://challenge.isic-archive.com/data/#2018 · metadados: doi:10.7910/DVN/DBW86T. Os dados não são versionados (tamanho e licença não comercial); o git guarda só IDs, manifestos e código.

## 2. Pipeline

1. `scripts/preprocess.py` + `configs/preprocess.yaml` — redimensiona uma vez para `data/processed/`: Task 3 → 341×256 (mantém 4:3); Task 1 → 256×256 (máscara com vizinho mais próximo e rebinarizada). Gera `data/processed/manifest_v1.csv`.
2. `scripts/make_split_v2.py` — gera `data/splits/split_v2.csv` (ver seção 4).
3. `src/skinlesion/data.py` — `ClassificationDataset` → (img[3,224,224], y, id); `SegmentationDataset` → (img[3,256,256], mask[1,256,256] ∈ {0,1}, id). O split vem de `configs/data.yaml → split_file`.
4. `configs/transforms.yaml` — augmentations só no treino; validação/teste: resize + crop central + normalização ImageNet.
5. `scripts/m2_evidencias.py hash | scores | analyze | figures` — auditoria completa e exportação dos manifestos.

## 3. Auditoria (medido)

- **Integridade/formato:** 14.121/14.121 arquivos lidos; 100% JPEG RGB de 3 canais; Task 3 toda em 600×450; Task 1 de 576–6.748 × 540–4.499 px; 2.594/2.594 máscaras binárias e do mesmo tamanho da imagem.
- **Rótulos:** 11.527/11.527 imagens com exatamente 1 classe; 0 divergências entre o GT e os metadados. Só 53% foram confirmados por histopatologia (100% em MEL/BCC/AKIEC; 37% em NV).
- **Duplicatas exatas (MD5):** 2 pares, ambos da mesma lesão e no mesmo split → mantidos e marcados.
- **Quase-duplicatas (pHash ≤ 4 + checagem de pixels + inspeção visual):**
  - Task 3: 524 pares; os 238 entre splits são lesões diferentes (menor diferença de pixels 6,5 vs mediana 8,2 entre imagens da mesma lesão). Sem vazamento.
  - **Task 1: vazamento encontrado.** No split_v1 (por imagem), 21 pares cruzavam splits; a inspeção confirmou a mesma lesão (uma cópia quase idêntica e várias re-aquisições com outra cor/enquadramento) — `reports/figures/12_*.jpg`.
- **Máscaras:** área mediana 13,8% (P5–P95: 1,5%–65%); 48 lesões < 1% da imagem; 10,6% tocam a borda.
- **Artefatos:** 200 imagens rotuladas manualmente (rotuladas pelo assistente, conferidas integralmente por Valmir): pelo 42%, régua 18%, vinheta 13%, bolha 5,5%, caneta 1,5%. Detector de vinheta validado (AUC 0,98; P 0,79; R 0,85) → MEL 18,9% vs NV 5,9% (χ² = 302,9; gl = 6; p < 1e-60). Detectores de pelo e régua com AUC 0,60 e 0,55 → **descartados**.
- **Subgrupos:** splits equilibrados em idade (≈ 52 anos), sexo (≈ 54% masculino) e tipo de confirmação. Teste oficial e Task 1 sem metadados; nenhuma base informa fototipo.

## 4. Split

| | train | validation | test | holdout |
|---|---|---|---|---|
| Task 3 (cls) | 7.017 (70,1%) | 1.487 (14,9%) | 1.511 (15,1%) | 1.512 (teste oficial) |
| Task 1 (seg) | 1.816 (70,0%) | 389 (15,0%) | 389 (15,0%) | — |

- Task 3: StratifiedGroupKFold por `lesion_id`, seed 42 — **idêntico ao split_v1** da M1.
- Task 1 (**decisão nova, split_v2**): pares com pHash ≤ 4 viram um grupo (33 grupos, 80 imagens); grupos embaralhados com seed 42 e alocados até 70/15/15. Critério conservador: um falso positivo só junta duas imagens no mesmo split.
- Testes de vazamento (todos = 0): lesões em mais de um split; grupos da Task 1 em mais de um split; duplicatas confirmadas entre splits; IDs comuns entre Task 1 e Task 3.
- Teste interno e holdout ficam bloqueados até a avaliação final; na M2 só foram contados e auditados.
- Manifestos: `data/splits/manifest_m2_v1.csv` e `train.csv`, `validation.csv`, `test.csv`, `test_official.csv`.

## 5. Transformações

Determinísticas (todos os splits): leitura RGB, redimensionamento, crop central 224 (cls), normalização ImageNet — nenhuma estatística é calculada nos dados. Aleatórias (só treino): flips, rotação, ColorJitter leve, RandomResizedCrop 0,7–1,0, CoarseDropout p = 0,3 (cls). A máscara recebe a mesma transformação geométrica (`reports/figures/09_augmentations.png`). Justificativa de cada uma em `configs/transforms.yaml`.

## 6. Riscos e limitações (inferido, com a evidência ao lado)

1. **Atalho por vinheta** (associada a MEL) → CoarseDropout, métricas com/sem vinheta, Grad-CAM.
2. **Desbalanceamento de 58×** → macro-F1, pesos de classe (`data/splits/class_weights_v1.json`), focal loss no experimento.
3. **Rótulo ≠ verdade clínica** (53% biópsia) → resultados descritos como concordância com o rótulo.
4. **Holdout com outra distribuição** (NV 60% vs 67%) → reportado à parte.
5. **População pouco diversa, sem fototipo** → não generalizar.
6. **Re-aquisições muito diferentes podem escapar do pHash** na Task 1 → risco residual de vazamento.

Sistema educacional — **não substitui avaliação clínica.**

## 7. Pendências

- Re-executar no PC (GPU): `python scripts/smoke_test_data.py` e o `notebooks/02_auditoria.ipynb` (Run All), para salvar outputs com o split_v2.
- Teste de reprodução por outra pessoa.
