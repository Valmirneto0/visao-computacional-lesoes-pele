# Notebook 01 — Exploração inicial, explicado célula a célula

Companheiro de leitura de `01_exploracao_inicial.ipynb`. Para cada célula: **o que faz**, **como**, **por quê** e **como ler a saída**. Os números citados são os que o notebook produziu em 18/09/2026 (mesmos de `reports/m1_proposta/evidencias.json`).

Convenção: **medido** = veio direto dos arquivos; **derivado** = conta feita sobre o medido; **inferido** = interpretação nossa.

---

## Célula 1 — Imports, configuração e caminhos

```python
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
CFG = yaml.safe_load(open(ROOT / "configs/data.yaml"))
P = {k: ROOT / v for k, v in CFG.items() if isinstance(v, str) and k != "priority_class"}
SEED = CFG["seed"]; random.seed(SEED); np.random.seed(SEED)
EVID = {}
```

**O que faz.** Descobre a raiz do projeto (funciona tanto rodando de dentro de `notebooks/` quanto da raiz), lê `configs/data.yaml` e monta um dicionário `P` com todos os caminhos de dados. Fixa a semente dos geradores aleatórios. Cria `EVID`, um dicionário vazio que vai acumulando os números importantes ao longo do notebook.

**Por quê.** Nenhum caminho é digitado no notebook — tudo vem do YAML. Se os dados mudarem de pasta, muda-se um arquivo. A semente fixa (`42`) garante que qualquer amostra aleatória (imagens sorteadas, split) seja idêntica em qualquer máquina. `EVID` é a "ponte" entre o notebook e os slides: no final ele é salvo em JSON e é de lá que os slides tiram os números.

**Saída.** Um dicionário `{nome: (caminho, existe?)}`. Todos com `True` = os dados estão onde o YAML diz. Se algum vier `False`, nada abaixo funciona.

---

## Célula 3 — Inventário dos arquivos

```python
t1_img = listar(P["task1_images"], "jpg"); t1_msk = listar(P["task1_masks"], "png")
ids_img = {p.stem for p in t1_img}
ids_msk = {p.stem.replace("_segmentation", "") for p in t1_msk}
inter = ids_img & ids_t3
```

**O que faz.** Conta os arquivos de cada pasta e confere três coisas: (a) toda imagem da Task 1 tem exatamente uma máscara e vice-versa; (b) quantas imagens tem a Task 3 (treino e teste oficial); (c) se há IDs repetidos entre Task 1 e Task 3.

**Como.** `p.stem` é o nome do arquivo sem extensão (`ISIC_0000000`). As máscaras chamam-se `ISIC_0000000_segmentation.png`; removendo o sufixo dá para comparar os dois conjuntos com operações de conjunto (`-` diferença, `&` interseção).

**Saída e leitura.**

```
Task1 imagens: 2594 | máscaras: 2594
Task3 treino: 10015 | Task3 teste oficial: 1512
Task1 imagens sem máscara: 0 | máscaras sem imagem: 0
IDs em comum Task1 ∩ Task3: 0
```

- 2.594 = 2.594 e zero órfãos: o pareamento imagem↔máscara está íntegro (**medido**).
- Interseção zero: nenhuma imagem de classificação tem máscara oficial. Consequência para a M4: um experimento "recorte pela máscara" na classificação vai depender de máscaras **preditas** pela U-Net, não anotadas (**inferido**).

---

## Célula 5 — Rótulos de classificação e agrupamento por lesão

```python
gt = pd.read_csv(P["task3_labels"])
gt["dx"] = gt[CLASSES].idxmax(axis=1)
assert (gt[CLASSES].sum(axis=1) == 1).all()
grp = pd.read_csv(P["task3_groupings"]).rename(columns={"diagnosis_confirm_type": "dx_type"})
df = gt[["image", "dx"]].merge(grp, on="image", how="left")
```

**O que faz.** Lê o CSV de rótulos (formato *one-hot*: 7 colunas com 0.0/1.0) e o converte para uma coluna única `dx` com a sigla da classe. Depois junta o arquivo suplementar com `lesion_id` e tipo de confirmação. Se `HAM10000_metadata.csv` existir, junta também idade, sexo e localização.

**Como.** `idxmax(axis=1)` devolve, em cada linha, o nome da coluna com o maior valor — como só uma coluna vale 1.0, é a classe. O `assert` garante que cada linha soma exatamente 1 (um rótulo por imagem, nenhuma sem rótulo). `merge(..., how="left")` mantém todas as 10.015 linhas do rótulo e traz as colunas extras onde houver correspondência.

**Saída e leitura.**

```
linhas: 10015 | sem lesion_id: 0
rótulo GT ≠ metadado: 0
```
Nenhuma imagem sem lesão associada; o rótulo do CSV oficial bate 100% com o do arquivo de metadados — duas fontes independentes concordam (**medido**).

Tabela de classes:

| dx | n | pct |
|---|---|---|
| MEL | 1.113 | 11,1% |
| NV | 6.705 | 67,0% |
| BCC | 514 | 5,1% |
| AKIEC | 327 | 3,3% |
| BKL | 1.099 | 11,0% |
| DF | 115 | 1,1% |
| VASC | 142 | 1,4% |
| razão maior/menor | 58,3 | |

- A razão 58,3 é 6.705 ÷ 115 (**derivado**). É o número que justifica macro-F1 em vez de acurácia e o experimento com perda focal.
- Um classificador que responde "NV" sempre acerta 67% — esse é o piso que qualquer modelo tem de superar.

Comparação com o teste oficial:

| | treino % | teste oficial % |
|---|---|---|
| NV | 66,9 | 60,1 |
| BKL | 11,0 | 14,4 |
| DF | 1,1 | 2,9 |

O teste oficial tem um pouco menos de NV e um pouco mais das classes raras. Diferença pequena o bastante para servir como avaliação externa, mas grande o bastante para lembrar que resultados nele podem sair ligeiramente diferentes do split interno (**inferido**).

A figura `01_contagem_classes.png` é só a tabela em barras.

---

## Célula 6 — Imagens por lesão, metadados e tipo de confirmação

```python
img_por_lesao = df.groupby("lesion_id")["image"].count()
ct2 = pd.crosstab(df["dx"], df["dx_type"], normalize="index")
```

**O que faz.** Conta quantas fotos cada lesão tem; lista valores faltantes; distribui sexo; cruza classe × tipo de confirmação do diagnóstico.

**Como.** `groupby("lesion_id").count()` agrupa as linhas pelo identificador da lesão. `crosstab(..., normalize="index")` faz uma tabela cruzada onde cada linha soma 1 — ou seja, "dentro da classe X, que fração veio de cada tipo de confirmação".

**Saída e leitura.**

```
lesões únicas: 7470 | lesões com >1 imagem: 1956 | máx imagens/lesão: 6
```
- 10.015 imagens, mas só 7.470 lesões. 1.956 lesões têm de 2 a 6 fotos (**medido**). É o fato que obriga o split a ser feito por lesão (célula 18): se fotos da mesma lesão caem em treino e teste, o modelo "decora" a lesão e o teste fica inflado.

```
age: 57 faltantes | sex: male 5406, female 4552, unknown 57
```
Metadados quase completos. 57 sem idade e 57 com sexo "unknown" — provavelmente as mesmas linhas (não verificado no notebook).

```
dx_type: histopathology 5340 | serial imaging showing no change 3704 | single image expert consensus 902 | confocal 69
histopatologia por classe (%): MEL 100 | BCC 100 | AKIEC 100 | BKL 69,7 | DF 47,8 | VASC 47,2 | NV 37,3
```
- 5.340 / 10.015 = 53,3% dos rótulos vieram de biópsia (**derivado**).
- As três classes malignas/pré-malignas foram 100% biopsiadas; a benigna mais comum (NV), só 37%. Leitura: o rótulo "NV" é, em 63% dos casos, a opinião do dermatologista ou o fato de a lesão não ter mudado no tempo — não a patologia. O modelo aprende a concordar com o médico (**inferido**; é o risco 3 dos slides).
- Nuance: como "ser biopsiada" está correlacionado com "ser maligna", analisar desempenho "só nas biopsiadas" na prática é analisar as classes malignas. Não confundir isso com um atalho visual — o tipo de confirmação não está na imagem.

Figura `02_metadados.png`: barras empilhadas de tipo de confirmação por classe (à esquerda), histograma de idade e sexo por classe (se houver metadados).

---

## Célula 7 — Localização anatômica por classe

```python
loc = pd.crosstab(df["localization"], df["dx"]).reindex(columns=CLASSES)
```

**O que faz.** Tabela de contagem: linhas = local do corpo, colunas = classe.

**Por quê.** É o "subgrupo disponível nos metadados" que o professor pede para analisar. Se o modelo errar mais no rosto ou nas mãos, isso precisa aparecer no relatório.

**Leitura.** Costas (2.192) e membros inferiores (2.077) dominam. O rosto tem 745 imagens mas concentra AKIEC (113 de 327) e BCC (101 de 514) — lesões ligadas a exposição solar (**medido**; a explicação solar é **inferida**). Pé, mão, genital e acral têm dezenas de imagens: qualquer métrica nesses subgrupos vem com incerteza grande.

---

## Célula 9 — Dimensões das imagens

```python
def dims(paths, n=None):
    for p in paths:
        with Image.open(p) as im: out.append((p.stem, *im.size, im.mode))
```

**O que faz.** Lê largura, altura e modo de cor de cada imagem — todas as 10.015 da Task 3 e uma amostra de 400 da Task 1.

**Como.** `Image.open` do Pillow lê só o cabeçalho do JPG; `im.size` não decodifica os pixels. Por isso 10.015 imagens levam ~50 s. A Task 1 tem imagens de até 6.708 px, então usa-se amostra.

**Saída e leitura.**

```
Task3: 10015 imagens | modos: ['RGB'] | 600x450: 10015
Task1 (amostra): 1024x768 (103) | 4288x2848 (75) | 3008x2000 (45) ... | largura 576–6708 | altura 542–4479
```
- Task 3 é uniforme: 600×450, RGB, sem exceção (**medido**). Facilita o pipeline.
- Task 1 mistura câmeras diferentes: de 576 px a 6.708 px de largura, proporção variável. Consequência: redimensionar para tamanho fixo (256×256 ou 512×512) e decidir como tratar a proporção (recorte ou *padding*). A máscara tem de sofrer exatamente a mesma transformação da imagem, senão desalinha.

Figura `03_dimensoes.png`: dispersão largura × altura. Task 3 é um único ponto vermelho; Task 1 espalha.

---

## Célula 11 — Amostras por classe

**O que faz.** Sorteia 4 imagens por classe (com a semente fixa) e monta uma grade 7×4.

**Por quê.** Olhar os dados antes de modelar. Nessas 28 imagens já aparecem: vinheta (canto escuro), pelos, réguas, bolhas e a diferença de "aparência" entre classes (VASC tende a vermelho/roxo, DF a nódulo pálido, MEL a assimetria e cores múltiplas).

**Leitura.** Não há saída numérica. Serve para o slide 4 e para começar a "galeria de artefatos".

---

## Célula 12 — Amostras de segmentação

**O que faz.** Sorteia 6 imagens da Task 1, mostra a imagem em cima e a mesma imagem com a máscara sobreposta em vermelho semitransparente embaixo.

**Como.** `np.array(m) > 127` binariza a máscara (branco = lesão). `imshow(..., alpha=.35, cmap="Reds")` desenha por cima da foto.

**Leitura.** Confirma visualmente que a máscara corresponde ao contorno da lesão. Nas 6 sorteadas já aparecem régua (2), pelos (1), vinheta (2) e uma lesão que ocupa quase o campo inteiro.

---

## Célula 14 — Máscaras: área e contato com a borda

```python
m = np.array(Image.open(p).convert("L").resize((256, 256))) > 127
borda = m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()
rows.append((p.stem, m.mean(), bool(borda), np.unique(...).size))
```

**O que faz.** Em 400 máscaras sorteadas, calcula: fração da imagem que é lesão; se a lesão encosta em alguma das 4 bordas; quantos valores distintos a máscara tem.

**Como.** Reduzida a 256×256 para velocidade. `m.mean()` de uma matriz booleana = fração de `True` = fração de área. `m[0]` é a primeira linha (borda superior), `m[-1]` a última, `m[:, 0]` a primeira coluna, `m[:, -1]` a última; `.any()` = "tem algum pixel de lesão ali".

**Saída e leitura.**

```
area_frac: média 0,22 | mediana 0,127 | mín 0,004 | máx 0,956
toca borda: 0,125 | máscaras binárias: 1,0
```
- Metade das lesões ocupa menos de 12,7% da imagem; há casos de 0,4% (lesão minúscula) e de 95,6% (lesão maior que o campo) (**medido**).
- 12,5% tocam a borda — a lesão continua fora da foto. Nesses casos Dice tende a cair e a rede não vê o contorno completo (**inferido**).
- 100% binárias (só 0 e 255): não há máscaras "suaves" nem valores intermediários para tratar.

Figura `06_area_mascaras.png`: histograma da fração de área, assimétrico à direita (muitas pequenas, poucas enormes).

---

## Célula 16 — Proxy de vinheta

```python
def vinheta(p, f=0.05):
    im = np.array(Image.open(p).convert("L").resize((256, 256)), dtype=float); k = int(256 * f)
    cantos = [im[:k, :k], im[:k, -k:], im[-k:, :k], im[-k:, -k:]]
    return np.mean([c.mean() for c in cantos])
amostra["vinheta"] = amostra.canto < 40
```

**O que faz.** Em 1.000 imagens sorteadas da Task 3, mede o brilho médio dos quatro cantos (quadrados de 5% do lado = 12×12 px após redimensionar para 256) e marca como "vinheta" se a média ficar abaixo de 40 numa escala 0–255.

**Por quê.** O tubo do dermatoscópio deixa um círculo escuro. Se esse artefato for mais comum numa classe, a rede pode usá-lo como atalho. Precisávamos de um número, mesmo grosseiro, para levantar a hipótese.

**Limite do método.** É um proxy: pele muito escura ou lesão no canto também disparam; vinheta suave (cantos cinza) não dispara. Não substitui inspeção visual.

**Saída e leitura.**

```
proporção com vinheta (amostra 1000): 0,049
MEL 0,149 | NV 0,031 | BCC 0,000 | AKIEC 0,036 | BKL 0,086 | DF 0,000 | VASC 0,000
```
- 4,9% no geral, mas 14,9% em melanoma contra 3,1% em nevo — quase 5× (**medido** na amostra; a extrapolação para o dataset inteiro é **inferida**).
- BCC/DF/VASC com 0% têm poucas imagens na amostra de 1.000 (≈50, 12, 14) — zero ali não significa muito.
- Hipótese que isso gera: as imagens de melanoma vieram, em parte, de outro aparelho ou outro centro. Se a rede aprender "canto escuro → MEL", vai acertar no teste (que tem o mesmo viés) e falhar no mundo real. Teste que confirma: na M3, Grad-CAM nos acertos de MEL e métricas separadas com/sem vinheta.

---

## Célula 18 — Split treino / validação / teste

```python
sgk = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
for k, (_, idx) in enumerate(sgk.split(df, df.dx, groups=df.lesion_id)): folds[idx] = k
df["split"] = np.select([df.fold < 7, df.fold == 7], ["train", "val"], "test")
g8 = df.loc[df.fold == 8, "lesion_id"].drop_duplicates().sample(frac=0.5, random_state=SEED)
df.loc[df.lesion_id.isin(g8), "split"] = "val"
vaz = df.groupby("lesion_id").split.nunique(); assert (vaz == 1).all()
```

**O que faz.** Divide as 10.015 imagens em 70% treino, 15% validação, 15% teste, garantindo duas propriedades ao mesmo tempo: (1) todas as fotos de uma mesma lesão ficam no mesmo conjunto; (2) cada conjunto tem a mesma proporção de classes.

**Como, passo a passo.**
1. `StratifiedGroupKFold(n_splits=10)` corta os dados em 10 pedaços ("folds") de ~10% cada, respeitando grupos (`lesion_id`) e estratificação (`dx`).
2. Folds 0–6 (7 pedaços = 70%) → treino. Fold 7 (10%) → validação. Folds 8 e 9 (20%) → teste, por enquanto.
3. Para chegar a 15/15, metade das **lesões** do fold 8 (não metade das imagens — senão quebraria o agrupamento) é movida para validação.
4. `assert (vaz == 1).all()`: para cada lesão, conta em quantos splits diferentes ela aparece; tem de ser exatamente 1 para todas. Se não fosse, o notebook pararia com erro.

**Por que não `train_test_split` simples.** Ele não conhece grupos. Sortearia por imagem e colocaria fotos da mesma lesão em conjuntos diferentes.

**Task 1.** As imagens de segmentação não têm `lesion_id` nem classe, então recebem um sorteio simples 70/15/15 com a mesma semente. Se alguma existisse também na Task 3 ela herdaria o split de lá — mas a interseção é zero (célula 3).

**Saída e leitura.**

| dx | train | val | test | %train | %val | %test |
|---|---|---|---|---|---|---|
| MEL | 776 | 166 | 171 | 11,1 | 11,2 | 11,3 |
| NV | 4.671 | 1.002 | 1.032 | 66,6 | 67,4 | 68,3 |
| DF | 73 | 19 | 23 | 1,0 | 1,3 | 1,5 |

```
proporção: train 70,1% | val 14,8% | test 15,1%
lesões em mais de um split: 0
Task1 split: train 1811 | val 384 | test 399
```
- As colunas de % são quase iguais entre os três conjuntos: estratificação funcionou (**medido**).
- Zero lesões em mais de um split: sem vazamento (**medido**).
- DF tem 23 imagens no teste. Um único erro muda o recall de DF em 4,3 pontos. Por isso qualquer métrica por classe rara precisa de intervalo de confiança (bootstrap) na M3 (**inferido**).
- O resultado vai para `data/splits/split_v1.csv`, com colunas `image, lesion_id, dx, split, task`. É o único split do projeto: todo treino e avaliação lê esse arquivo.

---

## Célula 20 — Resumo de evidências

**O que faz.** Salva o dicionário `EVID` em `reports/m1_proposta/evidencias.json`.

**Por quê.** Rastreabilidade: cada número dos slides tem um campo neste JSON, que foi gerado por uma célula identificável deste notebook, que leu arquivos com nome e caminho conhecidos. Se alguém perguntar "de onde veio o 58×", a resposta é `imbalance_ratio` ← célula 5 ← `ISIC2018_Task3_Training_GroundTruth.csv`.

---

## O que este notebook **não** faz (de propósito)

- Não treina nada — a M1 proíbe.
- Não redimensiona nem salva imagens processadas — isso é a M2.
- Não mede pelos, réguas ou bolhas — só vinheta. Os demais artefatos ficam para inspeção visual (galeria) ou detector dedicado na M2.
- Não faz split estratificado para a Task 1 — não há classe para estratificar. É uma escolha, não uma limitação técnica.
