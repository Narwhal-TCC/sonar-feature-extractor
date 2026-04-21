# Cenários de Teste — sss-feature-extractor

Execute os comandos abaixo a partir da raiz do projeto (`sss-feature-extractor/`).  
Substitua `.\data\` pelo caminho real onde estão suas imagens.

---

## Antes de tudo — verificar o ambiente

```powershell
# Confirma que o comando está instalado e os extractors carregam
sss-feature-extractor --list-extractors
sss-feature-extractor --list-sensors
```

**Saída esperada de `--list-extractors`:**
```
Extractors de IMAGEM:
  basic_stats          ← BasicStatsExtractor
  color_channels       ← ColorChannelExtractor
  frequency            ← FrequencyExtractor
  glcm                 ← GLCMExtractor
  gradient             ← GradientExtractor
  haar_wavelet         ← HaarWaveletExtractor
  histogram            ← HistogramExtractor
  hog                  ← HOGExtractor
  spatial_grid         ← SpatialGridExtractor

Extractors de ROI:
  roi                  ← ROIExtractor

Total: 10 extractors
```

---

## Cenário 0 — Imagem única (teste mais rápido)

Processa **uma única imagem** para confirmar que a leitura, a extração e o CSV funcionam.  
Ideal para verificar rapidamente após instalação ou após alterar um extractor.

```powershell
sss-feature-extractor `
    --image .\data\2010\0001_2010.jpg `
    --output .\outputs\single_image.csv `
    --verbose
```

**O que verificar no CSV gerado:**
- Deve ter **1 linha por objeto anotado** (ou 1 linha com `ann_class_name=negative` se o .txt estiver vazio)
- Colunas `obj_highlight_ratio`, `obj_roi_std`, `obj_local_contrast` devem estar presentes e não-nulas
- Coluna `label` deve ser `0` (NOMBO) ou `1` (MILCO)

```powershell
# Verificação rápida no PowerShell
python -c "
import pandas as pd
df = pd.read_csv('outputs/single_image.csv')
print('Shape:', df.shape)
print('Labels:', df['label'].value_counts().to_dict())
print('NaNs:', df.isnull().sum().sum())
print('Cols obj_*:', [c for c in df.columns if c.startswith('obj_')])
"
```

---

## Cenário 1 — Pipeline completo (per_object, dois models)

Processa múltiplas pastas e gera **dois CSVs diferentes** com features distintas.  
Este é o cenário principal de uso em produção.

```powershell
sss-feature-extractor `
    --pipeline .\tests\pipeline_full.json `
    --folders .\data\2010\ .\data\2015\ .\data\2021\ `
    --output-dir .\outputs\test_full `
    --workers 4
```

**Com recursão** (se os anos estiverem em subpastas dentro de `data/`):

```powershell
sss-feature-extractor `
    --pipeline .\tests\pipeline_full.json `
    --folder .\data\ --recursive `
    --output-dir .\outputs\test_full `
    --workers 4
```

**Saída esperada:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✅  Pipeline concluído — 2 CSV(s)
  Saída                                      Linhas   Colunas
  ────────────────────────────────────────────────────────────
  sss_sonar/model_regression                    NNN        91
  sss_sonar/model_tree                          NNN       135
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**O que verificar:**
- `model_tree.csv` deve ter **135 colunas** (inclui `obj_*` do ROI extractor)
- `model_regression.csv` deve ter **91 colunas** (sem `obj_*` — ROI não listado)
- Ambos devem ter a coluna `source_folder` com os nomes das pastas de origem
- Coluna `label` deve estar na última posição

```powershell
python -c "
import pandas as pd
tree = pd.read_csv('outputs/test_full/sss_sonar/model_tree.csv')
regr = pd.read_csv('outputs/test_full/sss_sonar/model_regression.csv')

print('=== model_tree ===')
print('Shape:', tree.shape)
print('Sources:', sorted(tree['source_folder'].unique()))
print('Labels:', tree['label'].value_counts().to_dict())
print('Has ROI features:', any(c.startswith('obj_') for c in tree.columns))
print('Has Haar features:', any(c.startswith('haar_') for c in tree.columns))

print()
print('=== model_regression ===')
print('Shape:', regr.shape)
print('Has ROI features:', any(c.startswith('obj_') for c in regr.columns))
print('Has Haar features:', any(c.startswith('haar_') for c in regr.columns))
print('Has FFT features:', any(c.startswith('fft_') for c in regr.columns))
"
```

---

## Cenário 2 — Smoke test (pipeline mínimo, verificação rápida)

Usa apenas as 4 features mais importantes para confirmar que o pipeline funciona  
**sem esperar o processamento completo** do dataset.

```powershell
sss-feature-extractor `
    --pipeline .\tests\pipeline_minimal.json `
    --folder .\data\2010\ `
    --output-dir .\outputs\test_minimal
```

**Saída esperada:**
```
sss_sonar/smoke_test → N linhas × ~70 colunas
```

---

## Cenário 3 — Modo per_image (1 linha por imagem)

Gera features para **classificação binária no nível da imagem**:  
"esta imagem contém um objeto balístico?" → `label = 0` ou `1`.

```powershell
sss-feature-extractor `
    --pipeline .\tests\pipeline_per_image.json `
    --folder .\data\2010\ `
    --output-dir .\outputs\test_per_image
```

**O que verificar:**
- Número de linhas deve ser igual ao número de imagens (não de objetos)
- Coluna `label` = `1` para imagens que contêm pelo menos 1 MILCO
- Não deve existir nenhuma coluna `obj_*` (ROI não disponível neste modo)
- Deve existir `ann_xc_mean`, `ann_yc_mean` (médias das anotações)

```powershell
python -c "
import pandas as pd
df = pd.read_csv('outputs/test_per_image/sss_sonar/binary_classifier.csv')
print('Shape:', df.shape)
print('Labels:', df['label'].value_counts().to_dict())
print('ROI cols present:', any(c.startswith('obj_') for c in df.columns))
print('Ann cols present:', 'ann_xc_mean' in df.columns)
"
```

---

## Cenário 4 — Checkpoint e retomada

Simula uma interrupção no processamento e verifica que a retomada funciona corretamente.

```powershell
# Passo 1: Inicia o processamento (vai gerar o arquivo .ckpt.json)
sss-feature-extractor `
    --pipeline .\tests\pipeline_full.json `
    --folder .\data\ --recursive `
    --output-dir .\outputs\test_checkpoint `
    --checkpoint-every 10

# Passo 2: Interrompa com Ctrl+C durante o processamento
# (ou deixe terminar normalmente para testar que resume não reprocessa)

# Passo 3: Executa novamente — deve pular as imagens já processadas
sss-feature-extractor `
    --pipeline .\tests\pipeline_full.json `
    --folder .\data\ --recursive `
    --output-dir .\outputs\test_checkpoint `
    --resume --verbose
```

**Log esperado na segunda execução:**
```
Checkpoint: N imagens já processadas, pulando.
```

---

## Cenário 5 — Multi-pasta com lista em arquivo

```powershell
# Cria o arquivo de lista (um caminho por linha)
@"
# Dataset principal
.\data\2010
.\data\2015
.\data\2021
# .\data\2023  ← comentado, será ignorado
"@ | Out-File -Encoding utf8 .\tests\pastas.txt

# Processa usando a lista
sss-feature-extractor `
    --pipeline .\tests\pipeline_full.json `
    --folder-list .\tests\pastas.txt `
    --output-dir .\outputs\test_folder_list `
    --tag-source
```

---

## Cenário 6 — Modo legado (1 CSV, todos os extractors)

```powershell
# Pasta única
sss-feature-extractor `
    --folder .\data\2010\ `
    --output .\outputs\legado_2010.csv `
    --mode per_object `
    --tag-source

# Múltiplas pastas
sss-feature-extractor `
    --folders .\data\2010\ .\data\2021\ `
    --output .\outputs\legado_merged.csv `
    --tag-source `
    --workers 4
```

---

## Validação final — comparar os dois CSVs do Cenário 1

```powershell
python -c "
import pandas as pd

tree = pd.read_csv('outputs/test_full/sss_sonar/model_tree.csv')
regr = pd.read_csv('outputs/test_full/sss_sonar/model_regression.csv')

# Colunas em tree mas não em regr (devem ser as features de ROI, histogram, spatial, hog)
only_tree = set(tree.columns) - set(regr.columns)
only_regr = set(regr.columns) - set(tree.columns)

print('Colunas APENAS em model_tree:', sorted(only_tree)[:10], '...')
print('Colunas APENAS em model_regression:', sorted(only_regr)[:5])
print()
print('Colunas em comum:', len(set(tree.columns) & set(regr.columns)))
print('Linhas iguais:', len(tree) == len(regr))
"
```

**Saída esperada:**
- Colunas apenas em `model_tree`: `obj_*`, `hist_bin_*`, `grid_*`
- Colunas apenas em `model_regression`: `fft_*`
- Colunas em comum: metadados + `basic_stats` + `glcm_*` + `haar_*`
- Número de linhas igual nos dois CSVs (mesmas imagens processadas)
