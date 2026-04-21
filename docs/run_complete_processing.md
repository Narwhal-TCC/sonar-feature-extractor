# Guia de Processamento Completo — sonar-feature-extractor

Passo a passo para processar os 2 datasets disponíveis e gerar os CSVs finais para treinamento de ML.

---

## Pré-requisito: Instalação

```powershell
cd sonar-feature-extractor
pip install .

# Verificar (deve listar 11 extractors e 2 sensores)
sonar-feature-extractor --list-extractors
sonar-feature-extractor --list-sensors
```

> **Windows com caminho acentuado (ex: `Área de Trabalho`):** use `pip install .`, não `pip install -e .`

---

## Dataset 1 — SSS (Santos & Moura 2024)

**Dados:** `data/sss/2010/` + `data/sss/2015/`  
**Sensor:** Side-Scan Sonar JPEG 1024×1024, labels YOLO `.txt`  
**Modo:** `per_object` — 1 linha por objeto anotado (bbox)

```powershell
sonar-feature-extractor `
    --pipeline tests/pipeline_full.json `
    --folders data/sss/2010 data/sss/2015 `
    --output-dir outputs/dataset_sss `
    --workers 4
```

**Saída esperada:**
```
outputs/dataset_sss/sss_sonar/
├── model_tree.csv        # 167 colunas: basic_stats + histogram + glcm + haar + gradient + spatial_grid + ROI
└── model_regression.csv  #  97 colunas: basic_stats + glcm + gradient + frequency + haar
```

**Adicionar mais anos:**
```powershell
sonar-feature-extractor `
    --pipeline tests/pipeline_full.json `
    --folders data/sss/2010 data/sss/2015 data/sss/2017 data/sss/2021 `
    --output-dir outputs/dataset_sss `
    --workers 4
```

---

## Dataset 2 — FLS Sintético (Ściegienka & Blachnik 2024)

**Dados:** `data/fls-uxo-synthetic/Fold_1/`  
**Sensor:** Forward-Looking Sonar PNG 512×399, label codificado no nome do arquivo  
**Modo:** `per_image` (sem bbox — sempre)

> **IMPORTANTE:** usar `--recursive` pois as PNGs estão em `Fold_1/train/uxo/` e `Fold_1/train/nonuxo/`

```powershell
sonar-feature-extractor `
    --pipeline tests/pipeline_fls_synthetic.json `
    --folder data/fls-uxo-synthetic/Fold_1 `
    --output-dir outputs/dataset_fls_synthetic `
    --recursive
```

**Saída esperada:**
```
outputs/dataset_fls_synthetic/fls_uxo_synthetic/
├── model_classification.csv  # 174 colunas: todas as features + metadados do filename
└── model_minimal.csv         #  52 colunas: basic_stats + glcm + metadados
```

**Para múltiplos folds:**
```powershell
sonar-feature-extractor `
    --pipeline tests/pipeline_fls_synthetic.json `
    --folders data/fls-uxo-synthetic/Fold_1 data/fls-uxo-synthetic/Fold_2 `
    --output-dir outputs/dataset_fls_synthetic `
    --recursive `
    --tag-source
```

---

## Validação dos CSVs Gerados

```python
import pandas as pd

# Dataset 1 — SSS
tree = pd.read_csv("outputs/dataset_sss/sss_sonar/model_tree.csv")
regr = pd.read_csv("outputs/dataset_sss/sss_sonar/model_regression.csv")

print(f"SSS model_tree:       {len(tree)} linhas x {tree.shape[1]} colunas")
print(f"SSS model_regression: {len(regr)} linhas x {regr.shape[1]} colunas")
print(f"Labels: {tree['label'].value_counts().to_dict()}")

# Verificacoes
assert any(c.startswith("obj_") for c in tree.columns),   "model_tree deve ter ROI features"
assert not any(c.startswith("obj_") for c in regr.columns), "model_regression nao deve ter ROI"
assert any(c.startswith("fft_") for c in regr.columns),  "model_regression deve ter FFT"
assert tree["label"].isin([0, 1]).all(),  "label deve ser 0 (NOMBO) ou 1 (MILCO)"
assert "source_folder" in tree.columns,  "deve ter source_folder"

# NaN esperados: apenas ann_* e obj_* para imagens negativas (sem bbox)
unexpected_nan = [c for c in tree.columns
                  if tree[c].isnull().any()
                  and not c.startswith(("ann_", "obj_"))]
assert len(unexpected_nan) == 0, f"NaN inesperado em: {unexpected_nan}"
print("SSS: OK")

# Dataset 2 — FLS Synthetic
fls = pd.read_csv("outputs/dataset_fls_synthetic/fls_uxo_synthetic/model_classification.csv")
print(f"\nFLS model_classification: {len(fls)} linhas x {fls.shape[1]} colunas")
print(f"Labels: {fls['label'].value_counts().to_dict()}")

assert any(c.startswith("fls_syn_") for c in fls.columns), "deve ter metadados do filename"
assert fls["label"].isin([0, 1]).all(),   "label deve ser 0 (nonUXO) ou 1 (UXO)"
assert fls.isnull().sum().sum() == 0,     "FLS nao deve ter NaN"
print("FLS Synthetic: OK")
```

---

## Notas Importantes

### SSS — Imagens negativas
Imagens SSS sem objetos anotados (`.txt` vazio) geram 1 linha por imagem com:
- `label = 0`, `n_annotations = 0`, `has_milco = 0`, `has_nombo = 0`
- Colunas `ann_*` e `obj_*` = NaN (sem bounding box, sem ROI)

Isso é comportamento correto — as features de textura/frequência da imagem inteira são preenchidas normalmente.

### FLS Synthetic — Estrutura de pastas
Os dados estão em subpastas `uxo/` e `nonuxo/` dentro de cada fold. **Sempre usar `--recursive`.**

### Retomada de processamento
Para conjuntos grandes (centenas de imagens), use `--checkpoint-every 50` (padrão) para salvar progresso. Se interromper, execute o mesmo comando com `--resume` (padrão):

```powershell
sonar-feature-extractor --pipeline tests/pipeline_full.json --folders data/sss/... --output-dir outputs/ --resume
```

### Performance
- Aumentar `--workers` para melhorar velocidade (ex: `--workers 8` em máquinas com 8+ cores)
- O engine processa cada imagem **uma única vez** e filtra as colunas por modelo — eficiente independentemente do número de modelos no pipeline
