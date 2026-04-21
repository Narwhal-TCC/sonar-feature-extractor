# sss-feature-extractor — Documento Técnico sobre Features Extraídas

> **Propósito deste documento:** fornecer informações relevantes e técnicas sobre todas as features que o componente extrai das imagens, auxiliando os profissionais de Machine Learning à identificar as melhores features para treino de seus modelos.

## Guia de Comandos

# 1. Pasta única
python sonar_feature_extractor.py --folder ./dataset/

# 2. Várias pastas explícitas na linha de comando
python sonar_feature_extractor.py --folders ./missao_2010/ ./missao_2015/ ./missao_2021/

# 3. Arquivo de texto com uma pasta por linha (linhas com # são ignoradas)
python sonar_feature_extractor.py --folder-list pastas.txt

# 4. Recursivo — desce automaticamente em todas as subpastas
python sonar_feature_extractor.py --folder ./dataset/ --recursive


# Documentação Técnica de Features — Side-Scan Sonar (SSS)
## Dataset: *Side-scan sonar imaging for Mine detection*
> Santos & Moura (2024) · DOI: `10.6084/m9.figshare.24574879.v2`

---

## Sumário

1. [Contexto do Problema](#1-contexto-do-problema)
2. [Estrutura dos Dados de Entrada](#2-estrutura-dos-dados-de-entrada)
3. [Grupos de Features](#3-grupos-de-features)
   - [Grupo 0 — Metadados da Imagem](#grupo-0--metadados-da-imagem)
   - [Grupo 1 — Estatísticas de Primeira Ordem](#grupo-1--estatísticas-de-primeira-ordem)
   - [Grupo 2 — Histograma de Intensidade](#grupo-2--histograma-de-intensidade)
   - [Grupo 3 — Textura GLCM (Haralick)](#grupo-3--textura-glcm-haralick)
   - [Grupo 4 — Gradiente e Bordas](#grupo-4--gradiente-e-bordas)
   - [Grupo 5 — Domínio da Frequência (FFT)](#grupo-5--domínio-da-frequência-fft)
   - [Grupo 6 — Grid Espacial](#grupo-6--grid-espacial)
   - [Grupo 7 — HOG (Histogram of Oriented Gradients)](#grupo-7--hog-histogram-of-oriented-gradients)
   - [Grupo 8 — Canais de Cor](#grupo-8--canais-de-cor)
   - [Grupo 9 — Rótulos e Coordenadas (Labels)](#grupo-9--rótulos-e-coordenadas-labels)
   - [Grupo 10 — Features da Região do Objeto (ROI)](#grupo-10--features-da-região-do-objeto-roi)
4. [Tabela Consolidada de Relevância para ML](#4-tabela-consolidada-de-relevância-para-ml)
5. [Guia de Uso por Modelo de ML](#5-guia-de-uso-por-modelo-de-ml)
6. [Recomendações de Pré-processamento](#6-recomendações-de-pré-processamento)

---

## 1. Contexto do Problema

O **Side-Scan Sonar (SSS)** é um sensor acústico que emite pulsos sonoros lateralmente ao veículo subaquático e registra o eco retornado. O resultado é uma imagem 2D em tons de cinza onde:

- **Highlight (reflexo):** Região de alta intensidade (branca/clara). Indica a face do objeto voltada para o sonar. Objetos sólidos e densos — como minas e bombas — criam reflexos fortes.
- **Shadow (sombra acústica):** Região escura atrás do objeto, onde o som não chegou. O tamanho e a forma da sombra permitem inferir a geometria e a altura do objeto.
- **Background (fundo):** Textura do fundo marinho — areia, lama, rochas.

O objetivo do pipeline de features é **transformar esse padrão visual em números** que modelos de ML possam separar entre:
- `NOMBO` (class 0) — *Non-Mine-like Bottom Object*: objetos comuns do fundo marinho
- `MILCO` (class 1) — *Mine-Like Contact*: objetos balísticos, minas, bombas

---

## 2. Estrutura dos Dados de Entrada

### 2.1 Imagem JPEG

```
Arquivo : 0257_2010.jpg
Dimensão: 1024 × 1024 pixels
Modo    : RGB (convertido internamente para grayscale)
```

**Como é carregada no código:**

```python
# sonar_feature_extractor.py — função load_image()
img_bgr  = cv2.imread(image_path, cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
```

> **Por que converter para grayscale?**  
> O SSS produz imagens essencialmente monocromáticas. A cor nas imagens do dataset é pseudocor (aplicada em pós-processamento para facilitar a visualização humana). A intensidade do canal de luminância (gray) é o dado físico real. A conversão elimina redundância e reduz o custo computacional em ~3×.

---

### 2.2 Arquivo de Anotação YOLO (`.txt`)

O arquivo de anotação segue o **formato YOLO v5/v8**, onde cada linha representa um objeto anotado por especialistas:

```
<class_id> <x_center> <y_center> <width> <height>
```

**Exemplo real do arquivo `0257_2010.txt`:**

```
0 0.75341796875 0.5732421875 0.0693359375 0.02734375
0 0.75537109375 0.5751953125 0.0791015625 0.033203125
```

| Campo | Tipo | Descrição |
|---|---|---|
| `class_id` | int | `0` = NOMBO, `1` = MILCO |
| `x_center` | float [0,1] | Centro horizontal do bounding box, normalizado pela largura da imagem |
| `y_center` | float [0,1] | Centro vertical do bounding box, normalizado pela altura da imagem |
| `width`    | float [0,1] | Largura do bounding box, normalizada pela largura da imagem |
| `height`   | float [0,1] | Altura do bounding box, normalizada pela altura da imagem |

**Como é lido e convertido para pixels absolutos no código:**

```python
# sonar_feature_extractor.py — função load_annotations()
cls, xc, yc, w, h = int(parts[0]), float(parts[1]), float(parts[2]),
                     float(parts[3]), float(parts[4])

x1 = int((xc - w / 2) * img_w)   # canto superior esquerdo
y1 = int((yc - h / 2) * img_h)
x2 = int((xc + w / 2) * img_w)   # canto inferior direito
y2 = int((yc + h / 2) * img_h)
```

A fórmula desfaz a normalização YOLO, convertendo centros e dimensões proporcionais em coordenadas absolutas de pixel, necessárias para recortar a ROI (Region of Interest).

---

## 3. Grupos de Features

---

### Grupo 0 — Metadados da Imagem

> **Função no código:** montagem inline dentro de `extract_features()`  
> **Quantidade de colunas:** 6

Estas colunas não são features de ML por si só, mas são essenciais para rastreabilidade, debug e para construir features derivadas.

---

#### `image_path`
- **Tipo:** `string`
- **Descrição:** Caminho completo do arquivo de imagem no sistema de arquivos.
- **Código:**
  ```python
  global_feats = {"image_path": image_path, ...}
  ```
- **Uso em ML:** Não entra no modelo. Serve para remontar a imagem original a partir de uma linha do CSV durante análise de erros (debugging de falsos positivos/negativos).

---

#### `img_width` e `img_height`
- **Tipo:** `int`
- **Valores esperados:** `1024` × `1024` para o dataset de referência.
- **Descrição:** Dimensões em pixels da imagem. Todas as imagens do dataset têm tamanho fixo, mas é boa prática registrar isso para detectar imagens corrompidas ou de resolução diferente.
- **Código:**
  ```python
  img_bgr, gray = load_image(image_path)
  h, w = gray.shape
  global_feats["img_width"]  = w
  global_feats["img_height"] = h
  ```
- **Uso em ML:** Pode servir para filtrar outliers (imagens com dimensão inesperada). Em pipelines com imagens de resolução variável, normalizar coordenadas por `img_width`/`img_height` é obrigatório.

---

#### `n_annotations`
- **Tipo:** `int`
- **Exemplo:** `2` (duas anotações na imagem de exemplo)
- **Descrição:** Número total de objetos anotados na imagem (soma de NOMBO + MILCO).
- **Código:**
  ```python
  annotations = load_annotations(label_path, w, h)
  global_feats["n_annotations"] = len(annotations)
  ```
- **Uso em ML:** Imagens com `n_annotations = 0` são **amostras negativas** puras (sem objetos identificados). Imagens com muitas anotações podem indicar cenas mais complexas. Pode ser usada como feature auxiliar em modelos de contagem ou multi-instância.

---

#### `has_milco`
- **Tipo:** `int` (0 ou 1)
- **Descrição:** Flag booleana. Vale `1` se a imagem contém pelo menos um objeto da classe MILCO (Mine-Like Contact), `0` caso contrário.
- **Código:**
  ```python
  global_feats["has_milco"] = int(any(a["class_id"] == 1 for a in annotations))
  ```
- **Uso em ML:** **Label primário** no modo `per_image`. É o alvo da classificação binária: "Esta imagem contém uma mina/objeto balístico?" Modelos simples como Regressão Logística e SVM usam esta coluna como `y`.

---

#### `has_nombo`
- **Tipo:** `int` (0 ou 1)
- **Descrição:** Flag booleana. Vale `1` se a imagem contém pelo menos um objeto NOMBO.
- **Código:**
  ```python
  global_feats["has_nombo"] = int(any(a["class_id"] == 0 for a in annotations))
  ```
- **Uso em ML:** Útil para análise exploratória e para entender o balanceamento das classes. Uma imagem pode ter `has_milco = 1` e `has_nombo = 1` simultaneamente (cena mista), o que é um caso difícil para modelos.

---

### Grupo 1 — Estatísticas de Primeira Ordem

> **Função no código:** `features_basic_stats(gray)`  
> **Quantidade de colunas:** 13  
> **Biblioteca:** `numpy`, `scipy.stats`

As **estatísticas de primeira ordem** tratam a imagem como um conjunto de valores escalares independentes (ignora relações espaciais). São as features mais simples e computacionalmente baratas. Capturam o **perfil de brilho global** da imagem.

```python
def features_basic_stats(gray: np.ndarray) -> dict:
    flat = gray.flatten().astype(np.float32)
    return {
        "mean":     float(np.mean(flat)),
        "std":      float(np.std(flat)),
        ...
    }
```

A imagem 2D é **achatada** em um vetor 1D de `1024 × 1024 = 1.048.576` valores, e estatísticas são aplicadas sobre esse vetor.

---

#### `mean`
- **Fórmula:** `μ = (1/N) · Σ I(x,y)`  
- **Exemplo:** `39.75`
- **Descrição:** Intensidade média de todos os pixels. No contexto do sonar SSS, reflete o **brilho global da cena**. Uma imagem com muitos reflexos de objetos terá `mean` maior que uma imagem apenas com fundo escuro.
- **Relevância ML:** ⭐⭐⭐ Alta. Separação direta entre imagens ruidosas e limpas. Cenas com objetos MILCO tendem a ter `mean` ligeiramente maior pela presença do highlight.

---

#### `std`
- **Fórmula:** `σ = sqrt((1/N) · Σ (I - μ)²)`  
- **Exemplo:** `24.23`
- **Descrição:** Desvio padrão dos pixels — mede o **contraste global** da imagem. Uma imagem com highlight claro e sombra escura terá `std` alto. Imagem de fundo uniforme terá `std` baixo.
- **Relevância ML:** ⭐⭐⭐⭐ Muito alta. Objetos sólidos criam descontinuidades de intensidade que aumentam o `std`. É uma das features mais discriminativas para presença de objeto.

---

#### `min` e `max`
- **Exemplo:** `min = 0.0`, `max = 254.0`
- **Descrição:** Valores mínimo e máximo de intensidade na imagem. No sonar, `max` próximo de 255 indica presença de um reflexo forte (highlight de objeto sólido). `min` próximo de 0 confirma sombra acústica profunda.
- **Relevância ML:** ⭐⭐ Média. Informação útil mas redundante com percentis e `std`. Sensível a ruído de um único pixel.

---

#### `p10`, `p25`, `p50`, `p75`, `p90`
- **Exemplos:** `p10 = 8.0`, `p25 = 19.0`, `p50 = 39.0`, `p75 = 60.0`, `p90 = 72.0`
- **Descrição:** Percentis da distribuição de intensidade. São **mais robustos** que `min`/`max` pois resistem a pixels ruidosos.
  - `p50` (mediana): brilho "típico" da cena.
  - `p90 - p10`: amplitude efetiva (similar ao `iqr` mas mais extrema).
  - A diferença `p90 - p50` indica assimetria da cauda superior (presença de highlights).
- **Relevância ML:** ⭐⭐⭐ Alta. Perfil de percentis captura a "forma" da distribuição sem ruído de extremos. Útil em modelos lineares.

---

#### `skewness`
- **Fórmula:** `γ₁ = E[(X - μ)³] / σ³`
- **Exemplo:** `0.664`
- **Descrição:** Assimetria da distribuição de intensidade. Valor positivo (cauda à direita) indica que há poucos pixels muito brilhantes — compatível com a presença de um highlight de objeto. Valor próximo de zero indica distribuição simétrica (fundo uniforme).
- **Relevância ML:** ⭐⭐⭐⭐ Muito alta. Minas e objetos balísticos criam assimetria positiva forte por causa do highlight. É um discriminador de presença de objeto especialmente útil em Regressão Logística.

---

#### `kurtosis`
- **Fórmula:** `γ₂ = E[(X - μ)⁴] / σ⁴ - 3`
- **Exemplo:** `3.775`
- **Descrição:** "Achatamento" da distribuição. Kurtosis alta (leptocúrtica) indica distribuição com cauda pesada — muitos pixels em valores extremos (muito escuros ou muito claros). É característica de imagens com objetos: fundo uniforme + highlight + sombra.
- **Relevância ML:** ⭐⭐⭐ Alta. Complementa `skewness`. Juntas, as duas capturam a forma completa da distribuição de intensidade.

---

#### `iqr`
- **Fórmula:** `IQR = p75 - p25`
- **Exemplo:** `41.0`
- **Descrição:** Intervalo interquartil — spread robusto dos pixels centrais da distribuição. Mede a dispersão dos 50% centrais da imagem, ignorando extremos.
- **Relevância ML:** ⭐⭐ Média. Proxy de contraste robusto a outliers. Menos informativo que `std` em imagens SSS, mas mais estável quando há ruído de speckle.

---

#### `energy`
- **Fórmula:** `E = (1/N) · Σ I(x,y)²`
- **Exemplo:** `2166.93`
- **Descrição:** Energia normalizada da imagem. No contexto acústico, energia é proporcional à **intensidade total do sinal retornado**. Áreas com mais reflexos (objetos) têm maior energia.
- **Relevância ML:** ⭐⭐⭐ Alta. Captura a "potência" da cena. Correlacionado com `mean` mas penaliza mais os pixels brilhantes (quadrático), sendo mais sensível a highlights de objetos.

---

### Grupo 2 — Histograma de Intensidade

> **Função no código:** `features_histogram(gray, bins=32)`  
> **Quantidade de colunas:** 32 (`hist_bin_00` a `hist_bin_31`)  
> **Biblioteca:** `numpy`

O histograma divide o intervalo `[0, 255]` em 32 bins iguais (cada bin cobre ~8 valores de intensidade) e conta a densidade de pixels em cada intervalo.

```python
def features_histogram(gray: np.ndarray, bins: int = 32) -> dict:
    hist, _ = np.histogram(gray.flatten(), bins=bins, range=(0, 256), density=True)
    return {f"hist_bin_{i:02d}": float(v) for i, v in enumerate(hist)}
```

O parâmetro `density=True` normaliza a contagem para que a integral seja 1 (é uma densidade de probabilidade), tornando o histograma **invariante ao tamanho da imagem**.

---

#### `hist_bin_00` a `hist_bin_31`
- **Fórmula:** `hist_bin_k = P(8k ≤ I < 8(k+1))`
- **Faixas de cada bin:**

| Coluna | Faixa de Intensidade | Interpretação no SSS |
|--------|---------------------|----------------------|
| `hist_bin_00` | [0, 8) | Sombras acústicas muito profundas |
| `hist_bin_01`–`hist_bin_09` | [8, 80) | Fundo marinho escuro, ruído |
| `hist_bin_10`–`hist_bin_19` | [80, 160) | Fundo médio, texturas do fundo |
| `hist_bin_20`–`hist_bin_27` | [160, 224) | Semi-highlight, reflexos fracos |
| `hist_bin_28`–`hist_bin_31` | [224, 255] | Highlight forte — objetos sólidos |

- **Exemplo dos primeiros bins:**
  - `hist_bin_00 = 0.012494` → ~1,25% dos pixels são sombra profunda
  - `hist_bin_04 = 0.012439` → distribuição uniforme no fundo (imagem de fundo limpo)

- **Relevância ML:** ⭐⭐⭐ Alta (coletivamente). O histograma completo é uma **assinatura espectral** da imagem. Imagens com MILCO tendem a ter `hist_bin_28` a `hist_bin_31` maiores (mais highlight) e `hist_bin_00` maior (mais sombra). Modelos como SVM com kernel RBF e Random Forest conseguem explorar esse padrão. Cuidado: como conjunto, são 32 features correlacionadas — considerar PCA antes de usar em regressão linear.

---

### Grupo 3 — Textura GLCM (Haralick)

> **Função no código:** `features_texture_glcm(gray)`  
> **Quantidade de colunas:** 12 (6 propriedades × 2 estatísticas cada)  
> **Biblioteca:** `skimage.feature.graycomatrix`, `skimage.feature.graycoprops`

A **Gray Level Co-occurrence Matrix (GLCM)** é uma matriz quadrada onde o elemento `M[i,j]` conta com que frequência o valor de intensidade `i` aparece adjacente ao valor `j` na imagem, em uma dada direção e distância. É a base das **Features de Haralick**, propostas em 1973 e ainda amplamente usadas em textura de imagens de sonar e radar.

```python
def features_texture_glcm(gray: np.ndarray) -> dict:
    gray_64 = (gray // 4).astype(np.uint8)          # reduz para 64 níveis
    distances = [1, 3]                               # vizinhança imediata e a 3 pixels
    angles    = [0, np.pi/4, np.pi/2, 3*np.pi/4]    # 0°, 45°, 90°, 135°

    glcm = graycomatrix(gray_64, distances=distances, angles=angles,
                        levels=64, symmetric=True, normed=True)
```

> **Redução para 64 níveis (`// 4`):** A GLCM de 256 níveis resultaria numa matriz 256×256, custosa de computar e esparsa. Reduzir para 64 níveis (`0-63`) mantém a estrutura de textura e reduz a complexidade em 16×.

> **Ângulos múltiplos:** Calculamos a GLCM em 4 direções (horizontal, diagonal, vertical, antidiagonal) para capturar texturas em todas as orientações. O resultado final (média e desvio entre os ângulos) é **invariante à rotação**.

> **Distâncias [1, 3]:** Distância 1 captura padrões de micro-textura (grão fino do fundo); distância 3 captura padrões de macro-textura (ripples do fundo, padrões maiores).

Para cada propriedade, calculamos `_mean` e `_std` sobre as 8 combinações de distância e ângulo.

---

#### `glcm_contrast_mean` / `glcm_contrast_std`
- **Fórmula:** `Contrast = Σᵢⱼ (i-j)² · M[i,j]`
- **Exemplos:** `mean = 66.73`, `std = 0.225`
- **Descrição:** Mede as diferenças de intensidade entre pixels vizinhos. **Contraste alto** = muitas transições bruscas = presença de bordas ou texturas rugosas. O fundo marinho arenoso tem contraste médio; o par highlight/shadow de um objeto balístico gera contraste muito alto na borda.
- **Relevância ML:** ⭐⭐⭐⭐ Muito alta. Uma das features de Haralick mais discriminativas para SSS. Objetos sólidos elevam o contraste local dramaticamente.

---

#### `glcm_dissimilarity_mean` / `glcm_dissimilarity_std`
- **Fórmula:** `Dissimilarity = Σᵢⱼ |i-j| · M[i,j]`
- **Exemplo:** `mean = 6.64`
- **Descrição:** Similar ao contraste, mas com crescimento linear (não quadrático) nas diferenças. É mais tolerante a diferenças extremas de intensidade. Mede a heterogeneidade da textura.
- **Relevância ML:** ⭐⭐⭐ Alta. Correlacionado com contraste, mas capta a tendência geral de variação de forma mais suave. Útil como feature complementar.

---

#### `glcm_homogeneity_mean` / `glcm_homogeneity_std`
- **Fórmula:** `Homogeneity = Σᵢⱼ M[i,j] / (1 + |i-j|)`
- **Exemplo:** `mean = 0.139`
- **Descrição:** Inverso do contraste. **Homogeneidade alta** = pixels vizinhos têm valores similares = fundo uniforme ou textura lisa. É alta no fundo marinho aberto e cai drasticamente perto de objetos.
- **Relevância ML:** ⭐⭐⭐⭐ Muito alta. Complementar ao contraste. Imagens sem objetos terão `homogeneity_mean` alto; imagens com MILCO, baixo. Boa feature para regressão logística.

---

#### `glcm_energy_mean` / `glcm_energy_std`
- **Fórmula:** `Energy = Σᵢⱼ M[i,j]²` (também chamado de Angular Second Moment)
- **Exemplo:** `mean = 0.0499`
- **Descrição:** Mede a **uniformidade** da GLCM. Quando a GLCM tem poucos pares dominantes (textura muito regular/repetitiva), a energia é alta. Textura complexa ou aleatória distribui os valores e reduz a energia.
- **Relevância ML:** ⭐⭐⭐ Alta. Fundo arenoso regular tem energia de textura maior que uma cena com objetos. Distingue bem background puro de cenas com objetos.

---

#### `glcm_correlation_mean` / `glcm_correlation_std`
- **Fórmula:** `Correlation = Σᵢⱼ [(i·j · M[i,j] - μᵢ·μⱼ) / (σᵢ·σⱼ)]`
- **Exemplo:** `mean = 0.089`
- **Descrição:** Correlação linear entre os valores de pixels em pares. Mede **dependência linear** entre pixels vizinhos. Alta correlação = estrutura espacial linear na textura (ex: ripples paralelos do fundo). Baixa correlação = padrão aleatório.
- **Relevância ML:** ⭐⭐ Média. Menos intuitiva para SSS, mas captura padrões direcionais da textura do fundo que podem mudar entre imagens.

---

#### `glcm_ASM_mean` / `glcm_ASM_std`
- **Nota:** ASM = *Angular Second Moment*, idêntico numericamente ao `energy` da GLCM.
- **Fórmula:** `ASM = Σᵢⱼ M[i,j]²`
- **Exemplo:** `mean = 0.002492`
- **Descrição:** Calculado como propriedade separada pelo `graycoprops` do scikit-image. Coincide com `energy` mas pode diferir em implementações específicas de normalização.
- **Relevância ML:** ⭐⭐ Média (quando combinado com `energy`). Manter ambos permite que o modelo decida qual usar.

---

### Grupo 4 — Gradiente e Bordas

> **Função no código:** `features_gradient(gray)`  
> **Quantidade de colunas:** 6  
> **Biblioteca:** `skimage.filters.sobel`, `cv2.Laplacian`

Features baseadas em **derivadas espaciais** da imagem. O gradiente detecta onde a intensidade **muda rapidamente** — ou seja, onde estão as bordas. No sonar, bordas são as transições entre highlight, shadow e background — exatamente onde os objetos se revelam.

---

#### `sobel_mean`, `sobel_std`, `sobel_max`
- **Operador:** Filtro de Sobel (derivada de primeira ordem)
- **Fórmula:**
  ```
  Gx = [[-1,0,1],[-2,0,2],[-1,0,1]] * I
  Gy = [[-1,-2,-1],[0,0,0],[1,2,1]] * I
  |G| = sqrt(Gx² + Gy²)
  ```
- **Código:**
  ```python
  from skimage.filters import sobel
  sobel_mag = sobel(gray.astype(np.float32))
  ```
- **Exemplos:** `sobel_mean = 17.92`, `sobel_std = 9.14`, `sobel_max = 171.72`

| Coluna | Descrição | Relevância ML |
|--------|-----------|---------------|
| `sobel_mean` | Magnitude média de bordas na imagem inteira. Alta = muitas transições de intensidade | ⭐⭐⭐⭐ |
| `sobel_std` | Variação da magnitude de bordas. Alta = bordas concentradas (objeto isolado vs. ruído difuso) | ⭐⭐⭐ |
| `sobel_max` | Borda mais nítida da imagem. No SSS, a transição highlight→shadow é a borda mais forte | ⭐⭐⭐ |

- **Interpretação no SSS:** O par highlight/shadow de uma mina cria uma borda muito forte e localizada. `sobel_max` alto com `sobel_mean` moderado indica presença de objeto isolado — padrão discriminativo para MILCO.

---

#### `laplacian_mean`, `laplacian_std`, `laplacian_var`
- **Operador:** Laplaciano (derivada de segunda ordem)
- **Fórmula:**
  ```
  ∇²I = ∂²I/∂x² + ∂²I/∂y²
  Kernel: [[0,1,0],[1,-4,1],[0,1,0]]
  ```
- **Código:**
  ```python
  laplacian = cv2.Laplacian(gray, cv2.CV_64F)
  ```
- **Exemplos:** `laplacian_mean = 86.52`, `laplacian_std = 103.20`, `laplacian_var = 10650.13`

| Coluna | Descrição | Relevância ML |
|--------|-----------|---------------|
| `laplacian_mean` | Média do valor absoluto do Laplaciano — mede a "riqueza de bordas" global | ⭐⭐⭐ |
| `laplacian_std` | Desvio do Laplaciano — indica se as bordas são uniformes ou concentradas | ⭐⭐⭐ |
| `laplacian_var` | **Proxy de nitidez da imagem** (variância do Laplaciano). Imagens nítidas com objetos têm `laplacian_var` muito alto | ⭐⭐⭐⭐ |

> **`laplacian_var` como detector de nitidez:** Uma imagem de fundo homogêneo tem Laplaciano próximo de zero em toda parte → variância baixa. Uma imagem com objeto sólido tem picos e vales fortes no Laplaciano na borda do objeto → variância alta. É um dos melhores proxies globais de "há um objeto saliente aqui?".

---

### Grupo 5 — Domínio da Frequência (FFT)

> **Função no código:** `features_frequency(gray)`  
> **Quantidade de colunas:** 4  
> **Biblioteca:** `numpy.fft`

A **Transformada de Fourier 2D** decompõe a imagem em suas componentes de frequência espacial. Frequências baixas correspondem a variações suaves (background, iluminação geral); frequências altas correspondem a detalhes finos e bordas (objetos, ruído).

```python
def features_frequency(gray: np.ndarray) -> dict:
    fft   = np.fft.fft2(gray.astype(np.float32))
    fft_s = np.fft.fftshift(fft)         # move frequência zero para o centro
    mag   = np.abs(fft_s)                # magnitude do espectro

    # Define raios de corte para as 3 bandas
    r_low, r_mid = min(h, w) // 8, min(h, w) // 4
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)

    low_energy  = np.sum(mag[dist <= r_low]  ** 2)
    mid_energy  = np.sum(mag[(dist > r_low) & (dist <= r_mid)] ** 2)
    high_energy = np.sum(mag[dist > r_mid]   ** 2)
```

A energia em cada banda é calculada como a soma dos quadrados das magnitudes (teorema de Parseval: energia no espaço = energia na frequência).

---

#### `fft_low_energy_ratio`
- **Banda:** Raio 0 a `min(H,W)/8` pixels no espaço de frequência
- **Exemplo:** `0.766` → 76,6% da energia está nas baixas frequências
- **Descrição:** Proporção da energia concentrada nas frequências baixas — corresponde ao **componente DC e variações lentas** (background, gradientes de iluminação, fundo uniforme).
- **Relevância ML:** ⭐⭐⭐ Alta. Imagens de fundo puro têm `fft_low_energy_ratio` muito alto (quase toda energia no DC). Imagens com objetos têm razão menor (energia distribuída em frequências maiores).

---

#### `fft_mid_energy_ratio`
- **Banda:** Raio `min(H,W)/8` a `min(H,W)/4`
- **Exemplo:** `0.037`
- **Descrição:** Proporção de energia nas **frequências médias** — textura do fundo marinho, ripples, padrões de sedimentação.
- **Relevância ML:** ⭐⭐ Média. Sensível ao tipo de fundo (areia → ripples → pedras). Pode ajudar a distinguir fundos problemáticos com muita textura, que são confundidos com objetos.

---

#### `fft_high_energy_ratio`
- **Banda:** Raio acima de `min(H,W)/4`
- **Exemplo:** `0.197`
- **Descrição:** Proporção de energia nas **altas frequências** — bordas nítidas, detalhes finos, ruído de speckle do sonar, e **padrões geométricos de objetos sólidos**.
- **Relevância ML:** ⭐⭐⭐⭐ Muito alta. Objetos balísticos (geometria regular e superfície dura) introduzem padrões de alta frequência característicos. Fundo sedimentar suave tem `fft_high_energy_ratio` baixo. É uma das features de frequência mais discriminativas.

---

#### `fft_total_log_energy`
- **Fórmula:** `log(1 + low + mid + high)`
- **Exemplo:** `35.41`
- **Descrição:** Log da energia total do espectro. A escala logarítmica comprime a variação enorme entre imagens muito diferentes (espectros variam em ordens de grandeza).
- **Relevância ML:** ⭐⭐ Média. Captura a "intensidade geral" do sinal de toda a imagem. Útil como feature de normalização global.

---

### Grupo 6 — Grid Espacial

> **Função no código:** `features_spatial_grid(gray, grid=4)`  
> **Quantidade de colunas:** 32 (16 células × 2 estatísticas)  
> **Formato de nomes:** `grid_rRcC_mean` e `grid_rRcC_std` onde R=linha (0-3), C=coluna (0-3)

A imagem é dividida em um **grid 4×4 de 16 células** iguais (cada célula com 256×256 pixels para imagens 1024×1024). Para cada célula, calcula-se a média e o desvio padrão de intensidade.

```python
def features_spatial_grid(gray: np.ndarray, grid: int = 4) -> dict:
    h, w   = gray.shape
    ch, cw = h // grid, w // grid
    for r in range(grid):
        for c in range(grid):
            cell = gray[r*ch:(r+1)*ch, c*cw:(c+1)*cw].astype(np.float32)
            feats[f"grid_r{r}c{c}_mean"] = float(np.mean(cell))
            feats[f"grid_r{r}c{c}_std"]  = float(np.std(cell))
```

---

#### `grid_rRcC_mean` e `grid_rRcC_std`
- **Exemplo:** `grid_r0c0_mean = 39.54`, `grid_r0c0_std = 23.10`
- **Mapa das células no sonar SSS:**

```
                 Coluna 0     Coluna 1     Coluna 2     Coluna 3
Linha 0:   [ r0c0        | r0c1        | r0c2        | r0c3        ]
Linha 1:   [ r1c0        | r1c1        | r1c2        | r1c3        ]  ← Nadir (centro)
Linha 2:   [ r2c0        | r2c1        | r2c2        | r2c3        ]
Linha 3:   [ r3c0        | r3c1        | r3c2        | r3c3        ]
           ↑                                                       ↑
        Flanco esquerdo                               Flanco direito
```

- **Descrição:**
  - `_mean` de cada célula revela onde na imagem há regiões brilhantes (highlight) ou escuras (shadow).
  - `_std` de cada célula indica se aquela região tem textura heterogênea (presença de objeto) ou uniforme (fundo limpo).
  - A **assimetria espacial** entre células é fundamental: objetos no flanco esquerdo do sonar terão `grid_rXc0_mean` e `grid_rXc1_mean` maiores que o lado direito.

- **Relevância ML:** ⭐⭐⭐⭐ Muito alta (como conjunto). O grid é a feature mais poderosa para **localização implícita** de objetos sem usar detecção explícita. Modelos de árvore (Random Forest, XGBoost) conseguem explorar o padrão espacial das células para inferir onde o objeto está e qual a sua classe. Em modo `per_image`, é a principal alternativa às coordenadas absolutas do bounding box.

> **Insight:** Se um MILCO está no quadrante superior direito, as células `r0c2`, `r0c3` terão `_mean` alto e `_std` alto. O modelo aprende esse padrão espacial implicitamente.

---

### Grupo 7 — HOG (Histogram of Oriented Gradients)

> **Função no código:** `features_hog_condensed(gray)`  
> **Quantidade de colunas:** 324 (`hog_000` a `hog_323`)  
> **Biblioteca:** `skimage.feature.hog`

O **HOG** é um descritor clássico de Computer Vision que captura a **estrutura direcional de bordas** na imagem. A imagem é dividida em células e, em cada célula, conta-se em quais direções os gradientes são mais intensos (9 orientações de 0° a 180°). Os histogramas são então normalizados em blocos de células adjacentes.

```python
def features_hog_condensed(gray: np.ndarray) -> dict:
    resized = cv2.resize(gray, (128, 128))          # redimensiona para velocidade
    hog_vec = hog(
        resized,
        orientations=9,          # 9 bins de direção (0° a 180°, steps de 20°)
        pixels_per_cell=(32, 32),# células grandes → vetor compacto
        cells_per_block=(2, 2),  # normalização em blocos 2×2 células
        feature_vector=True
    )
    return {f"hog_{i:03d}": float(v) for i, v in enumerate(hog_vec)}
```

> **Por que redimensionar para 128×128?**  
> Com `pixels_per_cell=(32,32)`, a imagem de 128×128 resulta em 4×4 = 16 células. Blocos 2×2 geram 9 blocos distintos com 4 células cada, resultando em `9 × 4 × 9 = 324` features — compacto o suficiente para ML sem ser tão custoso quanto o HOG completo de 1024×1024.

---

#### `hog_000` a `hog_323`
- **Tipo:** `float` (valores entre 0 e 1 após normalização L2)
- **Descrição:** Cada valor representa a **energia de gradiente em uma direção específica** em uma região específica da imagem. As 9 orientações representam:
  - `bin_0` = gradientes horizontais (0°)
  - `bin_4` = gradientes diagonais (≈ 80°)
  - `bin_8` = gradientes quase verticais (≈ 160°)

- **Relevância ML:** ⭐⭐⭐ Alta (como conjunto), mas **com ressalvas importantes:**

| Ponto Positivo | Ponto Negativo |
|----------------|----------------|
| Captura a orientação das bordas do objeto | 324 features = risco de maldição da dimensionalidade |
| Invariante a pequenas translações | Correlacionado com outras features de borda |
| Muito usado em detecção de objetos clássica (pedestres, veículos) | Custo de cálculo moderado |
| Captura forma do par highlight/shadow | Pode dominar o espaço de features se não normalizado |

> **Recomendação:** Para modelos lineares (Regressão Logística, Ridge), aplique **PCA** nas features HOG antes de treinar. Para Random Forest e XGBoost, podem ser usadas diretamente. Para SVM com kernel RBF, são extremamente eficazes.

---

### Grupo 8 — Canais de Cor

> **Função no código:** `features_color_channels(img_bgr)`  
> **Quantidade de colunas:** 9 (3 canais × 3 estatísticas)  
> **Biblioteca:** `numpy`

Embora imagens SSS sejam fisicamente monocromáticas, as imagens deste dataset são **pseudocoloradas** — o software de processamento do sonar aplica um mapa de cores (LUT) para facilitar a interpretação humana. Isso significa que os canais B, G, R carregam informação sobre a intensidade original, mas com ênfases diferentes.

```python
def features_color_channels(img_bgr: np.ndarray) -> dict:
    for i, ch_name in enumerate(["B", "G", "R"]):
        ch = img_bgr[:, :, i].astype(np.float32)
        feats[f"ch_{ch_name}_mean"] = float(np.mean(ch))
        feats[f"ch_{ch_name}_std"]  = float(np.std(ch))
        feats[f"ch_{ch_name}_p50"]  = float(np.percentile(ch, 50))
```

---

#### `ch_B_mean`, `ch_B_std`, `ch_B_p50`
- **Canal Blue:** Nas pseudocores de sonar, o canal azul tende a representar as **sombras e regiões de baixa intensidade acústica**.

#### `ch_G_mean`, `ch_G_std`, `ch_G_p50`
- **Canal Green:** O canal verde frequentemente captura as **intensidades intermediárias** do fundo.

#### `ch_R_mean`, `ch_R_std`, `ch_R_p50`
- **Canal Red:** O canal vermelho geralmente representa as **altas intensidades** — highlights dos objetos.

> As estatísticas de cada canal são: média (`_mean`), desvio padrão (`_std`) e mediana (`_p50`).

- **Relevância ML:** ⭐⭐ Média. Para o dataset específico desta pesquisa (tons âmbar), a razão entre os canais pode ser informativa. Contudo, como a pseudocorização é determinística a partir do grayscale, existe alta correlação entre canais e entre estas features e as features do Grupo 1. Útil como **verificação cruzada**, mas pode ser descartado em seleção de features.

---

### Grupo 9 — Rótulos e Coordenadas (Labels)

> **Origem:** Arquivo `.txt` YOLO + montagem em `extract_features()`  
> **Quantidade de colunas:** Varia por modo (5–7 colunas principais)

Este grupo contém as informações **derivadas diretamente das anotações humanas** do arquivo `.txt`. São as variáveis que ligam cada linha do CSV ao objeto físico identificado pelo especialista no sonar.

---

#### `ann_class_id` *(modo per_object)*
- **Tipo:** `int`
- **Valores:** `0` (NOMBO), `1` (MILCO), `-1` (imagem negativa sem anotações)
- **Código:**
  ```python
  row["ann_class_id"] = ann["class_id"]
  ```
- **Relevância ML:** Não entra como feature. É a origem do `label`. Mantida para rastreabilidade.

---

#### `ann_class_name` *(modo per_object)*
- **Tipo:** `string`
- **Valores:** `"NOMBO"`, `"MILCO"`, `"negative"`
- **Relevância ML:** Versão textual do `label`. Útil para `groupby`, visualizações e relatórios. Nunca entra em modelos numéricos.

---

#### `ann_x_center_norm`, `ann_y_center_norm` *(modo per_object)*
- **Tipo:** `float` [0, 1]
- **Exemplo:** `xc = 0.7534`, `yc = 0.5732`
- **Descrição:** Posição normalizada do centro do objeto na imagem. `xc = 0.75` significa que o objeto está a 75% da largura da imagem (flanco direito).
- **Relevância ML:** ⭐⭐⭐⭐ Muito alta (em modo `per_object`). A **posição relativa** dentro do swath do sonar tem significado físico:
  - Objetos próximos ao nadir (centro vertical, `yc ≈ 0.5`) são imageados com melhor resolução
  - Objetos no flanco externo (`xc próximo de 0 ou 1`) têm sombra mais pronunciada
  
> **Atenção:** Em modo `per_image`, estas coordenadas não existem. Nesse caso, usam-se as agregações `ann_xc_mean`, `ann_yc_mean`.

---

#### `ann_w_norm`, `ann_h_norm` *(modo per_object)*
- **Tipo:** `float` [0, 1]
- **Exemplo:** `w = 0.0693`, `h = 0.0273`
- **Descrição:** Largura e altura do bounding box, normalizadas pelas dimensões da imagem. Representam o **tamanho aparente** do objeto.
- **Relevância ML:** ⭐⭐⭐⭐ Muito alta.
  - Minas/bombas têm tamanho aparente característico (geralmente 3–12% da imagem)
  - Razão `w/h` (aspect ratio) discrimina objetos alongados (torpedos) de esféricos (minas)
  - Objetos muito pequenos (`w < 0.01`) podem ser ruído

---

#### `ann_xc_mean`, `ann_yc_mean`, `ann_w_mean`, `ann_h_mean`, `ann_area_mean` *(modo per_image)*
- **Tipo:** `float` (ou `NaN` se sem anotações)
- **Descrição:** Médias das coordenadas e dimensões de **todos os objetos** na imagem. Em imagens com apenas um objeto, são iguais às coordenadas do objeto. Em imagens com múltiplos objetos, representam o centroide e tamanho típico.
- **Relevância ML:** ⭐⭐⭐ Alta. `ann_area_mean` captura o tamanho médio dos objetos detectados. É um proxy da distância média dos objetos ao sensor.

---

#### `label`
- **Tipo:** `int`
- **Valores por modo:**
  - `per_object`: `0` (NOMBO) ou `1` (MILCO) — **target de classificação binária**
  - `per_image`: `0` (sem MILCO) ou `1` (tem MILCO) — **target de detecção de ameaça**
- **Código:**
  ```python
  # per_object:
  row["label"] = ann["class_id"]

  # per_image:
  global_feats["label"] = global_feats["has_milco"]
  ```
- **Relevância ML:** 🎯 **É a variável alvo (`y`) de todos os modelos.** Sempre movida para a última coluna do CSV por convenção. **Nunca usar como feature de entrada.**

---

### Grupo 10 — Features da Região do Objeto (ROI)

> **Função no código:** `features_object_region(gray, ann)`  
> **Quantidade de colunas:** 10  
> **Disponível em:** modo `per_object` apenas

Este grupo extrai features diretamente da **região do bounding box** de cada objeto, ao invés de usar a imagem inteira. São as features mais localmente informativas e têm alta relevância para discriminar NOMBO de MILCO.

```python
def features_object_region(gray: np.ndarray, ann: dict) -> dict:
    x1, y1, x2, y2 = ann["x1"], ann["y1"], ann["x2"], ann["y2"]
    roi = gray[y1:y2, x1:x2].astype(np.float32)   # recorte do bounding box

    # Contexto: região expandida em 50% ao redor do objeto
    pad = max(int((x2-x1)*0.5), int((y2-y1)*0.5), 10)
    ctx = gray[cy1:cy2, cx1:cx2].astype(np.float32)

    # Separação highlight/shadow por limiar de Otsu
    _, thresh = cv2.threshold(roi.astype(np.uint8), 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```

---

#### `obj_roi_mean`
- **Exemplo:** `98.26`
- **Descrição:** Intensidade média dentro do bounding box. Um objeto MILCO terá `obj_roi_mean` tipicamente **maior** que o fundo (highlight) e maior que NOMBO (objetos menores e menos reflectivos).
- **Relevância ML:** ⭐⭐⭐⭐ Muito alta. É a feature local mais direta de brilho do objeto.

---

#### `obj_roi_std`
- **Exemplo:** `103.44`
- **Descrição:** Desvio padrão de intensidade dentro do bounding box. Um objeto balístico com par highlight/shadow bem definido terá `obj_roi_std` muito alto (pixels claros E escuros no mesmo bbox). Objetos menores ou menos contrastados terão `obj_roi_std` menor.
- **Relevância ML:** ⭐⭐⭐⭐⭐ Máxima. A coexistência de pixels muito claros e muito escuros no bbox é a assinatura física do par highlight/shadow — o principal indicador acústico de um objeto sólido elevado do fundo.

---

#### `obj_roi_min` e `obj_roi_max`
- **Exemplos:** `min = 0.0`, `max = 254.0`
- **Descrição:** Intensidade mínima (sombra mais escura) e máxima (highlight mais brilhante) dentro do bounding box.
- **Relevância ML:** ⭐⭐⭐ Alta. `obj_roi_max` alto confirma reflexo forte; `obj_roi_min` baixo confirma sombra profunda. A diferença `obj_roi_max - obj_roi_min` é o **contraste interno do objeto**.

---

#### `obj_roi_skewness`
- **Exemplo:** `0.339`
- **Descrição:** Assimetria da distribuição de intensidade dentro do bounding box. Análoga à `skewness` global, mas aplicada apenas à região do objeto.
- **Relevância ML:** ⭐⭐⭐ Alta. MILCO com highlight predominante tende a ter distribuição positivamente assimétrica dentro do bbox.

---

#### `obj_roi_energy`
- **Fórmula:** `(1/N_roi) · Σ I²` (apenas pixels do bbox)
- **Exemplo:** `20355.31`
- **Descrição:** Energia normalizada dentro da região do objeto. Objetos mais brilhantes têm energia maior.
- **Relevância ML:** ⭐⭐⭐ Alta. Captura potência do sinal retornado especificamente pelo objeto, complementando `obj_roi_mean`.

---

#### `obj_highlight_ratio`
- **Fórmula:** `pixels acima do limiar de Otsu / total de pixels no bbox`
- **Exemplo:** `0.429` → 42,9% dos pixels dentro do bbox são "highlight"
- **Descrição:** Proporção de pixels classificados como **highlight** dentro do bounding box, usando **limiar de Otsu** (limiar automático que maximiza a separação bimodal da distribuição).
- **Código:**
  ```python
  _, thresh = cv2.threshold(roi.astype(np.uint8), 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  highlight_ratio = np.sum(thresh > 0) / thresh.size
  ```
- **Relevância ML:** ⭐⭐⭐⭐⭐ Máxima. Esta é possivelmente a feature **mais fisicamente interpretável** de todo o conjunto. No modelo acústico do SSS:
  - Um objeto elevado do fundo gera roughly 30–50% highlight e 50–70% shadow no seu bbox
  - Fundo plano sem objeto tem distribuição unimodal → Otsu divide arbitrariamente → ratio tende a ~50% com baixo contraste
  - Objetos muito refletivos (metálicos) terão `highlight_ratio` alta (>40%) e `obj_roi_std` muito alto

---

#### `obj_local_contrast`
- **Fórmula:** `|mean(ROI) - mean(contexto expandido)|`
- **Exemplo:** `43.15`
- **Descrição:** Diferença entre a intensidade média dentro do bounding box e a intensidade média da região vizinha imediata (bbox expandido em 50%). Mede o quanto o objeto **se destaca do fundo local**.
- **Relevância ML:** ⭐⭐⭐⭐ Muito alta. Objetos que surgem como "pontos quentes" no fundo terão `obj_local_contrast` alto. Objetos ambíguos (NOMBO confundível com fundo) terão contraste local baixo. É um discriminador direto de "objeto saliente vs. textura de fundo".

---

#### `obj_area_pixels`
- **Fórmula:** `(x2 - x1) × (y2 - y1)`
- **Exemplo:** `4200` pixels
- **Descrição:** Área em pixels quadrados do bounding box do objeto. Depende tanto do tamanho físico do objeto quanto da distância ao sensor.
- **Relevância ML:** ⭐⭐⭐ Alta. MILCO (minas) geralmente são maiores que NOMBO em datasets de treino supervisionado. Contudo, a área em pixels varia com a profundidade → **normalizar pela área total da imagem** para comparações robustas:
  ```python
  df["obj_area_norm"] = df["obj_area_pixels"] / (df["img_width"] * df["img_height"])
  ```

---

#### `obj_aspect_ratio`
- **Fórmula:** `(x2 - x1) / max(y2 - y1, 1)`
- **Exemplo:** `0.857`
- **Descrição:** Razão entre largura e altura do bounding box. Valores menores que 1 indicam objeto mais alto (vertical) que largo. No SSS, o eixo horizontal corresponde ao **alcance** (distância ao sensor) e o vertical ao **tempo de varredura** (trajetória do AUV).
- **Relevância ML:** ⭐⭐⭐⭐ Muito alta. Objetos balísticos elongados (torpedos, bombas cilíndricas) têm aspect ratio característico. Minas esféricas tendem a aspect ratio próximo de 1. NOMBO (pedras, entulho) tendem a ter formas irregulares.

---

## 4. Tabela Consolidada de Relevância para ML

| Grupo | Features | Total | Relevância Geral | Melhor Modelo |
|---|---|---|---|---|
| 0 — Metadados | `image_path`, `img_width`, `img_height`, `n_annotations`, `has_milco`, `has_nombo` | 6 | ⚙️ Diagnóstico | Não entra no modelo |
| 1 — Estatísticas | `mean`, `std`, `skewness`, `kurtosis`, `energy`, etc. | 13 | ⭐⭐⭐⭐ Alta | Regressão Logística, SVM |
| 2 — Histograma | `hist_bin_00` a `hist_bin_31` | 32 | ⭐⭐⭐ Alta (com PCA) | SVM RBF, Random Forest |
| 3 — GLCM | `glcm_contrast_*`, `glcm_homogeneity_*`, etc. | 12 | ⭐⭐⭐⭐ Muito alta | Todos os modelos |
| 4 — Gradiente | `sobel_*`, `laplacian_*` | 6 | ⭐⭐⭐⭐ Muito alta | Regressão Logística, SVM |
| 5 — FFT | `fft_*_ratio`, `fft_total_log_energy` | 4 | ⭐⭐⭐ Alta | XGBoost, Random Forest |
| 6 — Grid Espacial | `grid_rRcC_mean/std` | 32 | ⭐⭐⭐⭐ Muito alta | Árvores, XGBoost |
| 7 — HOG | `hog_000` a `hog_323` | 324 | ⭐⭐⭐ Alta (com PCA) | SVM RBF, Redes Neurais |
| 8 — Cor | `ch_B/G/R_mean/std/p50` | 9 | ⭐⭐ Média | Complementar |
| 9 — Labels | `ann_*`, `label` | 5–7 | 🎯 Alvo/Auxiliar | `label` = target |
| 10 — ROI Objeto | `obj_roi_*`, `obj_highlight_ratio`, etc. | 10 | ⭐⭐⭐⭐⭐ Máxima | Todos os modelos |

### Features de Maior Prioridade (Top 10 recomendadas para modelos iniciais)

1. `obj_highlight_ratio` — assinatura acústica de objeto sólido
2. `obj_roi_std` — contraste interno do par highlight/shadow
3. `obj_local_contrast` — o objeto se destaca do fundo?
4. `glcm_contrast_mean` — rugosidade da textura local
5. `std` — contraste global da imagem
6. `skewness` — assimetria da distribuição de brilho
7. `laplacian_var` — nitidez/riqueza de bordas
8. `fft_high_energy_ratio` — componentes de alta frequência
9. `obj_aspect_ratio` — forma geométrica do objeto
10. `ann_w_norm` × `ann_h_norm` (= área normalizada) — tamanho do objeto

---

## 5. Guia de Uso por Modelo de ML

### Regressão Logística
```python
# Features recomendadas: estatísticas + GLCM + gradiente + ROI
# HOG e histograma devem passar por PCA antes

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

features_lr = [
    "mean", "std", "skewness", "kurtosis", "energy",
    "glcm_contrast_mean", "glcm_homogeneity_mean", "glcm_energy_mean",
    "sobel_mean", "laplacian_var",
    "fft_high_energy_ratio", "fft_low_energy_ratio",
    "obj_roi_std", "obj_highlight_ratio", "obj_local_contrast",
    "obj_aspect_ratio", "obj_area_pixels"
]

pipeline = Pipeline([
    ("scaler", StandardScaler()),      # obrigatório para Reg. Logística
    ("clf", LogisticRegression(C=1.0, class_weight="balanced"))
])
```

> **`StandardScaler` é obrigatório:** Features estão em escalas muito diferentes (`energy ≈ 2000` vs `fft_ratio ≈ 0.19`). Sem normalização, o gradiente da regressão é dominado pelas features de maior magnitude.

---

### SVM com Kernel RBF
```python
# Funciona bem com HOG + GLCM + estatísticas após PCA
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# Reduz HOG de 324 para 50 componentes
features_svm = features_lr + [f"hog_{i:03d}" for i in range(324)]

pipeline_svm = Pipeline([
    ("scaler", StandardScaler()),
    ("pca",    PCA(n_components=50, random_state=42)),   # comprime HOG
    ("clf",    SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced"))
])
```

---

### Random Forest / XGBoost
```python
# Pode usar todas as features sem normalização
# Árvores são invariantes à escala
from sklearn.ensemble import RandomForestClassifier

# Todas as colunas numéricas exceto metadados e label
exclude = {"image_path", "ann_class_name", "label", "has_milco", "has_nombo"}
feature_cols = [c for c in df.select_dtypes(include="number").columns
                if c not in exclude]

clf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",   # corrige desbalanceamento NOMBO vs MILCO
    random_state=42,
    n_jobs=-1
)
```

---

## 6. Recomendações de Pré-processamento

### 6.1 Tratamento de NaN
Features do Grupo 10 (ROI) são `NaN` para imagens sem anotações no modo `per_image`. Estratégias:

```python
# Opção A: imputar com mediana (robusto a outliers)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

# Opção B: criar flag de NaN como feature separada
df["obj_highlight_ratio_missing"] = df["obj_highlight_ratio"].isna().astype(int)
df["obj_highlight_ratio"] = df["obj_highlight_ratio"].fillna(0)
```

### 6.2 Seleção de Features
```python
from sklearn.feature_selection import SelectKBest, f_classif

# Seleciona as 30 features mais relevantes via ANOVA F-test
selector = SelectKBest(f_classif, k=30)
X_selected = selector.fit_transform(X, y)

# Ver quais foram selecionadas
selected_names = np.array(feature_cols)[selector.get_support()]
```

### 6.3 Balanceamento de Classes
O dataset NOMBO >> MILCO. Sempre usar:
```python
# No modelo: class_weight="balanced"
# Ou oversample a classe minoritária:
from imblearn.over_sampling import SMOTE
X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
```

### 6.4 Features Derivadas Recomendadas
```python
# Contraste interno do objeto
df["obj_contrast_range"] = df["obj_roi_max"] - df["obj_roi_min"]

# Área normalizada pela imagem
df["obj_area_norm"] = df["obj_area_pixels"] / (df["img_width"] * df["img_height"])

# Razão highlight/background
df["highlight_to_global"] = df["obj_roi_mean"] / (df["mean"] + 1e-9)

# Feature de forma combinada
df["obj_compactness"] = df["obj_area_pixels"] / (df["ann_w_norm"] * df["ann_h_norm"] * 1024**2 + 1e-9)
```

---

*Documentação gerada para o pipeline `sonar_feature_extractor.py` — versão 1.0*  
*Dataset: Santos & Moura (2024) · DOI: `10.6084/m9.figshare.24574879.v2`*