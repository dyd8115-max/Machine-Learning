# 기계학습 수식 요약본

---

## 1장. 선형대수학

### 벡터 기초

| 연산 | 수식 |
|---|---|
| 크기 (Norm) | $\|v\| = \sqrt{x_1^2 + x_2^2 + \cdots + x_n^2}$ |
| 단위벡터 | $u = v / \|v\|$ |
| 내적 | $A \cdot B = \sum A_i B_i = \|A\|\|B\|\cos\theta$ |
| 외적 크기 | $\|A \times B\| = \|A\|\|B\|\sin\theta$ |

### 외적 계산

```math
A \times B = \begin{vmatrix} i & j & k \\ A_1 & A_2 & A_3 \\ B_1 & B_2 & B_3 \end{vmatrix} = i(A_2B_3-A_3B_2) - j(A_1B_3-A_3B_1) + k(A_1B_2-A_2B_1)
```

### 내적값 해석

| 값 | 의미 |
|---|---|
| 양수 | 같은 방향 |
| 0 | 직교 (90°) |
| 음수 | 반대 방향 |

### 기저 조건

- 선형 독립 + 벡터 수 = 공간의 차원

### 투영

```math
\text{proj}(x) = (x \cdot u) \cdot u, \quad r = x - \text{proj}(x)
```

### 행렬 곱셈

```math
C_{ij} = \sum_l A_{il} B_{lj}
```

### Rank

- 행 축소 후 0이 아닌 행의 수
- $\text{rank}(A) = n$ → 역행렬 존재 / $\text{rank}(A) < n$ → 역행렬 없음

### 역행렬 (2×2)

```math
A^{-1} = \frac{1}{\det(A)} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}, \quad \det(A) = ad - bc
```

### 선형 변환

```math
y = Wx \quad (W: M \times N)
```

- $M < N$: 차원 축소 / $M > N$: 차원 확대

---

## 2장. 확률

### 확률 공간

| 기호 | 이름 | 설명 |
|---|---|---|
| $\omega$ | 근원 사건 | 실험 결과 하나 |
| $\Omega$ | 표본 공간 | 모든 $\omega$의 집합 |
| $A$ | 사건 | $\Omega$의 부분집합 |
| $F$ | σ-대수 | 확률 계산 가능한 사건들의 모음 |

### Kolmogorov 공리

| 공리 | 수식 |
|---|---|
| 비음수성 | $P(A) \geq 0$ |
| 규범성 | $P(\Omega) = 1$ |
| 가산 가법성 | $A_i \cap A_j = \emptyset$ 이면 $P(A_1 \cup A_2 \cup \cdots) = \sum P(A_i)$ |

유도 성질: $P(A^c) = 1 - P(A)$, $P(A \cup B) = P(A) + P(B) - P(A \cap B)$

### 확률변수

```math
X : \Omega \to \mathbb{R} \quad (\omega \text{를 숫자로 바꾸는 함수})
```

### CDF / PMF / PDF

| | 수식 | 대상 |
|---|---|---|
| CDF | $F_X(x) = P(X \leq x)$ | 이산 / 연속 |
| PMF | $p(x) = P(X = x)$, $\sum p(x) = 1$ | 이산 |
| PDF | $P(a \leq X \leq b) = \int_a^b f(x)dx$, $\int f(x)dx = 1$ | 연속 |

### 주변 확률

```math
p(x) = \sum_y p(x,y), \quad f(x) = \int f(x,y)dy
```

### 조건부 확률 / 연쇄 법칙

```math
P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(A \cap B) = P(A|B) \cdot P(B)
```

### 독립 / 조건부 독립

```math
P(A \cap B) = P(A) \cdot P(B) \quad \text{(독립)}
```
```math
X \perp Y | Z \Leftrightarrow P(X,Y|Z) = P(X|Z) \cdot P(Y|Z) \quad \text{(조건부 독립)}
```

### 베이즈 정리

```math
P(\theta|X) = \frac{P(X|\theta) \cdot P(\theta)}{P(X)}
```

### 기댓값 / 분산

```math
E[X] = \sum_x x \cdot p(x) \quad \text{(이산)}, \quad E[X] = \int x \cdot f(x)dx \quad \text{(연속)}
```
```math
Var[X] = E[X^2] - (E[X])^2
```

기댓값 성질: $E[aX+bY+c] = aE[X]+bE[Y]+c$

분산 성질: $Var[aX+b] = a^2 \cdot Var[X]$, $X \perp Y$ 이면 $Var[X+Y] = Var[X]+Var[Y]$

---

## 3장. 기초 통계

### 통계 파라미터

```math
\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i, \quad s^2 = \frac{1}{n-1}\sum_{i=1}^n(x_i-\bar{x})^2, \quad s = \sqrt{s^2}
```

### 공분산 / 상관계수

```math
C(x,y) = \frac{1}{n-1}\sum_{i=1}^n(x_i-\bar{x})(y_i-\bar{y})
```
```math
\rho = \frac{C(x,y)}{\sigma_x \sigma_y} \quad (0 \leq |\rho| \leq 1)
```

### 왜도 / 첨도

```math
Skewness = \frac{\sum(x_i-\bar{x})^3}{\left[\sum(x_i-\bar{x})^2\right]^{3/2}}, \quad Kurtosis = \frac{\sum(x_i-\bar{x})^4}{\left[\sum(x_i-\bar{x})^2\right]^2}
```

- 왜도: 0이면 대칭, 양수면 오른쪽 꼬리, 음수면 왼쪽 꼬리
- 첨도: 3이면 정규분포, 3 초과면 뾰족, 3 미만이면 평편

### Bias-Variance Tradeoff

```math
E\left[(y-\hat{f}(x))^2\right] = \text{Bias}^2 + \text{Variance} + \text{Noise}
```

- Bias ↑ ↔ Variance ↓ (모델 복잡도에 따라 반비례)

### 회귀 분석

```math
\hat{y} = \theta_0 + \theta_1 x, \quad J(\theta) = \frac{1}{N}\sum_i(\hat{y}_i - y_i)^2 \quad \text{(MSE)}
```
```math
\theta = (X^\top X)^{-1} X^\top y \quad \text{(정규방정식)}
```

### 정규분포 / 중심극한정리

```math
N(x;\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
```
```math
\bar{X}_n = \frac{1}{n}\sum X_i \xrightarrow{n \to \infty} N\left(\mu, \frac{\sigma^2}{n}\right) \quad \text{(CLT)}
```

### MLE

```math
L(\theta;X) = \prod_i p(x_i|\theta) \xrightarrow{\log} \ell(\theta) = \sum_i \log p(x_i|\theta)
```
```math
\hat{\theta}_{MLE} = \underset{\theta}{\text{argmax}}\ \ell(\theta), \quad \nabla_\theta \ell(\theta) = 0
```

Gaussian MLE: $\bar{\mu} = \frac{1}{n}\sum x_i$, $\bar{\sigma}^2 = \frac{1}{n}\sum(x_i-\bar{\mu})^2$

### MAP

```math
\hat{\theta}_{MAP} = \underset{\theta}{\text{argmax}}\ [\log P(X|\theta) + \log P(\theta)]
```

| Prior | 정규화 |
|---|---|
| Gaussian $N(0,1/\lambda)$ | L2 (Ridge): $-\lambda\|\|\theta\|\|^2$ |
| Laplace $L(0,1/\lambda)$ | L1 (Lasso): $-\lambda\|\|\theta\|\|_1$ |

### 주요 분포 요약

| 분포 | PMF/PDF | $E[X]$ | $Var[X]$ |
|---|---|---|---|
| 베르누이 $Ber(p)$ | $p^x(1-p)^{1-x}$ | $p$ | $p(1-p)$ |
| 이항 $Bin(n,p)$ | $C(n,k)p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ |
| 포아송 $Poi(\lambda)$ | $\lambda^k e^{-\lambda}/k!$ | $\lambda$ | $\lambda$ |
| 정규 $N(\mu,\sigma^2)$ | $\frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x-\mu)^2/2\sigma^2}$ | $\mu$ | $\sigma^2$ |

### MLE vs MAP vs 베이지안

| | MLE | MAP | 베이지안 |
|---|---|---|---|
| 목적 | $\text{argmax}\ P(X|\theta)$ | $\text{argmax}\ P(X|\theta)P(\theta)$ | 전체 $P(\theta|X)$ 활용 |
| Prior | 무시 | 반영 | 반영 |
| 비용 | 낮음 | 낮음 | 높음 |
