# 기계학습 — 기초 통계 및 확률 이론 연습문제

---

## 문제 1 | 기댓값과 분산

확률변수 $X$는 불량품 수를 나타내며, PMF가 아래 표와 같이 주어진다.

| $k$ | 0 | 1 | 2 | 3 |
|-----|------|------|------|------|
| $P(X=k)$ | 0.50 | 0.30 | 0.15 | 0.05 |

1. $E[X]$와 $E[X^2]$을 각각 계산하라.
2. 분산 $\text{Var}(X) = E[X^2] - (E[X])^2$을 이용하여 $\text{Var}(X)$를 구하라.
3. $P(X \geq 2)$를 구하라.
4. $Y = 3X + 2$로 정의할 때, $\text{Var}(aX+b) = a^2\text{Var}(X)$를 이용하여 $E[Y]$와 $\text{Var}(Y)$를 구하라.

### 풀이

**(1) $E[X]$와 $E[X^2]$**

```math
E[X] = 0(0.50) + 1(0.30) + 2(0.15) + 3(0.05) = 0 + 0.30 + 0.30 + 0.15 = 0.75
```

```math
E[X^2] = 0^2(0.50) + 1^2(0.30) + 2^2(0.15) + 3^2(0.05) = 0 + 0.30 + 0.60 + 0.45 = 1.35
```

**(2) $\text{Var}(X)$**

```math
\text{Var}(X) = E[X^2] - (E[X])^2 = 1.35 - (0.75)^2 = 1.35 - 0.5625 = 0.7875
```

**(3) $P(X \geq 2)$**

```math
P(X \geq 2) = P(X=2) + P(X=3) = 0.15 + 0.05 = 0.20
```

**(4) $Y = 3X + 2$**

기댓값의 선형성과 분산의 성질을 적용한다.

```math
E[Y] = 3E[X] + 2 = 3(0.75) + 2 = 4.25
```

```math
\text{Var}(Y) = 3^2\,\text{Var}(X) = 9 \times 0.7875 = 7.0875
```

> $b$를 더하는 이동 변환은 분산에 영향을 주지 않고, 스케일 변환 $a$는 분산에 $a^2$배 영향을 준다.

---

## 문제 2 | Bias–Variance Tradeoff 수식 유도

회귀 문제에서 모델 예측값 $\hat{f}(x)$와 실제값 $y = f(x) + \epsilon$이 주어진다.  
($\epsilon$은 $E[\epsilon]=0$, $\text{Var}(\epsilon)=\sigma^2$이며, $\hat{f}(x)$와 독립인 노이즈)

예측 오차의 기댓값이 아래와 같이 세 항으로 분해됨을 단계별로 수식 유도하라.

```math
E\left[(y - \hat{f}(x))^2\right] = \underbrace{(E[\hat{f}(x)] - f(x))^2}_{\text{Bias}^2} + \underbrace{E\left[(\hat{f}(x) - E[\hat{f}(x)])^2\right]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Noise}}
```

각 항의 기계학습적 의미도 함께 서술하라.

### 풀이

표기 편의를 위해 $\hat{f} = \hat{f}(x)$, $f = f(x)$, $\bar{f} = E[\hat{f}(x)]$로 쓴다.

**Step 1.** $y = f + \epsilon$을 대입하고 전개한다.

```math
E\left[(y-\hat{f})^2\right] = E\left[((f-\hat{f})+\epsilon)^2\right] = E\left[(f-\hat{f})^2\right] + 2E\left[\epsilon(f-\hat{f})\right] + E[\epsilon^2]
```

$\epsilon$은 $\hat{f}$와 독립이고 $E[\epsilon]=0$이므로:

```math
E\left[\epsilon(f-\hat{f})\right] = E[\epsilon]\,E[f-\hat{f}] = 0, \qquad E[\epsilon^2] = \sigma^2
```

따라서:

```math
E\left[(y-\hat{f})^2\right] = E\left[(f-\hat{f})^2\right] + \sigma^2 \quad \cdots (*)
```

**Step 2.** $\pm\bar{f}$를 삽입하여 $E[(f-\hat{f})^2]$를 분해한다.

```math
f - \hat{f} = \underbrace{(f - \bar{f})}_{\text{상수}} + \underbrace{(\bar{f} - \hat{f})}_{\text{확률변수}}
```

전개하면 교차항 $2(f-\bar{f})E[\bar{f}-\hat{f}] = 0$이므로:

```math
E\left[(f-\hat{f})^2\right] = \underbrace{(\bar{f}-f)^2}_{\text{Bias}^2} + \underbrace{E\left[(\hat{f}-\bar{f})^2\right]}_{\text{Variance}}
```

**Step 3.** $(*)$에 대입하면 최종 결과를 얻는다.

```math
E\left[(y-\hat{f})^2\right] = \underbrace{(E[\hat{f}]-f)^2}_{\text{Bias}^2} + \underbrace{E\left[(\hat{f}-E[\hat{f}])^2\right]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Noise}}
```

| 항 | 의미 | 높아지는 상황 |
|----|------|--------------|
| $\text{Bias}^2$ | 평균 예측이 정답에서 벗어난 정도 | 모델이 너무 단순 |
| Variance | 학습 데이터 변화에 민감한 정도 | 모델이 노이즈까지 외움 |
| $\sigma^2$ (Noise) | 데이터 자체의 내재 불확실성 | 제거 불가 |

---

## 문제 3 | 공분산과 상관계수 계산

어떤 강의의 수강생 5명에 대해 주당 학습 시간 $x$와 기말 점수 $y$를 조사한 결과이다.

| 학생 | 1 | 2 | 3 | 4 | 5 |
|------|---|---|---|---|---|
| $x$ (시간) | 2 | 4 | 6 | 8 | 10 |
| $y$ (점수) | 55 | 65 | 70 | 80 | 85 |

1. 표본 평균 $\bar{x}$, $\bar{y}$를 구하라.
2. 표본 공분산 $C(x,y) = \frac{1}{n-1}\sum_{i=1}^n (x_i-\bar{x})(y_i-\bar{y})$를 계산하라.
3. 표본 분산 $V(x)$, $V(y)$를 각각 계산하고, 상관계수 $\rho = \frac{C(x,y)}{\sqrt{V(x)}\sqrt{V(y)}}$를 구하라.
4. 계산한 상관계수의 크기와 부호를 해석하라.

### 풀이

**(1) 표본 평균**

```math
\bar{x} = \frac{2+4+6+8+10}{5} = 6, \qquad \bar{y} = \frac{55+65+70+80+85}{5} = 71
```

**(2) 표본 공분산**

| $i$ | $x_i - \bar{x}$ | $y_i - \bar{y}$ | $(x_i-\bar{x})(y_i-\bar{y})$ |
|-----|----------------|----------------|-------------------------------|
| 1 | $-4$ | $-16$ | $64$ |
| 2 | $-2$ | $-6$ | $12$ |
| 3 | $0$ | $-1$ | $0$ |
| 4 | $2$ | $9$ | $18$ |
| 5 | $4$ | $14$ | $56$ |
| **합계** | | | **150** |

```math
C(x,y) = \frac{150}{5-1} = 37.5
```

**(3) 표본 분산과 상관계수**

```math
\sum(x_i-\bar{x})^2 = 16+4+0+4+16 = 40 \implies V(x) = \frac{40}{4} = 10
```

```math
\sum(y_i-\bar{y})^2 = 256+36+1+81+196 = 570 \implies V(y) = \frac{570}{4} = 142.5
```

```math
\rho = \frac{37.5}{\sqrt{10} \times \sqrt{142.5}} = \frac{37.5}{\sqrt{1425}} \approx \frac{37.5}{37.75} \approx 0.993
```

**(4) 해석**

$\rho \approx 0.993$: 크기가 1에 매우 가까우므로 **강한 선형 관계**가 있다. 부호가 양수이므로 학습 시간이 늘어날수록 점수도 높아지는 **양의 상관관계**이다.

---

## 문제 4 | 정규분포와 표준화

시험 점수 $X \sim N(70, 100)$ (평균 $\mu=70$, 분산 $\sigma^2=100$, $\sigma=10$)

1. 68–95–99.7 규칙을 이용하여 다음을 구하라: (a) $P(60 \leq X \leq 80)$, (b) $P(50 \leq X \leq 90)$
2. $Z = (X-70)/10 \sim N(0,1)$을 이용하여 $P(X > 85)$를 구하라. (참고: $\Phi(1.5) \approx 0.9332$)
3. 오차 분포가 $N(5,4)$인 모델 A와 $N(0,25)$인 모델 B 중 Bias-Variance 관점에서 어느 모델이 더 나은지 서술하라.

### 풀이

**(1) 68–95–99.7 규칙**

```math
P(60 \leq X \leq 80) = P(\mu-\sigma \leq X \leq \mu+\sigma) \approx 68\%
```

```math
P(50 \leq X \leq 90) = P(\mu-2\sigma \leq X \leq \mu+2\sigma) \approx 95\%
```

**(2) $P(X > 85)$**

```math
P(X > 85) = P\!\left(Z > \frac{85-70}{10}\right) = P(Z > 1.5) = 1 - \Phi(1.5) \approx 1 - 0.9332 = 0.0668
```

상위 약 **6.7%**에 해당하는 점수이다.

**(3) Bias–Variance 비교**

| 모델 | $\text{Bias}^2 = \mu_\epsilon^2$ | $\text{Variance} = \sigma_\epsilon^2$ | 총 오차 |
|------|----------------------------------|---------------------------------------|---------|
| A: $N(5,4)$ | 25 | 4 | **29** |
| B: $N(0,25)$ | 0 | 25 | **25** |

총 오차 기준으로는 **모델 B**가 더 낫다. 그러나 모델 A는 Variance가 작아 안정적이며 데이터가 늘어날수록 Bias 보정이 가능하다. 모델 B는 Bias는 없지만 예측 불안정성이 크다. 어느 쪽을 선택할지는 데이터 양과 문제 맥락에 따라 달라진다.

---

## 문제 5 | 상관계수와 독립성의 관계

$X \sim \text{Uniform}(-1,1)$이고 $Y = X^2$으로 정의된다. (PDF: $f_X(x) = 1/2$, $-1 \leq x \leq 1$)

1. $E[X]$, $E[Y]$, $E[XY]$를 각각 계산하라.
2. $\text{Cov}(X,Y)$와 상관계수 $\rho_{XY}$를 구하라.
3. $X$와 $Y$는 독립인가? 수학적 근거와 함께 서술하라.
4. "$\rho = 0 \Rightarrow$ 독립"이라는 명제가 참인지 거짓인지, 이 예시를 근거로 논하라.

### 풀이

**(1) 기댓값 계산**

```math
E[X] = \int_{-1}^{1} x \cdot \frac{1}{2}\,dx = 0 \quad \text{(기함수, 대칭 구간)}
```

```math
E[Y] = E[X^2] = \int_{-1}^{1} x^2 \cdot \frac{1}{2}\,dx = \frac{1}{2}\left[\frac{x^3}{3}\right]_{-1}^{1} = \frac{1}{2} \cdot \frac{2}{3} = \frac{1}{3}
```

```math
E[XY] = E[X^3] = \int_{-1}^{1} x^3 \cdot \frac{1}{2}\,dx = 0 \quad \text{(기함수, 대칭 구간)}
```

**(2) 공분산과 상관계수**

```math
\text{Cov}(X,Y) = E[XY] - E[X]E[Y] = 0 - 0 \cdot \frac{1}{3} = 0 \implies \rho_{XY} = 0
```

**(3) 독립성 판별**

$X$와 $Y = X^2$은 **독립이 아니다.**

$Y$는 $X$의 결정론적 함수이므로 $X$의 값이 주어지면 $Y$도 완전히 결정된다.

```math
P(Y \leq 0.01 \mid X = 0.5) = 0 \neq P(Y \leq 0.01)
```

독립의 정의 $P(A \cap B) = P(A)P(B)$에 위배된다.

**(4) 시사점: 상관계수의 한계**

$$\text{독립} \Rightarrow \rho = 0 \quad \text{이지만} \quad \rho = 0 \not\Rightarrow \text{독립}$$

상관계수는 **선형 관계만** 측정한다. $Y = X^2$과 같은 비선형 종속성은 상관계수에서 사라진다. 따라서 공분산·상관계수가 0이어도 두 변수 사이에 강한 비선형 관계가 존재할 수 있다.

---

## 문제 6 | 중심극한정리 응용

공장 제품의 불량률 $p = 0.2$. 제품 $n = 100$개를 임의로 추출할 때 불량품 수를 $S_n$이라 하자.

1. $S_n$의 정확한 분포를 밝히고, $E[S_n]$과 $\text{Var}(S_n)$을 구하라.
2. CLT를 이용하여 $P(S_n > 25)$를 근사하라. (참고: $\Phi(1.25) \approx 0.8944$)
3. $n$이 커질수록 표본 비율 $\hat{p} = S_n/n$의 추정 정밀도가 높아지는 이유를 $\text{Var}(\hat{p})$의 식을 통해 설명하라.

### 풀이

**(1) 분포, 기댓값, 분산**

각 제품은 독립적으로 불량일 확률이 $p=0.2$이므로:

```math
S_n \sim \text{Binomial}(n=100,\ p=0.2)
```

```math
E[S_n] = np = 100 \times 0.2 = 20, \qquad \text{Var}(S_n) = np(1-p) = 100 \times 0.2 \times 0.8 = 16
```

**(2) CLT를 이용한 근사**

$n=100$으로 충분히 크므로 CLT에 의해:

```math
Z = \frac{S_n - 20}{\sqrt{16}} = \frac{S_n - 20}{4} \approx N(0,1)
```

```math
P(S_n > 25) = P\!\left(Z > \frac{25-20}{4}\right) = P(Z > 1.25) = 1 - \Phi(1.25) \approx 1 - 0.8944 = 0.1056
```

**(3) $\text{Var}(\hat{p})$과 추정 정밀도**

```math
\text{Var}(\hat{p}) = \text{Var}\!\left(\frac{S_n}{n}\right) = \frac{1}{n^2}\text{Var}(S_n) = \frac{p(1-p)}{n}
```

$n$이 커질수록 $\text{Var}(\hat{p}) \rightarrow 0$이 되어 $\hat{p}$가 참값 $p$ 주변에 점점 집중된다. 표준오차가 $\sigma/\sqrt{n}$ 속도로 감소하므로 표본이 많을수록 신뢰 구간의 폭이 좁아지고 추정 정밀도가 향상된다.

---

## 문제 7 | 주변 확률과 조건부 확률

공정한 동전을 독립적으로 던져서 **처음 뒷면이 나올 때까지 던진 횟수를 $X$** 라 하고, 이후 공정한 동전을 다시 $X$번 던졌을 때 **$X$번 모두 앞면이 나오는 사건을 $A$** 라 하자.

1. $X$의 PMF를 구하고, 올바른 이산 확률분포임을 확인하라.
2. $E[X]$를 계산하라.
3. 다음 중 옳은 것을 고르고 근거를 서술하라: (a) $A$와 $X$는 독립이다. (b) $A$와 $X$는 독립이 아니다.
4. 주변 확률 $P(A) = \sum_{k=1}^{\infty} P(A, X=k)$와 조건부 확률을 이용하여 $P(A)$를 구하라.

### 풀이

**(1) PMF 유도 및 확인**

처음 뒷면이 $k$번째에 나오려면, 앞의 $k-1$번은 모두 앞면이고 $k$번째가 뒷면이어야 한다.

```math
P(X=k) = \left(\frac{1}{2}\right)^k, \quad k = 1, 2, 3, \ldots
```

전체 합이 1임을 등비급수로 확인:

```math
\sum_{k=1}^{\infty} \left(\frac{1}{2}\right)^k = \frac{1/2}{1-1/2} = 1 \quad \checkmark
```

**(2) $E[X]$ 계산 — 멱급수의 대수적 조작**

$S = E[X] = \sum_{k=1}^{\infty} k(1/2)^k$로 놓으면:

```math
S = 1 \cdot \frac{1}{2} + 2 \cdot \frac{1}{4} + 3 \cdot \frac{1}{8} + \cdots
```

```math
\frac{S}{2} = 1 \cdot \frac{1}{4} + 2 \cdot \frac{1}{8} + 3 \cdot \frac{1}{16} + \cdots
```

빼면:

```math
\frac{S}{2} = \frac{1}{2} + \frac{1}{4} + \frac{1}{8} + \cdots = \sum_{k=1}^{\infty}\left(\frac{1}{2}\right)^k = 1
```

```math
\therefore E[X] = S = 2
```

동전을 처음 뒷면이 나올 때까지 던지면 **평균 2번** 던지게 된다.

**(3) $A$와 $X$의 독립성**

**(b) $A$와 $X$는 독립이 아니다.**

$P(A \mid X=k) = (1/2)^k$이므로 $k$가 커질수록 $P(A \mid X=k) \rightarrow 0$으로 변한다. 즉 $X$의 값에 따라 $A$의 확률이 달라지므로 독립의 정의 $P(A \mid X=k) = P(A)$에 위배된다.

> $(1/2)^k$의 근거는 $A$와 $X$의 독립성이 아니라, 매 동전 던지기들 사이의 독립성에서 비롯된다.

**(4) $P(A)$ 계산**

```math
P(A) = \sum_{k=1}^{\infty} P(A \mid X=k) \cdot P(X=k) = \sum_{k=1}^{\infty} \left(\frac{1}{2}\right)^k \cdot \left(\frac{1}{2}\right)^k = \sum_{k=1}^{\infty} \left(\frac{1}{4}\right)^k = \frac{1/4}{1-1/4} = \frac{1}{3}
```

> $E[X]=2$이므로 평균 2번 던지지만, 그 2번이 모두 앞면일 확률은 $1/3$에 불과하다. $X$가 클수록 모두 앞면일 확률이 기하급수적으로 줄어들기 때문이다.

---

## 핵심 개념 정리

### 기댓값 (Expected Value)

```math
E[X] = \sum_x x \cdot P(X=x) \quad \text{[이산]}, \qquad E[X] = \int x \cdot f(x)\,dx \quad \text{[연속]}
```

- **선형성**: $E[aX+b] = aE[X]+b$
- $E[X^2]$는 $X^2$의 기댓값으로, 분산 계산에 활용

### 분산 (Variance)

```math
\text{Var}(X) = E[X^2] - (E[X])^2
```

- **스케일 변환**: $\text{Var}(aX+b) = a^2\text{Var}(X)$ — 이동($b$)은 분산에 무영향
- **표준편차**: $\text{SD}(X) = \sqrt{\text{Var}(X)}$

### 공분산 & 상관계수

```math
\text{Cov}(X,Y) = E[XY] - E[X]E[Y]
```

```math
\rho_{XY} = \frac{\text{Cov}(X,Y)}{\sqrt{\text{Var}(X)}\sqrt{\text{Var}(Y)}}, \quad |\rho| \leq 1
```

- $\rho > 0$: 양의 상관, $\rho < 0$: 음의 상관, $\rho = 0$: 무상관 (≠ 독립)
- **중요**: 독립 $\Rightarrow$ $\rho=0$ 이지만, $\rho=0$ $\not\Rightarrow$ 독립 (비선형 관계 존재 가능)

### 정규분포 & 표준화

```math
X \sim N(\mu, \sigma^2) \implies Z = \frac{X-\mu}{\sigma} \sim N(0,1)
```

- **68–95–99.7 규칙**: $P(\mu\pm\sigma)\approx68\%$, $P(\mu\pm2\sigma)\approx95\%$, $P(\mu\pm3\sigma)\approx99.7\%$
- $P(X > a) = 1 - \Phi\!\left(\frac{a-\mu}{\sigma}\right)$

### Bias–Variance Tradeoff

```math
E\left[(y-\hat{f})^2\right] = \text{Bias}^2 + \text{Variance} + \sigma^2
```

| 항 | 정의 | 줄이려면 |
|----|------|---------|
| $\text{Bias}^2$ | $(E[\hat{f}]-f)^2$ | 모델 복잡도 ↑ |
| Variance | $E[(\hat{f}-E[\hat{f}])^2]$ | 모델 복잡도 ↓, 데이터 ↑ |
| $\sigma^2$ (Noise) | 데이터 내재 불확실성 | 제거 불가 |

### 이항분포 (Binomial)

```math
S_n \sim \text{Bin}(n,p) \implies E[S_n]=np,\quad \text{Var}(S_n)=np(1-p)
```

### 중심극한정리 (CLT)

$n$이 충분히 클 때 ($n \geq 30$):

```math
\frac{S_n - np}{\sqrt{np(1-p)}} \approx N(0,1)
```

표본 비율의 분산: $\text{Var}(\hat{p}) = \frac{p(1-p)}{n}$ → $n$ 증가 시 $\text{Var}(\hat{p}) \rightarrow 0$

### 기하분포 & 전체 확률의 법칙

**기하분포**: 처음 성공까지의 시행 횟수 $X$

```math
P(X=k) = (1-p)^{k-1}p, \quad E[X] = \frac{1}{p}
```

**전체 확률의 법칙**:

```math
P(A) = \sum_k P(A \mid X=k) \cdot P(X=k)
```

### 조건부 확률 & 독립성

```math
P(A \mid B) = \frac{P(A \cap B)}{P(B)}
```

- **독립**: $P(A \cap B) = P(A)P(B)$ $\iff$ $P(A \mid B) = P(A)$
- 등비급수: $\sum_{k=1}^{\infty} r^k = \frac{r}{1-r}$ ($|r|<1$)
