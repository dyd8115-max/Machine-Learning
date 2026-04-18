# 기계학습 연습문제 2 — 선형 회귀, 이진 분류, 확률 밀도 함수 추정

인천대학교 컴퓨터공학부 | 2026년 1학기

---

## 문제 1 \[선형 회귀 — MSE 최소화\]

아래 $N = 3$개의 데이터 포인트가 주어졌다.

| $i$ | $x_i$ | $y_i$ |
|-----|--------|--------|
| 1   | 1      | 1      |
| 2   | 2      | 3      |
| 3   | 3      | 2      |

선형 회귀 모델 $\hat{y} = \theta_1 x + \theta_0$과 MSE 손실 함수

$$J(\theta_0, \theta_1) = \frac{1}{N} \sum_{i=1}^{N} \left(\theta_1 x_i + \theta_0 - y_i\right)^2$$

가 주어졌다. 이 손실 함수는 $\theta_0, \theta_1$에 대해 볼록(convex)하므로 편미분이 0이 되는 점이 전역 최솟값이다.

1. 다음 합산값을 직접 계산하라.

$$\sum_{i=1}^{3} x_i, \quad \sum_{i=1}^{3} y_i, \quad \sum_{i=1}^{3} x_i^2, \quad \sum_{i=1}^{3} x_i y_i$$

2. $J$를 최소화하는 필요조건 $\dfrac{\partial J}{\partial \theta_0} = 0$, $\dfrac{\partial J}{\partial \theta_1} = 0$을 각각 전개하면 다음과 같다.

$$\frac{\partial J}{\partial \theta_0} = \frac{2}{N}\sum_{i=1}^{N}\left(\theta_1 x_i + \theta_0 - y_i\right) = 0$$

$$\frac{\partial J}{\partial \theta_1} = \frac{2}{N}\sum_{i=1}^{N} x_i\left(\theta_1 x_i + \theta_0 - y_i\right) = 0$$

합산 기호를 분배하고 (1)의 값을 대입하여, $\theta_0$와 $\theta_1$에 관한 연립방정식 두 개를 세워라.

3. 연립방정식을 풀어 최적 파라미터 $\theta_0^*, \theta_1^*$을 구하라.  
   > 힌트: 한 방정식에서 $\theta_0$을 $\theta_1$의 식으로 표현한 뒤 다른 방정식에 대입하라.

4. 최적 파라미터를 이용하여 $x = 4$일 때의 예측값 $\hat{y}$를 구하라.

<details>
<summary>풀이 보기</summary>

### (1) 합산값 계산

$$\sum x_i = 1+2+3 = 6, \quad \sum y_i = 1+3+2 = 6$$

$$\sum x_i^2 = 1+4+9 = 14, \quad \sum x_i y_i = (1)(1)+(2)(3)+(3)(2) = 13$$

### (2) 연립방정식 세우기

$\dfrac{\partial J}{\partial \theta_0} = 0$ 조건:

$$\theta_1 \cdot 6 + 3\theta_0 = 6 \implies 2\theta_1 + \theta_0 = 2 \quad \cdots (I)$$

$\dfrac{\partial J}{\partial \theta_1} = 0$ 조건:

$$\theta_1 \cdot 14 + \theta_0 \cdot 6 = 13 \implies 14\theta_1 + 6\theta_0 = 13 \quad \cdots (II)$$

### (3) 연립방정식 풀기

$(I)$에서 $\theta_0 = 2 - 2\theta_1$을 $(II)$에 대입:

$$14\theta_1 + 6(2 - 2\theta_1) = 13 \implies 2\theta_1 = 1$$

$$\boxed{\theta_1^* = 0.5, \quad \theta_0^* = 1}$$

최적 모델: $\hat{y} = 0.5x + 1$

### (4) $x = 4$에서 예측

$$\hat{y} = 0.5 \times 4 + 1 = \boxed{3}$$

> 직관적 확인: 데이터 평균 $\bar{x} = 2, \bar{y} = 2$일 때 $\hat{y}(\bar{x}) = 0.5 \times 2 + 1 = 2 = \bar{y}$ ✓  
> 선형 회귀 직선은 항상 데이터의 평균점을 지난다.

</details>

---

## 문제 2 \[이진 분류 — Chain Rule과 경사하강법\]

아래 $N = 4$개의 데이터 포인트가 주어졌다.

| $i$ | $x_i$ | $y_i$ |
|-----|--------|--------|
| 1   | 0      | 0      |
| 2   | 1      | 0      |
| 3   | 2      | 1      |
| 4   | 3      | 1      |

시그모이드 함수 $\sigma(z) = \dfrac{1}{1+e^{-z}}$를 이용한 로지스틱 회귀 모델은 $\hat{y}_i = \sigma(z_i)$, $z_i = \theta_0 + \theta_1 x_i$이다. 단일 샘플에 대한 Binary Cross-Entropy 손실은:

$$L = -\left[y\log\hat{y} + (1-y)\log(1-\hat{y})\right]$$

전체 손실은 $J = \dfrac{1}{N}\displaystyle\sum_{i=1}^{N} L_i$. 초기 파라미터 $\theta_0 = 0, \theta_1 = 0$, 학습률 $\alpha = 1$.

1. 초기 파라미터에서 $\sigma(0) = 0.5$를 이용하여 모든 $i$에 대한 $\hat{y}_i$와 오차 $\hat{y}_i - y_i$를 구하라.

2. 초기 파라미터에서의 전체 손실 $J$를 계산하라. ($\log(0.5) = -\log 2 \approx -0.693$)

3. **Chain Rule을 이용한 그래디언트 유도**

   (a) $z = \theta_0 + \theta_1 x$에서 $\dfrac{\partial z}{\partial \theta_1}$을 구하라.

   (b) 시그모이드 미분 성질 $\sigma'(z) = \sigma(z)(1-\sigma(z))$를 이용하여 $\dfrac{\partial L}{\partial z}$를 계산하고 $\hat{y} - y$ 형태로 정리하라.

   (c) (a), (b) 결과를 Chain Rule에 대입하여 $\dfrac{\partial L}{\partial \theta_1} = (\hat{y}-y) \cdot x$임을 보여라.

   (d) $N$개 샘플에 대해 평균을 취하면 $\dfrac{\partial J}{\partial \theta_1} = \dfrac{1}{N}\displaystyle\sum_{i=1}^{N} x_i(\hat{y}_i - y_i)$가 됨을 설명하라. 같은 방식으로 $\dfrac{\partial J}{\partial \theta_0}$의 공식도 써라.

4. (3)에서 유도한 공식으로 초기 파라미터에서의 $\dfrac{\partial J}{\partial \theta_0}$와 $\dfrac{\partial J}{\partial \theta_1}$을 계산하라.

5. 경사하강법 $\theta \leftarrow \theta - \alpha \cdot \dfrac{\partial J}{\partial \theta}$를 한 번 적용하여 $\theta_0^{(1)}, \theta_1^{(1)}$을 구하고, $x = 2$에서의 예측 확률 $P(y=1 \mid x=2)$를 구하라. ($\sigma(1) \approx 0.731$)

<details>
<summary>풀이 보기</summary>

### (1) 초기 예측값과 오차

$\theta_0 = 0, \theta_1 = 0$이면 $z_i = 0$이므로 모든 $i$에 대해 $\hat{y}_i = \sigma(0) = 0.5$.

| $i$ | $x_i$ | $y_i$ | $\hat{y}_i$ | $\hat{y}_i - y_i$ |
|-----|--------|--------|-------------|-------------------|
| 1   | 0      | 0      | 0.5         | +0.5              |
| 2   | 1      | 0      | 0.5         | +0.5              |
| 3   | 2      | 1      | 0.5         | −0.5              |
| 4   | 3      | 1      | 0.5         | −0.5              |

### (2) 초기 손실 계산

$\hat{y}_i = 0.5$이므로:

$$J = -\frac{1}{4} \cdot 4 \cdot \log(0.5) = -\log(0.5) = \log 2 \approx \boxed{0.693}$$

> 모든 예측이 0.5일 때 손실은 $\log 2$로, 최대 불확실성 상태에 해당한다.

### (3) Chain Rule을 이용한 그래디언트 유도

**(a)**

$$\frac{\partial z}{\partial \theta_1} = x$$

**(b)**

$$\frac{\partial L}{\partial z} = -\left[y(1-\sigma(z)) - (1-y)\sigma(z)\right] = -(y - \sigma(z)) = \sigma(z) - y = \hat{y} - y$$

> 핵심: 시그모이드와 로그의 조합이 상쇄되어 단순한 잔차(residual) $\hat{y}-y$ 형태가 된다.

**(c)**

$$\frac{\partial L}{\partial \theta_1} = \frac{\partial L}{\partial z} \cdot \frac{\partial z}{\partial \theta_1} = (\hat{y} - y) \cdot x \quad \checkmark$$

**(d)**

$$\frac{\partial J}{\partial \theta_1} = \frac{1}{N}\sum_{i=1}^{N} x_i(\hat{y}_i - y_i), \qquad \frac{\partial J}{\partial \theta_0} = \frac{1}{N}\sum_{i=1}^{N} (\hat{y}_i - y_i)$$

### (4) 초기 파라미터에서 그래디언트 계산

$$\frac{\partial J}{\partial \theta_0} = \frac{1}{4}(0.5 + 0.5 - 0.5 - 0.5) = 0$$

$$\frac{\partial J}{\partial \theta_1} = \frac{1}{4}\left[(0)(0.5) + (1)(0.5) + (2)(-0.5) + (3)(-0.5)\right] = \frac{-2}{4} = -\frac{1}{2}$$

### (5) 파라미터 업데이트 및 새로운 예측 ($\alpha = 1$)

$$\theta_0^{(1)} = 0 - 1 \cdot 0 = 0, \qquad \theta_1^{(1)} = 0 - 1 \cdot \left(-\frac{1}{2}\right) = 0.5$$

$$P(y=1 \mid x=2) = \sigma(0 + 0.5 \times 2) = \sigma(1) \approx \boxed{0.731}$$

> 초기 예측(0.5)에서 0.731로 상승. $x$가 클수록 $y=1$임을 단 한 번의 업데이트로 학습하기 시작했다.

</details>

---

## 문제 3 \[가우시안 분류 — 결정 경계\]

두 클래스 $C_1, C_2$의 1차원 학습 데이터:

| 클래스 | 데이터       | $n_k$ |
|--------|------------|-------|
| $C_1$  | $\{1,2,3\}$ | 3     |
| $C_2$  | $\{5,6,7\}$ | 3     |

각 클래스의 데이터가 동일한 분산 $\sigma^2$을 가지는 가우시안 분포를 따른다고 가정:

$$P(x \mid C_k) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x-\mu_k)^2}{2\sigma^2}\right)$$

사전 확률 $P(C_1) = P(C_2) = \dfrac{1}{2}$.

1. MLE 평균 추정량(표본 평균)으로 $\hat{\mu}_1, \hat{\mu}_2$를 구하라.

2. 공통 분산 추정:

$$\hat{\sigma}^2 = \frac{1}{n_1+n_2}\left[\sum_{x \in C_1}(x-\hat{\mu}_1)^2 + \sum_{x \in C_2}(x-\hat{\mu}_2)^2\right]$$

3. 로그 우도비 $\log\dfrac{P(x \mid C_1)}{P(x \mid C_2)}$를 전개하라. 정규화 상수 $-\dfrac{1}{2}\log(2\pi\sigma^2)$ 항이 상쇄됨을 보이고, 남은 식이 $x$의 일차식이 됨을 보여라.  
   > 힌트: $-(x-\mu_1)^2 + (x-\mu_2)^2$을 전개하면 $x^2$ 항이 소거된다.

4. 등 사전 확률 조건에서 결정 경계 방정식은 $\log P(x \mid C_1) = \log P(x \mid C_2)$. (3)의 결과로 결정 경계 $x^*$를 구하라.

5. 테스트 데이터 $x = 3.5$와 $x = 5$를 각각 어느 클래스로 분류하는가?

<details>
<summary>풀이 보기</summary>

### (1) MLE 평균 추정

$$\hat{\mu}_1 = \frac{1+2+3}{3} = 2, \qquad \hat{\mu}_2 = \frac{5+6+7}{3} = 6$$

### (2) 공통 분산 MLE 추정

$$C_1: (1-2)^2+(2-2)^2+(3-2)^2 = 2, \quad C_2: (5-6)^2+(6-6)^2+(7-6)^2 = 2$$

$$\hat{\sigma}^2 = \frac{2+2}{3+3} = \frac{4}{6} = \boxed{\frac{2}{3}}$$

### (3) 로그 우도비 전개

$$\log\frac{P(x \mid C_1)}{P(x \mid C_2)} = \frac{1}{2\sigma^2}\left[-(x-\mu_1)^2 + (x-\mu_2)^2\right]$$

정규화 상수 $-\dfrac{1}{2}\log(2\pi\sigma^2)$가 상쇄됨. 괄호 안을 전개하면:

$$-(x-\mu_1)^2 + (x-\mu_2)^2 = \underbrace{-x^2+x^2}_{=0} + 2(\mu_1-\mu_2)x + (\mu_2^2 - \mu_1^2)$$

$x^2$ 항이 소거되어 $x$의 **일차식**이 된다. ✓

### (4) 결정 경계 $x^*$ 계산

로그 우도비 $= 0$으로 놓으면:

$$(x-\mu_1)^2 = (x-\mu_2)^2 \implies x^* = \frac{\mu_1+\mu_2}{2} = \frac{2+6}{2} = \boxed{4}$$

> 결정 경계는 두 평균의 중점이다.

### (5) 테스트 데이터 분류

| 테스트 $x$ | $x$와 $x^* = 4$ 비교       | 분류 결과 |
|------------|---------------------------|-----------|
| 3.5        | $3.5 < 4 \Rightarrow C_1$ | $C_1$     |
| 5          | $5 > 4 \Rightarrow C_2$   | $C_2$     |

</details>

---

## 문제 4 \[다중 클래스 MLE — 라그랑주 승수법\]

세 클래스 A, B, C로 이루어진 분류 문제, 총 $n = 10$개의 관측값.

| 클래스 | A | B | C |
|--------|---|---|---|
| $n_k$  | 4 | 3 | 3 |

각 클래스 확률 $\pi_A, \pi_B, \pi_C$ (Categorical 분포), 제약 조건: $\pi_A + \pi_B + \pi_C = 1, \; \pi_k \geq 0$.

로그 우도 함수:

$$\ell(\pi_A, \pi_B, \pi_C) = 4\log\pi_A + 3\log\pi_B + 3\log\pi_C$$

1. 제약 조건을 무시하고 $\dfrac{\partial \ell}{\partial \pi_A} = 0$을 풀면 어떤 결과가 나오는가? 왜 확률로서 의미가 없는지 설명하라.

2. 제약 함수 $g(\pi_A, \pi_B, \pi_C) = \pi_A + \pi_B + \pi_C - 1$의 그래디언트 $\nabla g$를 계산하라.

3. $\ell$의 그래디언트 $\nabla \ell$을 계산하라. $\left(\dfrac{d}{d\pi}\log\pi = \dfrac{1}{\pi}\right)$

4. 제약 평면 위의 최댓값에서 $\nabla \ell = \lambda \cdot \nabla g$가 성립한다. 이를 성분별로 써서 $\pi_A, \pi_B, \pi_C$를 $\lambda$의 식으로 표현하라.

5. 라그랑주 함수 $\mathcal{L}(\pi_A, \pi_B, \pi_C, \lambda) = \ell - \lambda(\pi_A+\pi_B+\pi_C-1)$에 대해:

   (a) $\dfrac{\partial \mathcal{L}}{\partial \pi_A} = 0$으로 놓으면 (4)의 $\pi_A$ 성분 조건이 나옴을 확인하라.

   (b) $\dfrac{\partial \mathcal{L}}{\partial \lambda} = 0$으로 놓으면 제약 조건 $\pi_A+\pi_B+\pi_C=1$이 나옴을 확인하라.

6. (4)의 결과를 제약 조건에 대입하여 $\lambda$를 구하고, $\hat{\pi}_A, \hat{\pi}_B, \hat{\pi}_C$를 계산하라.

7. (6)의 결과가 각 클래스의 관측 비율 $n_k/n$과 일치함을 확인하고, 기계학습 관점에서의 의미를 한 문장으로 서술하라.

<details>
<summary>풀이 보기</summary>

### (1) 제약 없이 최대화하면 왜 안 되는가?

$\dfrac{\partial \ell}{\partial \pi_A} = \dfrac{4}{\pi_A} = 0$을 풀려 하면 $\pi_A \to \infty$이어야 한다. 이는 확률로서 말이 되지 않는다. 제약 조건 없이는 로그 우도가 $\pi_A$가 커질수록 계속 증가하므로 유한한 최댓값이 존재하지 않는다.

### (2) 제약 평면의 법선 벡터

$$\nabla g = \left(\frac{\partial g}{\partial \pi_A}, \frac{\partial g}{\partial \pi_B}, \frac{\partial g}{\partial \pi_C}\right) = (1, 1, 1)$$

### (3) 목적함수의 그래디언트

$$\nabla \ell = \left(\frac{4}{\pi_A}, \frac{3}{\pi_B}, \frac{3}{\pi_C}\right)$$

### (4) 평행 조건으로 $\pi_k$ 표현

$\nabla \ell = \lambda \cdot \nabla g$를 성분별로:

$$\frac{4}{\pi_A} = \lambda, \quad \frac{3}{\pi_B} = \lambda, \quad \frac{3}{\pi_C} = \lambda$$

$$\implies \pi_A = \frac{4}{\lambda}, \quad \pi_B = \frac{3}{\lambda}, \quad \pi_C = \frac{3}{\lambda}$$

### (5) 라그랑주 함수 확인

**(a)**

$$\frac{\partial \mathcal{L}}{\partial \pi_A} = \frac{4}{\pi_A} - \lambda = 0 \implies \frac{4}{\pi_A} = \lambda \quad \checkmark$$

**(b)**

$$\frac{\partial \mathcal{L}}{\partial \lambda} = -(\pi_A + \pi_B + \pi_C - 1) = 0 \implies \pi_A + \pi_B + \pi_C = 1 \quad \checkmark$$

### (6) 최적 추정량 계산

$$\frac{4}{\lambda} + \frac{3}{\lambda} + \frac{3}{\lambda} = 1 \implies \frac{10}{\lambda} = 1 \implies \lambda = 10$$

$$\boxed{\hat{\pi}_A = 0.4, \quad \hat{\pi}_B = 0.3, \quad \hat{\pi}_C = 0.3}$$

검증: $0.4 + 0.3 + 0.3 = 1$ ✓

### (7) 결과 해석

| 클래스 | MLE 추정량 $\hat{\pi}_k$ | 관측 비율 $n_k/n$ |
|--------|--------------------------|-------------------|
| A      | 0.4                      | 4/10              |
| B      | 0.3                      | 3/10              |
| C      | 0.3                      | 3/10              |

완전히 일치한다. ✓

> **기계학습적 의미:** Categorical 분포의 MLE 추정량은 각 클래스의 관측 빈도(경험적 확률)이며, 이는 Softmax + Cross-Entropy로 학습하는 딥러닝 분류기가 수렴하는 출력 확률값이기도 하다.

> **$\lambda$의 의미:** 결과적으로 $\lambda = n = 10$. $\pi_k = n_k / \lambda$이고 $\lambda = n$이므로 $\pi_k = n_k/n$. $\lambda$는 세 $\pi_k$의 합이 정확히 1이 되도록 분자들을 정규화하는 역할을 한다.

</details>

---

## 개념 및 수식 정리

### 선형 회귀 (Linear Regression)

| 항목 | 내용 |
|------|------|
| 모델 | $\hat{y} = \theta_1 x + \theta_0$ |
| 손실 함수 (MSE) | $J = \dfrac{1}{N}\displaystyle\sum_{i=1}^{N}(\hat{y}_i - y_i)^2$ |
| 최적 조건 | $\dfrac{\partial J}{\partial \theta_0} = 0$, $\dfrac{\partial J}{\partial \theta_1} = 0$ (볼록 함수이므로 전역 최솟값) |
| 핵심 성질 | 회귀 직선은 반드시 데이터 평균점 $(\bar{x}, \bar{y})$를 통과 |

---

### 로지스틱 회귀 / 이진 분류

| 항목 | 수식 |
|------|------|
| 시그모이드 | $\sigma(z) = \dfrac{1}{1+e^{-z}}$ |
| 미분 성질 | $\sigma'(z) = \sigma(z)(1-\sigma(z))$ |
| BCE 손실 | $L = -[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ |
| 그래디언트 | $\dfrac{\partial L}{\partial \theta_1} = (\hat{y}-y) \cdot x$ |
| 경사하강법 | $\theta \leftarrow \theta - \alpha \cdot \dfrac{\partial J}{\partial \theta}$ |

> **Chain Rule 핵심:** 시그모이드와 로그의 조합이 상쇄 → 그래디언트가 잔차 $\hat{y}-y$ 형태로 단순화

---

### 가우시안 분류 (LDA)

| 항목 | 수식 |
|------|------|
| 가우시안 PDF | $P(x \mid C_k) = \dfrac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\dfrac{(x-\mu_k)^2}{2\sigma^2}\right)$ |
| MLE 평균 | $\hat{\mu}_k = \dfrac{1}{n_k}\displaystyle\sum_{x \in C_k} x$ |
| 공통 분산 추정 | $\hat{\sigma}^2 = \dfrac{1}{n_1+n_2}\left[\displaystyle\sum_{C_1}(x-\hat{\mu}_1)^2 + \sum_{C_2}(x-\hat{\mu}_2)^2\right]$ |
| 결정 경계 | $x^* = \dfrac{\mu_1 + \mu_2}{2}$ (등 사전 확률, 공통 분산 조건) |

> 로그 우도비에서 정규화 상수와 $x^2$ 항이 소거되어 **선형 결정 경계**가 만들어짐

---

### 라그랑주 승수법 (Lagrange Multiplier)

| 항목 | 내용 |
|------|------|
| 목적 | 등호 제약 $g(\mathbf{x}) = 0$ 하에서 $f(\mathbf{x})$ 최적화 |
| 라그랑주 함수 | $\mathcal{L} = f(\mathbf{x}) - \lambda \cdot g(\mathbf{x})$ |
| 최적 조건 | $\nabla f = \lambda \cdot \nabla g$ (목적함수 그래디언트 ∥ 제약 법선) |
| KKT 조건 | $\dfrac{\partial \mathcal{L}}{\partial \mathbf{x}} = 0$, $\dfrac{\partial \mathcal{L}}{\partial \lambda} = 0$ |

> **Categorical MLE 결과:** $\hat{\pi}_k = n_k / n$ (경험적 확률 = Softmax 수렴값)
