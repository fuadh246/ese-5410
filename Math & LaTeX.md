# 📐 Math & LaTeX Cheatsheet

> Quick reference for writing math in Jupyter Markdown cells.

---

## 1. Basic Math

### Inline Math

Put `$...$` around the equation:

```markdown
The mean is $\mu = \frac{1}{n}\sum_{i=1}^n x_i$.
```

Renders as:

The mean is $\mu = \frac{1}{n}\sum_{i=1}^n x_i$.

### Display Math

Use `$$...$$`:

```markdown
$$
\mu = \frac{1}{n}\sum_{i=1}^n x_i
$$
```

---

# 2. Basic Operators

| Math | LaTeX |
|---|---|
| $+$ | `+` |
| $-$ | `-` |
| $\times$ | `\times` |
| $\div$ | `\div` |
| $=$ | `=` |
| $\neq$ | `\neq` |
| $<$ | `<` |
| $>$ | `>` |
| $\leq$ | `\leq` |
| $\geq$ | `\geq` |
| $\approx$ | `\approx` |
| $\propto$ | `\propto$ |
| $\pm$ | `\pm` |

---

# 3. Fractions & Powers

### Fraction

```latex
$\frac{a}{b}$
```

$$
\frac{a}{b}
$$

### Powers

```latex
$x^2$
```

$$
x^2
$$

Multiple characters need `{}`:

```latex
$x^{10}$
```

$$
x^{10}
$$

### Subscripts

```latex
$x_i$
```

$$
x_i
$$

Multiple characters:

```latex
$x_{ij}$
```

$$
x_{ij}
$$

### Both

```latex
$x_i^2$
```

$$
x_i^2
$$

---

# 4. Roots

### Square Root

```latex
$\sqrt{x}$
```

$$
\sqrt{x}
$$

### nth Root

```latex
$\sqrt[n]{x}$
```

$$
\sqrt[n]{x}
$$

---

# 5. Greek Letters

## Lowercase

| Symbol | LaTeX | Symbol | LaTeX |
|---|---|---|---|
| $\alpha$ | `\alpha` | $\beta$ | `\beta` |
| $\gamma$ | `\gamma` | $\delta$ | `\delta` |
| $\epsilon$ | `\epsilon` | $\theta$ | `\theta` |
| $\lambda$ | `\lambda` | $\mu$ | `\mu` |
| $\pi$ | `\pi` | $\rho$ | `\rho` |
| $\sigma$ | `\sigma` | $\tau$ | `\tau` |
| $\phi$ | `\phi` | $\omega$ | `\omega` |

## Uppercase

| Symbol | LaTeX | Symbol | LaTeX |
|---|---|---|---|
| $\Gamma$ | `\Gamma` | $\Delta$ | `\Delta` |
| $\Theta$ | `\Theta` | $\Lambda$ | `\Lambda` |
| $\Sigma$ | `\Sigma` | $\Phi$ | `\Phi` |
| $\Omega$ | `\Omega` | $\Pi$ | `\Pi` |

---

# 6. Summation & Products

### Summation

```latex
$\sum_{i=1}^{n} x_i$
```

$$
\sum_{i=1}^{n} x_i
$$

### Product

```latex
$\prod_{i=1}^{n} x_i$
```

$$
\prod_{i=1}^{n} x_i
$$

---

# 7. Calculus

### Derivative

```latex
$\frac{dy}{dx}$
```

$$
\frac{dy}{dx}
$$

### Partial Derivative

```latex
$\frac{\partial f}{\partial x}$
```

$$
\frac{\partial f}{\partial x}
$$

### Second Derivative

```latex
$\frac{d^2y}{dx^2}$
```

$$
\frac{d^2y}{dx^2}
$$

### Integral

```latex
$\int f(x)\,dx$
```

$$
\int f(x)\,dx
$$

### Definite Integral

```latex
$\int_a^b f(x)\,dx$
```

$$
\int_a^b f(x)\,dx
$$

### Limit

```latex
$\lim_{x\to\infty} f(x)$
```

$$
\lim_{x\to\infty} f(x)
$$

---

# 8. Probability

### Probability

```latex
$P(X=x)$
```

$$
P(X=x)
$$

### Conditional Probability

```latex
$P(A\mid B)$
```

$$
P(A\mid B)
$$

### Bayes' Theorem

```latex
$$
P(A\mid B)
=
\frac{P(B\mid A)P(A)}{P(B)}
$$
```

$$
P(A\mid B)
=
\frac{P(B\mid A)P(A)}{P(B)}
$$

### Probability of Complement

```latex
$P(A^c)$
```

$$
P(A^c)
$$

### Union

```latex
$P(A\cup B)$
```

$$
P(A\cup B)
$$

### Intersection

```latex
$P(A\cap B)$
```

$$
P(A\cap B)
$$

---

# 9. Random Variables & Statistics

### Expected Value

```latex
$\mathbb{E}[X]$
```

$$
\mathbb{E}[X]
$$

### Expected Value of a Function

```latex
$$
\mathbb{E}[g(X)]
=
\sum_x g(x)P(X=x)
$$
```

### Variance

```latex
$$
\operatorname{Var}(X)
=
\mathbb{E}[(X-\mu)^2]
$$
```

### Alternative Variance Formula

```latex
$$
\operatorname{Var}(X)
=
\mathbb{E}[X^2]-\mathbb{E}[X]^2
$$
```

### Standard Deviation

```latex
$\sigma = \sqrt{\operatorname{Var}(X)}$
```

$$
\sigma = \sqrt{\operatorname{Var}(X)}
$$

### Covariance

```latex
$$
\operatorname{Cov}(X,Y)
=
\mathbb{E}[(X-\mu_X)(Y-\mu_Y)]
$$
```

### Correlation

```latex
$$
\rho_{X,Y}
=
\frac{\operatorname{Cov}(X,Y)}
{\sigma_X\sigma_Y}
$$
```

---

# 10. Common Distributions

### Binomial PMF

```latex
$$
P(X=k)
=
\binom{n}{k}
p^k(1-p)^{n-k}
$$
```

### Binomial Mean

```latex
$$
\mathbb{E}[X]=np
$$
```

### Binomial Variance

```latex
$$
\operatorname{Var}(X)=np(1-p)
$$
```

### Normal Distribution

```latex
$$
X\sim N(\mu,\sigma^2)
$$
```

### Normal PDF

```latex
$$
f(x)
=
\frac{1}{\sigma\sqrt{2\pi}}
e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$
```

### Uniform Distribution

```latex
$$
X\sim U(a,b)
$$
```

### Uniform Mean

```latex
$$
\mathbb{E}[X]=\frac{a+b}{2}
$$
```

### Uniform Variance

```latex
$$
\operatorname{Var}(X)
=
\frac{(b-a)^2}{12}
$$
```

---

# 11. Sets

| Meaning | LaTeX |
|---|---|
| $\in$ | `\in` |
| $\notin$ | `\notin` |
| $\subset$ | `\subset` |
| $\subseteq$ | `\subseteq` |
| $\cup$ | `\cup` |
| $\cap$ | `\cap` |
| $\emptyset$ | `\emptyset` |
| $\mathbb{R}$ | `\mathbb{R}` |
| $\mathbb{N}$ | `\mathbb{N}` |
| $\mathbb{Z}$ | `\mathbb{Z}` |

Example:

```latex
$x\in\mathbb{R}$
```

$$
x\in\mathbb{R}
$$

---

# 12. Linear Algebra

### Vector

```latex
$\mathbf{x}$
```

$$
\mathbf{x}
$$

### Vector

```latex
$$
\mathbf{x}
=
\begin{bmatrix}
x_1\\
x_2\\
x_3
\end{bmatrix}
$$
```

### Matrix

```latex
$$
A=
\begin{bmatrix}
1 & 2\\
3 & 4
\end{bmatrix}
$$
```

### Matrix Multiplication

```latex
$$
\mathbf{y}=A\mathbf{x}
$$
```

### Transpose

```latex
$A^T$
```

$$
A^T
$$

### Inverse

```latex
$A^{-1}$
```

$$
A^{-1}
$$

### Dot Product

```latex
$\mathbf{x}^\top\mathbf{y}$
```

$$
\mathbf{x}^\top\mathbf{y}
$$

### Norm

```latex
$\|\mathbf{x}\|$
```

$$
\|\mathbf{x}\|
$$

---

# 13. Machine Learning

### Linear Regression

```latex
$$
\hat{y}
=
\mathbf{w}^\top\mathbf{x}+b
$$
```

### Mean Squared Error

```latex
$$
MSE
=
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
$$
```

### Gradient Descent

```latex
$$
\theta
\leftarrow
\theta
-
\alpha\nabla_\theta J(\theta)
$$
```

### Sigmoid

```latex
$$
\sigma(x)
=
\frac{1}{1+e^{-x}}
$$
```

### Softmax

```latex
$$
P(y=k\mid\mathbf{x})
=
\frac{e^{z_k}}
{\sum_j e^{z_j}}
$$
```

---

# 14. Useful Text Formatting

### Bold Math

```latex
$\mathbf{x}$
```

$$
\mathbf{x}
$$

### Roman / Normal Text

```latex
$\text{if } x>0$
```

$$
\text{if } x>0
$$

### Function Names

Use `\operatorname{}`:

```latex
$\operatorname{Var}(X)$
```

$$
\operatorname{Var}(X)
$$

---

# 15. Brackets

```latex
$(x)$
```

```latex
$[x]$
```

```latex
$\{x\}$
```

For automatically sized brackets:

```latex
$$
\left(
\frac{x}{y}
\right)
$$
```

$$
\left(
\frac{x}{y}
\right)
$$

---

# 16. Cases / Piecewise Functions

```latex
$$
f(x)=
\begin{cases}
x^2, & x\geq0\\
-x, & x<0
\end{cases}
$$
```

$$
f(x)=
\begin{cases}
x^2, & x\geq0\\
-x, & x<0
\end{cases}
$$

---

# 17. Multiple Equations

Use `aligned`:

```latex
$$
\begin{aligned}
y &= mx+b\\
m &= \frac{y_2-y_1}{x_2-x_1}
\end{aligned}
$$
```

$$
\begin{aligned}
y &= mx+b\\
m &= \frac{y_2-y_1}{x_2-x_1}
\end{aligned}
$$

---

# ⭐ Most Important Ones to Memorize

You **do not need to memorize everything above**.

Start with these:

```text
$...$                  Inline math
$$...$$                Display math

\frac{a}{b}            Fraction
x^2                    Power
x_i                    Subscript
\sqrt{x}               Square root

\sum_{i=1}^{n}         Sum
\prod_{i=1}^{n}        Product
\int_a^b               Integral
\lim_{x\to\infty}      Limit

\mu                    Mu
\sigma                 Sigma
\alpha                 Alpha
\beta                  Beta
\lambda                Lambda

\mathbb{E}[X]          Expected value
\operatorname{Var}(X)  Variance
\operatorname{Cov}(X,Y) Covariance

P(A\mid B)             Conditional probability
P(A\cup B)             Union
P(A\cap B)             Intersection

\mathbf{x}             Vector
A^T                    Transpose
A^{-1}                 Inverse
\begin{bmatrix}...\end{bmatrix} Matrix
```

## 💡 Tip

Don't try to memorize LaTeX like a programming language.

**Write your math normally → look up the LaTeX command → use it repeatedly.**

After a few weeks of writing probability/statistics notes, you'll naturally remember things like `\frac`, `\sum`, `\mu`, `\sigma`, `\mathbb{E}`, and `\operatorname{Var}`.