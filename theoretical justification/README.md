Formal Proof of the Yaw Angle Bound for the Bézier Curve

*This document provides the formal proof for the property discussed in the main paper. It demonstrates that the yaw angle of the generated Bézier curve is inherently bounded by the geometry of its control points, which ensures the angular feasibility of our designed action space.*

---

### Lemma 1

For a sequence of complex numbers $\{z_i\}_{i=1}^{n}$, if for any $i$, the condition $\text{Arg}(z_i)\in (-\pi/2, \pi/2)$ holds, where $\text{Arg}(z)\in(-\pi, \pi]$ represents the principal argument of a complex number $z$, then the following inequality is true:
$$
\text{Arg}\left(\sum_{i=1}^{n}z_i\right) \in \left[\min_i\{\text{Arg}(z_i)\}, \max_i\{\text{Arg}(z_i)\}\right].
$$

**Proof:**

For the case of $n=2$, if $z_1=0$ or $z_2=0$, the inequality holds trivially. Otherwise, the following equality holds:
$$
\text{Arg}(z_1+z_2) = \arctan\left(\frac{r_1\sin\varphi_1+r_2\sin\varphi_2}{r_1\cos\varphi_1+r_2\cos\varphi_2}\right), 
$$
where $r_i$ and $\varphi_i$ represent the magnitude and argument of $z_i$, respectively.

The partial derivative of the fraction with respect to $r_i$ is:
$$
\frac{\partial}{\partial r_i}\left(\frac{r_1\sin\varphi_1+r_2\sin\varphi_2}{r_1\cos\varphi_1+r_2\cos\varphi_2}\right) = \frac{r_j\sin(\varphi_i-\varphi_j)}{(r_1\cos\varphi_1+r_2\cos\varphi_2)^2},
$$
where $i,j$ are distinct elements in $\{1,2\}$. Without loss of generality, assume $\varphi_1 > \varphi_2$. Given the condition $\varphi_1, \varphi_2 \in (-\pi/2, \pi/2)$, we have $\varphi_1-\varphi_2 \in (0, \pi)$, which implies $\sin(\varphi_1 - \varphi_2) > 0$. Since $\arctan(\cdot)$ is a strictly increasing function, the signs of the partial derivatives of the argument are:
$$
\frac{\partial}{\partial r_1}\text{Arg}(z_1+z_2) > 0 \quad \text{and} \quad \frac{\partial}{\partial r_2}\text{Arg}(z_1+z_2) < 0.
$$
Consequently, the argument $\text{Arg}(z_1+z_2)$ reaches its maximum and minimum values at the boundaries of the domain of $r_1, r_2$:
$$
\begin{aligned}
\max \text{Arg}(z_1+z_2) &= \lim_{\substack{r_1 \to +\infty \\ r_2 \to 0^+}}\text{Arg}(z_1+z_2) = \varphi_1, \\
\min \text{Arg}(z_1+z_2) &= \lim_{\substack{r_1 \to 0^+ \\ r_2 \to +\infty}}\text{Arg}(z_1+z_2) = \varphi_2.
\end{aligned}
$$
Therefore, for $n=2$, we have shown:
$$
\text{Arg}(z_1+z_2) \in [\min\{\varphi_1,\varphi_2\}, \max\{\varphi_1,\varphi_2\}].
$$
For the general case of $n$ complex numbers, we can apply this result inductively:
$$
\text{Arg}\left(\sum_{i=1}^{n}z_i\right) = \text{Arg}\left(z_n+\sum_{i=1}^{n-1}z_i\right) \in \left[\min\left\{\varphi_n, \text{Arg}\left(\sum_{i=1}^{n-1}z_i\right)\right\}, \max\left\{\varphi_n, \text{Arg}\left(\sum_{i=1}^{n-1}z_i\right)\right\}\right].
$$
By induction, it can be readily shown that the conclusion holds for any $n$.

---

### Theorem 2

For a Bézier curve parameterized by the control points defined in the main paper, the tangent direction at any point on the curve is bounded by the minimum and maximum angles of the vectors connecting adjacent control points.

**Proof:**

According to the definition, the derivative of an $n$-th order Bézier Curve $\mathbf{B}(\tau)$ is:
$$
\mathbf{B}^\prime(\tau) = n \sum_{i=0}^{n-1}(\mathbf{P}_{i+1}-\mathbf{P}_{i}) b_{i,n-1}(\tau),
$$
which means for any $\tau \in [0,1]$, $\mathbf{B}^\prime(\tau)$ is a conic combination of the vectors connecting adjacent control points.

The yaw angle $\varphi(\mathbf{l})$ of a vector $\mathbf{l}$ is defined counter-clockwise from the positive longitudinal axis. We map vectors to the complex domain by aligning the real axis with the positive longitudinal axis and the imaginary axis with the negative radial axis. This specific mapping ensures that a complex number $z$'s argument $\text{Arg}(z)$ directly corresponds to its vector's yaw angle.

We identify the vector $\mathbf{v}_i = \mathbf{P}_{i+1}-\mathbf{P}_{i}$ with a complex number $z_i$. As per the action space design, the longitudinal coordinates of the control points are strictly increasing. Consequently, $\mathbf{v}_i$ lies strictly within the upper half-plane, which implies that $\text{Re}(z_i)>0$. Therefore, $\text{Arg}(z_i) \in (-\pi/2, \pi/2)$.

Consequently, by Lemma 1, the following holds:
$$
\begin{aligned}
\varphi(\mathbf{B}^\prime(\tau)) &= \varphi\left(n\sum_{i=0}^{n-1}(\mathbf{P}_{i+1}-\mathbf{P}_{i}) b_{i,n-1}(\tau)\right) \\
&= \varphi\left(n\sum_{i=0}^{n-1}\mathbf{v}_i b_{i,n-1}(\tau)\right) \\
&= \text{Arg}\left(n\sum_{i=0}^{n-1}z_i b_{i,n-1}(\tau)\right) \\
&\in \left[\min_{i}\{\text{Arg}(n z_i b_{i,n-1}(\tau))\}, \max_{i}\{\text{Arg}(n z_i b_{i,n-1}(\tau))\}\right] \\
&= \left[\min_{i}\{\text{Arg}(z_i)\}, \max_{i}\{\text{Arg}(z_i)\}\right] \\
&= \left[\min_{i}\{\varphi(\mathbf{v}_i)\}, \max_{i}\{\varphi(\mathbf{v}_i)\}\right],
\end{aligned}
$$
which concludes the proof.