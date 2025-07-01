---
layout: section
---

# A brief introduction to Artificial Neural Networks

---
level: 2
---

# Biological Neuron

<br><br/>

<div class="w-full">
        <img src="/images/Neuron3-dark.svg" alt="Neuron Diagram" />
    <div class="text-xs text-gray-500 italic mt-1">
        By <a href="//commons.wikimedia.org/wiki/User:Egm4313.s12" title="User:Egm4313.s12">Egm4313.s12</a> (Prof. Loc Vu-Quoc) - <span class="int-own-work" lang="en">Own work</span>, <a href="https://creativecommons.org/licenses/by-sa/4.0" title="Creative Commons Attribution-Share Alike 4.0">CC BY-SA 4.0</a>, <a href="https://commons.wikimedia.org/w/index.php?curid=72801384">Link</a>
    </div>
</div>



---
level: 2
---

# First Artificial Neuron

<br><br/>

<div grid="~ cols-2 gap-4">
<div class="col-span-1">

- Threshold Logic Unit (TLU), or Linear Threshold Unit
- Warren McCulloch and Walter Pitts in 1943
- Simple model of a biological neuron
- Fixed number of inputs and outputs
- Logical gate-like behavior (AND, OR, NOT)

</div>
<div class="col-span-1">

<div class="w-full">
        <img src="/images/tlu-dark.svg" alt="TLU Diagram" />
</div>

</div>
</div>


---
level: 2
---

# Perceptron

<br><br/>

<div grid="~ cols-2 gap-4">
<div class="col-span-1">

- Rosenblatt's Perceptron (1958)
- First algorithm for supervised learning of binary classifiers
- Maps its input features to a single output value

</div>
<div class="col-span-1">

<div class="w-full">
        <img src="/images/perceptron-dark.svg" alt="TLU Diagram" />
</div>

</div>
</div>

---
level: 2
---

# Multi-Layer Perceptron


<div grid="~ cols-2 gap-4" class="mt-4">
<div class="col-span-1">

<div class="w-1/1">
        <img src="/images/mlp-dark.svg" alt="TLU Diagram" />
</div>

<div
  v-click
  class="absolute bottom-45 left-58 w-16 h-16 border-2 border-red-500 rounded-full"
/>

</div>
<div class="col-span-1">

$$
\begin{aligned}
x_i &~:~\text{input data} \\
a_{ij} &~:~\text{activations of the $j$-th neuron in the $i$-th layer} \\
w^{(i)}_{jk} &~:~\text{model parameters} \\
\hat{y}_j &~:~\text{predicted output}
\end{aligned}
$$

<div class="w-full">
    <img
        v-click
        src="/images/mlp-a23-dark.svg"
        alt=""
    />
</div>


</div>
</div>

---
level: 2
---

# Deep Neural Networks

<div grid="~ cols-3 gap-0" class="mt-4">
<div class="col-span-2">

<div class="w-full">
    <img
        src="/images/dnn-dark.svg"
        alt=""
    />
</div>

- given an input $\mathbf{x}$ MLP computes the output <br/> $\mathbf{\hat{y}} = f^\ast \left( \mathbf{x} \right) \in \mathbb{R}^O$
$$ {none|1|2|3}
\begin{aligned}
f^\ast \left( \mathbf{x} \right) = & & \cdots & ~\sigma \circ c^1(\mathcal{x}) \\
f^\ast \left( \mathbf{x} \right) = & & \cdots \sigma \circ c^2 \circ & ~\sigma \circ c^1(\mathcal{x}) \\
f^\ast \left( \mathbf{x} \right) = & \sigma^O \circ c^L \circ \sigma \circ c^{L-1} & \cdots \sigma \circ c^2 \circ & ~\sigma \circ c^1(\mathcal{x})
\end{aligned}
$$

</div>
<div class="col-span-1">

<v-clicks>

for each layer $l \in \lbrace 1, \cdots, L \rbrace$ 

$$ {none|1|1-2|1-3|all}
\begin{aligned}
\mathbf{a}^{l-1} & \in \mathbb{R}^{d_{l-1}}&~:~&\text{input} \\
\mathbf{z}^{l} & \in \mathbb{R}^{d_{l}}&~:~&\text{weighted input} \\
\mathbf{a}^{l} & \in \mathbb{R}^{d_{l}}&~:~&\text{output (activations)} \\
\mathbf{W}^l & \in \mathbb{R}^{d_l \times d_{l-1}}&~:~&\text{weights} \\
\mathbf{b}^l & \in \mathbb{R}^{d_{l}}&~:~&\text{biases} 
\end{aligned}
$$
$$
\begin{aligned}
\mathbf{z}^l &= c^l\left(\mathbf{a}^{l-1} \right) = \mathbf{W}^l \mathbf{a}^{l-1} + \mathbf{b}^l \\
\mathbf{a}^l &= \sigma \left(\mathbf{z}^l \right)
\end{aligned}
$$

</v-clicks>

<v-clicks>

- weights $\mathbf{W} = \left\lbrace \mathbf{W}^l \right\rbrace_{l=1}^L$
- biases $\mathbf{B} = \left\lbrace \mathbf{b}^l \right\rbrace_{l=1}^L$
- parameters $\mathbf\theta = \left\lbrace \mathbf{W}, \mathbf{B} \right\rbrace$

</v-clicks>

</div>
</div>

---
level: 2
---

# Activation Functions $\sigma$

<div grid="~ cols-3 gap-0" class="mt-4">
<div class="col-span-1">

<v-clicks>

<div class="w-full">
  <img src="/images/activation_plots/heaviside.svg" />
</div>

</v-clicks>

</div>
<div class="col-span-2">

<v-clicks>

- **Heaviside step function**: $\sigma(x) = \begin{cases} 0 & \text{if } x < 0 \\ 1 & \text{if } x \geq 0 \end{cases}$
- McCullock-Pitts neuron
- Zero gradient, bad for backpropagation
- Not used anymore

</v-clicks>

</div>
</div>

<div grid="~ cols-3 gap-0" class="mt-4">
<div class="col-span-1">

<v-clicks>

<div class="w-full">
  <img src="/images/activation_plots/logistic_variants.svg" />
</div>

</v-clicks>

</div>
<div class="col-span-2">

<v-clicks>

- **Logistic function**: $\sigma(x) = \frac{1}{1 + e^{-x}}$
- Smooth approximation to Heaviside function
- Good for binary classification
- Not symmetric

</v-clicks>

</div>
</div>


---
level: 2
---

# Activation Functions $\sigma$

<div grid="~ cols-3 gap-0" class="mt-4">
<div class="col-span-1">

<v-clicks>

<div class="w-full">
  <img src="/images/activation_plots/tanh.svg" />
</div>

</v-clicks>

</div>
<div class="col-span-2">

<v-clicks>

- **Tanh function**: $\sigma(x) = \frac{1 - e^{-2x}}{1 + e^{-2x}}$
- Symmetric unlike Logisitic function
- Smooth
- Vanishing gradients away from 0

</v-clicks>

</div>
</div>

<div grid="~ cols-3 gap-0" class="mt-4">
<div class="col-span-1">

<v-clicks>

<div class="w-full">
  <img src="/images/activation_plots/relu.svg" />
</div>

</v-clicks>

</div>
<div class="col-span-2">

<v-clicks>

- **ReLU (Rectified Linear Unit) function**: $\sigma(x) = \max(0, x)$
- Easy to compute
- Reduces vanishing gradient problem
- Scale invariant
- Issue of dying neurons

</v-clicks>

</div>
</div>

---
level: 2
---

# Activation Functions $\sigma$

<div grid="~ cols-3 gap-0" class="mt-4">
<div class="col-span-1">

<v-clicks>

<div class="w-full">
  <img src="/images/activation_plots/leaky_relu.svg" />
</div>

</v-clicks>

</div>
<div class="col-span-2">

<v-clicks>

- **Leaky ReLU function**: $\sigma(x) = \begin{cases} x & \text{if } x > 0 \\ \nu x & \text{if } x \leq 0 \end{cases}$, where $\nu$ is a small constant (e.g., 0.01)
- Addresses the dying ReLU problem
- Allows a small, non-zero gradient when $x < 0$
- Still retains the benefits of ReLU
- Commonly used in deep learning architectures

</v-clicks>

</div>
</div>

<div grid="~ cols-3 gap-0" class="mt-4">
<div class="col-span-1">

<v-clicks>

- many more activation functions

</v-clicks>

</div>
<div class="col-span-2">


</div>
</div>

---
level: 2
---

# Supervised Learning

How can we use the data to adjust our model $f$ parameters $\theta$ to minimize prediction error and ensure accuracy within an acceptable margin?

<v-clicks>

- Labelled data: $\left\lbrace \left( \mathbf{y}_i, \mathbf{f}_i \right) \right \rbrace$
- Input space: $\mathbf{y}_i \in \mathbf{Y} \subset \mathbb{R}^I$
- Output space: $\mathbf{f}_i = f\left( \mathbf{y}_i \right) \in \mathbb{R}^O$
- Training Set: $\mathcal{S} = \left\lbrace \left( \mathbf{y}_i, \mathbf{f}_i \right) \right\rbrace_{i=1}^N$
- Choice of training set
    - Random sampling of the input space: $\mathbf{y}_i ~~\text{i.i.d.} \sim ~ \text{Prob}\left( \mathbf{Y} \right)$

</v-clicks>

---
level: 2
---

# Loss Function


- Difference between predicted and actual values: $\mathbf{f}_i$ and $f^\ast_\theta \left( \mathbf{y}_i \right)$
- Evaluation: for all $\mathbf{y}_i \in \mathcal{S}$, and labels $\mathbf{f}_i \in \mathbb{R}^O$
- popular chioce is the $p$-norm loss function
    - $p=2$: mean squared error (MSE)
    - $p=1$: mean absolute error (MAE), induces sparsity in the model parameters
$$
J(\theta) = \frac{1}{N} \sum_{i=1}^{N} \underbrace{ \left \lvert f^\ast_\theta \left( \mathbf{y}_i \right) - \mathbf{f}_i \right \rvert^p_p }_{J_i(\theta)}
$$

---
level: 2
---

# Training (Supervised Learning)

- **Objective**: Find the model parameters $\theta$ that minimize the loss function $J(\theta)$
$$
\theta^\ast = \arg \min_{\theta} J(\theta) = \arg \min_{\theta} \frac{1}{N} \sum_{i=1}^{N} J_i(\theta)
$$
- **Observations**:
    - $J(\theta)$ is non-convex (in general) and may have multiple local minima
    - $J(\theta)$ is differentiable with respect to $\theta$

---
level: 2
---

# Gradient Descent

Iteratively update the model parameters in the direction of the negative gradient of the loss function
$$
\theta_{l+1} = \theta_l - \eta_l \nabla_\theta J\left(\theta_l \right)
$$

- Adaptive learning rate $\eta_l$
- Requires the computation of the gradient $\nabla_\theta J\left(\theta_l \right)$
- Algorithm:
  - forward pass: compute the loss $J(\theta_l)$
  - backward pass: compute the gradient $\nabla_\theta J\left(\theta_l \right)$
  - update the model parameters $\theta_{l+1}$ with learning rate $\eta_l$

---
level: 2
---

# Gradient Descent

<div grid="~ cols-3 gap-0" class="mt-4">
<div class="col-span-2">
<div class="relative w-[500px] h-[400px] bg-black">

  <!-- Background loss surface -->
  <img src="/images/sgd4-dark.png" class="absolute w-full h-full object-cover" />

  <div
    v-motion
    :initial="{ x: -320, y: 134}"
    :enter="{ x: -220, y: 134}"
    v-click
  >
  <svg
    class="absolute left-1/2 top-1/2 w-12 h-24 transform -translate-x-1/2 -translate-y-1/2"
    style="transform: rotate(270deg) scale(0.4);"
    viewBox="0 0 20 100"
    fill="white"
    xmlns="http://www.w3.org/2000/svg"
  >
    <!-- Shaft -->
    <rect x="6" y="10" width="8" height="72" fill="white" />
    <!-- Filled triangle head -->
    <polygon points="0,80 10,100 20,80" fill="white" />
  </svg>
  </div>

  <div
    v-motion
    :initial="{ x: -220, y: 134}"
    :enter="{ x: -184, y: 134}"
    v-click
  >
  <svg
    class="absolute left-1/2 top-1/2 w-12 h-24 transform -translate-x-1/2 -translate-y-1/2"
    style="transform: rotate(270deg) scale(0.4);"
    viewBox="0 0 20 100"
    fill="white"
    xmlns="http://www.w3.org/2000/svg"
  >
    <!-- Shaft -->
    <rect x="6" y="10" width="8" height="72" fill="white" />
    <!-- Filled triangle head -->
    <polygon points="0,80 10,100 20,80" fill="white" />
  </svg>
  </div>

  <div
    v-motion
    :initial="{ x: -184, y: 134}"
    :enter="{ x: -148, y: 134}"
    v-click
  >
  <svg
    class="absolute left-1/2 top-1/2 w-12 h-24 transform -translate-x-1/2 -translate-y-1/2"
    style="transform: rotate(270deg) scale(0.4);"
    viewBox="0 0 20 100"
    fill="white"
    xmlns="http://www.w3.org/2000/svg"
  >
    <!-- Shaft -->
    <rect x="6" y="10" width="8" height="72" fill="white" />
    <!-- Filled triangle head -->
    <polygon points="0,80 10,100 20,80" fill="white" />
  </svg>
  </div>

</div>
</div>

<div class="col-span-1">

<v-clicks>

Problems with gradient descent:

- Converges to local minima
- Slow down near saddle points
- Slow convergence due to vanishing gradients
- Divergence due to exploding gradients
- Large computational cost
  - Computation of gradients
  - Requires large memory

</v-clicks>

</div>
</div>

---
level: 2
---

# Stochastic Gradient Descent

- At $l$-th iteration of Gradient descent
  - choose a random sample $\left( \mathbf{y}_i, \mathbf{f}_i \right)$ from the training set $\mathcal{S}$
  - compute the gradient of the loss function with respect to the model parameters $\theta_l$ using only this sample
  $$
  \theta_{l+1} = \theta_l - \eta_l \nabla_\theta J_i\left(\theta_l \right)
  $$
- Convergences to a local minimum of the loss function

---
level: 2
---

# Stochastic Gradient Descent

<div grid="~ cols-3 gap-0" class="mt-4">
<div class="col-span-2">
<div class="relative w-[500px] h-[400px] bg-black">

  <!-- Background loss surface -->
  <img src="/images/sgd4-dark.png" class="absolute w-full h-full object-cover" />

  <div
    v-motion
    :initial="{ x: -320, y: 134}"
    :enter="{ x: -205, y: 140, rotate: 30}"
    v-click
  >
  <svg
    class="absolute left-1/2 top-1/2 w-12 h-24 transform -translate-x-1/2 -translate-y-1/2"
    style="transform: rotate(270deg) scale(0.2);"
    viewBox="0 0 20 100"
    fill="white"
    xmlns="http://www.w3.org/2000/svg"
  >
    <!-- Shaft -->
    <rect x="6" y="10" width="8" height="72" fill="white" />
    <!-- Filled triangle head -->
    <polygon points="0,80 10,100 20,80" fill="white" />
  </svg>
  </div>

  <div
    v-motion
    :initial="{ x: -205, y: 140, rotate: 30}"
    :enter="{ x: -250, y: 195, rotate: -70}"
    v-click
  >
  <svg
    class="absolute left-1/2 top-1/2 w-12 h-24 transform -translate-x-1/2 -translate-y-1/2"
    style="transform: rotate(270deg) scale(0.2);"
    viewBox="0 0 20 100"
    fill="white"
    xmlns="http://www.w3.org/2000/svg"
  >
    <!-- Shaft -->
    <rect x="6" y="10" width="8" height="72" fill="white" />
    <!-- Filled triangle head -->
    <polygon points="0,80 10,100 20,80" fill="white" />
  </svg>
  </div>

  <div
    v-motion
    :initial="{ x: -250, y: 195, rotate: -70}"
    :enter="{ x: -240, y: 202, rotate: -100}"
    v-click
  >
  <svg
    class="absolute left-1/2 top-1/2 w-12 h-24 transform -translate-x-1/2 -translate-y-1/2"
    style="transform: rotate(270deg) scale(0.2);"
    viewBox="0 0 20 100"
    fill="white"
    xmlns="http://www.w3.org/2000/svg"
  >
    <!-- Shaft -->
    <rect x="6" y="10" width="8" height="72" fill="white" />
    <!-- Filled triangle head -->
    <polygon points="0,80 10,100 20,80" fill="white" />
  </svg>
  </div>

</div>
</div>

<div class="col-span-1">

<v-clicks>

Problems with stochastic gradient descent:

- similar to gradient descent
- fast per iteration, but requires many iterations to converge
- high variance (noisy updates)

</v-clicks>

</div>
</div>


---
level: 2
---

# Mini-Batch Stochastic Gradient Descent


- Randomly shuffle training set $\mathcal{S}$ into mini-batches $\mathcal{S}_j$ of size $n$
- At $l$-th iteration of Stochastic Gradient descent
  - select a mini-batch $\mathcal{S}_l$
  - compute the gradient of the loss function with respect to the model parameters $\theta_l$ using only this batch
  $$
  \theta_{l+1} = \theta_l - \eta_l \sum_{j \in \mathcal{S}_j}  \nabla_\theta J_j \left(\theta_l \right)
  $$
- After $\frac{N}{n}$ iterations: 1 Epoch (all samples in the training set have been used once)
- Reshuffle after each epoch

---
level: 2
---

# Mini-Batch Gradient Descent

<div grid="~ cols-3 gap-0" class="mt-4">
<div class="col-span-2">
<div class="relative w-[500px] h-[400px] bg-black">

  <!-- Background loss surface -->
  <img src="/images/sgd4-dark.png" class="absolute w-full h-full object-cover" />

  <div
    v-motion
    :initial="{ x: -320, y: 134}"
    :enter="{ x: -250, y: 140, rotate: -30}"
    v-click
  >
  <svg
    class="absolute left-1/2 top-1/2 w-12 h-24 transform -translate-x-1/2 -translate-y-1/2"
    style="transform: rotate(270deg) scale(0.3);"
    viewBox="0 0 20 100"
    fill="white"
    xmlns="http://www.w3.org/2000/svg"
  >
    <!-- Shaft -->
    <rect x="6" y="10" width="8" height="72" fill="white" />
    <!-- Filled triangle head -->
    <polygon points="0,80 10,100 20,80" fill="white" />
  </svg>
  </div>

  <div
    v-motion
    :initial="{ x: -250, y: 140, rotate: -30}"
    :enter="{ x: -170, y: 120, rotate: 40}"
    v-click
  >
  <svg
    class="absolute left-1/2 top-1/2 w-12 h-24 transform -translate-x-1/2 -translate-y-1/2"
    style="transform: rotate(270deg) scale(0.3);"
    viewBox="0 0 20 100"
    fill="white"
    xmlns="http://www.w3.org/2000/svg"
  >
    <!-- Shaft -->
    <rect x="6" y="10" width="8" height="72" fill="white" />
    <!-- Filled triangle head -->
    <polygon points="0,80 10,100 20,80" fill="white" />
  </svg>
  </div>

  <div
    v-motion
    :initial="{ x: -170, y: 120, rotate: 40}"
    :enter="{ x: -178, y: 130, rotate: 5}"
    v-click
  >
  <svg
    class="absolute left-1/2 top-1/2 w-12 h-24 transform -translate-x-1/2 -translate-y-1/2"
    style="transform: rotate(270deg) scale(0.3);"
    viewBox="0 0 20 100"
    fill="white"
    xmlns="http://www.w3.org/2000/svg"
  >
    <!-- Shaft -->
    <rect x="6" y="10" width="8" height="72" fill="white" />
    <!-- Filled triangle head -->
    <polygon points="0,80 10,100 20,80" fill="white" />
  </svg>
  </div>

</div>
</div>

<div class="col-span-1">

<v-clicks>

Observations:

- similar to stochastic gradient descent
- faster convergence than stochastic gradient descent
- less noisy updates

</v-clicks>

</div>
</div>

---
level: 2
---

# (Stocastic) Gradient Descent

<div grid="~ cols-3 gap-4" class="mt-4">
<div class="col-span-1">

<img
    src="/images/sgd1-dark.png"
    class="w-full h-auto"
/>

Gradient Descent

- `batch_size = N`
- `1 epoch = 1 iteration`

</div>

<div class="col-span-1">

<img
    src="/images/sgd2-dark.png"
    class="w-full h-auto"
/>

Stochastic Gradient Descent

- `batch_size = 1`
- `1 epoch = N iterations`

</div>

<div class="col-span-1">

<img
    src="/images/sgd3-dark.png"
    class="w-full h-auto"
/>

Mini-Batch Gradient Descent

- `batch_size = n`
- `1 epoch = N/n iterations`

</div>
</div>

---
level: 2
---

# Adam Optimizer

- Adaptive Moment Estimation (Adam) is an extension of stochastic gradient descent that computes adaptive learning rates for each parameter
- Combines the advantages of two other extensions of SGD: AdaGrad and RMSProp
- Maintains two moving averages for each parameter:
  - **First moment** (mean): $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
  - **Second moment** (uncentered variance): $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t} + \epsilon} m_t
$$
- typically, $\beta_1 = 0.9$, $\beta_2 = 0.999$, and $\epsilon = 10^{-8}$

---
level: 2
---

# Computation of Gradients: Backpropagation

- For all optimizers: gradients $\nabla_\theta J(\theta)$ need to be computed: **backpropagation algorithm**

<div grid="~ cols-2 gap-4" class="mt-4">
<div class="col-span-1">

<v-clicks>

- Recall the MLP architecture ($l=1, \cdots, L$)
$$
\begin{aligned}
z_i^{l} &= \sum_{j=1}^{d_{l-1}} w_{ij}^{l} a_j^{l-1} + b_i^{l} \\
a_i^{l} &= \sigma\left(z_i^{l}\right) \\
\hat{y}_i &= \sigma^L\left(z_i^{L}\right) = f^\ast_\theta\left(\mathbf{y}_i\right)
\end{aligned}
$$

- Chain rule for derivatives:
$$
\begin{aligned}
\frac{\partial J}{\partial w_{ij}^{l}} &= \frac{\partial J}{\partial z_i^{l}} \frac{\partial z_i^{l}}{\partial w_{ij}^{l}} = \frac{\partial J}{\partial z_i^{l}} a_j^{l-1} \\
\frac{\partial J}{\partial b_i^{l}} &= \frac{\partial J}{\partial z_i^{l}} \frac{\partial z_i^{l}}{\partial b_i^{l}} = \frac{\partial J}{\partial z_i^{l}} 
\end{aligned}
$$

</v-clicks>

</div>
<div class="col-span-1">

<v-clicks>

- Chain rule (backward direction):
$$
\begin{aligned}
\frac{\partial J}{\partial z_i^{l}} &= \sum_{k=1}^{d_{l+1}} \frac{\partial J}{\partial z_k^{l+1}} \frac{\partial z_k^{l+1}}{\partial z_i^{l}} = 
\sum_{k=1}^{d_{l+1}} \frac{\partial J}{\partial z_k^{l+1}} \frac{\partial z_k^{l+1}}{\partial a_i^l} \frac{\partial a_i^l}{\partial z_i^l} \\
&= \sum_{k=1}^{d_{l+1}} \frac{\partial J}{\partial z_k^{l+1}} w_{ki}^{l+1} \sigma'\left(z_i^{l}\right) 
\end{aligned}
$$

- Recursive relationship:
$$
\begin{aligned}
\frac{\partial J}{\partial z_i^{l}} \equiv \delta_i^{l} &= \sigma'\left(z_i^{l}\right) \sum_{k=1}^{d_{l+1}} \delta_k^{l+1} w_{ki}^{l+1} \\
\delta_i^{L} &= \frac{\partial J}{\partial \hat{y}_i} \sigma'^L\left(z_i^{L}\right)
\end{aligned}
$$

</v-clicks>

</div>
</div>


---
level: 2
---

# Computation of Gradients: Matrix Multiplications and Autodiff

- Backpropagation can be efficiently implemented using matrix multiplications
- Forward pass:
  - For each layer $l$:
    - compute $\mathbf{z}^l = \mathbf{W}^l \mathbf{a}^{l-1} + \mathbf{b}^l$, where $\mathbf{W}^l \in \mathbb{R}^{d_{l} \times d_{l-1}}$ and $\mathbf{a}^{l-1} \in \mathbb{R}^{d_{l-1}}$ and $\mathbf{b}^l \in \mathbb{R}^{d_{l}}$
    - compute $\mathbf{a}^l = \sigma\left(\mathbf{z}^l\right)$
    - compute and cache $\sigma'\left(\mathbf{z}^l\right)$ using automatic differentiation
- Backward pass:
  - compute $\mathbf{\delta}^L = \frac{\partial J}{\partial \hat{\mathbf{y}}} \mathbb{I} \sigma'^L\left(\mathbf{z}^L\right)$ 
  - For each layer $l$:
    - compute $\mathbf{\delta}^l = \sigma'\left(\mathbf{z}^l\right) \odot \left( {\mathbf{W}^{l+1}}^T \mathbf{\delta}^{l+1} \right)$, where $\odot$ is the element-wise product
    - compute $\frac{\partial J}{\partial \mathbf{W}^l} = \mathbf{\delta}^l {\mathbf{a}^{l-1}}^T$ and $\frac{\partial J}{\partial \mathbf{b}^l} = \mathbf{\delta}^l$

---
level: 2
---

# Mini-Batch Version: More Matrices

<div grid="~ cols-2 gap-4" class="mt-4">

<div class="col-span-1">

- $n^l$: number of neurons in layer $l$
- $m$: number of samples in the mini-batch
- $\mathbf{X} \in \mathbb{R}^{n^0 \times m}$: input data matrix
- $\mathbf{W}^l \in \mathbb{R}^{n^l \times n^{l-1}}$: weight matrix for layer $l$



</div>
<div class="col-span-1">

- $\mathbf{b}^l \in \mathbb{R}^{n^l}$: bias vector for layer $l$ (broadcasted)
- $\mathbf{Z}^l \in \mathbb{R}^{n^l \times m}$: weighted input matrix for layer $l$
- $\mathbf{A}^l \in \mathbb{R}^{n^l \times m}$: activation matrix for layer $l$
- $\mathbf{\Delta}^l \in \mathbb{R}^{n^l \times m}$: error matrix for layer $l$

</div>
</div>


| Pass         | Operation                                | Shape(s)                                   |
|--------------|------------------------------------------|--------------------------------------------|
| Forward      | $Z^{l} = W^{l} A^{l-1} + b^l$            | $[n^l \times m] = [n^l \times n^{l-1}] \cdot [n^{l-1} \times m]$
|              | $A^{l} = \sigma(Z^{l})$                  | element-wise operation $[n^l \times m]$ |
| Backward     | $\Delta^{l} = \sigma'(Z^{l}) \odot {W^{l+1}}^T \Delta^{l+1}$ | $[n^l \times m] = [n^l \times m] \odot [n^l \times n^{l+1}] \times [n^{l+1} \times m]$ |
|              | $\frac{\partial J}{\partial W^{l}} = \Delta^{l} {A^{l-1}}^T$, $\frac{\partial J}{\partial b^{l}} = \Delta^{l}$| $[n^l \times n^{l-1}] = [n^l \times m] \cdot [m \times n^{l-1}]$ |

---
level: 2
---

# 🤔 But Why Tensors, Then?
_Even if MLPs only use matrices..._

- **Frameworks generalize beyond MLPs**:
  - CNNs: 4D tensors → `(batch, channels, height, width)`
  - RNNs/Transformers: 3D+ tensors → `(batch, sequence, embedding)`
- **Auto-diff engines** work with arbitrary-rank tensors
- Tensors = uniform abstraction for:
  - Batched computation
  - Broadcasting
  - GPU acceleration



---
level: 2
---

# Deep Learning Frameworks

- Simplify the implementation of neural networks
- Provide high-level APIs for building, training, and evaluating models
- Support for automatic differentiation
- Optimized matrix operations and GPU acceleration

