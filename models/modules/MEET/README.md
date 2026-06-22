# MEET: Memory-Efficient Equivariant Transformer

## Introduction

The Memory-Efficient Equivariant Transformer (MEET) is a neural network architecture designed to incorporate geometric equivariance into modern transformer architectures. This project has two main aims:

1. **Alignment with modern LLMs and VLMs**: This project transplants useful tricks from large language models and image generation models, including flash attention, SwiGLU activation, adaptive LayerNorm-zero (adaLN-zero) to ensure optimal performance and efficiency. You may find more information about modern LLMs and VLMs in:
> - [2 OLMo 2 Furious (arXiv:2501.00656)](https://arxiv.org/abs/2501.00656)  
> - [Scalable Diffusion Models with Transformer (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)

2. **Equivariant versions of core components**: The library provides equivariant implementations of fundamental neural network components, including multi-head attention (MHA), multi-layer perceptrons (MLPs), activations, and normalizations, enabling geometrically consistent representations across rigid transformations. More mathematical details about the equivariant mechanisms can be found in:
> - [An Equivariant Pretrained Transformer for Unified 3D Molecular Representation Learning (arXiv:2402.12714)](https://arxiv.org/abs/2402.12714)  
> - [Vector Neurons: A General Framework for SO(3)-Equivariant Networks (arXiv:2104.12229)](https://arxiv.org/abs/2104.12229)

In the following sections, we will provide detailed documentation for each neural network layer, explaining their mathematical foundations, implementation details, and configuration options.

## Understanding Equivariant Mechanisms

This tutorial introduces the mathematical foundations behind **equivariance** and **invariance** in geometric deep learning, with a focus on 3D rigid transformations.

### 1. Rigid Transformations and Equivariance & Invariance

Rigid transformations are geometric operations that **preserve distances and angles** between points in (usually 3D) space, primarily **rotations** and **translations**. To build models that are physically meaningful, we require them to behave consistently under such transformations. This behavior is formalized through **equivariance** and **invariance**.

**Equivariance**: A function $f$ is *equivariant* if transforming the input and then applying $f$ is equivalent to applying $f$ and then transforming the output: $f(Tx) = T f(x)$.

**Invariance**: A function $f$ is *invariant* if its output does not change when the input is transformed: $f(Tx) = f(x)$. 

### 2. Distance Matrices Are E(n)-Invariant Features

The **pairwise distance matrix** provides a fundamental **E(n)-invariant** representation.

For a coordinate set of $m$ points in $n$-dimensional space $X \in \mathbb{R}^{m \times n}$:

$$
D_{ij} = \| X_i - X_j \|_2
$$

This matrix remains unchanged under any **Euclidean transformation** (rotation, translation, or reflection):

$$
D(X) = D(RX + t)
$$

Hence, it is a powerful invariant feature in geometric deep learning models.

### 3. Difference Vectors are E(n)-Equivariant Features

The **difference vector** between two coordinates is **E(n)-equivariant**:

$$
\Delta = X_i - X_j
$$

Under a rotation $R$ and translation $t$:

$$
(RX_i + t) - (RX_j + t) = R(X_i - X_j) = R \Delta
$$

This makes difference vectors ideal for encoding directional geometric information while maintaining consistency under rigid transformations.

### 4. Basic Operations and Their Equivariance Properties

#### 4.1 Dot Product is E(n)-Invariant

The **dot product** between two vector features $\mathbf{v}, \mathbf{w}$ is invariant under rotation:

$$
(R\mathbf{v}) \cdot (R\mathbf{w}) = \mathbf{v}^T R^T R \mathbf{w} = \mathbf{v} \cdot \mathbf{w}
$$

Thus, the dot product captures **rotation-invariant** relationships, such as angles or magnitudes.

#### 4.2 Linear Transformation (Matmul without Bias) is E(n)-Equivariant

A **bias-free linear transformation** of vector features preserves equivariance.

For $V \in \mathbb{R}^{*, n, D}$ and weights $W \in \mathbb{R}^{D \times D'}$:

$$
R(VW) = (RV)W
$$

This ensures that learned linear projections of vectors maintain their geometric meaning under rotation.

#### 4.3 Norm is E(n)-Invariant

The **Euclidean norm** (magnitude) of a vector feature is invariant under rotation:

$$
\| R\mathbf{v} \|_2 = \| \mathbf{v} \|_2
$$

The norm provides a scalar, geometry-independent measure of vector strength.

## Adopted Tricks from LLMs and VLMs

The MEET incorporates several state-of-the-art techniques pioneered in Large Language Models (LLMs) and AI generative models (e.g. DiT) to achieve optimal performance and efficiency. These techniques have been adapted to work with geometric data while maintaining the core principles that made them successful in language and vision domains.

### 1. XFormers Attention

XFormers provides the only attention backend kept in this clean MEET package. It supplies memory-efficient attention kernels while keeping the package interface small and self-contained.

**Implementation Details:**
- Set as the required attention backend in the configuration (`backend: xformers`)
- Uses `xops.memory_efficient_attention` for optimized memory usage
- Implements block diagonal masking for efficient computation on batched inputs
- Other backends were removed from this clean package

### 2. SwiGLU Activation

Swish-Gated Linear Units (SwiGLU) is an activation function that combines Swish (SiLU) and Gated Linear Units (GLU). It has become the standard activation in modern LLMs like PaLM, LLaMA, and GPT-4 due to its superior performance in deep networks.

**Implementation Details:**
- Configured as the default activation function for both scalar and vector features
- Provides better expressivity than traditional activation functions like ReLU or GELU
- Helps with training stability in deep networks
- Has been shown to improve performance in transformer models

### 3. Adaptive LayerNorm-zero (adaLN-zero)

Adaptive LayerNorm-zero is a specialized normalization technique used in diffusion models, particularly in Diffusion Transformers (DiT). It allows for classifier-free guidance by controlling the scale and shift parameters of layer normalization.

**Implementation Details:**
- Uses adaptive scaling and biasing for normalization layers
- Takes conditioning variables to control the normalization parameters
- Enables classifier-free guidance when used in diffusion models
- Provides flexibility in controlling the denoising process

### 4. RMSNorm

Root Mean Square Layer Normalization (RMSNorm) is a simplified version of LayerNorm that omits the mean-centering step. It has been widely adopted in models like GPT-3 due to its computational efficiency and similar performance.

**Implementation Details:**
- Configured as the default normalization type (`norm_type: rms`)
- Reduces computational overhead compared to standard LayerNorm
- Maintains training stability in deep networks
- Works well with the other optimization techniques in modern transformers

### 5. Pre-Normalization

Pre-normalization refers to applying normalization before the attention and feed-forward layers rather than after. This technique has become standard in deep transformer models as it helps stabilize training.

**Implementation Details:**
- Set as the default normalization position (`norm_position: pre`)
- Implemented in the sublayer wrapper with support for 'pre', 'post', and 'reordered' positions
- Enables training of deeper networks by normalizing the inputs to each sub-layer
- Works in conjunction with residual connections to improve gradient flow

### 6. Parameter Initialization

Careful parameter initialization is crucial for training deep neural networks effectively. The project implements truncnorm initialization strategies inspired by modern LLMs.

**Implementation Details:**
- Configurable initialization standard deviation (`init_std: 0.02`)
- Cutoff factor for initialization (`init_cutoff_factor: 3`)
- Ensures proper scaling of activations and gradients
- Helps prevent issues like vanishing or exploding gradients during training
- Similar to strategies used in models like BERT and GPT


## Public API

The clean package exports the following public classes:

- `MEET` and `MEETWrapper` from `MEET`
- `InitLayer`, `MEETAttnLayer`, `MEETFFNLayer`, `MEETLayer`, `MEETBondLayer`, `BondAdapter`, and `FinalLayer` from `MEET.model`

## Neural Network Layers

### 1. InitLayer

The `InitLayer` layer is responsible for initializing both scalar and vector features from input embeddings and coordinates. It implements a distance-based attention mechanism to generate E(3)-equivariant vector features that capture geometric relationships between points, and can be easily modified to work with higher-dimensional spaces.

#### 1.1 Mathematical Formulation

#### 1.1.1 Distance-Aware Attention
The model incorporates distance information into attention mechanisms  use a **distance encoding method** that embeds geometric information directly into the query and key projections. For a pair of points $(i, j)$ with coordinates $\mathbf{r}_i, \mathbf{r}_j \in \mathbb{R}^n$:

$$
\mathbf{r}_i = [x_{1,i}, x_{2,i}, \ldots, x_{k,i}, \ldots, x_{n,i}]
$$

**Distance encoding vectors** are constructed:

   $$
   \mathbf{q}^{\text{dist}}_i = s_h \cdot \bigoplus_{k=1}^{n}[x_{k,i}^2, 1, -2x_{k,i}]
   $$

   $$
   \mathbf{k}^{\text{dist}}_j = s_h \cdot \bigoplus_{k=1}^{n}[1, x_{k,j}^2, x_{k,j}]
   $$

   where $s_h$ is a learnable scaling parameter per head.

**Augmented queries and keys** are formed by concatenating the original query and key vectors of each head with their distance encoding vectors:

   $$
   \mathbf{Q}_i^{(h)} = \mathrm{concat}(\mathbf{q}_i^{(h)}, -\mathbf{q}^{\text{dist}}_i), \quad \mathbf{K}_j^{(h)} = \mathrm{concat}(\mathbf{k}_j^{(h)}, \mathbf{k}^{\text{dist}}_j)
   $$

**Distance-aware attention score** is computed as a dot product:

   $$
   A_{ij}^{(h)} = \frac{\langle \mathbf{Q}_i^{(h)}, \mathbf{K}_j^{(h)} \rangle}{\sqrt{d_h}} = \frac{\langle \mathbf{q}_i^{(h)}, \mathbf{k}_j^{(h)} \rangle - \langle \mathbf{q}^{\text{dist}}_i, \mathbf{k}^{\text{dist}}_j \rangle}{\sqrt{d_h}}
   $$

The key insight is that $\langle \mathbf{q}^{\text{dist}}_i, \mathbf{k}^{\text{dist}}_j \rangle = s_h^2 \cdot \|\mathbf{r}_i - \mathbf{r}_j\|^2$, so:

   $$
   A_{ij}^{(h)} = \frac{\langle \mathbf{q}_i^{(h)}, \mathbf{k}_j^{(h)} \rangle - s_h^2 \cdot \|\mathbf{r}_i - \mathbf{r}_j\|^2}{\sqrt{d_h}}
   $$

#### 1.1.2 Vector Feature Construction

The vector feature construction process incorporates geometric information through a coordinate-based attention mechanism that maintains O(n) equivariance. 

Using the distance-aware attention weights $\alpha_{ij}^{(h)}$ computed in the previous section, we compute attended coordinates for each atom $i$ and head $h$:

$$
\hat{\mathbf{r}}_i^{(h)} = \sum_j \alpha_{ij}^{(h)} \mathbf{r}_j
$$

Vector features are constructed by computing the displacement between attended coordinates and original coordinates:

$$
\mathbf{v}_i^{(h)} = \hat{\mathbf{r}}_i^{(h)} - \mathbf{r}_i = \sum_j \alpha_{ij}^{(h)} (\mathbf{r}_j - \mathbf{r}_i)
$$

Vector features from all attention heads are combined and projected through a linear transformation:

$$
\mathbf{V}_i = \text{Linear}\left( \text{concat}_h(\mathbf{v}_i^{(h)}) \right)
$$

This construction ensures **O(n) equivariance** because:
1. The attention weights $\alpha_{ij}^{(h)}$ are rotationally invariant (they depend only on distances)
2. The displacement vectors transform naturally under rotation

For any rotation $R \in \mathbf{O}(n)$:

$$
\mathbf{r}_i \mapsto R\mathbf{r}_i \implies \mathbf{V}_i \mapsto R\mathbf{V}_i
$$

The subtraction of $\mathbf{r}_i$ from the attended coordinates is crucial for maintaining equivariance, as it centers the representation around each point.

#### 1.2 Class Definition

#### Signature

```python
class InitLayer(nn.Module):
    def __init__(self, model_config):
        # Initializes the feature initialization layer
        pass
    
    def forward(self, atom_feat, coords, seqlen):
        # Processes input data and returns initialized features
        pass
        return (atom_feat, vector_feat), attention_mask
```

#### Parameters

**model_config Required Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `d_model` | int | Hidden dimension size for the model |
| `n_heads` | int | Number of attention heads |
| `use_atom_embedding` | bool | Whether `atom_feat` contains atom type indices requiring embedding |
| `atom_types` | int | Number of atom types; required when `use_atom_embedding=True` |
| `include_bias` | bool | Whether to include bias in linear layers |
| `att_norm` | bool | Whether to apply normalization to attention queries and keys |
| `norm_type` | str | Type of normalization to use (e.g., 'rms') |
| `norm_with_affine` | str | Whether to use learnable affine parameters in normalization |
| `backend` | str | Attention backend to use. Only `'xformers'` is supported |
| `dropout` | float | Dropout probability used by xFormers attention |
| `init_std` | float | Standard deviation for parameter initialization |
| `init_cutoff_factor` | float | Cutoff factor for truncated normal initialization |

#### Inputs

- `atom_feat`: Tensor of shape `[sum(L)]` with atom type indices when `use_atom_embedding=True`; otherwise tensor of shape `[sum(L), d_model]` with precomputed scalar features
- `coords`: Tensor of shape `[sum(L), 3]` containing 3D coordinates for all atoms
- `seqlen`: **List** but **NOT tensor** of shape `[batch_size]` indicating the number of atoms in each sample

#### Outputs

- Tuple of `(features, attention_mask)` where:
  - `features`: Tuple of `(scalar_features, vector_features)`
    - `scalar_features`: Tensor of shape `[sum(L), d_model]` representing scalar embeddings
    - `vector_features`: Tensor of shape `[sum(L), 3, d_model]` representing equivariant vector features
  - `attention_mask`: xFormers `BlockDiagonalMask` built from `seqlen`

### 2. Attention Mechanisms

The model implements the classic scaled dot-product self attention layers that maintain E(n)-equivariance named `MEETAttnLayer`.

#### 2.1 General Attention Formulation

The attention layer follows a mathematical formulation with E(n)-equivariant properties:

**Query, Key, Value Preparation:**
For each atom $i$, scalar features $\mathbf{H}_i \in \mathbb{R}^{d_{\text{model}}}$ and vector features $\mathbf{V}_i \in \mathbb{R}^{n \times d_{\text{model}}}$ are transformed into query, key, and value representations using learned linear projections:

$$
\mathbf{q}_i^{(h)}, \mathbf{k}_i^{(h)}, \mathbf{v}_i^{(h)} = \text{QKV\_Prep}(\mathbf{H}_i, \mathbf{V}_i, \mathbf{r}_i)
$$

In the function `QKV_Prep`, vector features are flattened and concatenated with scalar features, and the distance encoding mentioned in Section 1.1.1 is also concatenated for Q and K.

The following processing is identical to the classic scaled dot-product self attention layers.

#### 2.2 Layer-Specific Definitions

#### Signature

```python
class MEETAttnLayer(nn.Module):

    def __init__(self,model_config,layer_idx=-1):
        # Initializes the self-attention layer
        pass
    
    def forward(self, feats, coords,seqlen_info):
        # Processes input data and returns updated features
        pass
        return (scalar_feat, vector_feat)

```

#### Parameters

**model_config Required Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `d_model` | int | Hidden dimension size for the model |
| `n_heads` | int | Number of attention heads |
| `include_bias` | bool | Whether to include bias in linear layers |
| `att_norm` | bool | Whether to apply normalization to attention queries and keys |
| `norm_type` | str | Type of normalization to use (e.g., 'rms') |
| `norm_with_affine` | str | Whether to use learnable affine parameters in normalization |
| `backend` | str | Attention backend to use. Only `'xformers'` is supported |
| `dropout` | float | Dropout probability used by xFormers attention |
| `init_std` | float | Standard deviation for parameter initialization |
| `init_cutoff_factor` | float | Cutoff factor for truncated normal initialization |

#### Inputs

- `feats`: a tuple of (scalar_feat, vector_feat), represented by `(H,V)` in the code. 
- `coords`: a tensor of shape `[sum(L), 3]` containing 3D coordinates for all points.
- `seqlen_info`: an xFormers `BlockDiagonalMask` used to handle variable sequence lengths in batching, generated by `InitLayer`.

#### Outputs

- `MEETAttnLayer` produces a single set of updated scalar and vector features $(\mathbf{H}_{\text{out}}, \mathbf{V}_{\text{out}})$.
  
  
### 3. Feedforward Neural Network for Mixing Scalar and Vector Features

The `MEETFFNLayer` layer is a hybrid feed-forward network that processes both scalar and vector features while maintaining E(n) equivariance. It implements a novel architecture that blends information between scalar and vector representations through cross-feature interactions.

#### 3.1 Mathematical Formulation
The `MEETFFNLayer` layer processes a pair of features $(\mathbf{H}, \mathbf{V})$, where $\mathbf{H} \in \mathbb{R}^{d}$ are scalar features (invariant) and $\mathbf{V} \in \mathbb{R}^{n \times d}$ are vector features (equivariant). Let $d$ be the model dimension and $d_{ffn}$ be the hidden dimension determined by `mlp_ratio`.

The operation proceeds in the following steps:

**Vector Projection and Splitting**
The vector features are projected linearly and split into a mix-to-scalar component ($\mathbf{V}_1$) and a hidden component ($\mathbf{V}_2$):

$$
\mathbf{V}_{\text{proj}} = \mathbf{V}\mathbf{W}_{v1} \quad \text{where } \mathbf{W}_{v1} \in \mathbb{R}^{d \times (d + d_{ffn})}
$$

$$
\mathbf{V}_1, \mathbf{V}_2 = \text{split}(\mathbf{V}_{\text{proj}}) \quad \text{s.t. } \mathbf{V}_1 \in \mathbb{R}^{n \times d}, \mathbf{V}_2 \in \mathbb{R}^{n \times d_{ffn}}
$$

**Scalar Enhancement and Projection**
The Euclidean norm of the mix-to-scalar vector component is calculated and concatenated with the original scalar features. This allows geometric information to influence invariant features:

$$
\mathbf{H}_{\text{in}} = [\mathbf{H}, \|\mathbf{V}_1\|_2] \in \mathbb{R}^{2d}
$$

These enhanced scalars are processed through a linear layer and split into a gating signal ($\mathbf{H}_1$) and a hidden component ($\mathbf{H}_2$):

$$
\mathbf{H}_{\text{proj}} = \mathbf{H}_{\text{in}}\mathbf{W}_{h1} + \mathbf{b}_{h1}
$$

$$
\mathbf{H}_1, \mathbf{H}_2 = \text{split}(\mathbf{H}_{\text{proj}}) \quad \text{s.t. } \mathbf{H}_1 \in \mathbb{R}^{d_{ff}}, \mathbf{H}_2 \in \mathbb{R}^{2d_{ff}}
$$

**Activation Function**
The hidden vector features are gated by the activated scalar gating signal, inspired by the SwishGLU activation function. This serves as the core "blending" mechanism where scalar semantics gate vector directions:

$$
\mathbf{V}_2 = \sigma_v(\mathbf{H}_1) \odot \mathbf{V}_2
$$

The hidden scalar features are activated by the SwishGLU activation function:

$$
\mathbf{H}_2 = \text{SwishGLU}(\mathbf{H}_2)
$$

**Final Projections**
Both streams are projected back to the original model dimension $d$.

#### 3.2 Class Definition

#### Signature

```python
class MEETFFNLayer(nn.Module):
    def __init__(self, model_config, se3=False):
        # Initializes the feed-forward layer with configuration and SE(3) mode
        pass
    
    def forward(self, feats):
        # Processes input features and returns updated scalar and vector features
        pass
        return (H_out, V_out)
```

#### Parameters

**model_config Required Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `d_model` | int | Hidden dimension size for the model |
| `mlp_ratio` | float | Ratio of hidden dimension in MLP layers to the input dimension |
| `include_bias` | bool | Whether to include bias in linear layers (for scalar MLP) |
| `h_act_fn` | str | Activation function for scalar features |
| `v_act_fn` | str | Activation function for vector features |
| `dropout` | float | Dropout rate for regularization |
| `init_std` | float | Standard deviation for parameter initialization |
| `init_cutoff_factor` | float | Cutoff factor for truncated normal initialization |

#### Inputs

- `feats`: Tuple of `(scalar_features, vector_features)`

#### Outputs

- Tuple of `(updated_scalar_features, updated_vector_features)`

### 4. Sublayer Wrappers

The sublayer wrappers (`VNSubLayerWrapper`) provide a common framework for normalizing inputs, applying residual connections, and supporting adaptive layer normalization in different contexts within the network. These wrappers enforce consistent preprocessing and postprocessing across various layer types.

#### 4.1 Mathematical Formulation

Sublayer wrappers implement a unified processing pattern with configurable normalization placement and residual connections:

#### 4.1.1 Normalization and Residual Connections

The core processing flow follows these steps:

**Input Preservation**: Store original inputs for residual connections:

   $$
   \mathbf{H}_0 = \mathbf{H}, \quad \mathbf{V}_0 = \mathbf{V}
   $$

**Normalization** (position-dependent):
   - **Pre-normalization**: Apply before sublayer processing:

     $$
     \mathbf{H}_{\text{norm}} = \text{Norm}(\mathbf{H}_0), \quad \mathbf{V}_{\text{norm}} = \text{Norm}(\mathbf{V}_0)
     $$

   - **Reordered-normalization**: Apply after sublayer processing but before residual connection
   - **Post-normalization**: Apply after residual connection

**Sublayer Processing**:

   $$
   \mathbf{H}_{\text{out}}, \mathbf{V}_{\text{out}} = \text{SubLayer}(\mathbf{H}_{\text{norm}}, \mathbf{V}_{\text{norm}})
   $$

**Residual Connection**:

   $$
   \mathbf{H}_{\text{final}} = \mathbf{H}_{\text{out}} + \mathbf{H}_0
   $$

   $$
   \mathbf{V}_{\text{final}} = \mathbf{V}_{\text{out}} + \mathbf{V}_0
   $$

#### 4.1.2 Adaptive Layer Normalization (adaLN_zero)

When `adaLN_zero` is enabled, the wrappers support conditional normalization parameters that can be generated by an external controller (e.g., a diffusion model's timestep embedding):

$$
\mathbf{H}_{\text{norm}} = \text{Norm}({\mathbf{H}}) \odot \text{scale}_H + \text{bias}_H
$$

Where `scale_H`, `bias_H`, `gate_H`, `scale_V`, `bias_V`, and `gate_V` are dynamically computed parameters.

Additionally, `adaLN_zero` requires a gating mechanism at the residual connection:

   $$
   \mathbf{H}_{\text{final}} = \text{gate}_H \cdot \mathbf{H}_{\text{out}} + \mathbf{H}_0
   $$

   $$
   \mathbf{V}_{\text{final}} = \text{gate}_V \cdot \mathbf{V}_{\text{out}} + \mathbf{V}_0
   $$

#### 4.2 Class Definitions

#### Common Signature and Interface

```python
class VNSubLayerWrapper(nn.Module):
    def __init__(self, sub_layer, model_config):
        # Initializes the wrapper with normalization layers
        pass
    
    def forward(self, feats, conditions=None, **kwargs):
        # Applies normalization, sublayer processing, and residual connection
        pass
        # Returns processed features
```

#### Common Parameters

**model_config Required Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `d_model` | int | Hidden dimension size for the model |
| `norm_type` | str | Type of normalization to use (e.g., 'rms') |
| `norm_with_affine` | str | Whether normalization uses `'learnable'`, `'adaptive'`, or `'none'` affine parameters |
| `norm_position` | str | Position of normalization: 'pre', 'post', or 'reordered' |
| `adaLN_zero` | bool | Whether to use adaptive LayerNorm zero with external conditions |

## Usage

The package can be used independently when `models/modules` is on `PYTHONPATH`:

```python
from MEET import MEET, MEETWrapper
from MEET.model import InitLayer, MEETAttnLayer, MEETFFNLayer, MEETLayer, MEETBondLayer, BondAdapter
```

### MEET

```python
model = MEET(d_model=512, n_layers=6, n_heads=16, backend='xformers')
H_out, V_out, X_out = model(atom_feat, coords, batch_ids, ligand_mask=None, edge_index=None, edge_attr=None, condition=None)
```

`MEET.forward` returns scalar features, vector features, and reordered coordinates.

### MEETWrapper

```python
model = MEETWrapper(d_model=512, n_layers=6, n_heads=16, backend='xformers')
H_out, V_out = model(atom_feat, vector_feat, coords, batch_ids, ligand_mask=None, edge_index=None, edge_attr=None, condition=None)
```

`MEETWrapper` is a compatibility wrapper for callers that pass an existing `vector_feat`; the current implementation recomputes vector features from coordinates through `MEET` and returns only scalar/vector outputs.
