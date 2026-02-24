# Data Generation Using Modelling and Simulation

##  Overview

This project demonstrates a complete pipeline for **synthetic data generation using physics-based simulation**, followed by **machine learning model training**, **performance evaluation**, and **multi-criteria model selection using the TOPSIS method**. The simulation environment used is the classic **CartPole-v1** from OpenAI Gymnasium, which provides a controlled, reproducible source of data.

---

##  Objectives

- Generate realistic synthetic data through a physics-based simulation environment (CartPole-v1)
- Train and evaluate multiple Machine Learning regression models on the generated data
- Apply the **TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)** multi-criteria decision-making method to rank the models objectively
- Identify the best-performing model based on a balanced, weighted scoring approach

---

##  Technologies & Libraries

| Library | Purpose |
|---|---|
| `gymnasium` | Physics-based simulation environment (CartPole-v1) |
| `numpy` | Numerical operations |
| `pandas` | Data manipulation and storage |
| `scikit-learn` | ML models and evaluation metrics |
| `xgboost` | Gradient boosted tree implementation |
| `matplotlib` | Visualization of TOPSIS rankings |

---

##  Project Structure

```
├── Data_generation_using_Modelling_and_Simulation.ipynb   # Main notebook
├── simulation_data.csv                                     # Generated simulation dataset
├── model_ranking_topsis.csv                                # TOPSIS ranking output
└── README.md                                               # Project documentation
```

---

##  Methodology

### Step 1: Simulation Environment Setup

The **CartPole-v1** environment from OpenAI Gymnasium is used as the simulation engine. This is a classic control problem where a pole is balanced on a cart. The environment provides a physically accurate, deterministic simulation governed by four state variables:

| Parameter | Range | Description |
|---|---|---|
| `cart_pos` | [-2.4, 2.4] | Horizontal position of the cart |
| `cart_vel` | [-2.0, 2.0] | Velocity of the cart |
| `pole_angle` | [-0.2, 0.2] | Angle of the pole (radians) |
| `pole_vel` | [-2.0, 2.0] | Angular velocity of the pole tip |

### Step 2: Synthetic Data Generation (1000 Simulations)

For each simulation run:
1. A random set of initial state parameters is sampled from the defined ranges.
2. The environment is initialized with these parameters.
3. A **random action policy** is applied (actions are sampled from the action space) until the episode terminates.
4. The **cumulative reward** (total timesteps survived) is recorded as the target variable (`reward`).

This process is repeated **1000 times**, producing a dataset of 1000 samples with 4 features and 1 continuous target. The dataset is saved as `simulation_data.csv`.

```python
for i in range(1000):
    params = [cart_pos, cart_vel, pole_angle, pole_vel]  # randomly sampled
    reward = run_simulation(params)
    data.append(params + [reward])
```

### Step 3: Data Preparation

The generated dataset is split into training and testing sets using an **80/20 split** with a fixed random seed (`random_state=42`) to ensure reproducibility:
- **Training Set:** 800 samples
- **Test Set:** 200 samples

### Step 4: Model Training & Evaluation

Eight regression models are trained on the simulation data and evaluated on the held-out test set. The following metrics are computed for each model:

| Metric | Description | Preference |
|---|---|---|
| **MSE** | Mean Squared Error — penalizes large errors heavily | Lower is better |
| **MAE** | Mean Absolute Error — average magnitude of errors | Lower is better |
| **RMSE** | Root Mean Squared Error — interpretable in target units | Lower is better |
| **R²** | Coefficient of Determination — proportion of variance explained | Higher is better |
| **Train Time** | Wall-clock time to fit the model (seconds) | Lower is better |

The eight models evaluated are:
1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Decision Tree Regressor
5. Random Forest Regressor
6. Gradient Boosting Regressor
7. XGBoost Regressor
8. K-Nearest Neighbors (KNN)

### Step 5: TOPSIS Multi-Criteria Ranking

**TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)** is a multi-criteria decision analysis method that ranks alternatives by their geometric distance from an ideal best solution and an ideal worst solution. A model with the **highest TOPSIS score** is closest to the ideal best and furthest from the ideal worst.

#### TOPSIS Algorithm Steps:
1. **Normalize** the decision matrix so all criteria are on a comparable scale.
2. **Apply weights** to each criterion to reflect their relative importance.
3. Determine the **Ideal Best** (best value for each criterion) and **Ideal Worst** (worst value for each criterion).
4. Calculate the **Euclidean distance** of each model from the ideal best and ideal worst.
5. Compute the **TOPSIS Score** = `dist_worst / (dist_best + dist_worst)`.
6. **Rank** models in descending order of their TOPSIS score.

#### Weights & Impacts Used:

| Criterion | Weight | Impact | Rationale |
|---|---|---|---|
| MSE | 0.20 | − (lower is better) | Penalizes large prediction errors |
| MAE | 0.15 | − (lower is better) | Measures average error magnitude |
| RMSE | 0.20 | − (lower is better) | Error in target variable units |
| R² | 0.35 | + (higher is better) | Most important — explained variance |
| Train Time | 0.10 | − (lower is better) | Efficiency consideration |

---

##  Results

### Model Evaluation Metrics

| Model | MSE | MAE | RMSE | R² | Train Time (s) |
|---|---|---|---|---|---|
| Linear Regression | 93.9075 | 7.0874 | 9.6906 | 0.000186 | 0.002228 |
| Ridge | 93.8719 | 7.0899 | 9.6888 | 0.000565 | 0.001825 |
| Lasso | 93.9260 | 7.1970 | 9.6915 | -0.000010 | 0.001827 |
| Decision Tree | 141.0750 | 6.6550 | 11.8775 | -0.501997 | 0.006959 |
| Random Forest | 68.2613 | 5.1044 | 8.2620 | 0.273235 | 0.494874 |
| Gradient Boosting | 68.2644 | 5.3992 | 8.2622 | 0.273203 | 0.242785 |
| **XGBoost** | **69.4451** | **5.3942** | **8.3334** | **0.260633** | **0.118432** |
| KNN | 92.8428 | 6.6700 | 9.6355 | 0.011522 | 0.004623 |

### TOPSIS Ranking

| Rank | Model | MSE | MAE | RMSE | R² | Train Time | TOPSIS Score |
|---|---|---|---|---|---|---|---|
|  1 | **XGBoost** | 69.445 | 5.394 | 8.333 | 0.2606 | 0.1184 | **0.9483** |
|  2 | Gradient Boosting | 68.264 | 5.399 | 8.262 | 0.2732 | 0.2428 | 0.9041 |
|  3 | Random Forest | 68.261 | 5.104 | 8.262 | 0.2732 | 0.4949 | 0.8210 |
| 4 | KNN | 92.843 | 6.670 | 9.635 | 0.0115 | 0.0046 | 0.6725 |
| 5 | Ridge | 93.872 | 7.090 | 9.689 | 0.0006 | 0.0018 | 0.6588 |
| 6 | Linear Regression | 93.908 | 7.087 | 9.691 | 0.0002 | 0.0022 | 0.6582 |
| 7 | Lasso | 93.926 | 7.197 | 9.692 | -0.0000 | 0.0018 | 0.6578 |
| 8 | Decision Tree | 141.075 | 6.655 | 11.878 | -0.5020 | 0.0070 | 0.1777 |

###  TOPSIS Score Visualization

The bar chart below shows TOPSIS scores for all 8 models in descending order. A higher bar indicates a better overall model according to the multi-criteria evaluation.

```
TOPSIS Score (Higher = Better)
┌─────────────────────────────────────────────────────────────┐
│ XGBoost          ████████████████████████████████  0.9483   │
│ Gradient Boost.  ██████████████████████████████    0.9041   │
│ Random Forest    ████████████████████████████      0.8210   │
│ KNN              ████████████████████████          0.6725   │
│ Ridge            ██████████████████████            0.6588   │
│ Linear Regr.     ██████████████████████            0.6582   │
│ Lasso            ██████████████████████            0.6578   │
│ Decision Tree    ██████                            0.1777   │
└─────────────────────────────────────────────────────────────┘
```

>  **Best Model According to TOPSIS: XGBoost** (Score: 0.9483)

---

##  Key Findings & Discussion

**Ensemble methods dominate:** XGBoost, Gradient Boosting, and Random Forest occupy the top 3 positions. These models benefit from combining multiple weak learners and are well-suited to tabular regression tasks.

**XGBoost wins by TOPSIS despite not having the absolute best individual metrics:** While Random Forest and Gradient Boosting achieve slightly lower MSE and RMSE, XGBoost's combination of competitive accuracy *and* significantly faster training time (0.118s vs. 0.243s–0.495s) gives it an edge when all criteria are weighted together.

**Linear models perform poorly:** Linear Regression, Ridge, and Lasso all yield near-zero R² scores (~0.0002–0.0006), indicating they cannot capture the non-linear relationships between CartPole state parameters and cumulative reward.

**Decision Tree overfits:** The Decision Tree achieves the worst performance overall with an R² of -0.502 and the highest MSE (141.07), suggesting it overfits the training data and generalises poorly, even compared to linear models.

**Low R² across the board:** The highest R² achieved is ~0.27 (Random Forest / Gradient Boosting). This is expected because the target (`reward`) is generated by a **random action policy**, introducing high irreducible noise. Predicting cumulative reward from initial state alone is inherently difficult without a learned control policy.

---

## How to Run

### 1. Clone / Download the notebook

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Install Dependencies

```bash
pip install gymnasium gymnasium[classic-control] scikit-learn xgboost matplotlib pandas numpy
```

### 3. Run the Notebook

Open and run all cells in `Data_generation_using_Modelling_and_Simulation.ipynb` using Jupyter Notebook, JupyterLab, or Google Colab.

### 4. Outputs

After execution, the following files are generated:
- `simulation_data.csv` — 1000-row dataset from CartPole-v1 simulations
- `model_ranking_topsis.csv` — Full TOPSIS ranking table
- TOPSIS bar chart displayed inline

---

##  References

- [OpenAI Gymnasium — CartPole-v1](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
- [TOPSIS Method — Wikipedia](https://en.wikipedia.org/wiki/TOPSIS)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

---

##  Author

*Generated as part of a Data Science / AI coursework project on synthetic data generation and model evaluation using simulation-based approaches.*
