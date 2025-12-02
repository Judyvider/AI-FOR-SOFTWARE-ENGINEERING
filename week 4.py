import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def run_bias_mitigation_demo():
    print("--- Part 3: Ethical Reflection - Bias Mitigation Implementation ---\n")

    # 1. Load & Prep Data (Same as Task 3)
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    # Target: 1 = High Priority (Malignant), 0 = Low Priority (Benign)
    df['priority_target'] = np.where(data.target == 0, 1, 0)

    # 2. Simulate a "Protected Attribute" (Bias Injection)
    # Let's imagine the data comes from two teams: 'Team_Legacy' (0) and 'Team_Frontend' (1).
    # We will simulate a scenario where 'Team_Frontend' tickets are historically biased 
    # to be labeled 'Low Priority' (0) more often, even if they share similar features.
    
    np.random.seed(42)
    # Randomly assign teams
    df['team_id'] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])
    
    # INJECT BIAS: Artificially downgrade priority for Team 1 (Frontend)
    # If Team is 1 and originally High Priority, flip it to Low Priority 30% of the time.
    mask = (df['team_id'] == 1) & (df['priority_target'] == 1)
    flip_indices = df[mask].sample(frac=0.3, random_state=42).index
    df.loc[flip_indices, 'priority_target'] = 0

    print(f"Dataset enriched with protected attribute 'team_id'.")
    print("Simulated Bias: Team 1 (Frontend) tickets artificially downgraded.\n")

    # 3. Measure Bias (Disparate Impact)
    # Disparate Impact = P(High Priority | Team 1) / P(High Priority | Team 0)
    
    def calculate_disparate_impact(dataframe, protected_col, target_col):
        # Probability of getting a positive outcome (High Priority) for unprivileged group (Team 1)
        prob_unprivileged = dataframe[dataframe[protected_col] == 1][target_col].mean()
        # Probability of getting a positive outcome for privileged group (Team 0)
        prob_privileged = dataframe[dataframe[protected_col] == 0][target_col].mean()
        
        return prob_unprivileged / prob_privileged

    di_before = calculate_disparate_impact(df, 'team_id', 'priority_target')
    print(f"Disparate Impact (Before Mitigation): {di_before:.4f}")
    print("(A value < 0.8 usually indicates significant bias against the unprivileged group)\n")

    # 4. Mitigation Strategy: Reweighing (Pre-processing)
    # We calculate weights so that the training process pays more attention to 
    # underrepresented positive examples (Team 1 + High Priority).
    
    # Count frequencies
    n = len(df)
    n_privileged = len(df[df['team_id'] == 0])
    n_unprivileged = len(df[df['team_id'] == 1])
    n_positive = len(df[df['priority_target'] == 1])
    n_negative = len(df[df['priority_target'] == 0])
    
    # Calculate Expected Probability if independent
    # e.g. P(Team 1, High Priority) should be P(Team 1) * P(High Priority)
    
    # For demonstration, we will implement a simplified weight calculation for
    # the specific group: Team 1 (Unprivileged) with Outcome 1 (High Priority)
    
    # Expected count for (Team 1, High Priority)
    expected_pos_unpriv = (n_unprivileged * n_positive) / n
    # Actual count
    actual_pos_unpriv = len(df[(df['team_id'] == 1) & (df['priority_target'] == 1)])
    
    # Weight = Expected / Actual
    weight_pos_unpriv = expected_pos_unpriv / actual_pos_unpriv
    
    print(f"Calculated Re-weighting Factor for (Team 1, High Priority): {weight_pos_unpriv:.4f}")
    print("This weight > 1.0 means we up-weight these samples during training.\n")

    # 5. Train Model with Weights
    X = df.drop('priority_target', axis=1)
    y = df['priority_target']
    
    # Assign weights: default to 1.0, apply calculated weight to specific group
    sample_weights = np.ones(len(df))
    mask_target_group = (df['team_id'] == 1) & (df['priority_target'] == 1)
    sample_weights[mask_target_group] = weight_pos_unpriv

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y, sample_weight=sample_weights)
    
    print("Model trained with Fairness Weights.")
    print("In a full AIF360 pipeline, we would now re-evaluate Disparate Impact on predictions.")

if __name__ == "__main__":
    run_bias_mitigation_demo()