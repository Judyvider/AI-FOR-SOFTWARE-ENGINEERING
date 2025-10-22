# ==============================================================================
# PROJECT TITAN: Adaptive Carbon Policy Optimization
# Python Notebook Simulation Script (SUPERVISED LEARNING - RANDOM FOREST CLASSIFIER)
#
# This script implements a Supervised Learning approach to classify the optimal 
# policy action based on environment state variables (CI, RS, GR).
# ==============================================================================

import numpy as np
import random
import time
import matplotlib.pyplot as plt
import requests 

# Import Supervised Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# --- I. Environment Setup (Simplified for Action Space Definition) ---

class ClimateEnv:
    """
    A simplified environment to define the available action space.
    The step() and reset() methods are no longer used for training in 
    the Supervised Learning paradigm, but are kept for context.
    """
    def __init__(self):
        # State components (CI, RS, GR) are used as features for the classifier
        
        # Action Space (A): Discrete set of budget splits (A1: Renewable, A2: Carbon Pricing)
        # Actions represent the percentage of the budget allocated to A1 (Renewable Acceleration).
        # These are the classes (labels) the classifier must predict.
        self.action_space = [
            0.1,  # Action 0: 10% A1 (Aggressive Carbon Pricing)
            0.3,  # Action 1: 30% A1
            0.5,  # Action 2: 50% A1 (Balanced)
            0.7,  # Action 3: 70% A1
            0.9,  # Action 4: 90% A1 (Aggressive Renewable Acceleration)
        ]
        self.n_actions = len(self.action_space)

    def reset(self):
        # Retained for structure, but not used in training
        return np.array([50.0, 20.0, 3.0])

    def step(self, action_index):
        # Retained for structure, but not used in training
        return None, 0, False, {}

# --- II. Data Generation (Creating the Labeled Dataset) ---

def generate_synthetic_data(num_samples, action_space):
    """
    Creates a synthetic dataset of (State Features, Optimal Action Label).
    The 'Optimal Action' is determined by simple, policy-based rules, 
    simulating the desired output of a highly optimized RL agent.
    """
    X = [] # Features: [Carbon_Intensity, Renewable_Share, Economic_Growth]
    y = [] # Labels: [Optimal_Action_Index] (0 to 4)

    # State bounds for data generation
    ci_range = (10, 100)
    rs_range = (5, 95)
    eg_range = (-5, 10)

    for _ in range(num_samples):
        # Generate random state features within reasonable bounds
        ci = random.uniform(*ci_range)
        rs = random.uniform(*rs_range)
        eg = random.uniform(*eg_range)
        
        # Determine the Optimal Action (Label) based on policy heuristics:
        
        optimal_action_index = 2  # Default to 50/50 split (Action 2)

        # High Carbon Intensity (CI > 65) requires aggressive reduction
        if ci > 65:
            if rs < 40: # If renewables are low, push A1 aggressively
                optimal_action_index = 4 # 90% A1
            else: # If renewables are moderate, balance pricing
                optimal_action_index = 3 # 70% A1
        
        # Low Carbon Intensity (CI < 30) allows shift to efficiency/stability
        elif ci < 30:
            if eg < 0: # If growth is negative, prioritize carbon pricing (A2) to save funds
                optimal_action_index = 0 # 10% A1 (90% A2)
            else: # If growth is good, maintain tech funding
                optimal_action_index = 1 # 30% A1
        
        # Medium Carbon Intensity (30 <= CI <= 65) - Balance factors
        else: 
            if rs > 70: # High renewables, focus on efficiency
                optimal_action_index = 1
            elif eg < 1.0: # Stagnant growth, minimize budget impact with A2
                optimal_action_index = 1
            else: # Good growth, push moderate A1
                optimal_action_index = 3

        X.append([ci, rs, eg])
        y.append(optimal_action_index)

    return np.array(X), np.array(y)


# --- III. Supervised Learning Model Training ---

# 1. Setup Environment Context
env = ClimateEnv()
NUM_SAMPLES = 5000 

print(f"# Project Titan Supervised Training Initialized.")

# 2. Generate and Prepare Data
X, y = generate_synthetic_data(NUM_SAMPLES, env.action_space)
print(f"  Generated {len(X)} synthetic data points (Features: CI, RS, GR).")

# 3. Split Data for Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"  Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

# 4. Initialize and Train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
print("  Training Random Forest Classifier...")
start_time = time.time()
model.fit(X_train, y_train)
end_time = time.time()
print(f"  Training Complete! Total time: {end_time - start_time:.2f} seconds.")

# --- IV. Model Evaluation ---

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\n" + "="*50)
print(f"Model Evaluation (Random Forest Classifier)")
print(f"Accuracy on Test Data: {accuracy:.4f}")
print("="*50)

# Optional: Print classification report for deeper analysis
# print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=[f'{i*100:.0f}% A1' for i in env.action_space]))


# --- V. Policy Advisor & Prediction (Interactive Function) ---

def generate_advice_text(A1_share):
    """Generates detailed policy interpretation based on the recommended budget split."""
    A1_percent = A1_share * 100
    
    if A1_percent >= 90:
        return "The model recommends an **extremely aggressive investment in Renewable Acceleration (A1)**. This is typical when Carbon Intensity is very high, but economic growth is robust enough to absorb the pricing impact. Prioritize green tech deployment and infrastructure immediately."
    elif A1_percent >= 70:
        return "The optimal policy favors a **strong emphasis on Renewables Acceleration (A1)**. Maintain strong Carbon Pricing (A2) but allocate the majority of the budget to directly drive down emissions through technology and infrastructure deployment."
    elif A1_percent == 50:
        return "The model suggests a **perfectly balanced approach (50/50)**. This usually indicates a medium-risk, medium-reward state where both incentives (A1) and market mechanisms (A2) are equally important to achieve progress without destabilizing the economy."
    elif A1_percent >= 30:
        return "The optimal policy leans toward **strong Carbon Pricing and Efficiency (A2)**. This is often recommended when Renewable Share (RS) is already moderate, or when Economic Growth (EG) is weak, as A2 is generally more cost-effective in the short term."
    else: # 10% A1
        return "The model strongly advocates for an **Aggressive Carbon Pricing/Efficiency policy (A2)**. This usually happens in states where economic stability is paramount (low/negative EG) or when the current Renewable Share is sufficient, making efficiency and pricing the most rewarding next step."

def get_policy_advice(scenario_name, carbon_intensity, renewable_share, econ_growth):
    """
    Simulates the interactive search: takes a state and returns the ML prediction 
    and detailed advice based on the trained Random Forest model.
    """
    # Prepare input for the model: [CI, RS, GR]
    input_state = np.array([carbon_intensity, renewable_share, econ_growth]).reshape(1, -1)
    
    # 1. Model Prediction
    try:
        action_index = model.predict(input_state)[0]
    except Exception as e:
        print(f"🛑 Error during prediction: {e}")
        return

    # 2. Map Index to Action Share
    A1_share = env.action_space[action_index]
    A2_share = 1.0 - A1_share
    
    # 3. Generate Advice
    advice = generate_advice_text(A1_share)

    # 4. Print Results
    print("\n" + "="*80)
    print(f"🔮 POLICY ADVICE FOR SCENARIO: {scenario_name}")
    print(f"   Input State (CI, RS, GR): [Carbon Intensity: {carbon_intensity:.1f}, Renewable Share: {renewable_share:.1f}%, Economic Growth: {econ_growth:.1f}%]")
    print("-" * 80)
    print(f"   Optimal Budget Split Recommendation:")
    print(f"     > A1 (Renewables Acceleration): **{A1_share * 100:.0f}%**")
    print(f"     > A2 (Carbon Pricing/Efficiency): **{A2_share * 100:.0f}%**")
    print("\n   Policy Interpretation:")
    print(f"     {advice}")
    print("="*80)

# --- VI. Real-Time Data and Policy Advisor Demo ---

def simulate_real_time_fetch(url="https://api.example.com/climate_data", max_retries=5):
    """
    Simulates fetching real-time data with exponential backoff (for deployment planning).
    """
    print("  --- Simulating Real-Time API Fetch ---")
    # This section remains conceptual for deployment planning
    mock_data = {
        'temperature_anomaly': random.uniform(-2.0, 2.0),
        'gdp_growth_forecast': random.uniform(1.0, 5.0)
    }
    print("  ✅ Simulated data fetch successful.")
    return mock_data 

print("\n\n--- Interactive Policy Scenario Testing (Call these functions to 'search' for policy advice) ---")
simulate_real_time_fetch()


# Scenario 1: High Carbon, Low Renewables, Medium Growth (Needs strong action)
get_policy_advice(
    scenario_name="2025: High Risk State",
    carbon_intensity=75.0,
    renewable_share=25.0,
    econ_growth=3.0
)

# Scenario 2: Medium Carbon, Medium Renewables, Low/Negative Growth (Needs cautious action)
get_policy_advice(
    scenario_name="2035: Green Stagnation Risk",
    carbon_intensity=45.0,
    renewable_share=55.0,
    econ_growth=0.5
)

# Scenario 3: Low Carbon, High Renewables, High Growth (Focus shifts to maintenance/efficiency)
get_policy_advice(
    scenario_name="2045: Target Maintenance",
    carbon_intensity=25.0,
    renewable_share=80.0,
    econ_growth=6.0
)
