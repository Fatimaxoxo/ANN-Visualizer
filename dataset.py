"""
This file is meant to be imported by Erin's main.py during integration:
    from dataset import load_student_dataset
    X, y, feature_names = load_student_dataset()
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_student_dataset(filename='student_dataset.csv'):
    """
    Loads the student dataset from a CSV file.

    Args:
        filename (str): Path to the CSV file. Defaults to 'student_dataset.csv'.

    Returns:
        X (numpy array): Shape (10, 3) -> the 3 input features
                          (Study_Hours, Assignments, Classes_Missed)
        y (numpy array): Shape (10,)   -> the target variable (1 = Pass, 0 = Fail)
        feature_names (list): Column names for the 3 features, in order.
                               Isbat needs this to label the input circles
                               in the diagram (e.g. "Study_Hours" instead of "X1").
    """

    # 1. Load the CSV file using Pandas
    df = pd.read_csv(filename)

    # 2. Separate the Input Features from the Target
    features = df.drop(columns=['Target_Pass'])
    target = df['Target_Pass']

    # 3. Keep the feature column names before converting to NumPy.
    feature_names = features.columns.tolist()

    # 4. Convert Pandas DataFrames to NumPy Arrays.
    X = features.to_numpy()
    y = target.to_numpy()

    # 5. Print a quick summary
    print("Dataset successfully loaded!")
    print(f"Total Samples: {len(X)}")
    print(f"Features per sample: {X.shape[1]}")
    print(f"Feature names: {feature_names}")
    print("-" * 30)

    return X, y, feature_names


class MLPClassifier:
    """
    Complete MLP Classifier with Forward Pass, Loss Calculation, and Backpropagation.

    The code (and the printed output) always follows the neuron flow in order:

        FORWARD  (left  -> right) :  INPUT -> Z1 -> A1 -> Z2 -> A2 -> LOSS (L)
        BACKWARD (right -> left)  :  LOSS (L) -> dA2 -> dZ2 -> dA1 -> dZ1 -> INPUT

    Hidden Layer Activation: ReLU
    Output Layer Activation: Sigmoid (needed for a 0-1 probability output)
    """

    def __init__(self, input_size=3, hidden_size=4, output_size=1, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights with small random values
        np.random.seed(42)  # For reproducibility
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

        # Store intermediate values for visualization
        self.forward_cache = {}
        self.backward_cache = {}

    # ------------------------------------------------------------------
    # Activation functions
    # ------------------------------------------------------------------
    def relu(self, z):
        """ReLU activation function: max(0, z)"""
        return np.maximum(0, z)

    def relu_derivative(self, z):
        """Derivative of ReLU: 1 where z > 0, else 0."""
        return (z > 0).astype(float)

    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def sigmoid_derivative(self, z):
        """Derivative of sigmoid function"""
        s = self.sigmoid(z)
        return s * (1 - s)

    def binary_crossentropy_loss(self, y_true, y_pred):
        """
        Binary cross-entropy loss with partial derivatives

        Loss = -[y * log(y_pred) + (1-y) * log(1-y_pred)]
        dL/dy_pred = -(y/y_pred) + (1-y)/(1-y_pred)
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        dL_dy_pred = -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)
        return loss, dL_dy_pred

    # ==================================================================
    # FORWARD PASS   -->  ORDER:  INPUT -> Z1 -> A1 -> Z2 -> A2 -> LOSS
    # ==================================================================
    def forward_pass(self, X, y_true=None, verbose=True):
        """
        Forward pass through the network with detailed calculations,
        following the neuron flow in strict order:

            INPUT  ->  Z1 (hidden pre-activation)  ->  A1 (ReLU)
                   ->  Z2 (output pre-activation)   ->  A2 (Sigmoid)
                   ->  L  (loss, if y_true is given)
        """
        if verbose:
            print("\n" + "=" * 70)
            print("FORWARD PASS - Detailed Calculations  (INPUT -> Z1 -> A1 -> Z2 -> A2 -> L)")
            print("=" * 70)

        # ---------------- STEP 1: INPUT ----------------
        if verbose:
            print(f"\n[STEP 1: INPUT]")
            print(f"Input shape: {X.shape}")
            print(f"Input values: {X.flatten()}")

        # ---------------- STEP 2: HIDDEN LAYER (Z1, A1) ----------------
        # Z1 = X . W1 + b1
        z1 = np.dot(X, self.W1) + self.b1
        # A1 = ReLU(Z1)
        a1 = self.relu(z1)

        if verbose:
            print(f"\n[STEP 2: HIDDEN LAYER  (activation: ReLU)]")
            print(f"Weight matrix W1 shape: {self.W1.shape}")
            print(f"W1:\n{self.W1}")
            print(f"Bias b1: {self.b1.flatten()}")
            print(f"\nZ1 = X.W1 + b1:")
            print(f"Z1 shape: {z1.shape}")
            print(f"Z1 values: {z1.flatten()}")
            print(f"\nA1 = ReLU(Z1) = max(0, Z1):")
            print(f"A1 values: {a1.flatten()}")

        # ---------------- STEP 3: OUTPUT LAYER (Z2, A2) ----------------
        # Z2 = A1 . W2 + b2
        z2 = np.dot(a1, self.W2) + self.b2
        # A2 = sigmoid(Z2)
        a2 = self.sigmoid(z2)

        if verbose:
            print(f"\n[STEP 3: OUTPUT LAYER  (activation: Sigmoid)]")
            print(f"Weight matrix W2 shape: {self.W2.shape}")
            print(f"W2:\n{self.W2}")
            print(f"Bias b2: {self.b2.flatten()}")
            print(f"\nZ2 = A1.W2 + b2:")
            print(f"Z2 shape: {z2.shape}")
            print(f"Z2 values: {z2.flatten()}")
            print(f"\nA2 = sigmoid(Z2):")
            print(f"A2 values: {a2.flatten()}")
            print(f"Final prediction: {a2.flatten()[0]:.4f}")

        # ---------------- STEP 4: LOSS (L) ----------------
        loss = None
        if y_true is not None:
            loss, _ = self.binary_crossentropy_loss(y_true, a2)
            if verbose:
                print(f"\n[STEP 4: LOSS CALCULATION (L)]")
                print(f"True label: {y_true.flatten()[0]}")
                print(f"Predicted (A2): {a2.flatten()[0]:.4f}")
                print(f"Binary Cross-Entropy Loss (L): {loss:.6f}")

        # Cache everything needed for the backward pass and visualizations
        self.forward_cache = {
            'X': X,
            'z1': z1,
            'a1': a1,
            'z2': z2,
            'a2': a2,
            'loss': loss,
            'y_true': y_true
        }

        return a2

    # ==================================================================
    # BACKWARD PASS  -->  ORDER:  LOSS -> dA2 -> dZ2 -> dA1 -> dZ1
    #                     (mirror of forward, walked in reverse)
    # ==================================================================
    def backward_pass(self, y_true, verbose=True):
        """
        Backpropagation using chain rule, walking the SAME stages as the
        forward pass but in reverse:

            L  ->  dL/dA2  ->  dL/dZ2  ->  dL/dW2, dL/db2
               ->  dL/dA1  ->  dL/dZ1  ->  dL/dW1, dL/db1
        """
        if verbose:
            print("\n" + "=" * 70)
            print("BACKPROPAGATION - Chain Rule Calculations  (L -> dA2 -> dZ2 -> dA1 -> dZ1)")
            print("=" * 70)

        # Retrieve cached values from the forward pass
        X = self.forward_cache['X']
        z1 = self.forward_cache['z1']
        a1 = self.forward_cache['a1']
        z2 = self.forward_cache['z2']
        a2 = self.forward_cache['a2']

        # ---------------- STEP 1: LOSS (L) and dL/dA2 ----------------
        loss, dL_da2 = self.binary_crossentropy_loss(y_true, a2)

        if verbose:
            print(f"\n[STEP 1: LOSS (L) and dL/dA2]")
            print(f"True label: {y_true.flatten()[0]}")
            print(f"Predicted (A2): {a2.flatten()[0]:.4f}")
            print(f"Loss (L): {loss:.6f}")
            print(f"dL/dA2: {dL_da2.flatten()[0]:.6f}")

        # ---------------- STEP 2: OUTPUT LAYER (dZ2, dW2, db2) ----------------
        # dL/dZ2 = dL/dA2 * dA2/dZ2   (dA2/dZ2 = sigmoid derivative)
        da2_dz2 = self.sigmoid_derivative(z2)
        dL_dz2 = dL_da2 * da2_dz2

        # dL/dW2 = A1^T . dL/dZ2
        dL_dW2 = np.dot(a1.T, dL_dz2)
        # dL/db2 = sum(dL/dZ2)
        dL_db2 = np.sum(dL_dz2, axis=0, keepdims=True)

        if verbose:
            print(f"\n[STEP 2: OUTPUT LAYER GRADIENTS (dZ2, dW2, db2)]")
            print(f"dA2/dZ2 (sigmoid derivative): {da2_dz2.flatten()[0]:.6f}")
            print(f"dL/dZ2 = dL/dA2 * dA2/dZ2: {dL_dz2.flatten()[0]:.6f}")
            print(f"\ndL/dW2 = A1^T . dL/dZ2:")
            print(f"dL/dW2 shape: {dL_dW2.shape}")
            print(f"dL/dW2:\n{dL_dW2}")
            print(f"\ndL/db2 = sum(dL/dZ2): {dL_db2.flatten()}")

        # ---------------- STEP 3: HIDDEN LAYER (dA1, dZ1, dW1, db1) ----------------
        # dL/dA1 = dL/dZ2 . W2^T
        dL_da1 = np.dot(dL_dz2, self.W2.T)
        # dL/dZ1 = dL/dA1 * dA1/dZ1   (dA1/dZ1 = ReLU derivative)
        da1_dz1 = self.relu_derivative(z1)
        dL_dz1 = dL_da1 * da1_dz1

        # dL/dW1 = X^T . dL/dZ1
        dL_dW1 = np.dot(X.T, dL_dz1)
        # dL/db1 = sum(dL/dZ1)
        dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)

        if verbose:
            print(f"\n[STEP 3: HIDDEN LAYER GRADIENTS (dA1, dZ1, dW1, db1)]")
            print(f"dL/dA1 = dL/dZ2 . W2^T:")
            print(f"dL/dA1 shape: {dL_da1.shape}")
            print(f"dL/dA1 values: {dL_da1.flatten()}")
            print(f"\ndA1/dZ1 (ReLU derivative): {da1_dz1.flatten()}")
            print(f"dL/dZ1 = dL/dA1 * dA1/dZ1: {dL_dz1.flatten()}")
            print(f"\ndL/dW1 = X^T . dL/dZ1:")
            print(f"dL/dW1 shape: {dL_dW1.shape}")
            print(f"dL/dW1:\n{dL_dW1}")
            print(f"\ndL/db1 = sum(dL/dZ1): {dL_db1.flatten()}")

        # Cache gradients (used by visualize_backward_network)
        self.backward_cache = {
            'dL_dW1': dL_dW1,
            'dL_db1': dL_db1,
            'dL_dW2': dL_dW2,
            'dL_db2': dL_db2,
            'loss': loss,
            'y_true': y_true
        }

        return dL_dW1, dL_db1, dL_dW2, dL_db2, loss

    # ------------------------------------------------------------------
    def update_weights(self, dL_dW1, dL_db1, dL_dW2, dL_db2, verbose=True):
        """
        Update weights using gradient descent
        W_new = W_old - learning_rate * gradient
        """
        if verbose:
            print("\n" + "=" * 70)
            print("WEIGHT UPDATE - Gradient Descent")
            print("=" * 70)
            print(f"Learning rate: {self.learning_rate}")

        self.W1 -= self.learning_rate * dL_dW1
        self.b1 -= self.learning_rate * dL_db1
        self.W2 -= self.learning_rate * dL_dW2
        self.b2 -= self.learning_rate * dL_db2

        if verbose:
            print(f"\nUpdated W1:\n{self.W1}")
            print(f"Updated b1: {self.b1.flatten()}")
            print(f"\nUpdated W2:\n{self.W2}")
            print(f"Updated b2: {self.b2.flatten()}")

    def train_step(self, X, y, verbose=True):
        """Perform one training step: forward (INPUT->L), backward (L->dZ1), update"""
        y_pred = self.forward_pass(X, y_true=y, verbose=verbose)
        dL_dW1, dL_db1, dL_dW2, dL_db2, loss = self.backward_pass(y, verbose)
        self.update_weights(dL_dW1, dL_db1, dL_dW2, dL_db2, verbose)
        return loss, y_pred

    # ==================================================================
    # VISUALIZATION - FORWARD FLOW (INPUT -> HIDDEN -> OUTPUT -> LOSS)
    # ==================================================================
    def visualize_network(self, feature_names, sample_idx=0):
        """
        Visualize the MLP FORWARD pass: input -> hidden -> output -> loss.
        Shows weights, biases, activations and the loss (if available).
        Colors are bold/saturated and edges thicker for presentation use.
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title('MLP Network - FORWARD PASS  (INPUT -> Z1/A1 -> Z2/A2 -> LOSS)',
                      fontsize=16, fontweight='bold', pad=20)

        input_x, hidden_x, output_x = 1, 5, 9

        X = self.forward_cache.get('X', np.zeros((1, self.input_size)))
        a1 = self.forward_cache.get('a1', np.zeros((1, self.hidden_size)))
        a2 = self.forward_cache.get('a2', np.zeros((1, self.output_size)))
        z1 = self.forward_cache.get('z1', np.zeros((1, self.hidden_size)))
        z2 = self.forward_cache.get('z2', np.zeros((1, self.output_size)))
        loss = self.forward_cache.get('loss', None)

        input_y = np.linspace(2, 8, self.input_size)
        hidden_y = np.linspace(1.5, 8.5, self.hidden_size)
        output_y = [5]

        POS_COLOR = '#006400'   # bold dark green
        NEG_COLOR = '#B22222'   # bold firebrick red

        # ---- Input -> Hidden connections (weights W1) ----
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                weight = self.W1[i, j]
                color = POS_COLOR if weight > 0 else NEG_COLOR
                lw = 1.5 + min(abs(weight) * 6, 4)   # bolder, more visible line width
                ax.plot([input_x + 0.35, hidden_x - 0.35],
                        [input_y[i], hidden_y[j]],
                        color=color, alpha=0.85, linewidth=lw)
                ax.text((input_x + hidden_x) / 2,
                        (input_y[i] + hidden_y[j]) / 2,
                        f'{weight:.2f}', fontsize=6.5, ha='center',
                        fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7, edgecolor='none'))

        # ---- Hidden -> Output connections (weights W2) ----
        for j in range(self.hidden_size):
            weight = self.W2[j, 0]
            color = POS_COLOR if weight > 0 else NEG_COLOR
            lw = 1.5 + min(abs(weight) * 6, 4)
            ax.plot([hidden_x + 0.35, output_x - 0.35],
                    [hidden_y[j], output_y[0]],
                    color=color, alpha=0.85, linewidth=lw)
            ax.text((hidden_x + output_x) / 2,
                    (hidden_y[j] + output_y[0]) / 2,
                    f'{weight:.2f}', fontsize=6.5, ha='center',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7, edgecolor='none'))

        # ---- Input Layer Nodes ----
        for i in range(self.input_size):
            circle = plt.Circle((input_x, input_y[i]), 0.42,
                                 facecolor='#ADD8E6', edgecolor='#00008B', linewidth=3)
            ax.add_patch(circle)
            label = feature_names[i] if i < len(feature_names) else f'X{i+1}'
            ax.text(input_x - 0.85, input_y[i], label,
                    fontsize=9.5, ha='right', va='center', fontweight='bold', color='#00008B')
            ax.text(input_x, input_y[i], f'{X[0, i]:.2f}',
                    fontsize=8.5, ha='center', va='center', fontweight='bold')

        # ---- Hidden Layer Nodes (show activation A1 AND bias b1) ----
        for j in range(self.hidden_size):
            circle = plt.Circle((hidden_x, hidden_y[j]), 0.42,
                                 facecolor='#90EE90', edgecolor='#004d00', linewidth=3)
            ax.add_patch(circle)
            ax.text(hidden_x, hidden_y[j], f'H{j+1}\nZ1={z1[0, j]:.3f}\nA1={a1[0, j]:.3f}',
                    fontsize=7.5, ha='center', va='center', fontweight='bold')
            # Bias value printed just below/beside the node
            ax.text(hidden_x, hidden_y[j] - 0.62, f'b1={self.b1.flatten()[j]:.3f}',
                    fontsize=7.5, ha='center', va='center', color='#8B4500', fontweight='bold')

        # ---- Output Layer Node (show activation A2 AND bias b2) ----
        circle = plt.Circle((output_x, output_y[0]), 0.42,
                             facecolor='#FFD700', edgecolor='#8B4500', linewidth=3)
        ax.add_patch(circle)
        ax.text(output_x, output_y[0], f'Output\nZ2={z2[0, 0]:.3f}\nA2={a2[0, 0]:.3f}',
                fontsize=8, ha='center', va='center', fontweight='bold')
        ax.text(output_x, output_y[0] - 0.75, f'b2={self.b2.flatten()[0]:.3f}',
                fontsize=8, ha='center', va='center', color='#8B4500', fontweight='bold')

        # ---- Loss display (if available) ----
        if loss is not None:
            ax.text(output_x, output_y[0] - 1.6, f'LOSS (L) = {loss:.4f}',
                    fontsize=11, ha='center', va='center', fontweight='bold',
                    color='white', bbox=dict(boxstyle='round,pad=0.3', facecolor='#B22222', edgecolor='black', linewidth=1.5))

        # Layer labels
        ax.text(input_x, 9.5, 'Input Layer', fontsize=12, ha='center', fontweight='bold')
        ax.text(hidden_x, 9.5, 'Hidden Layer (ReLU)', fontsize=12, ha='center', fontweight='bold')
        ax.text(output_x, 9.5, 'Output Layer (Sigmoid)', fontsize=12, ha='center', fontweight='bold')

        ax.text(5, 0.4, 'Dark Green edge: Positive weight   |   Firebrick Red edge: Negative weight',
                fontsize=9.5, ha='center', style='italic', fontweight='bold')

        plt.tight_layout()
        plt.savefig('mlp_forward_visualization.png', dpi=150, bbox_inches='tight')
        print("\nForward pass visualization saved as 'mlp_forward_visualization.png'")
        plt.show()

    # ==================================================================
    # VISUALIZATION - BACKWARD FLOW (LOSS -> OUTPUT -> HIDDEN -> INPUT)
    # ==================================================================
    def visualize_backward_network(self, feature_names):
        """
        Visualize the MLP BACKWARD pass (backpropagation): gradients flowing
        from the loss back toward the input. Mirrors visualize_network's
        layout, but:
          - arrows point RIGHT -> LEFT (output -> hidden -> input), showing
            the true direction gradients travel during backprop
          - edges are colored/labeled by GRADIENT values (dL/dW), not weights
          - bias GRADIENTS (dL/db1, dL/db2) are shown at each node
          - the loss (L) that started the backward pass is shown prominently
          - colors are bold/saturated with thick edges for presenting
        """
        if not self.backward_cache:
            print("No backward pass has been run yet - call backward_pass() or train_step() first.")
            return

        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        ax.set_title('MLP Network - BACKWARD PASS  (LOSS -> dA2/dZ2 -> dA1/dZ1 -> INPUT)',
                      fontsize=16, fontweight='bold', pad=20)

        input_x, hidden_x, output_x = 1, 5, 9

        dL_dW1 = self.backward_cache['dL_dW1']
        dL_db1 = self.backward_cache['dL_db1'].flatten()
        dL_dW2 = self.backward_cache['dL_dW2']
        dL_db2 = self.backward_cache['dL_db2'].flatten()
        loss = self.backward_cache.get('loss', None)

        input_y = np.linspace(2, 8, self.input_size)
        hidden_y = np.linspace(1.5, 8.5, self.hidden_size)
        output_y = [5]

        POS_COLOR = '#006400'   # bold dark green  = gradient pushing weight up
        NEG_COLOR = '#B22222'   # bold firebrick red = gradient pushing weight down

        # ---- Output -> Hidden gradient arrows (dL/dW2), drawn RIGHT to LEFT ----
        for j in range(self.hidden_size):
            grad = dL_dW2[j, 0]
            color = POS_COLOR if grad > 0 else NEG_COLOR
            lw = 1.8 + min(abs(grad) * 4, 5)
            ax.annotate('', xy=(hidden_x + 0.35, hidden_y[j]), xytext=(output_x - 0.35, output_y[0]),
                        arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, alpha=0.9,
                                         mutation_scale=22, shrinkA=0, shrinkB=0))
            ax.text((hidden_x + output_x) / 2,
                    (hidden_y[j] + output_y[0]) / 2,
                    f'{grad:.3f}', fontsize=6.5, ha='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.75, edgecolor='none'))

        # ---- Hidden -> Input gradient arrows (dL/dW1), drawn RIGHT to LEFT ----
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                grad = dL_dW1[i, j]
                color = POS_COLOR if grad > 0 else NEG_COLOR
                lw = 1.8 + min(abs(grad) * 4, 5)
                ax.annotate('', xy=(input_x + 0.35, input_y[i]), xytext=(hidden_x - 0.35, hidden_y[j]),
                            arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, alpha=0.9,
                                             mutation_scale=18, shrinkA=0, shrinkB=0))
                ax.text((input_x + hidden_x) / 2,
                        (input_y[i] + hidden_y[j]) / 2,
                        f'{grad:.3f}', fontsize=6, ha='center', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.75, edgecolor='none'))

        # ---- Output Layer Node (starting point of backprop: shows LOSS + db2) ----
        circle = plt.Circle((output_x, output_y[0]), 0.42,
                             facecolor='#FFD700', edgecolor='#8B4500', linewidth=3)
        ax.add_patch(circle)
        ax.text(output_x, output_y[0], 'Output',
                fontsize=9, ha='center', va='center', fontweight='bold')
        ax.text(output_x, output_y[0] - 0.75, f'dL/db2={dL_db2[0]:.3f}',
                fontsize=8, ha='center', va='center', color='#8B4500', fontweight='bold')
        if loss is not None:
            ax.text(output_x, output_y[0] + 1.1, f'LOSS (L) = {loss:.4f}',
                    fontsize=11, ha='center', va='center', fontweight='bold',
                    color='white', bbox=dict(boxstyle='round,pad=0.3', facecolor='#B22222', edgecolor='black', linewidth=1.5))

        # ---- Hidden Layer Nodes (show bias gradient dL/db1) ----
        for j in range(self.hidden_size):
            circle = plt.Circle((hidden_x, hidden_y[j]), 0.42,
                                 facecolor='#90EE90', edgecolor='#004d00', linewidth=3)
            ax.add_patch(circle)
            ax.text(hidden_x, hidden_y[j], f'H{j+1}',
                    fontsize=9, ha='center', va='center', fontweight='bold')
            ax.text(hidden_x, hidden_y[j] - 0.62, f'dL/db1={dL_db1[j]:.3f}',
                    fontsize=7, ha='center', va='center', color='#8B4500', fontweight='bold')

        # ---- Input Layer Nodes (end point of backprop) ----
        for i in range(self.input_size):
            circle = plt.Circle((input_x, input_y[i]), 0.42,
                                 facecolor='#ADD8E6', edgecolor='#00008B', linewidth=3)
            ax.add_patch(circle)
            label = feature_names[i] if i < len(feature_names) else f'X{i+1}'
            ax.text(input_x - 0.85, input_y[i], label,
                    fontsize=9.5, ha='right', va='center', fontweight='bold', color='#00008B')

        # Layer labels
        ax.text(input_x, 9.5, 'Input Layer', fontsize=12, ha='center', fontweight='bold')
        ax.text(hidden_x, 9.5, 'Hidden Layer (ReLU)', fontsize=12, ha='center', fontweight='bold')
        ax.text(output_x, 9.5, 'Output Layer (Sigmoid)', fontsize=12, ha='center', fontweight='bold')

        ax.text(5, 0.4, 'Dark Green arrow: +ve gradient (dL/dW)   |   Firebrick Red arrow: -ve gradient   |   Arrows flow output -> input',
                fontsize=9, ha='center', style='italic', fontweight='bold')

        plt.tight_layout()
        plt.savefig('mlp_backward_visualization.png', dpi=150, bbox_inches='tight')
        print("Backward pass visualization saved as 'mlp_backward_visualization.png'")
        plt.show()

# Quick standalone test so Farzana, Isbat, and Erin can trust this file
# works correctly before they build their own code on top of it.
if __name__ == "__main__":
    print("=" * 70)
    print("MLP CLASSIFIER - Complete Implementation")
    print("=" * 70)

    X, y, feature_names = load_student_dataset('student_dataset.csv')

    mlp = MLPClassifier(input_size=3, hidden_size=4, output_size=1, learning_rate=0.1)

    print(f"\nNetwork Architecture:")
    print(f"  Input Layer: {mlp.input_size} neurons")
    print(f"  Hidden Layer: {mlp.hidden_size} neurons (ReLU)")
    print(f"  Output Layer: {mlp.output_size} neuron(s) (Sigmoid)")
    print(f"  Learning Rate: {mlp.learning_rate}")

    print("\n" + "=" * 70)
    print(f"TRAINING ON SAMPLE 1: {feature_names}")
    print("=" * 70)

    sample_X = X[0:1]
    sample_y = y[0:1]

    print(f"\nInput features:")
    for i, name in enumerate(feature_names):
        print(f"  {name}: {sample_X[0, i]}")
    print(f"Target: {sample_y[0]} (1=Pass, 0=Fail)")

    loss, y_pred = mlp.train_step(sample_X, sample_y, verbose=True)

    print(f"\n" + "=" * 70)
    print(f"TRAINING SUMMARY")
    print("=" * 70)
    print(f"Loss: {loss:.6f}")
    print(f"Prediction: {y_pred[0, 0]:.4f}")
    print(f"True Label: {sample_y[0]}")
    print(f"Prediction Class: {1 if y_pred[0, 0] > 0.5 else 0}")

    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    mlp.visualize_network(feature_names)          # forward flow diagram
    mlp.visualize_backward_network(feature_names)  # backward flow diagram

    print("\n" + "=" * 70)
    print("TRAINING FOR MULTIPLE EPOCHS")
    print("=" * 70)

    epochs = 10
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(len(X)):
            sample_X = X[i:i+1]
            sample_y = y[i:i+1]
            loss, _ = mlp.train_step(sample_X, sample_y, verbose=False)
            epoch_loss += loss

        avg_loss = epoch_loss / len(X)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), losses, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average Loss', fontsize=12)
    plt.title('Training Loss Over Epochs', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig('loss_curve.png', dpi=150, bbox_inches='tight')
    print("\nLoss curve saved as 'loss_curve.png'")
    plt.show()

    print("\n" + "=" * 70)
    print("FINAL PREDICTIONS ON ALL SAMPLES")
    print("=" * 70)

    correct = 0
    for i in range(len(X)):
        sample_X = X[i:i+1]
        y_pred = mlp.forward_pass(sample_X, verbose=False)
        prediction = 1 if y_pred[0, 0] > 0.5 else 0
        true_label = y[i]

        status = "OK" if prediction == true_label else "X"
        print(f"Sample {i+1}: {feature_names} -> Pred: {prediction:.0f} ({y_pred[0, 0]:.4f}), "
              f"True: {true_label} {status}")

        if prediction == true_label:
            correct += 1

    accuracy = correct / len(X) * 100
    print(f"\nAccuracy: {accuracy:.1f}% ({correct}/{len(X)} correct)")