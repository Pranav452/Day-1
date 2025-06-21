# Q3: Perceptron From Scratch

This project implements a single-neuron logistic regression model (perceptron) from scratch using pure NumPy to classify fruits (apples vs bananas) based on their physical characteristics.

## Files Structure

```
q3/
├── fruit.csv                    # Dataset with fruit characteristics
├── perceptron.py               # Main implementation
├── requirements.txt            # Python dependencies
├── reflection.md               # Analysis and insights
├── training_plots.png          # Loss and accuracy plots (generated)
├── learning_rate_comparison.png # LR comparison plots (generated)
└── README.md                   # This file
```

## Dataset

The `fruit.csv` contains 15 samples with features:
- **length_cm**: Fruit length in centimeters
- **weight_g**: Fruit weight in grams  
- **yellow_score**: How yellow the fruit is (0-1 scale)
- **label**: 0 = Apple, 1 = Banana

## Implementation Features

✅ **Pure NumPy Implementation**
- Sigmoid activation function
- Binary cross-entropy loss
- Batch gradient descent
- No external ML libraries (scikit-learn, PyTorch, etc.)

✅ **Training Requirements**
- 500+ epochs or loss < 0.05 (early stopping)
- Learning rate optimization
- Loss and accuracy tracking

✅ **Visualization**
- Loss curves over epochs
- Accuracy progression
- Learning rate comparison experiments

✅ **Comprehensive Analysis**
- Initial vs final performance comparison
- Learning rate impact study
- Detailed reflection with DJ-knob analogy

## Installation & Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the perceptron:**
   ```bash
   python perceptron.py
   ```

3. **Expected Output:**
   - Dataset statistics
   - Training progress (every 50 epochs)
   - Final model performance
   - Sample predictions
   - Training plots saved as PNG files

## Key Results

- **Initial Accuracy**: 6.7% (worse than random!)
- **Final Accuracy**: 100% (perfect classification)
- **Loss Reduction**: From 0.6945 to 0.0496 (target achieved)
- **Convergence**: Lightning-fast 86 epochs (vs 500 max)
- **Prediction Confidence**: 89-97% on all samples

## Learning Rate Analysis

| Learning Rate | Convergence Speed | Stability | Result |
|---------------|-------------------|-----------|---------|
| 0.01          | Very Slow         | Stable    | ~400 epochs |
| 0.1           | **OPTIMAL** ⭐     | Stable    | **86 epochs** |
| 0.5-1.0       | Fast but Unstable | Oscillates| Risk of overshoot |

## Learned Model Weights

**Final weights**: `[1.0495746, -1.02532912, 1.06347684]`
- **Length coefficient (+1.05)**: Longer fruits → more banana-like
- **Weight coefficient (-1.03)**: Heavier fruits → more apple-like  
- **Yellow coefficient (+1.06)**: More yellow → more banana-like
- **Bias**: +0.38 (slight banana preference)

## Reflection

See `reflection.md` for detailed analysis covering:
- **Actual results**: 6.7% → 100% accuracy in 86 epochs
- **Real learned weights** interpretation and meaning
- **Learning rate 0.1** proved perfectly tuned
- **DJ-knob analogy** with real training experience

---

**Assignment**: Q3 - Perceptron From Scratch  
**Objective**: Build and train a single-neuron classifier using pure NumPy  
**Dataset**: Custom fruit classification (15 samples, 3 features) 