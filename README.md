Purpose: Manual quantization of model weights.

Action:

Loads trained scikit-learn model.

Extracts weights and bias.

Performs manual 8-bit quantization.

Saves and compares size and performance (R² score) of quantized vs original model.

Key File: quantize.py