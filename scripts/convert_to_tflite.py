"""
Convert a trained flaxchat model to LiteRT (.tflite) format.

Requires: tensorflow (pip install tensorflow)
Run on Linux/x86 where TF has wheel support.

Usage:
    python -m scripts.convert_to_tflite --checkpoint=exports/model_local.pkl --output=exports/model.tflite
"""

import os
import argparse
import pickle

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from flaxchat.gpt import GPT, GPTConfig

parser = argparse.ArgumentParser(description="Convert to LiteRT/TFLite")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pkl)")
parser.add_argument("--output", type=str, default="exports/model.tflite", help="Output .tflite path")
parser.add_argument("--seq-len", type=int, default=None, help="Override sequence length for export")
parser.add_argument("--quantize", action="store_true", help="Apply int8 dynamic range quantization")
args = parser.parse_args()

# Load checkpoint
print(f"Loading checkpoint from {args.checkpoint}")
with open(args.checkpoint, "rb") as f:
    ckpt = pickle.load(f)

config_dict = ckpt["config"]
params_dict = ckpt["params"]

# Override seq len if specified
if args.seq_len:
    config_dict["sequence_len"] = args.seq_len

config = GPTConfig(**config_dict)
seq_len = config.sequence_len

# Rebuild model and load weights
model = GPT(config, rngs=nnx.Rngs(0))
model_state = nnx.state(model, nnx.Param)
model_dict = nnx.to_pure_dict(model_state)

# Load params back
restored = jax.tree.map(lambda x: jnp.array(x), params_dict)
nnx.replace_by_pure_dict(model, restored)
print(f"Model loaded: {model.num_params():,} params")

# Create pure inference function
graphdef, state = nnx.split(model)

def predict(input_ids):
    """Pure function: input_ids (1, T) -> logits (1, T, V)"""
    m = nnx.merge(graphdef, state)
    return m(input_ids)

# Convert via jax2tf → TF SavedModel → TFLite
print("Converting JAX → TF → TFLite...")

try:
    from jax.experimental import jax2tf
    import tensorflow as tf

    # Convert JAX function to TF
    tf_predict = jax2tf.convert(
        jax.jit(predict),
        polymorphic_shapes=None,  # fixed shape
        enable_xla=False,  # TFLite doesn't support XLA ops
    )

    # Create TF concrete function with fixed input shape
    input_spec = tf.TensorSpec((1, seq_len), tf.int32)

    @tf.function(input_signature=[input_spec])
    def tf_model(input_ids):
        return tf_predict(input_ids)

    # Save as SavedModel
    saved_model_dir = args.output.replace(".tflite", "_savedmodel")
    tf.saved_model.save(
        tf.Module(),
        saved_model_dir,
        signatures={"serving_default": tf_model.get_concrete_function()},
    )
    print(f"SavedModel saved to {saved_model_dir}")

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    if args.quantize:
        print("Applying int8 dynamic range quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(tflite_model)

    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"\nTFLite model saved to {args.output} ({size_mb:.2f} MB)")
    print(f"  Input shape: (1, {seq_len}) int32")
    print(f"  Output shape: (1, {seq_len}, {config.vocab_size}) float32")

    # Verify with LiteRT interpreter
    try:
        from ai_edge_litert.interpreter import Interpreter
        interp = Interpreter(model_path=args.output)
        interp.allocate_tensors()
        input_details = interp.get_input_details()
        output_details = interp.get_output_details()
        print(f"\nVerification with LiteRT interpreter:")
        print(f"  Input: {input_details[0]['shape']} {input_details[0]['dtype']}")
        print(f"  Output: {output_details[0]['shape']} {output_details[0]['dtype']}")

        # Test inference
        test_input = np.zeros((1, seq_len), dtype=np.int32)
        interp.set_tensor(input_details[0]['index'], test_input)
        interp.invoke()
        output = interp.get_tensor(output_details[0]['index'])
        print(f"  Test output shape: {output.shape}")
        print("  Verification passed!")
    except Exception as e:
        print(f"  Verification skipped: {e}")

except ImportError as e:
    print(f"\nError: {e}")
    print("\nTo convert to TFLite, you need tensorflow installed:")
    print("  pip install tensorflow")
    print("\nTensorFlow currently supports Python 3.10-3.12 on all platforms,")
    print("and Python 3.13 on Linux x86_64.")
    print("\nAlternative: use the exported StableHLO or .npz weights directly.")
