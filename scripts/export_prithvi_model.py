"""
scripts/export_prithvi_model.py
----------------------------
Download the Prithvi-EO-1.0-100M model from HuggingFace and export
it as TorchScript for use in the sentinelhub project.

Usage:
    .venv/bin/python scripts/export_prithvi_model.py

Requirements:
    pip install terratorch transformers accelerate
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn as nn

# Default model from HuggingFace
DEFAULT_MODEL_NAME = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M-multi-temporal-crop-classification"

# Output path
DEFAULT_OUT_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "sentinelhub",
    "api",
    "ml",
    "prithvi_classifier.pt",
)


def _download_and_load_prithvi_manual(model_name: str) -> nn.Module:
    """Download and load Prithvi model manually from HF Hub.

    This handles models that don't have standard config.json files.
    """
    import tempfile
    from pathlib import Path

    from huggingface_hub import hf_hub_download, list_repo_files

    # Parse model name
    if "/" in model_name:
        repo_id = model_name
    else:
        repo_id = f"ibm-nasa-geospatial/{model_name}"

    print(f"Downloading model files from {repo_id}...")

    # Check for HF_TOKEN
    import os

    if not os.environ.get("HF_TOKEN"):
        print("\nWARNING: HF_TOKEN not set. You may encounter rate limits.")
        print("Set it with: export HF_TOKEN=your_token_here")
        print("Get a token from: https://huggingface.co/settings/tokens\n")

    # Create a temporary directory to store files
    cache_dir = Path(tempfile.gettempdir()) / "prithvi_models"
    cache_dir.mkdir(exist_ok=True)

    try:
        # List files in the repo to find the .pth and .py files
        files = list_repo_files(repo_id)
        pth_files = [f for f in files if f.endswith(".pth")]
        py_files = [f for f in files if f.endswith(".py")]

        if not pth_files:
            raise RuntimeError(f"No .pth weight files found in {repo_id}")

        weights_filename = pth_files[0]
        config_filename = py_files[0] if py_files else None

        print(f"Found weights file: {weights_filename}")
        if config_filename:
            print(f"Found config file: {config_filename}")

        weights_path = hf_hub_download(
            repo_id=repo_id,
            filename=weights_filename,
            cache_dir=str(cache_dir),
            local_files_only=False,
        )
        print(f"Downloaded weights to: {weights_path}")

        # Try to download config file
        config_path = None
        if config_filename:
            try:
                config_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=config_filename,
                    cache_dir=str(cache_dir),
                    local_files_only=False,
                )
                print(f"Downloaded config to: {config_path}")
            except Exception as exc:
                print(f"Could not download config: {exc}")

        # Load the weights
        # weights_only=False is required for this model due to numpy objects in checkpoint
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

        # The checkpoint might contain state_dict directly or be wrapped
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        print(f"Loaded checkpoint with {len(state_dict)} keys")

        # Import and instantiate the model architecture if config available
        if config_path:
            # This is an MMSegmentation config file, not a standalone model.
            # The config uses dict(type='...') patterns that require mmseg/mmengine.
            # TerraTorch wraps this properly.
            raise RuntimeError(
                "This model requires TerraTorch/MMSegmentation to load. "
                "The config file is an MMSegmentation config, not a standalone model.\n\n"
                "To fix this, install terratorch:\n"
                "  pip install terratorch\n\n"
                "Then run this script again."
            )
        else:
            # Without config, we can't reconstruct the model
            raise RuntimeError(
                "Model config file not available. "
                "Please install terratorch: pip install terratorch"
            )

        return model

    except Exception as exc:
        raise RuntimeError(
            f"Failed to download/load model from {repo_id}. "
            f"Error: {exc}"
        ) from exc


class PrithviInferenceWrapper(nn.Module):
    """Wrapper for Prithvi model to make it TorchScript-compatible.

    The original model may have complex control flow that's not
    TorchScript-friendly. This wrapper simplifies the forward pass.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module | None = None,
        num_classes: int = 13,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

        # If no head provided, create a simple classification head
        if head is None:
            # PrithviViT returns a list of feature maps
            # We'll use adaptive pooling and a linear layer
            self.num_classes = num_classes
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(768, num_classes)  # 768 is embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, 18, 224, 224) - 3 timesteps × 6 bands

        Returns:
            Logits tensor (B, num_classes)
        """
        # Reshape from (B, 18, 224, 224) to (B, 6, 3, 224, 224)
        # 18 channels = 6 bands × 3 timesteps
        B = x.shape[0]
        x = x.view(B, 6, 3, 224, 224)

        features = self.backbone(x)

        if self.head is not None:
            return self.head(features)

        # Default: handle transformer token outputs
        if isinstance(features, (list, tuple)):
            # Use last layer's tokens
            tokens = features[-1]  # (B, 589, 768) where 589 = 14*14*3 + 1 CLS token

        # Average pool across all tokens (excluding CLS token which is first)
        # tokens shape: (B, num_tokens, embed_dim)
        tokens = tokens.mean(dim=1)  # (B, embed_dim)
        return self.fc(tokens)


def download_and_export(
    model_name: str = DEFAULT_MODEL_NAME,
    out_path: str = DEFAULT_OUT_PATH,
    use_terralith: bool = True,
) -> None:
    """Download Prithvi model and export to TorchScript.

    Args:
        model_name: HuggingFace model identifier
        out_path: Path to save the TorchScript model
        use_terralith: Whether to use terratorch (recommended) or transformers
    """
    print(f"Loading model: {model_name}")

    model: nn.Module

    if use_terralith:
        try:
            from terratorch import prithvi_vit

            print("Using TerraTorch to load Prithvi model...")
            # Load the Prithvi-EO-1.0-100M backbone
            # This is the backbone only; the HF model has a full encoder-decoder
            model = prithvi_vit.prithvi_eo_v1_100(pretrained=True)
        except ImportError as exc:
            print(f"terratorch import error: {exc}")
            print("Attempting manual download (may fail for MMSegmentation models)...")
            use_terralith = False
        except Exception as exc:
            print(f"terratorch loading failed: {exc}")
            print("Attempting manual download...")
            use_terralith = False

    if not use_terralith:
        # Download and load the model manually
        # Note: This only works for standard HF models, not MMSegmentation ones
        model = _download_and_load_prithvi_manual(model_name)

    model.eval()

    # Extract backbone and head if possible
    # Prithvi models typically have a backbone + decoder/head structure
    backbone: nn.Module = model
    head: nn.Module | None = None

    # Try to extract common structures
    if hasattr(model, "model"):
        backbone = model.model
    elif hasattr(model, "backbone"):
        backbone = model.backbone
        head = getattr(model, "decode_head", None) or getattr(model, "head", None)

    # Wrap for TorchScript compatibility
    # The HF model has 13 classes for crop classification
    wrapper = PrithviInferenceWrapper(backbone, head, num_classes=13)
    wrapper.eval()

    # Test with dummy input
    dummy_input = torch.randn(1, 18, 224, 224)
    print(f"Testing with dummy input: {dummy_input.shape}")

    with torch.no_grad():
        output = wrapper(dummy_input)
        print(f"Output shape: {output.shape}")
        num_classes = output.shape[1]
        print(f"Number of classes: {num_classes}")

    # Export to TorchScript
    print("Tracing model...")
    try:
        scripted = torch.jit.trace(wrapper, dummy_input)
        scripted.eval()

        # Verify the scripted model works
        with torch.no_grad():
            test_output = scripted(dummy_input)
            print(f"Scripted model output shape: {test_output.shape}")

        # Save
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        scripted.save(out_path)
        print(f"TorchScript model saved to: {out_path}")

    except Exception as exc:
        print(f"Tracing failed: {exc}")
        print("Trying scripting instead...")

        try:
            scripted = torch.jit.script(wrapper)
            scripted.save(out_path)
            print(f"TorchScript model saved to: {out_path}")
        except Exception as exc2:
            print(f"Scripting also failed: {exc2}")
            print("\nThe Prithvi model may contain control flow that is not")
            print("TorchScript-compatible. You may need to:")
            print("1. Modify the model architecture for TorchScript compatibility")
            print("2. Use the non-TorchScript version (PrithviCropClassifier)")
            print("3. Export via ONNX instead")
            raise


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Export Prithvi model to TorchScript"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUT_PATH,
        help="Output path for TorchScript model",
    )
    parser.add_argument(
        "--use-transformers",
        action="store_true",
        help="Use transformers instead of terratorch",
    )

    args = parser.parse_args()

    download_and_export(
        model_name=args.model,
        out_path=args.output,
        use_terralith=not args.use_transformers,
    )


if __name__ == "__main__":
    main()
