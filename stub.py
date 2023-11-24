import modal
from util import parse_args


args = parse_args()


stub = modal.Stub(
    image=(
        modal.Image.debian_slim()
        if args.dry or args.no_cuda
        else modal.Image.from_registry("nvcr.io/nvidia/pytorch:22.12-py3")
    )
    .pip_install(
        "torch==2.0.1+cu118", index_url="https://download.pytorch.org/whl/cu118"
    )
    .pip_install("transformers", "gdown", "pyext", "graphviz")
    .run_commands(
        # Download 1.5B model
        "gdown 1svUcwtqL6AD_Ti0eXJS03AaMdS7HDZ0d -O /root/",
        # Extract models
        "mkdir -p /root/models",
        "tar -xvf /root/models_1.5B.tar -C /root/models",
    )
    .run_commands(
        "apt-get update",
        "apt-get install -y graphviz",
    )
)
