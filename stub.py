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
        # Download 1.5B and 2.7B param model
        "gdown 16kXGKJaFS3MmW-kQMyUbinF828o4tBSY -O /root/",
        # Extract models
        "mkdir -p /root/models",
        "tar -xvf /root/models.tar -C /root/models",
    ),
)
