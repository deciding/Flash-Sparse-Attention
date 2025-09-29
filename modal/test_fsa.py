import modal
from modal import Image, App, Volume

from pathlib import Path
local_dir = Path(__file__).parent.parent

flash_attn_wheel_name = "flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64.whl"
flash_attn_wheel_file = local_dir / flash_attn_wheel_name
# NOTE: remember to copy the flash_mla/cuda...so manually to flash_mla dir

txl_wheel_name = "txl-3.4.0-cp312-cp312-linux_x86_64.whl"
txl_wheel_file = local_dir / txl_wheel_name

# Define the image: start from debian-slim + python3.12
nsa_image = (
    Image.debian_slim(python_version="3.12")
    #.pip_install("torch")
    #.apt_install("git", "build-essential", "cmake", "ninja-build")
    .pip_install_from_requirements(local_dir / 'requirements.txt') # local file not remote file
    .add_local_file(flash_attn_wheel_file, remote_path="/workspace/", copy=True) # copy the local code to the image
    .add_local_file(txl_wheel_file, remote_path="/workspace/", copy=True) # copy the local code to the image
    .run_commands(
        f"pip install /workspace/{flash_attn_wheel_name}",
        f"pip install /workspace/{txl_wheel_name}",
    )
    .workdir("/workspace")
    .add_local_dir(
        local_dir,
        remote_path='/workspace/',
        ignore=[".git", "*.whl"]
    )
)

# Define the app
app = App("flash-sparse-attention")

volume = Volume.from_name("fsa-dump", create_if_missing=True) # create a cloud volume to store compiled dump files

@app.function(
    image=nsa_image,
    gpu="H100",  # request NVIDIA H100 GPU
    timeout=60 * 20,  # 20 minutes just in case build is slow
)
def run_benchmark():
    import subprocess
    import sys
    import torch

    def get_gpu_type():
        import subprocess

        try:
            # Execute nvidia-smi command to query GPU details
            result = subprocess.run(['nvidia-smi', '-q'], capture_output=True, text=True, check=True)
            output = result.stdout

            # Look for indicators of SXM or PCIe in the output
            for line in output.split("\n"):
                if "Product Name" in line:
                    print(line)
                    if 'H100' in line and 'HBM3' in line:
                        return True
        except subprocess.CalledProcessError as e:
            print(f"Error running nvidia-smi: {e}")
        except FileNotFoundError:
            print("nvidia-smi not found. Please ensure NVIDIA drivers are installed and in your PATH.")
        return False

    if not get_gpu_type():
        return

    #from tests.fsa.test_cmp_attn_decode import test_cmp_attn_decode
    #test_cmp_attn_decode()

    #from tests.nsa.benchmark_nsa import benchmark
    #benchmark.run(print_data=True, save_path='.')

    #from tests.flash_mla.test_flash_mla_decoding import main
    #main(torch.bfloat16)

    from tests.flash_mla.test_flash_mla_prefill import main
    main()
