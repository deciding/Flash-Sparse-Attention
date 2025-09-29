import modal
from pathlib import Path
local_dir = Path(__file__).parent.parent

# Define the image: start from debian-slim + python3.12
nsa_image = (
    modal.Image.debian_slim(python_version="3.12")
    #.pip_install("torch")
    #.apt_install("git", "build-essential", "cmake", "ninja-build")
    .pip_install_from_requirements(local_dir / 'requirements.txt') # local file not remote file
    .workdir("/workspace")
    .add_local_dir(
        local_dir,
        remote_path='/workspace/',
        ignore=[".git"]
    )
)

# Define the app
app = modal.App("flash-sparse-attention")

@app.function(
    image=nsa_image,
    gpu="H100",  # request NVIDIA H100 GPU
    timeout=60 * 20,  # 20 minutes just in case build is slow
)
def run_benchmark():
    import subprocess
    import sys

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

    from test.test_cmp_attn_decode import test_cmp_attn_decode
    test_cmp_attn_decode()

