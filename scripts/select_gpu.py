import os
import subprocess

def get_free_gpu():
    # Query GPU usage
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True
    )

    # Parse the output
    gpus = [line.split(",") for line in result.stdout.strip().split("\n")]
    gpus = [(int(index), int(mem_used)) for index, mem_used in gpus]

    # Sort by least memory used (assumption: lowest memory used = least occupied)
    gpus.sort(key=lambda x: x[1])

    # Return the least used GPU index
    return gpus[0][0] if gpus else "0"

if __name__ == "__main__":
    print(get_free_gpu())  # Prints the selected GPU index
