import subprocess
import os

def launch_tensorboard():
    try:
        # Use the "start" command; shell=True is required on Windows.
        tb_process = subprocess.Popen("start tensorboard --logdir=runs", shell=True)
        print("TensorBoard launched on http://localhost:6006")
        return tb_process
    except Exception as e:
        print(f"Failed to launch TensorBoard: {e}")
        return None

if __name__ == "__main__":
    launch_tensorboard()