import subprocess
import os
import sys
import webbrowser
import time
from pathlib import Path
import socket
import logging
from torch.utils.tensorboard import SummaryWriter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def wait_for_tensorboard(port=6006, max_attempts=10):
    for i in range(max_attempts):
        if is_port_in_use(port):
            return True
        logger.info(f"Waiting for TensorBoard to start... (attempt {i+1}/{max_attempts})")
        time.sleep(2)
    return False

def run_tensorboard():
    logdir = os.path.join(os.getcwd(), "runs")
    
    # Create the runs directory if it doesn't exist
    Path(logdir).mkdir(parents=True, exist_ok=True)
    
    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=logdir)
    
    # Kill any existing TensorBoard processes
    if sys.platform == "win32":
        try:
            subprocess.run(["taskkill", "/f", "/im", "tensorboard.exe"], stderr=subprocess.DEVNULL)
            time.sleep(2)  # Give it time to fully terminate
        except Exception as e:
            logger.warning(f"Failed to kill existing tensorboard: {e}")
    else:
        try:
            subprocess.run(["pkill", "-f", "tensorboard"], stderr=subprocess.DEVNULL)
            time.sleep(2)  # Give it time to fully terminate
        except Exception as e:
            logger.warning(f"Failed to kill existing tensorboard: {e}")

    # Construct the TensorBoard command
    cmd = [
        sys.executable,  # Use the current Python interpreter
        "-m",
        "tensorboard.main",
        "--logdir", logdir,
        "--port", "6006",
        "--bind_all"  # Allow connections from other machines
    ]
    
    logger.info(f"Starting TensorBoard with command: {' '.join(cmd)}")
    
    try:
        env = os.environ.copy()
        # Clear any existing TensorBoard related environment variables
        for key in list(env.keys()):
            if key.startswith('TENSORBOARD_'):
                del env[key]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=1,
            universal_newlines=True
        )
        
        # Start threads to monitor output
        def log_output(pipe, level):
            for line in pipe:
                logger.log(level, line.strip())
                
        from threading import Thread
        Thread(target=log_output, args=(process.stdout, logging.INFO), daemon=True).start()
        Thread(target=log_output, args=(process.stderr, logging.ERROR), daemon=True).start()
        
        # Wait for TensorBoard to start
        if wait_for_tensorboard():
            logger.info("TensorBoard started successfully!")
            url = "http://localhost:6006"
            webbrowser.open(url)
            logger.info(f"TensorBoard should be available at: {url}")
            
            # Keep the script running
            try:
                while True:
                    if process.poll() is not None:
                        logger.error("TensorBoard process terminated unexpectedly")
                        break
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("\nShutting down TensorBoard...")
                process.terminate()
                process.wait(timeout=5)
        else:
            logger.error("Failed to start TensorBoard")
            # Print any error output
            error_output = process.stderr.read()
            if error_output:
                logger.error(f"TensorBoard error output:\n{error_output}")
            
    except Exception as e:
        logger.error(f"Error starting TensorBoard: {e}")
        return

if __name__ == "__main__":
    logger.info("Starting TensorBoard...")
    run_tensorboard()