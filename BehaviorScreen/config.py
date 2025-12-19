from pathlib import Path
import socket
import multiprocessing as mp

# multiprocessing --------------------------------------

def default_num_processes(reserve: int = 2) -> int:
    n = mp.cpu_count()
    return max(1, n - reserve)

NUM_PROCESSES = default_num_processes()

# path --------------------------------------

HOST = socket.gethostname()

DATA_ROOTS = {
    "O1-596": Path("/media/martin/DATA1/Behavioral_screen"),
    "O1-619": Path("/home/martin/Desktop/DATA"),
    "TheBeast": Path("/media/martin/MARTIN_8TB_0/Work/Baier/DATA/Behavioral_screen"),
}

try:
    ROOT_FOLDER = DATA_ROOTS[HOST]
except KeyError:
    raise RuntimeError(
        f"Unknown host '{HOST}'. "
        "Please add it to DATA_ROOTS in config.py"
    )
