from pathlib import Path
import socket
import multiprocessing as mp

LIGHTNING_POSE_MODEL_URL = "https://owncloud.gwdg.de/index.php/s/gbgFp5L8IPs369u/download"

# multiprocessing --------------------------------------

def default_num_processes(reserve: int = 2) -> int:
    n = mp.cpu_count()
    return max(1, n - reserve)

NUM_PROCESSES = default_num_processes()

# path --------------------------------------

HOST = socket.gethostname()

DATA_ROOTS = {
    "o1-596": Path("/media/martin/DATA1/Behavioral_screen/DATA/WT_dec_2025"),
    "O1-619": Path("/home/martin/Desktop/DATA/WT_Ronidazole"),
    "TheBeast": Path("/media/martin/MARTIN_8TB_0/Work/Baier/DATA/Behavioral_screen"),
}

try:
    ROOT_FOLDER = DATA_ROOTS[HOST]
except KeyError:
    raise RuntimeError(
        f"Unknown host '{HOST}'. "
        "Please add it to DATA_ROOTS in config.py"
    )
