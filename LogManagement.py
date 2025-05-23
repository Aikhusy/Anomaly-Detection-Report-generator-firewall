import logging
from datetime import datetime
import time

logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.FileHandler('log_aplikasi.txt')
formatter = logging.Formatter('[%(levelname)s][%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)

def tulis_log(pesan, level='info'):
    if level == 'info':
        logger.info(pesan)
    elif level == 'warning':
        logger.warning(pesan)
    elif level == 'error':
        logger.error(pesan)

if __name__ == "__main__":
    while True:
        waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tulis_log(f"Log aktif pada {waktu}")
        time.sleep(5)
