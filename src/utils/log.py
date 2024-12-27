import datetime
import random
import os

def get_log_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    return os.path.join(project_root, "logs")

def create_log_filename(prefix, ext):
    # prefix + date + random + ext
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    random_number = str(random.randint(100, 999))
    log_filename = prefix + '_' + current_date + '_' + random_number + ext
    return os.path.join(get_log_dir(), log_filename)
