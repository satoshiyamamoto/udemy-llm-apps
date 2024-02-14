import glob
import os
import subprocess

CSV_DIR = "100knocks-preprocess/docker/work/data"
DB_FILE = "sample.db"

if os.path.exists(DB_FILE):
    os.remove(DB_FILE)

csv_file_paths = glob.glob(f"{CSV_DIR}/*.csv")

for csv_file_path in csv_file_paths:
    basename = os.path.basename(csv_file_path)
    table = basename.rstrip(".csv")

    print(f"loading... csv_file_path = {csv_file_path}, table = {table}")
    subprocess.run(
        ["sqlite3", "-separator", ",", DB_FILE, f".import {csv_file_path} {table}"]
    )
    subprocess.run(["sqlite3", DB_FILE, f"select count(*) from {table}"])
