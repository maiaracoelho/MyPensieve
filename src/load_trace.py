import os


COOKED_TRACE_FOLDER = "./train/"


def load_trace(cooked_trace_folder=None):
    if cooked_trace_folder is None:
        cooked_trace_folder = "./train/"

    cooked_files = os.listdir(cooked_trace_folder)
    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []
    for cooked_file in cooked_files:
        file_path = os.path.join(cooked_trace_folder, cooked_file)
        cooked_time = []
        cooked_bw = []
        with open(file_path, "rb") as f:  # <-- mudar para "r", modo texto
            for line in f:
                parse = line.split()
                cooked_time.append(float(parse[0]))
                cooked_bw.append(float(parse[1]))
        all_cooked_time.append(cooked_time)
        all_cooked_bw.append(cooked_bw)
        all_file_names.append(cooked_file)

    return all_cooked_time, all_cooked_bw, all_file_names
