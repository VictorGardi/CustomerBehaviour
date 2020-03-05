import os
import fnmatch
import shutil
import datetime

def get_outdir(algo, case, n_experts, state_rep):

    if not os.path.exists('results'): os.makedirs('results')

    path = os.path.join('results', str(algo))
    if not os.path.exists(path): os.makedirs(path)

    path = os.path.join(path, case)
    if not os.path.exists(path): os.makedirs(path)

    path = os.path.join(path, str(n_experts) + '_expert(s)')
    if not os.path.exists(path): os.makedirs(path)

    path = os.path.join(path, 'case_' + str(state_rep))
    if not os.path.exists(path): os.makedirs(path)

    time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(path, time)

    return path


def get_env(case, n_experts):
    if case == 'discrete_events':
        env = 'discrete-buying-events-v0'
    elif case == 'full_receipt':
        env = ''

    return env

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'True', 't', 'y', '1', 'true'):
        return True
    elif v.lower() in ('no', 'False', 'f', 'n', '0', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def move_dir(src: str, dst: str, pattern: str = '*'):
    if not os.path.isdir(dst):
        pathlib.Path(dst).mkdir(parents=True, exist_ok=True)
    for f in fnmatch.filter(os.listdir(src), pattern):
        shutil.move(os.path.join(src, f), os.path.join(dst, f))

def read_npz_file(path, get_npz_as_list = False, print_npz = True):
    from numpy import load
    ls = [] if get_npz_as_list else None
    data = load(path, allow_pickle=True)
    lst = data.files
    for item in lst:
        print(data[item]) if print_npz else None
        ls.append(data[item])
    return ls

def convert_imports(dir):
    keyword = 'chainerrl'
    replacement = 'customer_behaviour.chainerrl'
    for subdir, dirs, files in os.walk(dir):
        for file in files:
        #print os.path.join(subdir, file)
            filepath = subdir + os.sep + file

            if filepath.endswith(".py"):
                with open(filepath) as f:
                    newText=f.read().replace(keyword, replacement)

                with open(filepath, 'w') as f:
                    f.write(newText)
                
    print('----- .py files that had ' + str(keyword) + ' in it has now been replaced to ' + str(replacement) + ' -----')

def save_plt_as_eps(obj, path):
    obj.savefig(path, format='eps')
    