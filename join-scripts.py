base_dir = './'
script_files = [
    'ROI/neural_network/helpers.py',
    'modules/datasets.py',
    'modules/inference.py',
    'modules/losses.py',
    'modules/metrics.py',
    'modules/misc.py',
    'modules/postprocessing.py',
    'modules/preprocessing.py',
    'modules/visualization.py',
    'training/binary.py',
    'training/cascade.py',
    'training/dual.py',
    'training/multiclass.py',
    'training/multilabel.py',
    'training/trainer.py',
    'networks/refunet3pluscbam.py',
    'networks/raunetplusplus.py',
    'networks/swinunet.py',
]
script_files = [base_dir + f for f in script_files]

imports = []
code = []

skipping = False

for file in script_files:
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('import') or line.startswith('from'):
                imports.append(line)
            else:
                if line.startswith('if __name__ =='):
                    break
                if line.startswith('__all__ ='):
                    skipping = True
                if not skipping:
                    code.append(line)
                if skipping and line.endswith(']\n'):
                    skipping = False


def non_local(x):
    if x.startswith('from .'):
        return False
    if x.startswith('from ..'):
        return False
    local_dirs = ['networks', 'modules', 'training', 'ROI']
    for d in local_dirs:
        if x.startswith(f'from {d}'):
            return False
    return True


def comment_problematic(x):
    problematic = ['torchview', 'torchviz', 'torchsummary', 'pydensecrf']
    if any(word in x for word in problematic):
        return f'# {x}'
    return x


imports = map(comment_problematic, sorted(filter(non_local, list(set(imports)))))

with open('joined-scripts.py', 'w', encoding='utf-8') as f:
    f.writelines(imports)
    f.writelines(code)
