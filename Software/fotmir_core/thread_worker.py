"""Private worker. Environment is configured by main before this process starts."""
import json,pickle,sys
from pathlib import Path
from main import main
if __name__ == '__main__':
 a=json.loads(Path(sys.argv[1]).read_text())
 r=main(a['input'],a['output'],'manual',a['tau'],a['ratio'],seed=a['seed'],initial_indices=a['initial_indices'],threads=None)
 with open(sys.argv[2],'wb') as f:pickle.dump(r,f)
