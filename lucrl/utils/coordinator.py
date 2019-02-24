from .persistence import load, save

import glob
import os


class Path:
    """
    **Encapsulates directory surfing**

    - Example::

        p = Path('.')
        p

        > '/opt/shared/project_name'

        p.join('data').join('raw')

        > '/opt/shared/project_name/data/raw'

        p.join('data').join('raw').back()

        > '/opt/shared/project_name/data'

        p.join('data').exists

        > True

        p.join('dataakjnfjknajkn').exists

        > False

        p.join('data').join('raw').contents()

        > ['/opt/shared/project_name/data/raw/file1.csv',
           '/opt/shared/project_name/data/raw/file2.csv',
           '/opt/shared/project_name/data/raw/file3.csv']

        p.save([1,2,3])

        > None

        p.load()

        > [1,2,3]
    """
    def __init__(self, s: str):
        self.path = os.path.abspath(s)

    def __repr__(self):
        return self.path

    def join(self, other):
        return Path(os.path.join(self.path, other))

    def back(self):
        return Path(os.path.dirname(self.path))

    def load(self):
        return load(self.path)

    def save(self, obj):
        save(obj, self.path)

    def contents(self, recursive: bool = True):
        if recursive:
            p = glob.glob(os.path.join(self.path, '**'), recursive=True)
        else:
            p = glob.glob(os.path.join(self.path, '*'), recursive=False)
        return sorted([x for x in p if not os.path.isdir(x)])

    @property
    def exists(self):
        return os.path.isdir(self.path) or os.path.isfile(self.path)

class Coordinator:
    """
    **Provides paths to main folders**

    Initializes paths to `root`, `src`, ...

    .. note:: Every field is a member of Path, not str!

    - Example::

        # From a notebook
        c = Coordinator('..')

        c.root

        > '/opt/shared/project_name'

        c.data_interim

        > '/opt/shared/project_name/data/interim'
    """
    def __init__(self, root_path: str):
        self.root = Path(root_path)
        self.config = self.root.join('config')
        self.logs = self.root.join('logs')
        self.data = self.root.join('data')
        self.data_raw = self.data.join('raw')
        self.data_interim = self.data.join('interim')
        self.data_features = self.data.join('features')
        self.data_processed = self.data.join('processed')
        self.docs = self.root.join('docs')
        self.models = self.root.join('models')
        self.notebooks = self.root.join('notebooks')
