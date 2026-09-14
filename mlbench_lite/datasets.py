import numpy as np
from sklearn.datasets import make_classification
def load_clover(return_X_y=False):
    X, y = make_classification(
        n_samples=400,
        n_features=4,
        n_informative=4,
        n_redundant=0,
        n_classes=4,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=42
    )
    feature_names = [
        'leaf_length',
        'leaf_width',
        'petiole_length',
        'leaflet_count'
    ]
    target_names = [
        'white_clover',
        'red_clover',
        'crimson_clover',
        'alsike_clover'
    ]
    DESCR = """
    Clover Dataset
    ==============
    A synthetic dataset representing different types of clover leaves.
    Features:
    - leaf_length: Length of the leaf in cm
    - leaf_width: Width of the leaf in cm
    - petiole_length: Length of the petiole in cm
    - leaflet_count: Number of leaflets per leaf
    Classes:
    - white_clover: Trifolium repens
    - red_clover: Trifolium pratense
    - crimson_clover: Trifolium incarnatum
    - alsike_clover: Trifolium hybridum
    Samples: 400
    Features: 4
    Classes: 4
    """
    if return_X_y:
        return X, y
    from sklearn.utils import Bunch
    return Bunch(
        data=X,
        target=y,
        feature_names=feature_names,
        target_names=target_names,
        DESCR=DESCR
    )
