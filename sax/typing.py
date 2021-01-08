""" Common datastructure types used in SAX """

from typing import Optional, Dict, Union, Tuple, Callable, Any



ParamsDict = Dict[str, Union[Dict, float]]

ModelDict = Dict[Union[Tuple[str, str], str], Union[Callable, ParamsDict]]

ComplexFloat = Union[complex, float]

ModelFunc = Callable[[ParamsDict], ComplexFloat]
