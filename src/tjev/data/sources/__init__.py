"""Every data source of the mix, by block.

======================  ==============================  ======================================
module                  registry                        block
======================  ==============================  ======================================
:mod:`.contenders`      ``JEV_SOURCES``                 jev (contenders' public decision data)
:mod:`.generators`      ``GENERATORS``                  generators / writing (code-labelled)
:mod:`.public`          ``SOURCES``                     replay (public datasets as decisions)
:mod:`.text_analysis`   ``TEXT_SOURCES``                analysis (authorship, tone, emotion…)
======================  ==============================  ======================================

A source adapter maps one dataset row to zero or more item dicts (:mod:`tjev.data.item`);
each :class:`~.base.Source` records its dataset, splits, license and commercial status.
"""

from .base import Source
from .contenders import JEV_SOURCES
from .generators import GENERATORS, generate
from .public import SOURCES
from .text_analysis import TEXT_SOURCES

# Every dataset-backed source by name (generators are separate: they have no dataset)
ALL_SOURCES: dict[str, Source] = {**SOURCES, **TEXT_SOURCES, **JEV_SOURCES}

__all__ = [
    "ALL_SOURCES",
    "GENERATORS",
    "JEV_SOURCES",
    "SOURCES",
    "TEXT_SOURCES",
    "Source",
    "generate",
]
