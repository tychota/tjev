"""Every data source of the mix, by block.

======================  ==============================  ======================================
module                  registry                        block
======================  ==============================  ======================================
:mod:`.contenders`      ``JEV_SOURCES``                 jev (contenders' public decision data)
:mod:`.generators`      ``GENERATORS``                  generators / writing (code-labelled)
:mod:`.generators`      ``LONG_GENERATORS``             generators (> 4096 tokens, opt-in)
:mod:`.public`          ``SOURCES``                     replay (public datasets as decisions)
:mod:`.text_analysis`   ``TEXT_SOURCES``                analysis (authorship, tone, emotion…)
:mod:`.targeted`        ``TARGETED_SOURCES``            targeted (JevBench weak spots, mix-v6)
:mod:`.jev_v7`          ``JEV_V7_SOURCES``              jev / breadth (typed-decision corpora)
:mod:`.targeted_v7`     ``TARGETED_V7_SOURCES``         targeted (temporal, numeric, legal)
:mod:`.tools_v7`        ``TOOLS_V7_SOURCES``            targeted (tool use, rules, reading)
:mod:`.judge`           ``JUDGE_SOURCES``               judge (+ analysis: x-stance)
======================  ==============================  ======================================

The mix-v7 registries (``V7_SOURCES``) set each source's block (``Source.block``) and the
licence use of its rows (``Source.license_use``, or per row); their card licences are in
``V7_LICENCES``. They may hold non-commercial sources: ``tjev data derive
--drop-license-use non-commercial`` derives the commercial-only mix.

A source adapter maps one dataset row to zero or more item dicts (:mod:`tjev.data.item`);
each :class:`~.base.Source` records its dataset, splits, license and commercial status.
"""

from .base import Source
from .contenders import JEV_SOURCES
from .generators import GENERATORS, LONG_GENERATORS, generate
from .jev_v7 import JEV_V7_SOURCES
from .jev_v7 import LICENCES as _JEV_V7_LICENCES
from .judge import JUDGE_SOURCES
from .judge import LICENCES as _JUDGE_LICENCES
from .public import SOURCES
from .targeted import TARGETED_SOURCES
from .targeted_v7 import LICENCES as _TARGETED_V7_LICENCES
from .targeted_v7 import TARGETED_V7_SOURCES
from .text_analysis import TEXT_SOURCES
from .tools_v7 import LICENCES as _TOOLS_V7_LICENCES
from .tools_v7 import TOOLS_V7_SOURCES

# The mix-v7 sources, in build order (each one names its block)
V7_SOURCES: dict[str, Source] = {
    **JEV_V7_SOURCES,
    **TARGETED_V7_SOURCES,
    **TOOLS_V7_SOURCES,
    **JUDGE_SOURCES,
}
V7_LICENCES: dict[str, tuple[str, str, str]] = {
    **_JEV_V7_LICENCES,
    **_TARGETED_V7_LICENCES,
    **_TOOLS_V7_LICENCES,
    **_JUDGE_LICENCES,
}

# Every dataset-backed source by name (generators are separate: they have no dataset)
# (the targeted sources, then the v7 ones, last: adding them does not change the build
# order of the others)
ALL_SOURCES: dict[str, Source] = {
    **SOURCES,
    **TEXT_SOURCES,
    **JEV_SOURCES,
    **TARGETED_SOURCES,
    **V7_SOURCES,
}

__all__ = [
    "ALL_SOURCES",
    "GENERATORS",
    "JEV_SOURCES",
    "LONG_GENERATORS",
    "SOURCES",
    "TARGETED_SOURCES",
    "TEXT_SOURCES",
    "V7_LICENCES",
    "V7_SOURCES",
    "Source",
    "generate",
]
