from typing import Callable, List
from dataclasses import dataclass

from slitheryn.tools.doctor.checks.paths import check_slither_path
from slitheryn.tools.doctor.checks.platform import compile_project, detect_platform
from slitheryn.tools.doctor.checks.versions import show_versions


@dataclass
class Check:
    title: str
    function: Callable[..., None]


ALL_CHECKS: List[Check] = [
    Check("PATH configuration", check_slither_path),
    Check("Software versions", show_versions),
    Check("Project platform", detect_platform),
    Check("Project compilation", compile_project),
]
