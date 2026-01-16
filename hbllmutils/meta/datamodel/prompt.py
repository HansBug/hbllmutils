import textwrap
from typing import Optional, List


def create_meta_prompt_for_datamodel(dm_cls: type, related_dm_clses: Optional[List[type]] = None):
    return textwrap.dedent(f"""



    """).strip()
