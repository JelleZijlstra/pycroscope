"""

Runs pycroscope on itself.

"""

import pycroscope
from pycroscope.test_node_visitor import skip_if_not_installed


class PycroscopeVisitor(pycroscope.name_check_visitor.NameCheckVisitor):
    should_check_environ_for_files = False
    config_filename = "../pyproject.toml"


@skip_if_not_installed("asynq")
def test_all() -> None:
    PycroscopeVisitor.check_all_files()


if __name__ == "__main__":
    PycroscopeVisitor.main()
