"""

Runs pycroscope on itself.

"""

import pycroscope


class PycroscopeVisitor(pycroscope.name_check_visitor.NameCheckVisitor):
    should_check_environ_for_files = False
    config_filename = "../pyproject.toml"


def test_all() -> None:
    PycroscopeVisitor.check_all_files()


if __name__ == "__main__":
    PycroscopeVisitor.main()
