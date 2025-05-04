# static analysis: ignore
from .implementation import assert_is_value
from .test_name_check_visitor import TestNameCheckVisitorBase
from .test_node_visitor import assert_passes
from .value import KnownValue


class TestImport(TestNameCheckVisitorBase):
    @assert_passes()
    def test_import(self):
        import pycroscope as P

        def capybara() -> None:
            import pycroscope
            import pycroscope as py
            import pycroscope.extensions as E

            assert_is_value(pycroscope, KnownValue(P))
            assert_is_value(py, KnownValue(P))
            assert_is_value(E, KnownValue(P.extensions))

    @assert_passes()
    def test_import_from(self):
        def capybara():
            import pycroscope as P

            def capybara():
                from pycroscope import extensions
                from pycroscope.extensions import assert_error

                assert_is_value(extensions, KnownValue(P.extensions))
                assert_is_value(assert_error, KnownValue(P.extensions.assert_error))

    def test_import_star(self):
        self.assert_passes(
            """
            import pycroscope as P

            if False:
                from pycroscope import *

                assert_is_value(extensions, KnownValue(P.extensions))
                not_a_name  # E: undefined_name
            """
        )


class TestDisallowedImport(TestNameCheckVisitorBase):
    @assert_passes()
    def test_top_level(self):
        import getopt  # E: disallowed_import
        import xml.etree.ElementTree  # E: disallowed_import
        from getopt import GetoptError  # E: disallowed_import

        print(getopt, GetoptError, xml)  # shut up flake8

        def capybara():
            import getopt  # E: disallowed_import
            import xml.etree.ElementTree  # E: disallowed_import
            from getopt import GetoptError  # E: disallowed_import

            print(getopt, GetoptError, xml)

    @assert_passes()
    def test_nested(self):
        import email.base64mime  # ok
        import email.quoprimime  # E: disallowed_import
        from email.quoprimime import unquote  # E: disallowed_import
        from xml.etree import ElementTree  # E: disallowed_import

        print(email, unquote, ElementTree)

        def capybara():
            import email.base64mime  # ok
            import email.quoprimime  # E: disallowed_import
            from email.quoprimime import unquote  # E: disallowed_import
            from xml.etree import ElementTree  # E: disallowed_import

            print(email, unquote, ElementTree)

    @assert_passes()
    def test_import_from(self):
        from email import base64mime, quoprimime  # ok  # E: disallowed_import

        print(quoprimime, base64mime)

        def capybara():
            from email import base64mime, quoprimime  # ok  # E: disallowed_import

            print(quoprimime, base64mime)
