import sys
sys.path.append('../')

import pytest
import LMR_utils2 as Utils


@pytest.mark.parametrize("doc", [
    """This is class docstring""",
    None])
def test_class_doc_inherit(doc):
    class foo:
        __doc__ = doc
        pass

    @Utils.class_docs_fixer
    class bar(foo):
        pass

    assert bar.__doc__ == doc


@pytest.mark.parametrize("doc", [
    """This is func docstring""",
    None])
def test_function_doc_inherit(doc):
    class foo:
        def lol(self):
            pass

    foo.lol.__func__.__doc__ = doc

    @Utils.class_docs_fixer
    class bar(foo):
        def lol(self):
            pass

    assert bar.lol.__func__.__doc__ == doc


def test_function_doc_augment():
    parent_doc = """This is the parents lol docstr"""
    child_doc = """%%aug%%
            The childs doc is here
            """

    class foo:
        def lol(self):
            pass

    foo.lol.__func__.__doc__ = parent_doc

    @Utils.class_docs_fixer
    class bar(foo):
        @Utils.augment_docstr
        def lol(self):
            """%%aug%%
            The childs doc is here
            """
            pass

    assert bar.lol.__func__.__doc__ == (parent_doc +
                                        child_doc.replace('%%aug%%', ''))
