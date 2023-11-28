"""This module contains Mkdocs macros to modify the documentation. The
documentation on mkdocs plugins is available.

[here](https://mkdocs-macros-plugin.readthedocs.io/en/latest/).
"""

BINDER_URL = "https://mybinder.org/v2/gh"
BRANCH = "master"
BINDER_TEMPLATE = """
-->
!!! info
    - Try this notebook in an executable environment with [Binder]({binder_link}).
    - Download this notebook [here]({download_link}).
<!--
"""


def define_env(env):
    """This is the hook for defining variables, macros and filters.

    - variables: the dictionary that contains the environment variables
    - macro: a decorator function, to declare a macro.
    """

    @env.macro
    def add_binder_block(page):
        """Add a block with binder and download link.

        In any page (.md or .ipynb), the string `{{ add_binder_block(page) }}`
        is replaced by an admonition box (note) with binder and download link.
        This is intended for Jupyter notebooks.

        Args:
            page: Mkdocs Page instance (described
                [here](https://www.mkdocs.org/user-guide/custom-themes/#page))

        Returns:
            str: admonition box to be inserted in the documentation.
        """
        repo_url = env.conf["repo_url"]
        repo_name = env.conf["repo_name"]
        docs_dirs = "docs"
        filepath = f"{docs_dirs}/{page.file.src_path}"
        binder_link = f"{BINDER_URL}/{repo_name}/{BRANCH}"
        binder_link = f"{binder_link}?filepath={filepath}"
        download_link = f"{repo_url}/blob/{BRANCH}/{filepath}"
        return BINDER_TEMPLATE.format(
            binder_link=binder_link, download_link=download_link
        )
