import json
from os import path, listdir

NOTEBOOKS_PATH = path.join("notebooks")
NOTEBOOKS_SITE_PATH = path.join("..")
JUPYTER_LOGO_PATH = path.join("..", "..", "images", "Jupyter_logo.svg")
DOWNLOAD_BUTTON = """Download this notebook : <a href="%s" download="%s"><img src="%s" alt="%s"></a>"""
NB_CELL = {"cell_type": "markdown", "metadata": {}}
BINDER_STR = "Open this notebook in an executable environment using Binder by clicking on the following icon : [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/deepcharles/ruptures/HEAD?filepath=notebooks%2F{})"
SECTION_TITLE = "## Ressources"

if __name__ == "__main__":
    for filename in listdir(NOTEBOOKS_PATH):
        if path.isfile(path.join(NOTEBOOKS_PATH, filename)):
            print("Dealing with " + path.join(NOTEBOOKS_PATH, filename))
            with open(path.join(NOTEBOOKS_PATH, filename), "r") as fd:
                data = json.load(fd)

                # Set section
                section_cell = NB_CELL.copy()
                section_cell["source"] = SECTION_TITLE

                # Binder
                binder_md = BINDER_STR.format(filename)
                binder_cell = NB_CELL.copy()
                binder_cell["source"] = binder_md

                # Download button
                c_download_button = DOWNLOAD_BUTTON % (
                    path.join(NOTEBOOKS_SITE_PATH, filename),
                    filename,
                    JUPYTER_LOGO_PATH,
                    filename,
                )
                download_cell = NB_CELL.copy()
                download_cell["source"] = c_download_button

                print(binder_cell)
                print(download_cell)

                data["cells"].extend([section_cell, binder_cell, download_cell])

            with open(path.join(NOTEBOOKS_PATH, filename), "w") as outfile:
                json.dump(data, outfile)
