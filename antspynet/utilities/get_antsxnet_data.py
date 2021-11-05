import tensorflow as tf
import ants
import os

def get_antsxnet_data(file_id=None,
                      target_file_name=None,
                      antsxnet_cache_directory=None):

    """
    Download data such as prefabricated templates and spatial priors.

    Arguments
    ---------

    file_id string
        One of the permitted file ids or pass "show" to list all
        valid possibilities. Note that most require internet access
        to download.

    target_file_name string
       Optional target filename.

    antsxnet_cache_directory string
       Optional target output.  If not specified these data will be downloaded
       to the subdirectory ~/.keras/ANTsXNet/.

    Returns
    -------
    A filename string

    Example
    -------
    >>> template_file = get_antsxnet_data('biobank')
    """

    from ..utilities import brain_extraction

    def switch_data(argument):
        switcher = {
            "biobank": "https://ndownloader.figshare.com/files/22429242",
            "croppedMni152": "https://ndownloader.figshare.com/files/22933754",
            "croppedMni152Priors": "https://ndownloader.figshare.com/files/27688437",
            "deepFlashPriors": "https://figshare.com/ndownloader/files/31208272",
            "deepFlashTemplateT1": "https://figshare.com/ndownloader/files/31207795",
            "deepFlashTemplateT1SkullStripped": "https://figshare.com/ndownloader/files/31339867",
            "deepFlashTemplateT2": "https://figshare.com/ndownloader/files/31207798",
            "deepFlashTemplateT2SkullStripped": "https://figshare.com/ndownloader/files/31339870",
            "mprage_hippmapp3r": "https://ndownloader.figshare.com/files/24984689",
            "protonLobePriors": "https://figshare.com/ndownloader/files/30678452",
            "protonLungTemplate": "https://ndownloader.figshare.com/files/22707338",
            "ctLungTemplate": "https://ndownloader.figshare.com/files/22707335",
            "luna16LungPriors": "https://ndownloader.figshare.com/files/28253796",
            "priorDktLabels": "https://ndownloader.figshare.com/files/24139802",
            "S_template3": "https://ndownloader.figshare.com/files/22597175",
            "priorDeepFlashLeftLabels": "https://ndownloader.figshare.com/files/25422098",
            "priorDeepFlashRightLabels": "https://ndownloader.figshare.com/files/25422101",
            "adni": "https://ndownloader.figshare.com/files/25516361",
            "ixi": "https://ndownloader.figshare.com/files/25516358",
            "kirby": "https://ndownloader.figshare.com/files/25620107",
            "mni152": "https://ndownloader.figshare.com/files/25516349",
            "nki": "https://ndownloader.figshare.com/files/25516355",
            "nki10": "https://ndownloader.figshare.com/files/25516346",
            "oasis": "https://ndownloader.figshare.com/files/25516352"
        }
        return(switcher.get(argument, "Invalid argument."))

    if file_id == None:
        raise ValueError("Missing file id.")

    valid_list = ("biobank",
                  "croppedMni152",
                  "croppedMni152Priors",
                  "ctLungTemplate",
                  "deepFlashPriors",
                  "deepFlashTemplateT1",
                  "deepFlashTemplateT1SkullStripped",
                  "deepFlashTemplateT2",
                  "deepFlashTemplateT2SkullStripped",
                  "luna16LungPriors",
                  "mprage_hippmapp3r",
                  "priorDktLabels",
                  "priorDeepFlashLeftLabels",
                  "priorDeepFlashRightLabels",
                  "protonLobePriors",
                  "protonLungTemplate",
                  "S_template3",
                  "adni",
                  "ixi",
                  "kirby",
                  "mni152",
                  "nki",
                  "nki10",
                  "oasis",
                  "show")

    if not file_id in valid_list:
        raise ValueError("No data with the id you passed - try \"show\" to get list of valid ids.")

    if file_id == "show":
       return(valid_list)

    url = switch_data(file_id)

    if target_file_name == None:
        target_file_name = file_id + ".nii.gz"

    if antsxnet_cache_directory == None:
        antsxnet_cache_directory = "ANTsXNet"

    target_file_name_path = tf.keras.utils.get_file(target_file_name, url,
        cache_subdir=antsxnet_cache_directory)

    return(target_file_name_path)
