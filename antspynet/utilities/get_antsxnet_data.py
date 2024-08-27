import tensorflow as tf
import ants
import os.path

_antsxnet_cache_directory = os.path.join(os.path.expanduser('~'), '.keras', 'ANTsXNet')

def get_antsxnet_cache_directory():
    """Get the cache directory for ANTsXNet data. Data and pre-trained models will be
    downloaded here.

    The default directory is ~/.keras/ANTsXNet.

    Returns
    -------
    antsxnet_cache_directory string
        The directory to store ANTsXNet data.

    """
    return(_antsxnet_cache_directory)


def set_antsxnet_cache_directory(antsxnet_cache_dir):
    """Set the cache directory for ANTsXNet data. Data and pre-trained models will be
    downloaded here.

    Arguments
    ---------
    antsxnet_cache_dir string
        The directory to store ANTsXNet data. It will be created if it does not exist.
    """
    global _antsxnet_cache_directory
    _antsxnet_cache_directory = os.path.abspath(antsxnet_cache_dir)

    if not os.path.exists(_antsxnet_cache_directory):
        os.makedirs(_antsxnet_cache_directory)


def get_antsxnet_data(file_id=None,
                      target_file_name=None):

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

    Returns
    -------
    A filename string

    Example
    -------
    >>> template_file = get_antsxnet_data('biobank')
    """

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
            "deepFlashTemplate2T1SkullStripped": "https://figshare.com/ndownloader/files/46461451",
            "deepFlashTemplate2Labels": "https://figshare.com/ndownloader/files/46461415",
            "mprage_hippmapp3r": "https://ndownloader.figshare.com/files/24984689",
            "protonLobePriors": "https://figshare.com/ndownloader/files/30678452",
            "protonLungTemplate": "https://ndownloader.figshare.com/files/22707338",
            "ctLungTemplate": "https://ndownloader.figshare.com/files/22707335",
            "luna16LungPriors": "https://ndownloader.figshare.com/files/28253796",
            "xrayLungPriors": "https://figshare.com/ndownloader/files/41965815",
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
            "oasis": "https://ndownloader.figshare.com/files/25516352",
            "magetTemplate": "https://figshare.com/ndownloader/files/41052572",
            "magetTemplateBrainMask": "https://figshare.com/ndownloader/files/41052569",
            "magetCerebellumTemplate": "https://figshare.com/ndownloader/files/41052581",
            "magetCerebellumTemplatePriors": "https://figshare.com/ndownloader/files/41052578",
            "magetCerebellumxTemplate0GenericAffine": "https://figshare.com/ndownloader/files/41052575",
            "mraTemplate": "https://figshare.com/ndownloader/files/46406695",
            "mraTemplateBrainMask": "https://figshare.com/ndownloader/files/46406698",
            "mraTemplateVesselPrior": "https://figshare.com/ndownloader/files/46406713",
            "hcpyaT1Template": "https://figshare.com/ndownloader/files/46746142",
            "hcpyaT2Template": "https://figshare.com/ndownloader/files/46746334",
            "hcpyaFATemplate": "https://figshare.com/ndownloader/files/46746349",
            "hcpyaTemplateBrainMask": "https://figshare.com/ndownloader/files/46746388",
            "hcpyaTemplateBrainSegmentation": "https://figshare.com/ndownloader/files/46746367",
            "bsplineT2MouseTemplate": "https://figshare.com/ndownloader/files/44706247",
            "bsplineT2MouseTemplateBrainMask": "https://figshare.com/ndownloader/files/44869285",
            "DevCCF_P56_MRI-T2_50um": "https://figshare.com/ndownloader/files/44706244",
            "DevCCF_P56_MRI-T2_50um_BrainParcellationNickMask": "https://figshare.com/ndownloader/files/44706238",
            "DevCCF_P56_MRI-T2_50um_BrainParcellationTctMask": "https://figshare.com/ndownloader/files/47214532",
            "DevCCF_P04_STPT_50um": "https://figshare.com/ndownloader/files/46711546",
            "DevCCF_P04_STPT_50um_BrainParcellationJayMask": "https://figshare.com/ndownloader/files/46712656"
        }
        return(switcher.get(argument, "Invalid argument."))

    if file_id is None:
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
                  "deepFlashTemplate2T1SkullStripped",
                  "deepFlashTemplate2Labels",
                  "luna16LungPriors",
                  "mprage_hippmapp3r",
                  "priorDktLabels",
                  "priorDeepFlashLeftLabels",
                  "priorDeepFlashRightLabels",
                  "protonLobePriors",
                  "protonLungTemplate",
                  "xrayLungPriors",
                  "S_template3",
                  "adni",
                  "ixi",
                  "kirby",
                  "mni152",
                  "nki",
                  "nki10",
                  "oasis",
                  "magetTemplate",
                  "magetTemplateBrainMask",
                  "magetCerebellumTemplate",
                  "magetCerebellumTemplatePriors",
                  "magetCerebellumxTemplate0GenericAffine",
                  "mraTemplate",
                  "mraTemplateBrainMask",
                  "mraTemplateVesselPrior",
                  "bsplineT2MouseTemplate",
                  "bsplineT2MouseTemplateBrainMask",
                  "DevCCF_P56_MRI-T2_50um",
                  "DevCCF_P56_MRI-T2_50um_BrainParcellationNickMask",
                  "DevCCF_P56_MRI-T2_50um_BrainParcellationTctMask",
                  "DevCCF_P04_STPT_50um",
                  "DevCCF_P04_STPT_50um_BrainParcellationJayMask",
                  "hcpyaT1Template",
                  "hcpyaT2Template",
                  "hcpyaFATemplate",
                  "hcpyaTemplateBrainMask",
                  "hcpyaTemplateBrainSegmentation",
                  "show")

    if not file_id in valid_list:
        raise ValueError("No data with the id you passed - try \"show\" to get list of valid ids.")

    if file_id == "show":
       return(valid_list)

    url = switch_data(file_id)

    if target_file_name is None:
        if file_id == "magetCerebellumxTemplate0GenericAffine":
            target_file_name = file_id + ".mat"
        else:
            target_file_name = file_id + ".nii.gz"

    target_file_name_path = os.path.join(get_antsxnet_cache_directory(), target_file_name)

    # keras get_file does not work on read-only file systems. It will attempt to download the file even
    # if it exists. This is a problem for shared cache directories and read-only containers.
    #
    # Check if the file exists here, and if so, return it. Else let keras handle the download
    if not os.path.exists(target_file_name_path):
        if not url:
            err_msg = 'The requested data "' + file_id + '" is not available'
            raise NotImplementedError(err_msg)

        target_file_name_path = tf.keras.utils.get_file(target_file_name, url,
                                                        cache_subdir=get_antsxnet_cache_directory())

    return(target_file_name_path)
