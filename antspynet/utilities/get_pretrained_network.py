import tensorflow as tf

def get_pretrained_network(file_id=None,
                           target_file_name=None,
                           antsxnet_cache_directory=None):

    """
    Download pretrained network/weights.

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
    >>> model_file = get_pretrained_network('dbpn4x')
    """

    def switch_networks(argument):
        switcher = {
            "brainAgeGender": "https://ndownloader.figshare.com/files/22179948",
            "brainAgeDeepBrainNet": "https://ndownloader.figshare.com/files/23573402",
            "brainExtraction": "https://ndownloader.figshare.com/files/22944632",
            "brainExtractionT1": "https://ndownloader.figshare.com/files/27334370",
            "brainExtractionT1v1": "https://ndownloader.figshare.com/files/28057626",
            "brainExtractionT2": "https://ndownloader.figshare.com/files/23066153",
            "brainExtractionFLAIR": "https://ndownloader.figshare.com/files/23562194",
            "brainExtractionBOLD": "https://ndownloader.figshare.com/files/22761977",
            "brainExtractionFA": "https://ndownloader.figshare.com/files/22761926",
            "brainExtractionNoBrainer": "https://ndownloader.figshare.com/files/22598039",
            "brainExtractionInfantT1T2": "https://ndownloader.figshare.com/files/22968833",
            "brainExtractionInfantT1": "https://ndownloader.figshare.com/files/22968836",
            "brainExtractionInfantT2": "https://ndownloader.figshare.com/files/22968830",
            "brainSegmentation": "https://ndownloader.figshare.com/files/13900010",
            "brainSegmentationPatchBased": "https://ndownloader.figshare.com/files/14249717",
            "claustrum_axial_0": "https://ndownloader.figshare.com/files/27844068",
            "claustrum_axial_1": "https://ndownloader.figshare.com/files/27844059",
            "claustrum_axial_2": "https://ndownloader.figshare.com/files/27844062",
            "claustrum_corona_0": "https://ndownloader.figshare.com/files/27844074",
            "claustrum_corona_1": "https://ndownloader.figshare.com/files/27844071",
            "claustrum_corona_2": "https://ndownloader.figshare.com/files/27844065",
            "ctHumanLung": "https://ndownloader.figshare.com/files/20005217",
            "dbpn4x": "https://ndownloader.figshare.com/files/13347617",
            "deepFlash": "https://ndownloader.figshare.com/files/22933757",
            "deepFlashLeft8": "https://ndownloader.figshare.com/files/25441007",
            "deepFlashRight8": "https://ndownloader.figshare.com/files/25441004",
            "deepFlashLeft16": "https://ndownloader.figshare.com/files/25465844",
            "deepFlashRight16": "https://ndownloader.figshare.com/files/25465847",
            "deepFlashLeft16new": "https://ndownloader.figshare.com/files/25991681",
            "deepFlashRight16new": "https://ndownloader.figshare.com/files/25991678",
            "denoising": "https://ndownloader.figshare.com/files/14235296",
            "dktInner": "https://ndownloader.figshare.com/files/23266943",
            "dktOuter": "https://ndownloader.figshare.com/files/23765132",
            "dktOuterWithSpatialPriors": "https://ndownloader.figshare.com/files/24230768",
            "elBicho": "https://ndownloader.figshare.com/files/26736779",
            "ewDavidWmhSegmentationWeights": "https://ndownloader.figshare.com/files/26827691",
            # "ewDavidWmhSegmentationSlicewiseWeights": "https://ndownloader.figshare.com/files/26787152",
            "ewDavidWmhSegmentationSlicewiseWeights": "https://ndownloader.figshare.com/files/26896703",
            "ewDavidWmhSegmentationSlicewiseT1OnlyWeights": "https://ndownloader.figshare.com/files/26920130",
            "ewDavidSysu": "",
            "ewDavidSysuT1Only": "",
            "ewDavidSysuFlairOnly": "",
            "ewDavidSysuPlus": "",
            "ewDavidSysuPlusT1Only": "",
            "ewDavidSysuPlusFlairOnly": "",
            "ewDavidSysuPlusSeg": "",
            "ewDavidSysuPlusSegT1Only": "",
            "ewDavidSysuWithSite": "",
            "ewDavidSysuWithSiteT1Only": "",
            "ewDavidSysuWithSiteFlairOnly": "",
            "functionalLungMri": "https://ndownloader.figshare.com/files/13824167",
            "hippMapp3rInitial": "https://ndownloader.figshare.com/files/18068408",
            "hippMapp3rRefine": "https://ndownloader.figshare.com/files/18068411",
            "hypothalamus": "https://ndownloader.figshare.com/files/28344378",
            "koniqMBCS": "https://ndownloader.figshare.com/files/24967376",
            "koniqMS": "https://ndownloader.figshare.com/files/25461887",
            "koniqMS2": "https://ndownloader.figshare.com/files/25474850",
            "koniqMS3": "https://ndownloader.figshare.com/files/25474847",
            "lungCtWithPriorsSegmentationWeights": "https://ndownloader.figshare.com/files/28357818",
            "mriSuperResolution": "https://ndownloader.figshare.com/files/24128618",
            "protonLungMri": "https://ndownloader.figshare.com/files/13606799",
            "sixTissueOctantBrainSegmentation": "https://ndownloader.figshare.com/files/23776025",
            "sixTissueOctantBrainSegmentationWithPriors1": "https://ndownloader.figshare.com/files/28159869",
            "sysuMediaWmhFlairOnlyModel0": "https://ndownloader.figshare.com/files/22898441",
            "sysuMediaWmhFlairOnlyModel1": "https://ndownloader.figshare.com/files/22898570",
            "sysuMediaWmhFlairOnlyModel2": "https://ndownloader.figshare.com/files/22898438",
            "sysuMediaWmhFlairT1Model0": "https://ndownloader.figshare.com/files/22898450",
            "sysuMediaWmhFlairT1Model1": "https://ndownloader.figshare.com/files/22898453",
            "sysuMediaWmhFlairT1Model2": "https://ndownloader.figshare.com/files/22898459",
            "tidsQualityAssessment": "https://ndownloader.figshare.com/files/24292895",
            "wholeTumorSegmentationT2Flair": "https://ndownloader.figshare.com/files/14087045"
        }
        return(switcher.get(argument, "Invalid argument."))

    if file_id == None:
        raise ValueError("Missing file id.")

    valid_list = ("dbpn4x",
                  "brainAgeGender",
                  "brainAgeDeepBrainNet",
                  "brainExtraction",
                  "brainExtractionT1",
                  "brainExtractionT1v1",
                  "brainExtractionT2",
                  "brainExtractionFLAIR",
                  "brainExtractionBOLD",
                  "brainExtractionFA",
                  "brainExtractionNoBrainer",
                  "brainExtractionInfantT1T2",
                  "brainExtractionInfantT1",
                  "brainExtractionInfantT2",
                  "brainSegmentation",
                  "brainSegmentationPatchBased",
                  "claustrum_axial_0",
                  "claustrum_axial_1",
                  "claustrum_axial_2",
                  "claustrum_coronal_0",
                  "claustrum_coronal_1",
                  "claustrum_coronal_2",
                  "ctHumanLung",
                  "deepFlash",
                  "deepFlashLeft8",
                  "deepFlashRight8",
                  "deepFlashLeft16",
                  "deepFlashRight16",
                  "deepFlashLeft16new",
                  "deepFlashRight16new",
                  "denoising",
                  "dktInner",
                  "dktOuter",
                  "dktOuterWithSpatialPriors",
                  "elBicho",
                  "ewDavidWmhSegmentationWeights",
                  "ewDavidWmhSegmentationSlicewiseWeights",
                  "ewDavidWmhSegmentationSlicewiseT1OnlyWeights",
                  "ewDavidSysu",
                  "ewDavidSysuT1Only",
                  "ewDavidSysuFlairOnly",
                  "ewDavidSysuPlus",
                  "ewDavidSysuPlusT1Only",
                  "ewDavidSysuPlusFlairOnly",
                  "ewDavidSysuPlusSeg",
                  "ewDavidSysuPlusSegT1Only",
                  "ewDavidSysuWithSite",
                  "ewDavidSysuWithSiteT1Only",
                  "ewDavidSysuWithSiteFlairOnly",
                  "functionalLungMri",
                  "hippMapp3rInitial",
                  "hippMapp3rRefine",
                  "hypothalamus",
                  "koniqMBCS",
                  "koniqMS",
                  "koniqMS2",
                  "koniqMS3",
                  "lungCtWithPriorsSegmentationWeights",
                  "mriSuperResolution",
                  "protonLungMri",
                  "sixTissueOctantBrainSegmentation",
                  "sixTissueOctantBrainSegmentationWithPriors1",
                  "sixTissueOctantBrainSegmentationWithPriors2",
                  "sysuMediaWmhFlairOnlyModel0",
                  "sysuMediaWmhFlairOnlyModel1",
                  "sysuMediaWmhFlairOnlyModel2",
                  "sysuMediaWmhFlairT1Model0",
                  "sysuMediaWmhFlairT1Model1",
                  "sysuMediaWmhFlairT1Model2",
                  "tidsQualityAssessment",
                  "wholeTumorSegmentationT2Flair",
                  "show")

    if not file_id in valid_list:
        raise ValueError("No data with the id you passed - try \"show\" to get list of valid ids.")

    if file_id == "show":
       return(valid_list)

    url = switch_networks(file_id)

    if target_file_name == None:
        target_file_name = file_id + ".h5"

    if antsxnet_cache_directory == None:
        antsxnet_cache_directory = "ANTsXNet"

    target_file_name_path = tf.keras.utils.get_file(target_file_name, url,
        cache_subdir = antsxnet_cache_directory)

    return(target_file_name_path)
