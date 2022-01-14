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
            "arterialLesionWeibinShi": "https://figshare.com/ndownloader/files/31624922",
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
            "claustrum_coronal_0": "https://ndownloader.figshare.com/files/27844074",
            "claustrum_coronal_1": "https://ndownloader.figshare.com/files/27844071",
            "claustrum_coronal_2": "https://ndownloader.figshare.com/files/27844065",
            "ctHumanLung": "https://ndownloader.figshare.com/files/20005217",
            "dbpn4x": "https://ndownloader.figshare.com/files/13347617",
            "deepFlashLeftT1": "https://ndownloader.figshare.com/files/28966269",
            "deepFlashRightT1": "https://ndownloader.figshare.com/files/28966266",
            "deepFlashLeftBoth": "https://ndownloader.figshare.com/files/28966275",
            "deepFlashRightBoth": "https://ndownloader.figshare.com/files/28966272",
            "deepFlashLeftT1Hierarchical": "https://figshare.com/ndownloader/files/31226449",
            "deepFlashRightT1Hierarchical": "https://figshare.com/ndownloader/files/31226452",
            "deepFlashLeftBothHierarchical": "https://figshare.com/ndownloader/files/31226458",
            "deepFlashRightBothHierarchical": "https://figshare.com/ndownloader/files/31226455",
            "deepFlashLeftT1Hierarchical_ri": "https://figshare.com/ndownloader/files/33198794",
            "deepFlashRightT1Hierarchical_ri": "https://figshare.com/ndownloader/files/33198800",
            "deepFlashLeftBothHierarchical_ri": "https://figshare.com/ndownloader/files/33198803",
            "deepFlashRightBothHierarchical_ri": "https://figshare.com/ndownloader/files/33198809",
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
            "ewDavidSysu": "https://ndownloader.figshare.com/files/28757622", # "https://ndownloader.figshare.com/files/28403973",
            "ewDavidSysuRankedIntensity": "https://ndownloader.figshare.com/files/28403937",
            "ewDavidSysuT1Only": "https://ndownloader.figshare.com/files/28757628", #"https://ndownloader.figshare.com/files/28403934",
            "ewDavidSysuFlairOnly": "https://ndownloader.figshare.com/files/28757625", # "https://ndownloader.figshare.com/files/28403931",
            "ewDavidSysuWithAttention": "https://ndownloader.figshare.com/files/28757631", # "https://ndownloader.figshare.com/files/28431312",
            "ewDavidSysuWithAttentionT1Only": "https://ndownloader.figshare.com/files/28757646", # "https://ndownloader.figshare.com/files/28403970",
            "ewDavidSysuWithAttentionFlairOnly": "https://ndownloader.figshare.com/files/28757643", # "https://ndownloader.figshare.com/files/28403943",
            "ewDavidSysuWithAttentionAndSite": "https://ndownloader.figshare.com/files/28757634",
            "ewDavidSysuWithAttentionAndSiteT1Only": "https://ndownloader.figshare.com/files/28757640",
            "ewDavidSysuWithAttentionAndSiteFlairOnly": "https://ndownloader.figshare.com/files/28757637",
            "ewDavidSysuPlus": "https://ndownloader.figshare.com/files/28403976",
            "ewDavidSysuPlusT1Only": "https://ndownloader.figshare.com/files/28403958",
            "ewDavidSysuPlusFlairOnly": "https://ndownloader.figshare.com/files/28403946",
            "ewDavidSysuPlusSeg": "https://ndownloader.figshare.com/files/28403940",
            "ewDavidSysuPlusSegT1Only": "https://ndownloader.figshare.com/files/28403955",
            "ewDavidSysuPlusSegWithSite": "https://ndownloader.figshare.com/files/28431375",
            "ewDavidSysuPlusSegWithSiteT1Only": "https://ndownloader.figshare.com/files/28431372",
            "ewDavidSysuWithSite": "https://ndownloader.figshare.com/files/28403964",
            "ewDavidSysuWithSiteT1Only": "https://ndownloader.figshare.com/files/28403952",
            "ewDavidSysuWithSiteFlairOnly": "https://ndownloader.figshare.com/files/28403979",
            "functionalLungMri": "https://ndownloader.figshare.com/files/13824167",
            "hippMapp3rInitial": "https://ndownloader.figshare.com/files/18068408",
            "hippMapp3rRefine": "https://ndownloader.figshare.com/files/18068411",
            "hypothalamus": "https://ndownloader.figshare.com/files/28344378",
            "koniqMBCS": "https://ndownloader.figshare.com/files/24967376",
            "koniqMS": "https://ndownloader.figshare.com/files/25461887",
            "koniqMS2": "https://ndownloader.figshare.com/files/25474850",
            "koniqMS3": "https://ndownloader.figshare.com/files/25474847",
            "lungCtWithPriorsSegmentationWeights": "https://ndownloader.figshare.com/files/28357818",
            "maskLobes": "https://figshare.com/ndownloader/files/30678458",
            "mriSuperResolution": "https://ndownloader.figshare.com/files/24128618",
            "protonLungMri": "https://ndownloader.figshare.com/files/13606799",
            "protonLobes": "https://figshare.com/ndownloader/files/30678455",
            "sixTissueOctantBrainSegmentation": "https://ndownloader.figshare.com/files/23776025",
            "sixTissueOctantBrainSegmentationWithPriors1": "https://ndownloader.figshare.com/files/28159869",
            "sysuMediaWmhFlairOnlyModel0": "https://ndownloader.figshare.com/files/22898441",
            "sysuMediaWmhFlairOnlyModel1": "https://ndownloader.figshare.com/files/22898570",
            "sysuMediaWmhFlairOnlyModel2": "https://ndownloader.figshare.com/files/22898438",
            "sysuMediaWmhFlairT1Model0": "https://ndownloader.figshare.com/files/22898450",
            "sysuMediaWmhFlairT1Model1": "https://ndownloader.figshare.com/files/22898453",
            "sysuMediaWmhFlairT1Model2": "https://ndownloader.figshare.com/files/22898459",
            "tidsQualityAssessment": "https://ndownloader.figshare.com/files/24292895",
            "wholeTumorSegmentationT2Flair": "https://ndownloader.figshare.com/files/14087045",
            "wholeLungMaskFromVentilation": "https://ndownloader.figshare.com/files/28914441"
        }
        return(switcher.get(argument, "Invalid argument."))

    if file_id == None:
        raise ValueError("Missing file id.")

    valid_list = ("dbpn4x",
                  "arterialLesionWeibinShi",
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
                  "deepFlashLeftT1",
                  "deepFlashRightT1",
                  "deepFlashLeftBoth",
                  "deepFlashRightBoth",
                  "deepFlashLeftT1Hierarchical",
                  "deepFlashRightT1Hierarchical",
                  "deepFlashLeftBothHierarchical",
                  "deepFlashRightBothHierarchical",
                  "deepFlashLeftT1Hierarchical_ri",
                  "deepFlashRightT1Hierarchical_ri",
                  "deepFlashLeftBothHierarchical_ri",
                  "deepFlashRightBothHierarchical_ri",
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
                  "ewDavidSysu",
                  "ewDavidSysuRankedIntensity",
                  "ewDavidSysuT1Only",
                  "ewDavidSysuFlairOnly",
                  "ewDavidSysuWithAttention",
                  "ewDavidSysuWithAttentionT1Only",
                  "ewDavidSysuWithAttentionFlairOnly",
                  "ewDavidSysuWithAttentionAndSite",
                  "ewDavidSysuWithAttentionAndSiteT1Only",
                  "ewDavidSysuWithAttentionAndSiteFlairOnly",
                  "ewDavidSysuPlus",
                  "ewDavidSysuPlusT1Only",
                  "ewDavidSysuPlusFlairOnly",
                  "ewDavidSysuPlusSeg",
                  "ewDavidSysuPlusSegT1Only",
                  "ewDavidSysuPlusSegWithSite",
                  "ewDavidSysuPlusSegWithSiteT1Only",
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
                  "maskLobes",
                  "mriSuperResolution",
                  "protonLungMri",
                  "protonLobes",
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
                  "wholeLungMaskFromVentilation",
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
