import os.path
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
    >>> model_file = get_pretrained_network('brainExtraction')
    """

    def switch_networks(argument):
        switcher = {
            "antsxnetWmh": "https://figshare.com/ndownloader/files/42301059",
            "antsxnetWmhOr": "https://figshare.com/ndownloader/files/42301056",
            "arterialLesionWeibinShi": "https://figshare.com/ndownloader/files/31624922",
            "brainAgeGender": "https://ndownloader.figshare.com/files/22179948",
            "brainAgeDeepBrainNet": "https://ndownloader.figshare.com/files/23573402",
            "brainExtraction": "https://ndownloader.figshare.com/files/22944632",
            "brainExtractionT1": "https://ndownloader.figshare.com/files/27334370",
            "brainExtractionT1v1": "https://ndownloader.figshare.com/files/28057626",
            "brainExtractionRobustT1": "https://figshare.com/ndownloader/files/34821874",
            "brainExtractionT2": "https://ndownloader.figshare.com/files/23066153",
            "brainExtractionRobustT2": "https://figshare.com/ndownloader/files/34870416",
            "brainExtractionRobustT2Star": "https://figshare.com/ndownloader/files/34870413",
            "brainExtractionFLAIR": "https://ndownloader.figshare.com/files/23562194",
            "brainExtractionRobustFLAIR": "https://figshare.com/ndownloader/files/34870407",
            "brainExtractionBOLD": "https://ndownloader.figshare.com/files/22761977",
            "brainExtractionRobustBOLD": "https://figshare.com/ndownloader/files/34870404",
            "brainExtractionFA": "https://ndownloader.figshare.com/files/22761926",
            "brainExtractionRobustFA": "https://figshare.com/ndownloader/files/34870410",
            "brainExtractionNoBrainer": "https://ndownloader.figshare.com/files/22598039",
            "brainExtractionInfantT1T2": "https://ndownloader.figshare.com/files/22968833",
            "brainExtractionInfantT1": "https://ndownloader.figshare.com/files/22968836",
            "brainExtractionInfantT2": "https://ndownloader.figshare.com/files/22968830",
            "brainSegmentation": "https://ndownloader.figshare.com/files/13900010",
            "brainSegmentationPatchBased": "https://ndownloader.figshare.com/files/14249717",
            "bratsStage1": "https://figshare.com/ndownloader/files/42384756",
            "bratsStage2": "https://figshare.com/ndownloader/files/42685150", #"https://figshare.com/ndownloader/files/42384750",
            "cerebellumWhole": "https://figshare.com/ndownloader/files/41460447",
            "cerebellumTissue": "https://figshare.com/ndownloader/files/41107724",
            "cerebellumLabels": "https://figshare.com/ndownloader/files/41168678",
            "claustrum_axial_0": "https://ndownloader.figshare.com/files/27844068",
            "claustrum_axial_1": "https://ndownloader.figshare.com/files/27844059",
            "claustrum_axial_2": "https://ndownloader.figshare.com/files/27844062",
            "claustrum_coronal_0": "https://ndownloader.figshare.com/files/27844074",
            "claustrum_coronal_1": "https://ndownloader.figshare.com/files/27844071",
            "claustrum_coronal_2": "https://ndownloader.figshare.com/files/27844065",
            "ctHumanLung": "https://ndownloader.figshare.com/files/20005217",
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
            "lesion_patch": "https://figshare.com/ndownloader/files/43882539",
            "lesion_whole_brain": "https://figshare.com/ndownloader/files/44032017",  # 44162156
            "lesion_flip_brain": "https://figshare.com/ndownloader/files/44032014",
            "lesion_flip_template_brain": "https://figshare.com/ndownloader/files/44032011",
            "ex5_coronal_weights": "https://figshare.com/ndownloader/files/42434193",
            "ex5_sagittal_weights": "https://figshare.com/ndownloader/files/42434202",
            "allen_brain_mask_weights" : "https://figshare.com/ndownloader/files/36999880",  #https://figshare.com/ndownloader/files/42481248
            "allen_brain_leftright_coronal_mask_weights" : "",
            "allen_cerebellum_coronal_mask_weights" : "",
            "allen_cerebellum_sagittal_mask_weights" : "",
            "allen_sr_weights" : "",
            "mouseMriBrainExtraction" : "https://figshare.com/ndownloader/files/44714947",
            "mouseT2wBrainExtraction3D" : "https://figshare.com/ndownloader/files/44714941",
            "mouseT2wBrainParcellation3DNick" : "https://figshare.com/ndownloader/files/44714944",
            "functionalLungMri": "https://ndownloader.figshare.com/files/13824167",
            "hippMapp3rInitial": "https://ndownloader.figshare.com/files/18068408",
            "hippMapp3rRefine": "https://ndownloader.figshare.com/files/18068411",
            "hyperMapp3r": "https://figshare.com/ndownloader/files/38790702",
            "hypothalamus": "https://ndownloader.figshare.com/files/28344378",
            "inpainting_sagittal_rmnet_weights" : "https://figshare.com/ndownloader/files/44367188",
            "inpainting_coronal_rmnet_weights" : "https://figshare.com/ndownloader/files/44294099",
            "inpainting_axial_rmnet_weights" : "https://figshare.com/ndownloader/files/44244446",
            "inpainting_sagittal_rmnet_flair_weights" : "https://figshare.com/ndownloader/files/44511356",
            "inpainting_coronal_rmnet_flair_weights" : "https://figshare.com/ndownloader/files/44468984",
            "inpainting_axial_rmnet_flair_weights" : "https://figshare.com/ndownloader/files/44406923",
            "koniqMBCS": "https://ndownloader.figshare.com/files/24967376",
            "koniqMS": "https://figshare.com/ndownloader/files/35295403",
            "koniqMS2": "https://figshare.com/ndownloader/files/35295397",
            "koniqMS3": "https://ndownloader.figshare.com/files/25474847",
            "lungCtWithPriorsSegmentationWeights": "https://ndownloader.figshare.com/files/28357818",
            "maskLobes": "https://figshare.com/ndownloader/files/30678458",
            "mriSuperResolution": "https://figshare.com/ndownloader/files/35290684",
            "mriModalityClassification": "https://figshare.com/ndownloader/files/41691681",
            "protonLungMri": "https://ndownloader.figshare.com/files/13606799",
            "protonLobes": "https://figshare.com/ndownloader/files/30678455",
            "pulmonaryArteryWeights": "https://figshare.com/ndownloader/files/42918565",
            "sixTissueOctantBrainSegmentation": "https://ndownloader.figshare.com/files/23776025",
            "sixTissueOctantBrainSegmentationWithPriors1": "https://ndownloader.figshare.com/files/28159869",
            "sysuMediaWmhFlairOnlyModel0": "https://ndownloader.figshare.com/files/22898441",
            "sysuMediaWmhFlairOnlyModel1": "https://ndownloader.figshare.com/files/22898570",
            "sysuMediaWmhFlairOnlyModel2": "https://ndownloader.figshare.com/files/22898438",
            "sysuMediaWmhFlairT1Model0": "https://ndownloader.figshare.com/files/22898450",
            "sysuMediaWmhFlairT1Model1": "https://ndownloader.figshare.com/files/22898453",
            "sysuMediaWmhFlairT1Model2": "https://ndownloader.figshare.com/files/22898459",
            "tidsQualityAssessment": "https://figshare.com/ndownloader/files/35295391",
            "xrayLungExtraction": "https://figshare.com/ndownloader/files/41965818",
            "xrayLungOrientation": "https://figshare.com/ndownloader/files/41965821",
            "chexnetClassification": "https://figshare.com/ndownloader/files/42423522",
            "chexnetANTsXNetClassification": "https://figshare.com/ndownloader/files/42428943",
            "wholeTumorSegmentationT2Flair": "https://ndownloader.figshare.com/files/14087045",
            "wholeLungMaskFromVentilation": "https://ndownloader.figshare.com/files/28914441"
        }
        return(switcher.get(argument, "Invalid argument."))

    if file_id == None:
        raise ValueError("Missing file id.")

    valid_list = ("antsxnetWmh",
                  "antsxnetWmhOr",
                  "arterialLesionWeibinShi",
                  "brainAgeGender",
                  "brainAgeDeepBrainNet",
                  "brainExtraction",
                  "brainExtractionT1",
                  "brainExtractionT1v1",
                  "brainExtractionRobustT1",
                  "brainExtractionT2",
                  "brainExtractionRobustT2",
                  "brainExtractionRobustT2Star",
                  "brainExtractionFLAIR",
                  "brainExtractionRobustFLAIR",
                  "brainExtractionBOLD",
                  "brainExtractionRobustBOLD",
                  "brainExtractionFA",
                  "brainExtractionRobustFA",
                  "brainExtractionNoBrainer",
                  "brainExtractionInfantT1T2",
                  "brainExtractionInfantT1",
                  "brainExtractionInfantT2",
                  "brainSegmentation",
                  "brainSegmentationPatchBased",
                  "bratsStage1",
                  "bratsStage2",
                  "cerebellumWhole",
                  "cerebellumTissue",
                  "cerebellumLabels",
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
                  "ex5_coronal_weights",
                  "ex5_sagittal_weights",
                  "allen_brain_mask_weights",
                  "allen_brain_leftright_coronal_mask_weights",
                  "allen_cerebellum_coronal_mask_weights",
                  "allen_cerebellum_sagittal_mask_weights",
                  "allen_sr_weights",
                  "mouseMriBrainExtraction",
                  "mouseT2wBrainExtraction3D",
                  "mouseT2wBrainParcellation3DNick",
                  "elBicho",
                  "functionalLungMri",
                  "hippMapp3rInitial",
                  "hippMapp3rRefine",
                  "hyperMapp3r",
                  "hypothalamus",
                  "inpainting_sagittal_rmnet_weights",
                  "inpainting_coronal_rmnet_weights",
                  "inpainting_axial_rmnet_weights",
                  "inpainting_sagittal_rmnet_flair_weights",
                  "inpainting_coronal_rmnet_flair_weights",
                  "inpainting_axial_rmnet_flair_weights",
                  "koniqMBCS",
                  "koniqMS",
                  "koniqMS2",
                  "koniqMS3",
                  "lesion_patch",
                  "lesion_whole_brain",
                  "lesion_flip_brain",
                  "lesion_flip_template_brain",
                  "lungCtWithPriorsSegmentationWeights",
                  "maskLobes",
                  "mriSuperResolution",
                  "mriModalityClassification",
                  "protonLungMri",
                  "protonLobes",
                  "pulmonaryArteryWeights",
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
                  "xrayLungExtraction",
                  "xrayLungOrientation",
                  "chexnetClassification",
                  "chexnetANTsXNetClassification",
                  "wholeTumorSegmentationT2Flair",
                  "wholeLungMaskFromVentilation",
                  "show")

    if not file_id in valid_list:
        raise ValueError("No data with the id you passed - try \"show\" to get list of valid ids.")

    if file_id == "show":
       return(valid_list)

    url = switch_networks(file_id)

    if target_file_name is None:
        target_file_name = file_id + ".h5"

    if antsxnet_cache_directory is None:
        antsxnet_cache_directory = os.path.join(os.path.expanduser('~'), ".keras/ANTsXNet")

    target_file_name_path = os.path.join(antsxnet_cache_directory, target_file_name)

    # keras get_file does not work on read-only file systems. It will attempt to download the file even
    # if it exists. This is a problem for shared cache directories and read-only containers.
    #
    # Check if the file exists here, and if so, return it. Else let keras handle the download
    if not os.path.exists(target_file_name_path):
        if not url:
            err_msg = 'The requested network "' + file_id + '" is not available'
            raise NotImplementedError(err_msg)

        target_file_name_path = tf.keras.utils.get_file(target_file_name, url,
                                                        cache_subdir=antsxnet_cache_directory)

    return(target_file_name_path)
