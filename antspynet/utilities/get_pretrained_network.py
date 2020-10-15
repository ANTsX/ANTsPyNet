import tensorflow as tf

def get_pretrained_network(file_id=None, target_file_name=None, output_directory=None):

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

    output_directory string
       Optional target output.  If not specified these data will be downloaded
       to the subdirectory ~/.keras/ANTsXNet/.

    Returns
    -------
    A filename string

    Example
    -------
    >>> model_file = get_pretrained_network( 'dbpn4x' )
    """

    def switch_networks(argument):
        switcher = {
            "brainAgeGender": "https://ndownloader.figshare.com/files/22179948",
            "brainAgeDeepBrainNet": "https://ndownloader.figshare.com/files/23573402",
            "brainExtraction": "https://ndownloader.figshare.com/files/22944632",
            # "brainExtraction": "https://ndownloader.figshare.com/files/13729661",  # old weights
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
            "ctHumanLung": "https://ndownloader.figshare.com/files/20005217",
            "dbpn4x": "https://ndownloader.figshare.com/files/13347617",
            "deepFlash": "https://ndownloader.figshare.com/files/22933757",
            "denoising": "https://ndownloader.figshare.com/files/14235296",
            "dktInner": "https://ndownloader.figshare.com/files/23266943",
            "dktOuter": "https://ndownloader.figshare.com/files/23765132",
            "dktOuterWithSpatialPriors": "https://ndownloader.figshare.com/files/24230768",
            "functionalLungMri": "https://ndownloader.figshare.com/files/13824167",
            "hippMapp3rInitial": "https://ndownloader.figshare.com/files/18068408",
            "hippMapp3rRefine": "https://ndownloader.figshare.com/files/18068411",
            "mriSuperResolution": "https://ndownloader.figshare.com/files/24128618",
            "protonLungMri": "https://ndownloader.figshare.com/files/13606799",
            "sixTissueOctantBrainSegmentation": "https://ndownloader.figshare.com/files/23776025",
            "sysuMediaWmhFlairOnlyModel0": "https://ndownloader.figshare.com/files/22898441",
            "sysuMediaWmhFlairOnlyModel1": "https://ndownloader.figshare.com/files/22898570",
            "sysuMediaWmhFlairOnlyModel2": "https://ndownloader.figshare.com/files/22898438",
            "sysuMediaWmhFlairT1Model0": "https://ndownloader.figshare.com/files/22898450",
            "sysuMediaWmhFlairT1Model1": "https://ndownloader.figshare.com/files/22898453",
            "sysuMediaWmhFlairT1Model2": "https://ndownloader.figshare.com/files/22898459",
            "wholeTumorSegmentationT2Flair": "https://ndownloader.figshare.com/files/14087045"
        }
        return(switcher.get(argument, "Invalid argument."))

    if file_id == None:
        raise ValueError("Missing file id.")

    valid_list = ("dbpn4x",
                  "brainAgeGender",
                  "brainAgeDeepBrainNet",
                  "brainExtraction",
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
                  "ctHumanLung",
                  "deepFlash",
                  "denoising",
                  "dktInner",
                  "dktOuter",
                  "dktOuterWithSpatialPriors",
                  "functionalLungMri",
                  "hippMapp3rInitial",
                  "hippMapp3rRefine",
                  "mriSuperResolution",
                  "protonLungMri",
                  "sixTissueOctantBrainSegmentation",
                  "sysuMediaWmhFlairOnlyModel0",
                  "sysuMediaWmhFlairOnlyModel1",
                  "sysuMediaWmhFlairOnlyModel2",
                  "sysuMediaWmhFlairT1Model0",
                  "sysuMediaWmhFlairT1Model1",
                  "sysuMediaWmhFlairT1Model2",
                  "wholeTumorSegmentationT2Flair",
                  "show")

    if not file_id in valid_list:
        raise ValueError("No data with the id you passed - try \"show\" to get list of valid ids.")

    if file_id == "show":
       return(valid_list)

    url = switch_networks(file_id)

    if target_file_name == None:
        target_file_name = file_id + ".h5"

    if output_directory == None:
        output_directory = "ANTsXNet"

    target_file_name_path = tf.keras.utils.get_file(target_file_name, url,
        cache_subdir = output_directory)

    return(target_file_name_path)
