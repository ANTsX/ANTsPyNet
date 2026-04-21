#!/usr/bin/env python

import argparse

import ants
import antspynet


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run mouse brain extraction with ANTsPyNet."
    )
    parser.add_argument(
        "input_image",
        type=str,
        help="Path to the input mouse image.",
    )
    parser.add_argument(
        "output_image",
        type=str,
        help="Path to write the extracted probability image.",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="t2",
        help='Mouse image modality passed to mouse_brain_extraction (default: "t2").',
    )
    parser.add_argument(
        "--isotropic-output",
        action="store_true",
        help="Return the probability image in isotropic space before writing it.",
    )
    parser.add_argument(
        "--axis",
        type=int,
        default=2,
        help="Axis index used for ex5 modalities (default: 2).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress while running the model.",
    )
    return parser


def run(input_image, output_image, modality="t2", isotropic_output=False, axis=2, verbose=False):
    image = ants.image_read(input_image)
    probability_image = antspynet.mouse_brain_extraction(
        image,
        modality=modality,
        return_isotropic_output=isotropic_output,
        which_axis=axis,
        verbose=verbose,
    )
    ants.image_write(probability_image, output_image)
    return output_image


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        run(
            args.input_image,
            args.output_image,
            modality=args.modality,
            isotropic_output=args.isotropic_output,
            axis=args.axis,
            verbose=args.verbose,
        )
    except Exception as error:
        parser.exit(1, f"Error: {error}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())