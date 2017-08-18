import argparse

from adultfilter.classifier import make_prediction_fun

def main():
    """
    Classify document as adult or not.

    Gets the document and prints its category
    """
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        "--title",
        required=True,
        help="Title of document (site) as string.",
        type=str
    )
    parser.add_argument(
        "--keywords",
        required=True,
        help="Keywords of document (site) as multiple strings.",
        type=str,
        nargs='+'
    )
    parser.add_argument(
        "--description",
        required=True,
        help="Description of document (site) as string.",
        type=str
    )
    parser.add_argument(
        "--body-text",
        required=True,
        help="Main (body) text of document (site) as string.",
        type=str
    )
    args = parser.parse_args()
    arguments = args.__dict__

    predict_fun = make_prediction_fun()

    print(predict_fun(**arguments))

if __name__ == "__main__":
    main()