# __main__.py
# Writen by Thomas Hilder

"""
Command line interface for hmcmoments.
"""

from .generate import generate
from .io import get_parser
from .settings import Settings


def main() -> None:
    # Get CLI argument values
    parser = get_parser()
    args = parser.parse_args()
    # Get settings object
    user_settings = Settings.from_dict(**vars(args))
    # Call moments generation function with settings
    generate(settings=user_settings)


if __name__ == "__main__":
    main()
