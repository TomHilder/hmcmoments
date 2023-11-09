# __main__.py
# Writen by Thomas Hilder

"""
Command line interface for hmcmoments.
"""

from .generate import generate_moments
from .io import get_parser, write_moments
from .settings import Settings


def main() -> None:
    # Get CLI argument values
    parser = get_parser()
    args = parser.parse_args()
    # Get settings object
    user_settings = Settings.from_dict(**vars(args))
    # Call moments generation function with settings
    moments = generate_moments(user_settings)
    # Write moments to files
    write_moments(moments, user_settings)


if __name__ == "__main__":
    main()
