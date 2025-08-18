"""A module for important set-up and configuration functionality, but doesn't implement the library's key features."""

import logging
import os
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Environment variable constants
LABEL_STUDIO_HOST = os.getenv("LABEL_STUDIO_HOST")
LABEL_STUDIO_TOKEN = os.getenv("LABEL_STUDIO_TOKEN")
MC_API_KEY = os.getenv("MC_API_KEY")


def configure_logging():
    """A helper method that configures logging, usable by any script in this library."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s : %(asctime)s : %(name)s : %(message)s",
    )


def validate_environment_variables(required_vars: Optional[List[str]] = None) -> Tuple[str, ...]:
    """Validate that required environment variables are set.

    Args:
        required_vars: List of environment variable names to validate.
                      If None, defaults to LABEL_STUDIO_HOST and LABEL_STUDIO_TOKEN.

    Returns:
        tuple: Values of the required environment variables in the order specified

    Raises:
        ValueError: If any of the required environment variables are missing
    """
    if required_vars is None:
        required_vars = ["LABEL_STUDIO_HOST", "LABEL_STUDIO_TOKEN"]

    missing_vars = []
    var_values = []

    for var_name in required_vars:
        var_value = os.getenv(var_name)
        if not var_value:
            missing_vars.append(var_name)
        else:
            var_values.append(var_value)

    if missing_vars:
        raise ValueError(
            f"Missing environment variables: {', '.join(missing_vars)}. Please set them in your .env file."
        )

    return tuple(var_values)


def validate_label_studio_env() -> Tuple[str, str]:
    """Validate Label Studio environment variables specifically.

    Returns:
        tuple: (LABEL_STUDIO_HOST, LABEL_STUDIO_TOKEN)

    Raises:
        ValueError: If either LABEL_STUDIO_HOST or LABEL_STUDIO_TOKEN is missing
    """
    return validate_environment_variables(["LABEL_STUDIO_HOST", "LABEL_STUDIO_TOKEN"])


def validate_mediacloud_env() -> str:
    """Validate Media Cloud environment variables specifically.

    Returns:
        str: MC_API_KEY value

    Raises:
        ValueError: If MC_API_KEY is missing
    """
    return validate_environment_variables(["MC_API_KEY"])[0]


def get_environment_variable(var_name: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """Get an environment variable with optional default value and required validation.

    Args:
        var_name: Name of the environment variable
        default: Default value if the environment variable is not set
        required: If True, raises ValueError when the variable is not set

    Returns:
        The environment variable value or default value

    Raises:
        ValueError: If required=True and the environment variable is not set
    """
    value = os.getenv(var_name)

    if value is None:
        if required:
            raise ValueError(f"Required environment variable '{var_name}' is not set.")
        return default

    return value


def check_environment_variables(vars_to_check: List[str]) -> Dict[str, bool]:
    """Check which environment variables are set without raising errors.

    Args:
        vars_to_check: List of environment variable names to check

    Returns:
        Dictionary mapping variable names to whether they are set (True) or not (False)
    """
    return {var_name: bool(os.getenv(var_name)) for var_name in vars_to_check}
