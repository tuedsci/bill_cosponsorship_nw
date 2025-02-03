"""
Module for processing XML files.

Author: Tue Nguyen
"""

import xml.etree.ElementTree as ET


def extract_xml_fields(
    element: ET.Element,
    fields: list[str],
    prefix: str = "",
) -> dict:
    """
    Extract specified fields from an XML element into a dictionary.
    Args:
        element: XML element to extract from
        fields: List of field names to extract
        prefix: Optional prefix for field names in output dictionary
    """
    result = {}

    for field in fields:
        field_elem = element.find(field)
        if field_elem is not None:
            key = f"{prefix}{field}" if prefix else field
            result[key] = field_elem.text
        else:
            key = f"{prefix}{field}" if prefix else field
            result[key] = None

    return result
