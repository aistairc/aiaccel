from __future__ import annotations

import xml.etree.ElementTree as ET


def xmltodict(xml_str: str) -> dict:
    def _element_to_dict(element: ET.Element) -> dict:
        result = {}
        for child in element:
            if len(child) == 0:
                result[child.tag] = child.text.strip() if child.text else None
            else:
                if child.tag not in result:
                    result[child.tag] = []
                result[child.tag].append(_element_to_dict(child))
        return result

    root = ET.fromstring(xml_str)
    return _element_to_dict(root)
