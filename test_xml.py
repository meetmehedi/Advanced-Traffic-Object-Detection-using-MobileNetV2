import glob
import os
import xml.etree.ElementTree as ET

xml_files = glob.glob('/Users/md.mehedihasan/Downloads/trafic/train/Final Train Dataset/*.xml')
for xml_file in xml_files:
    try:
        ET.parse(xml_file)
    except Exception as e:
        print(f"Error in {xml_file}: {e}")
