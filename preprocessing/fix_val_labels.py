import argparse
import json
import os
import xml.etree.ElementTree as ET
from typing import List

import tqdm


def collect_png_files(root_dir: str) -> List[str]:
    """Return list of .png image paths (relative to root_dir) sorted alphabetically."""
    files = []
    for current_root, _dirs, fnames in os.walk(root_dir):
        for fname in fnames:
            if fname.lower().endswith('.png'):
                rel_path = os.path.relpath(os.path.join(current_root, fname), start=root_dir)
                files.append(rel_path.replace('\\', '/'))
    return sorted(files)


def build_synset_mapping(xml_dir: str) -> dict[str, int]:
    """Build synset -> class_idx mapping using alphabetical ordering of synset ids in xml_dir."""
    synsets = set()
    for xml_fname in os.listdir(xml_dir):
        if not xml_fname.endswith('.xml'):
            continue
        tree = ET.parse(os.path.join(xml_dir, xml_fname))
        elem = tree.find('.//object/name')
        if elem is None or elem.text is None:
            continue
        synsets.add(elem.text.strip())
    sorted_synsets = sorted(synsets)
    return {synset: idx for idx, synset in enumerate(sorted_synsets)}


def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate dataset.json labels for processed ImageNet validation set")
    parser.add_argument('--images-dir', required=True, help='Path to processed validation images directory (root that contains 00000/, 00001/, ...)')
    parser.add_argument('--xml-dir', required=True, help='Path to ILSVRC/Annotations/CLS-LOC/val directory with XML files')
    parser.add_argument('--output', default='dataset.json', help='Output JSON filename (relative to images-dir)')
    return parser.parse_args()


def main():
    args = parse_args()

    images_dir = args.images_dir
    xml_dir = args.xml_dir
    output_fname = os.path.join(images_dir, args.output)

    print('Collecting PNG files...')
    png_files = collect_png_files(images_dir)
    print(f'Found {len(png_files)} images')

    print('Building synset mapping...')
    synset_to_class = build_synset_mapping(xml_dir)
    print(f'Detected {len(synset_to_class)} unique synsets')

    print('Generating labels...')
    labels_list = [None] * len(png_files)
    for idx, rel_png in tqdm.tqdm(list(enumerate(png_files)), desc='Processing images'):
        # Derive corresponding xml filename from original JPEG base name embedded in PNG name.
        # We assume original file order was preserved: idx matches order of sorted XML names.
        base_png = os.path.basename(rel_png)
        jpeg_stub = os.path.splitext(base_png)[0]  # img00000001 -> img00000001
        # But we don't have original JPEG name; fallback: pick xml by index order
        xml_files_sorted = sorted(os.listdir(xml_dir))
        if idx >= len(xml_files_sorted):
            raise IndexError('Mismatch between number of images and xml files')
        xml_path = os.path.join(xml_dir, xml_files_sorted[idx])
        tree = ET.parse(xml_path)
        elem = tree.find('.//object/name')
        if elem is None or elem.text is None:
            raise RuntimeError(f'Missing synset in {xml_files_sorted[idx]}')
        synset = elem.text.strip()
        label = synset_to_class[synset]
        labels_list[idx] = [rel_png, label]

    assert all(entry is not None for entry in labels_list)

    with open(output_fname, 'w') as f:
        json.dump({'labels': labels_list}, f)
    print(f'Saved labels to {output_fname}')


if __name__ == '__main__':
    main() 