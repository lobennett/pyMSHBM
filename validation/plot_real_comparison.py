#!/usr/bin/env python3
"""Export auditable cortical maps and numerical summaries (matplotlib extra)."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np
from scipy.io import loadmat

from pymshbm.io.assets import load_assets, sha256_file


def surface_panel(ax, geometry, labels, colors, *, side, title, binary=False):
    """Orthographic projection with depth-sorted triangles; no label smoothing."""
    coords, triangles = geometry
    depth = coords[triangles, 0].mean(axis=1)
    order = np.argsort(depth * side)
    face_labels = np.sort(labels[triangles], axis=1)
    majority = np.where(face_labels[:, 1] == face_labels[:, 2], face_labels[:, 1], face_labels[:, 0])
    if binary:
        majority = face_labels.max(axis=1)  # Show every affected vertex's incident faces.
    points = coords[:, [1, 2]].copy()
    points[:, 0] *= -side
    collection = PolyCollection(points[triangles[order]], facecolors=colors[majority[order]],
                                edgecolors='none', linewidths=0, rasterized=True, antialiased=False)
    ax.add_collection(collection)
    ax.set_xlim(points[:, 0].min() - 3, points[:, 0].max() + 3)
    ax.set_ylim(points[:, 1].min() - 3, points[:, 1].max() + 3)
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.set_title(title, fontsize=10)


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    preparation = json.loads((args.input / 'preparation.json').read_text())
    assets = load_assets(args.assets)
    subjects = preparation['selected_subjects']
    geometry = {h: nib.freesurfer.read_geometry(args.surfaces / f'{h.lower()}h.inflated') for h in ('L', 'R')}
    views = [('L', -1, 'Left lateral'), ('L', 1, 'Left medial'),
             ('R', -1, 'Right medial'), ('R', 1, 'Right lateral')]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'savefig.facecolor': 'white'})
    coverage_colors = np.array([[.84, .85, .87, 1], [.88, .28, .08, 1], [.47, .12, .65, 1]])

    def coverage_codes(record, hemi):
        with np.load(args.input / record['hemispheres'][hemi]['coverage_file']) as data:
            imputed = data['imputed_mask'] if 'imputed_mask' in data else data['missing_mask']
            unresolved = data['unresolved_zero_mask'] if 'unresolved_zero_mask' in data else np.zeros_like(imputed)
            return imputed.astype(int) + 2 * unresolved.astype(int)

    for subject in subjects:
        record = next(r for r in preparation['runs'] if r['subject'] == subject)
        masks = {h: coverage_codes(record, h) for h in ('L', 'R')}
        for other in (r for r in preparation['runs'] if r['subject'] == subject):
            for hemi in ('L', 'R'):
                other_mask = coverage_codes(other, hemi)
                if not np.array_equal(masks[hemi], other_mask):
                    raise ValueError('Session coverage masks differ; render and label them separately')
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.1))
        for ax, (hemi, side, title) in zip(axes, views, strict=True):
            surface_panel(ax, geometry[hemi], masks[hemi], coverage_colors, side=side, title=title, binary=True)
        counts = {h: int(np.count_nonzero(masks[h] == 1)) for h in ('L', 'R')}
        zero_count = sum(int(np.count_nonzero(masks[h] == 2)) for h in ('L', 'R'))
        seeds = sum(record['hemispheres'][h]['imputed_seed_vertices'] for h in ('L', 'R'))
        fig.suptitle(f'{subject}: cortical boundary coverage', fontsize=13)
        fig.text(.5, .015, f"Orange: {counts['L']} left + {counts['R']} right vertices; {seeds} of 1,175 seeds. "
                 f'Purple: {zero_count} unresolved vertices.\n'
                 'Imputation stays within 5 mm; colored faces touch affected vertices; counts use exact vertices.',
                 ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .05, 1, .92))
        fig.savefig(output / f'{subject}-input-coverage.png', dpi=180)
        plt.close(fig)
    if args.workdir is None or args.python_output is None:
        return
    comparison = json.loads((args.workdir / 'comparison.json').read_text())
    manifest = json.loads((args.workdir / 'manifest.json').read_text())
    reference = loadmat(args.workdir / 'reference-params.mat', simplify_cells=True)['Params']
    python = loadmat(args.python_output / 'model/priors/Params_Final.mat', simplify_cells=True)['Params']
    n, _, s, _ = manifest['shape_N_D_S_T']
    probabilities = np.asarray(reference['s_lambda']).reshape(n, 15, s)
    labels = probabilities.argmax(axis=1) + 1
    labels[probabilities.sum(axis=1) == 0] = 0
    colors = np.zeros((16, 4))
    colors[0] = [.84, .85, .87, 1]
    for name, index, red, green, blue, alpha in assets.colors:
        colors[index] = [red / 255, green / 255, blue / 255, alpha / 255]
    difference_colors = np.array([[.84, .85, .87, 1], [.84, .1, .15, 1]])
    for index, subject in enumerate(manifest['subjects']):
        predicted = {}
        expected = {}
        for hemi, section in [('L', slice(0, n // 2)), ('R', slice(n // 2, None))]:
            path = args.python_output / f'sub-{subject}/func/sub-{subject}_space-fsaverage6_atlas-DU15NET_hemi-{hemi}_dseg.label.gii'
            predicted[hemi] = nib.load(path).agg_data()
            expected[hemi] = labels[section, index]
        fig, axes = plt.subplots(3, 4, figsize=(12, 8.4))
        for col, (hemi, side, title) in enumerate(views):
            for row, (series, palette, label) in enumerate([
                (predicted[hemi], colors, 'Python'), (expected[hemi], colors, 'CBIG / Octave'),
                ((predicted[hemi] != expected[hemi]).astype(int), difference_colors, 'Disagreement (red)')]):
                surface_panel(axes[row, col], geometry[hemi], series, palette, side=side,
                              title=f'{label} · {title}', binary=row == 2)
        count = sum(np.count_nonzero(predicted[h] != expected[h]) for h in ('L', 'R'))
        fig.suptitle(f'{subject} · MS-HBM on OpenNeuro ds000224 · {count:,} differing vertex labels', fontsize=14)
        handles = [Patch(facecolor=colors[item[1]], label=f'{item[1]} {item[0]}') for item in assets.colors]
        fig.legend(handles=handles, loc='lower center', ncol=8, frameon=False, fontsize=8)
        fig.tight_layout(rect=(0, .085, 1, .955))
        fig.savefig(output / f'{subject}-network-comparison.png', dpi=180)
        fig.savefig(output / f'{subject}-network-comparison.pdf', dpi=180)
        plt.close(fig)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for data, label, style in [(python, 'Python', '-o'), (reference, 'CBIG / Octave', '--x')]:
        history = np.atleast_1d(data.get('Record', data.get('record'))).ravel()
        ax1.plot(np.arange(1, len(history) + 1), history, style, label=label, markersize=5)
    ax1.set(xlabel='Outer iteration', ylabel='Source objective', title='Stopping history')
    ax1.legend(frameon=False)
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    for index, subject in enumerate(manifest['subjects']):
        dice = comparison['labels']['subjects'][subject]['per_network_dice']
        ax2.plot(range(1, 16), [dice[str(i)] for i in range(1, 16)], 'o-', label=subject)
    ax2.set(xlabel='Unpermuted DU15NET network ID', ylabel='Dice', title='Network label agreement',
            ylim=(0, 1.025), xticks=range(1, 16))
    ax2.legend(frameon=False)
    for ax in (ax1, ax2):
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Real-data numerical comparison · shared prepared MSC input', fontsize=13)
    fig.tight_layout()
    fig.savefig(output / 'numerical-summary.png', dpi=180)
    fig.savefig(output / 'numerical-summary.pdf')
    plt.close(fig)
    (output / 'figure-provenance.json').write_text(json.dumps({
        'plot_script_sha256': sha256_file(Path(__file__)), 'matplotlib': matplotlib.__version__,
        'geometry': {str(p.name): sha256_file(p) for p in args.surfaces.glob('*.inflated')},
        'geometry_source': 'FreeSurfer 8.1.0-1 fsaverage6 inflated display meshes',
        'comparison_sha256': sha256_file(args.workdir / 'comparison.json'),
        'preparation_sha256': sha256_file(args.input / 'preparation.json'),
        'rendering': 'Orthographic depth-sorted triangles; majority network label per face; '
                     'binary overlays expand to faces touching any affected vertex; statistics use exact vertices',
    }, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'surfaces', 'assets', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--workdir', type=Path)
    parser.add_argument('--python-output', type=Path)
    run(parser.parse_args())
