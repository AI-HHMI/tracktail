"""Re-render the cover time-lapse panel (right side of fig:rat7m) from a saved track file.

Defaults are the paper settings (poses at t = 0, 114, 252 frames = 0, 0.95, 2.1 s; 200 tracks).
    pixi run python scripts/cover/render_timelapse.py --trial <rat7m>/test/s5-d1/101500 \
        --track out/cover_rat7m/track.npz --out out/cover_rat7m/rat7m_cloud_timelapse_tracks \
        [--ts 0 114 252] [--azim -24] [--elev 42] [--roll -6] [--mirror] [--n-tracks 200] [--track-alpha 0.4] \
        [--ghost-size 1.3] [--ghost-alpha 1.0]
Writes <out>.png and <out>.pdf (points/tracks rasterized at 300 dpi, text editable, JPEG-compressed).
Needs ghostscript (`gs`) on PATH for the PDF compression step.
"""
import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cover_rat7m_demo as C  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--trial', required=True, help='trial dir with metadata.yaml and vid/<cam>.mp4')
ap.add_argument('--track', required=True, help='track.npz from cover_rat7m_demo.py --stages track')
ap.add_argument('--ts', type=int, nargs='+', default=[0, 114, 252])
ap.add_argument('--azim', type=float, default=-24); ap.add_argument('--elev', type=float, default=42)
ap.add_argument('--roll', type=float, default=-6); ap.add_argument('--mirror', action='store_true')
ap.add_argument('--n-tracks', type=int, default=200); ap.add_argument('--track-alpha', type=float, default=0.4)
ap.add_argument('--ghost-size', type=float, default=1.3); ap.add_argument('--ghost-alpha', type=float, default=1.0)
ap.add_argument('--labels', action='store_true', help='draw pose labels (off by default)')
ap.add_argument('--qlabel-offset', type=float, nargs=2, default=[0, 30])
ap.add_argument('--out', required=True, help='output path without extension')
a = ap.parse_args()

tr = dict(np.load(a.track, allow_pickle=True))
p = C.Panels(a.trial, {k: tr[k] for k in ['coords', 'vis', 'vis_cams', 'frames']})
raw = a.out + '_raw'
p.cloud_timelapse_glow(raw, ts=tuple(a.ts), azim=a.azim, elev=a.elev, roll=a.roll, mirror=a.mirror, n_tracks=a.n_tracks,
                       track_alpha=a.track_alpha, ghost_size=a.ghost_size, ghost_alpha=a.ghost_alpha,
                       qlabel_offset=tuple(a.qlabel_offset), label=a.labels)
subprocess.run(['gs', '-q', '-dNOPAUSE', '-dBATCH', '-dSAFER', '-sDEVICE=pdfwrite', '-dCompatibilityLevel=1.5',
                '-dDownsampleColorImages=false', '-dAutoFilterColorImages=false', '-dColorImageFilter=/DCTEncode',
                '-dJPEGQ=90', '-dAutoFilterGrayImages=false', '-dGrayImageFilter=/FlateEncode',
                f'-sOutputFile={a.out}.pdf', raw + '.pdf'], check=True)
os.replace(raw + '.png', a.out + '.png'); os.remove(raw + '.pdf')
print('wrote', a.out + '.pdf', a.out + '.png')
