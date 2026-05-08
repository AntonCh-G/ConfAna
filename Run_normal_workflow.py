from pathlib import Path

import yaml
from src.io_coordinates import load_or_build_coordinate_table_from_config
from src.plots_static import make_density_png, make_transition_png
from src.plots_interactive import make_density_interactive
from src.states import assign_conformer_states_from_config
from src.transitions import analyze_grouped_transitions

cfg = yaml.safe_load(open('configs/default.yaml'))
out = Path(cfg['outputs']['dir'])
out.mkdir(parents=True, exist_ok=True)

df, cache_hit = load_or_build_coordinate_table_from_config(cfg)
print(f'cache_hit={cache_hit}, frames={len(df)}')

# Phase 7 — density plots
make_density_png(df, 'plane',    out / 'density_plane.png',    dpi=cfg['outputs']['dpi'], config=cfg)
make_density_png(df, 'dihedral', out / 'density_dihedral.png', dpi=cfg['outputs']['dpi'], config=cfg)

# Phase 8 — state assignment
df = assign_conformer_states_from_config(df, cfg['clustering'])

# Phase 9 — transition analysis (both definitions)
plane    = analyze_grouped_transitions(df, 'plane',    lag=cfg['transitions']['lag'], dt=cfg['transitions']['dt'])
dihedral = analyze_grouped_transitions(df, 'dihedral', lag=cfg['transitions']['lag'], dt=cfg['transitions']['dt'])
print(f"plane_groups={len(plane['per_group_counts'])}, dihedral_groups={len(dihedral['per_group_counts'])}")

# Phase 10 — static transition heatmaps
make_transition_png(plane,    out / 'transition_plane.png',    dpi=cfg['outputs']['dpi'], config=cfg)
make_transition_png(dihedral, out / 'transition_dihedral.png', dpi=cfg['outputs']['dpi'], config=cfg)
print("Saved transition_plane.png and transition_dihedral.png")

# Phase 11 — interactive HTML
make_density_interactive(df, 'plane',    out / 'density_plane.html',    config=cfg)
make_density_interactive(df, 'dihedral', out / 'density_dihedral.html', config=cfg)
print("Saved density_plane.html and density_dihedral.html")
