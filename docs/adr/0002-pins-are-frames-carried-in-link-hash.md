# ADR 0002 — A pin is a frame, carried between pair pages in the link hash

**Status:** Accepted

## Context

Each coordinate pair of a run is exported as its own standalone HTML page, opened from local
files with no server. Researchers wanted a pinned structure to stay visible when they switch
from one pair page to another, so they can see where the same conformer sits in the other
coordinates.

## Decision

A pin identifies one frame (the clicked bin's representative frame), not a bin. When the
researcher follows a header pair link, the page appends the current pins to the link hash
(`#pins=…`). Each pin carries its frame identity, its coordinates for every pair of the run,
its metadata and its structure text. The receiving page places the pin at the frame's own
coordinates in its pair and shows the carried structure. The hash is kept in step with the
pins, so reload and back/forward restore them.

## Considered options

- **Browser storage (rejected).** It would also work when a page is opened by double-click,
  but Firefox gives every local file its own origin, so it would silently fail there. Pins
  from an old run could also leak into a new run's pages.
- **Re-pin the bin's representative on each page (rejected).** No structure would need to be
  carried, but the card would show a different molecule on each page under the same number.

## Consequences

- Every page must embed, for each bin's representative frame, its `frame_id` and the values of
  all DoF used by any pair of the run, not only its own two axes.
- Pins only travel through the header links; a page opened on its own starts empty.
- The pinned set, the pin numbers and the pin limit are shared across the run's pages.
