# Read-only portfolio demo viewer

This fully static viewer renders the repository's synthetic evidence and
reconciliation fixtures. It performs no writes, executes no project commands,
and sends no data to a service.

## Run locally

From the repository root:

```powershell
cd agent-memory-os-demo
py -3.11 -m http.server 8000
```

Open <http://localhost:8000/demo/> in a browser. An HTTP server is required
because browsers do not permit the viewer's local JSON requests from a
`file://` page.

## Data source

The viewer reads only these committed public fixtures:

- `examples/sanitized-interview-agent/synthetic-reconciliation.json`
- `examples/sanitized-interview-agent/synthetic-evidence.json`

Every displayed project detail is synthetic. The interface is read-only and
does not derive new reconciliation decisions; it renders the recorded output.

## GitHub Pages

The relative asset and data paths work when the repository root is published
with GitHub Pages. The public URL will be documented only after visual review
and merge approval.
