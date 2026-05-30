"""
Report generator.

Renders a rich terminal report with:
  - Summary panel
  - Match table with per-signal score breakdown
  - Detail panels for top matches
  - Mermaid supply-chain diagram
  - Optional JSON export
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

_CONFIDENCE_COLORS = {
    'CRITICAL':   'bright_red',
    'HIGH':       'red',
    'MEDIUM':     'yellow',
    'LOW':        'cyan',
    'NEGLIGIBLE': 'dim',
}

_MERMAID_STYLES = {
    'CRITICAL':   'fill:#cc0000,color:#fff,stroke:#880000',
    'HIGH':       'fill:#ff6600,color:#fff,stroke:#cc4400',
    'MEDIUM':     'fill:#ffcc00,color:#000,stroke:#aa8800',
    'LOW':        'fill:#66aaff,color:#000,stroke:#3366cc',
    'NEGLIGIBLE': 'fill:#dddddd,color:#555,stroke:#aaaaaa',
}


def generate_report(
    matches: List[Dict[str, Any]],
    profile: Optional[Dict[str, Any]] = None,
    json_output: Optional[str] = None,
) -> None:
    """
    Print a rich terminal report for *matches*.

    Args:
        matches:     Output of ``searcher.search_repositories``.
        profile:     The .cd profile dict (used for the JSON export summary).
        json_output: If given, save a machine-readable report to this path.
    """
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')

    if not matches:
        console.print(Panel(
            '[bold green]No suspicious matches found.[/bold green]\n'
            f'Timestamp: {ts}',
            title='[bold]ADS CodeDetective Report[/bold]',
            border_style='green',
        ))
        return

    # ── Summary ───────────────────────────────────────────────────────────────
    high_count = sum(
        1 for m in matches
        if m.get('confidence') in {'CRITICAL', 'HIGH'}
    )
    summary_lines = [
        f'[bold]Scan complete.[/bold]  '
        f'Found [bold red]{len(matches)}[/bold red] potential match(es), '
        f'[bold red]{high_count}[/bold red] HIGH or CRITICAL.',
        f'Timestamp: {ts}',
    ]
    if profile:
        summary_lines.append(
            f"Profile: {profile.get('total_files', '?')} files · "
            f"{profile.get('total_size_bytes', 0) // 1024} KB · "
            f"{len(profile.get('identifiers', []))} identifiers · "
            f"{len(profile.get('winnow_fingerprint', []))} fingerprint hashes"
        )
    console.print()
    console.print(Panel(
        '\n'.join(summary_lines),
        title='[bold]ADS CodeDetective Report[/bold]',
        border_style='bright_blue',
    ))

    # ── Match table ───────────────────────────────────────────────────────────
    table = Table(
        title='Potential Code Copies — sorted by composite score',
        box=box.ROUNDED,
        show_lines=True,
        border_style='bright_blue',
    )
    table.add_column('#',                  style='dim',   width=3,  justify='right')
    table.add_column('Repository',         style='green', min_width=28)
    table.add_column('Confidence',         justify='center', width=11)
    table.add_column('Score',              justify='right', width=7)
    table.add_column('Hash\nOverlap',      justify='right', width=8)
    table.add_column('Winnow\nJaccard',    justify='right', width=9)
    table.add_column('Identifier\nOverlap',justify='right', width=10)
    table.add_column('Lang',               justify='center', width=8)
    table.add_column('Stars',              justify='right', width=6)

    for i, m in enumerate(matches, 1):
        conf  = m.get('confidence', 'NEGLIGIBLE')
        color = _CONFIDENCE_COLORS.get(conf, 'white')
        table.add_row(
            str(i),
            m.get('repo', 'unknown'),
            f'[{color}]{conf}[/{color}]',
            f"{m.get('composite', 0):.3f}",
            f"{m.get('hash_overlap', 0):.3f}",
            f"{m.get('winnow_jaccard', 0):.3f}",
            f"{m.get('identifier_overlap', 0):.3f}",
            m.get('language', '—')[:8],
            str(m.get('stars', '—')),
        )

    console.print()
    console.print(table)

    # ── Detail panels (top 5) ─────────────────────────────────────────────────
    console.print()
    for i, m in enumerate(matches[:5], 1):
        conf  = m.get('confidence', 'NEGLIGIBLE')
        color = _CONFIDENCE_COLORS.get(conf, 'white')
        url   = m.get('repo_url', m.get('file_url', ''))
        body  = (
            f"[link={url}]{url}[/link]\n"
            f"Language: {m.get('language', '—')}  |  Stars: {m.get('stars', '—')}\n\n"
            f"  Composite score:      [bold]{m.get('composite', 0):.4f}[/bold]\n"
            f"  Hash overlap:         {m.get('hash_overlap', 0):.4f}  "
            f"(weight 40% — exact byte matches)\n"
            f"  Winnow Jaccard:       {m.get('winnow_jaccard', 0):.4f}  "
            f"(weight 35% — structural similarity)\n"
            f"  Identifier overlap:   {m.get('identifier_overlap', 0):.4f}  "
            f"(weight 25% — shared function/class names)"
        )
        console.print(Panel(
            body,
            title=f'[{color}][{i}] {m.get("repo", "unknown")} — {conf}[/{color}]',
            border_style=color,
        ))

    # ── Mermaid diagram ───────────────────────────────────────────────────────
    mmd = _build_mermaid(matches)
    console.print()
    console.print('[bold]Supply Chain Graph (Mermaid):[/bold]')
    console.print(f'```mermaid\n{mmd}\n```')
    with open('supply_chain_graph.mmd', 'w') as fh:
        fh.write(mmd)
    console.print('[dim]Graph saved to supply_chain_graph.mmd[/dim]')

    # ── JSON export ───────────────────────────────────────────────────────────
    if json_output:
        report: Dict[str, Any] = {
            'generated_at': ts,
            'match_count':  len(matches),
            'matches':      matches,
        }
        if profile:
            report['profile_summary'] = {
                'total_files':       profile.get('total_files'),
                'total_size_bytes':  profile.get('total_size_bytes'),
                'languages':         profile.get('languages'),
                'identifier_count':  len(profile.get('identifiers', [])),
                'fingerprint_hashes': len(profile.get('winnow_fingerprint', [])),
            }
        with open(json_output, 'w') as fh:
            json.dump(report, fh, indent=2)
        console.print(f'[dim]JSON report saved to {json_output}[/dim]')


def _build_mermaid(matches: List[Dict[str, Any]]) -> str:
    lines = [
        'graph LR',
        '    Origin["Our Codebase"]:::origin',
    ]
    for i, m in enumerate(matches):
        conf    = m.get('confidence', 'LOW')
        score   = m.get('composite', 0.0)
        node_id = f'R{i}'
        # Sanitize label: keep alphanumerics, slashes, dots, hyphens
        safe    = re.sub(r'[^\w/.\-]', '_', m.get('repo', f'repo{i}'))
        label   = f'{safe}\\n{conf} ({score:.2f})'
        style   = _MERMAID_STYLES.get(conf, _MERMAID_STYLES['NEGLIGIBLE'])

        lines.append(f'    {node_id}["{label}"]')
        lines.append(f'    style {node_id} {style}')
        lines.append(f'    Origin -->|"score {score:.2f}"| {node_id}')

    lines.append('    classDef origin fill:#0055cc,color:#fff,stroke:#003399')
    return '\n'.join(lines)
