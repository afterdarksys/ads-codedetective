from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table

console = Console()

def generate_report(matches: List[Dict[str, Any]]):
    """
    Generate a report of the findings, including a Mermaid diagram.
    """
    if not matches:
        console.print("[yellow]No matches found.[/yellow]")
        return

    # 1. Console Table
    table = Table(title="Potential Code Matches")
    table.add_column("Source", style="cyan")
    table.add_column("Repository", style="green")
    table.add_column("Score", style="magenta")
    table.add_column("URL", style="blue")

    for match in matches:
        table.add_row(
            match.get('source', 'unknown'),
            match.get('repo', 'unknown'),
            str(match.get('score', 0)),
            match.get('file_url', '')
        )

    console.print(table)

    # 2. Mermaid Diagram
    mermaid_content = "graph TD\n"
    mermaid_content += "    Origin[Origin Codebase] -->|Suspected Copy| Clones\n"
    
    for i, match in enumerate(matches):
        node_id = f"Repo{i}"
        repo_name = match.get('repo', 'Unknown Repo').replace('/', '_').replace('-', '_')
        mermaid_content += f"    Clones --> {node_id}[{repo_name}]\n"
        mermaid_content += f"    {node_id} -->|Score: {match.get('score')}| Origin\n"
        
    console.print("\n[bold]Supply Chain Graph (Mermaid):[/bold]")
    console.print(f"```mermaid\n{mermaid_content}\n```")
    
    # Save to file
    with open('supply_chain_graph.mmd', 'w') as f:
        f.write(mermaid_content)
    console.print("[dim]Graph saved to supply_chain_graph.mmd[/dim]")
