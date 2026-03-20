import typer
import json
import os
from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.table import Table
from .scanner import scan_directory
from .profiler import generate_profile
from .searcher import search_repositories
from .reporter import generate_report

app = typer.Typer(help="ADS CodeDetective - Software Supply Chain Monitoring")
console = Console()

@app.command()
def scan(
    directory: str = typer.Argument(..., help="Directory to scan"),
    output: str = typer.Option("detectables.cd", help="Output file for the profile"),
    full: bool = typer.Option(False, help="Perform full analysis including ML profiling")
):
    """
    Scan a directory to create a fingerprint and ML profile.
    """
    console.print(f"[bold green]Scanning directory:[/bold green] {directory}")
    
    try:
        # Step 1: Scan and Fingerprint
        with console.status("Scanning files and generating hashes..."):
            fingerprints = scan_directory(directory)
        
        console.print(f"Found [bold]{len(fingerprints)}[/bold] files.")
        
        # Step 2: ML Profiling (if requested or default)
        # For MVP, we always do basic profiling
        with console.status("Generating code profile..."):
            profile = generate_profile(fingerprints, directory)
            
        # Save to output
        with open(output, 'w') as f:
            json.dump(profile, f, indent=2)
            
        console.print(f"[bold green]Success![/bold green] Profile saved to {output}")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")

@app.command()
def investigate(
    profile_path: str = typer.Argument("detectables.cd", help="Path to profile file"),
    threshold: float = typer.Option(0.8, help="Similarity threshold")
):
    """
    Search for potential rip-offs using a profile.
    """
    if not os.path.exists(profile_path):
        console.print(f"[bold red]Error:[/bold red] Profile {profile_path} not found.")
        return

    with open(profile_path, 'r') as f:
        profile = json.load(f)

    with console.status("Searching repositories..."):
        matches = search_repositories(profile)
    
    generate_report(matches)

if __name__ == "__main__":
    app()
