from typing import Dict, List, Any
import os
try:
    from github import Github
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False

def search_repositories(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Search for repositories matching the profile.
    For MVP, we search for repositories containing files with matching names/sizes 
    or just use a mock response if no API key.
    """
    matches = []
    
    # Check for GitHub Token
    token = os.environ.get("GITHUB_TOKEN")
    
    if GITHUB_AVAILABLE and token:
        g = Github(token)
        # Strategy: Pick the largest/most unique file and search for it by name
        # This is a naive search strategy for the MVP
        fingerprints = profile.get('fingerprints', [])
        if fingerprints:
            # Sort by size (descending) to find significant files
            sorted_files = sorted(fingerprints, key=lambda x: x['size'], reverse=True)
            candidate = sorted_files[0] if sorted_files else None
            
            if candidate:
                query = f"filename:{os.path.basename(candidate['path'])}"
                try:
                    result = g.search_code(query)
                    for item in result[:5]: # Limit to 5
                        matches.append({
                            'source': 'github',
                            'repo': item.repository.full_name,
                            'file_url': item.html_url,
                            'score': 0.8 # Placeholder score
                        })
                except Exception as e:
                    print(f"GitHub Search Error: {e}")
    else:
        # Mock Response for demo
        matches.append({
            'source': 'mock_db',
            'repo': 'afterdarksys/ads-codedetective-clone',
            'file_url': 'http://github.com/afterdarksys/clone',
            'score': 0.95,
            'reason': 'High structural similarity detected'
        })
        
    return matches
