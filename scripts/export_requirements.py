import tomllib
import os

def export_requirements():
    """Reads dependencies from pyproject.toml and writes them to requirements.txt."""
    try:
        with open("pyproject.toml", "rb") as f:
            data = tomllib.load(f)
        
        dependencies = data.get("project", {}).get("dependencies", [])
        
        with open("requirements.txt", "w") as f:
            f.write("# Simple requirements.txt generated from pyproject.toml\n")
            for dep in dependencies:
                f.write(f"{dep}\n")
        
        print(f"Successfully exported {len(dependencies)} dependencies to requirements.txt")
        
    except Exception as e:
        print(f"Error exporting requirements: {e}")
        exit(1)

if __name__ == "__main__":
    export_requirements()
