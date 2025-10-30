from pathlib import Path
import uvicorn


if __name__ == "__main__":
    # Minimal dev runner with safe reload scope
    root = Path(__file__).resolve().parent
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(root / "backend"), str(root / "src")],
        reload_excludes=[".git", "node_modules", "frontend/node_modules"],
        log_level="info",
    )
