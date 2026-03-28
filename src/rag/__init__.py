__all__ = ["rag_pipeline", "load_esi_handbook_prefix"]


def __getattr__(name: str):
    """Lazy imports to avoid circular dependencies."""
    if name == "rag_pipeline":
        from src.rag.pipeline import rag_pipeline
        return rag_pipeline
    if name == "load_esi_handbook_prefix":
        from src.rag.esi_handbook import load_esi_handbook_prefix
        return load_esi_handbook_prefix
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
