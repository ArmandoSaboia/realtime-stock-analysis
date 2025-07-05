import os
import logging
from typing import Set, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

def safe_extract_paths(module: Any) -> Set[str]:
    """
    Safely extracts module paths while avoiding PyTorch-related issues.
    
    Args:
        module: Python module to extract paths from
        
    Returns:
        Set[str]: Set of absolute paths found in the module
    """
    paths = set()
    
    try:
        # Extract from __file__ attribute if available
        if hasattr(module, '__file__') and module.__file__:
            paths.add(os.path.abspath(str(module.__file__)))
            
        # Extract from __spec__.origin if available
        if hasattr(module, '__spec__') and module.__spec__ and module.__spec__.origin:
            paths.add(os.path.abspath(str(module.__spec__.origin)))
            
        # Safely extract from __path__ avoiding PyTorch custom classes
        if hasattr(module, '__path__'):
            path_attr = getattr(module, '__path__')
            # Skip PyTorch modules to avoid custom class issues
            if hasattr(path_attr, '_path') and 'torch' not in str(type(module)):
                paths.update(os.path.abspath(p) for p in path_attr._path)
                
    except Exception as e:
        logger.debug(f"Path extraction warning for module {module.__name__}: {str(e)}")
        
    # Filter only existing paths
    return {p for p in paths if os.path.exists(p)}

def patch_streamlit_watcher() -> bool:
    """
    Applies the patch to Streamlit's file watcher system.
    
    This function replaces Streamlit's default module path extraction
    with our safer version that handles PyTorch modules correctly.
    
    Returns:
        bool: True if patch was successfully applied, False otherwise
    """
    try:
        from streamlit.watcher import local_sources_watcher
        # Replace the path extraction function with our safe version
        local_sources_watcher.get_module_paths = safe_extract_paths
        logger.info("Successfully patched Streamlit file watcher")
        return True
        
    except ImportError as e:
        logger.error(f"Failed to patch Streamlit watcher: {str(e)}")
        return False

def is_torch_module(module: Any) -> bool:
    """
    Checks if a module is PyTorch-related to handle it specially.
    
    Args:
        module: Python module to check
        
    Returns:
        bool: True if module is PyTorch-related, False otherwise
    """
    module_path = str(getattr(module, '__file__', ''))
    return 'torch' in module_path or 'pytorch' in module_path.lower()