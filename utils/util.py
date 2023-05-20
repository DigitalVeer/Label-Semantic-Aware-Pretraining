import os
import sys

def get_folders( path ):
    """
    Returns a list of all folders in the given path.
    
    Args:
        path (str): The path to the folder to search.
    """
    folders = []
    for name in os.listdir( path ):
        if os.path.isdir( os.path.join( path, name ) ):
            folders.append( name )
    return folders
