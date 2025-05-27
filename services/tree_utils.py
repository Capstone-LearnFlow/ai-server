from typing import Dict, List, Optional
from copy import deepcopy
from models import TreeNode


def find_node_by_id(tree: TreeNode, node_id: str) -> Optional[TreeNode]:
    """Find a node in the tree by its ID."""
    if tree.id == node_id:
        return tree
    
    # Search in child nodes
    for child in tree.child:
        result = find_node_by_id(child, node_id)
        if result:
            return result
    
    # Search in sibling nodes
    for sibling in tree.sibling:
        result = find_node_by_id(sibling, node_id)
        if result:
            return result
    
    return None


def get_all_nodes(tree: TreeNode) -> Dict[str, TreeNode]:
    """Get all nodes in the tree as a dictionary with node IDs as keys."""
    result = {tree.id: tree}
    
    # Process child nodes
    for child in tree.child:
        result.update(get_all_nodes(child))
    
    # Process sibling nodes
    for sibling in tree.sibling:
        result.update(get_all_nodes(sibling))
    
    return result


def get_parent_map(tree: TreeNode) -> Dict[str, TreeNode]:
    """Build a dictionary mapping node IDs to their parent nodes."""
    parent_map = {}
    
    def process_children(parent_node, children):
        for child in children:
            parent_map[child.id] = parent_node
            # Process this child's children
            process_children(child, child.child)
            # Process this child's siblings
            process_children(parent_node, child.sibling)
    
    # Process root node's children
    process_children(tree, tree.child)
    
    # Process root node's siblings
    for sibling in tree.sibling:
        parent_map[sibling.id] = None  # Root-level siblings don't have a parent in our context
        process_children(sibling, sibling.child)
    
    return parent_map


def extract_subtree_to_root(node: TreeNode, tree: TreeNode) -> TreeNode:
    """Extract a subtree that includes the path from the given node to the root.
    
    Args:
        node: The node to start from
        tree: The complete tree structure
        
    Returns:
        A new tree containing only the path from the node to the root
    """
    # Create a parent map for the entire tree
    parent_map = get_parent_map(tree)
    
    # Build a path from the node to the root
    path_nodes = [node]
    current_id = node.id
    
    while current_id in parent_map and parent_map[current_id]:
        parent_node = parent_map[current_id]
        path_nodes.append(parent_node)
        current_id = parent_node.id
    
    # Create a new subtree with only the nodes in the path
    # Start from the root (last node in the path)
    if not path_nodes:
        return deepcopy(node)  # Fallback to just the node if path is empty
    
    path_nodes.reverse()  # Reverse so root is first
    
    # Create a copy of the root node
    subtree = deepcopy(path_nodes[0])
    subtree.child = []  # Clear children
    subtree.sibling = []  # Clear siblings
    
    # Build the path down to the target node
    current_node = subtree
    for i in range(1, len(path_nodes)):
        child_copy = deepcopy(path_nodes[i])
        child_copy.child = []  # Clear children that aren't in the path
        child_copy.sibling = []  # Clear siblings
        
        current_node.child = [child_copy]  # Add as the only child
        current_node = child_copy  # Move down to the child for next iteration
    
    # If the last node in the path is not the original node, add it
    if path_nodes[-1].id != node.id:
        node_copy = deepcopy(node)
        node_copy.child = []
        node_copy.sibling = []
        current_node.child = [node_copy]
    
    return subtree


def find_new_nodes(current_tree: Dict[str, TreeNode], previous_tree: Dict[str, TreeNode]) -> List[TreeNode]:
    """Find nodes that were added since the previous tree state."""
    new_nodes = []
    
    # Build a dictionary of node IDs to their parent nodes
    # We need to reconstruct the tree first
    tree_root = None
    for node_id, node in current_tree.items():
        if tree_root is None or len(node.sibling) > 0:  # Simple heuristic to find a root node
            tree_root = node
            break
    
    if tree_root is None and current_tree:  # If we couldn't find a root but have nodes
        tree_root = next(iter(current_tree.values()))  # Just take the first node
        
    parent_map = get_parent_map(tree_root) if tree_root else {}
    
    for node_id, node in current_tree.items():
        # Check if this is a '근거' node with a '반론' parent, which we want to exclude
        should_exclude = False
        if node.type == "근거" and node_id in parent_map:
            parent_node = parent_map[node_id]
            if parent_node and parent_node.type == "반론":
                should_exclude = True
                
        # If node didn't exist before or is of type 근거 or 답변 (and not excluded) and has been updated
        if not should_exclude and (
            node_id not in previous_tree or 
            (node.type in ["근거", "답변"] and 
             (node.updated_at != previous_tree[node_id].updated_at if node_id in previous_tree else True))
        ):
            new_nodes.append(node)

    return new_nodes
