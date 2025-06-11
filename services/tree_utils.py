from typing import Dict, List, Optional, Tuple
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


def count_sibling_transitions(node_id: str, parent_map: Dict[str, TreeNode], sibling_map: Dict[str, List[str]]) -> int:
    """
    Count the number of sibling transitions from a node to the root.
    
    Args:
        node_id: The ID of the node to count from
        parent_map: Map of node IDs to their parent nodes
        sibling_map: Map of node IDs to their sibling node IDs
        
    Returns:
        The number of sibling transitions
    """
    transitions = 0
    current_id = node_id
    
    while current_id in parent_map and parent_map[current_id]:
        parent_id = parent_map[current_id].id
        
        # Check if any siblings of the current node have the parent as a child
        # This indicates a sibling transition
        if parent_id in sibling_map and sibling_map[parent_id]:
            transitions += 1
            
        current_id = parent_id
        
    return transitions

def build_sibling_map(tree_root: TreeNode) -> Dict[str, List[str]]:
    """
    Build a dictionary mapping node IDs to their sibling node IDs.
    
    Args:
        tree_root: The root node of the tree
        
    Returns:
        A dictionary mapping node IDs to their sibling node IDs
    """
    sibling_map = {}
    
    def process_node(node: TreeNode):
        # Map sibling IDs for this node
        sibling_map[node.id] = [sibling.id for sibling in node.sibling]
        
        # Process child nodes
        for child in node.child:
            process_node(child)
            
        # Process sibling nodes
        for sibling in node.sibling:
            process_node(sibling)
    
    process_node(tree_root)
    return sibling_map

def find_new_nodes(current_tree: Dict[str, TreeNode], previous_tree: Dict[str, TreeNode]) -> List[TreeNode]:
    """Find nodes that were added since the previous tree state."""
    new_nodes = []
    
    # Find any new node IDs that weren't in the previous tree
    new_node_ids = set(current_tree.keys()) - set(previous_tree.keys())
    
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
    sibling_map = build_sibling_map(tree_root) if tree_root else {}
    
    # First, add all new evidence nodes that weren't in the previous tree
    for node_id in new_node_ids:
        node = current_tree[node_id]
        if node.type == "근거":
            # Check if this is a '근거' node with a '반론' parent, which we want to exclude
            should_exclude = False
            if node_id in parent_map:
                parent_node = parent_map[node_id]
                if parent_node:
                    # Exclude if parent is a '반론'
                    if parent_node.type == "반론":
                        should_exclude = True
                    
                    # Exclude if sibling transitions > 2
                    # This handles cases like "주장 - 반론 - 주장" where the evidence
                    # for the last claim shouldn't be reviewed
                    if count_sibling_transitions(parent_node.id, parent_map, sibling_map) > 2:
                        should_exclude = True
            
            if not should_exclude:
                new_nodes.append(node)
    
    # If no new nodes were found, check for updated nodes
    if not new_nodes:
        for node_id, node in current_tree.items():
            # Skip nodes we've already processed
            if node_id in new_node_ids:
                continue
                
            # Only consider nodes of type "근거" (evidence)
            if node.type != "근거":
                continue
                
            # Check if this is a '근거' node with a '반론' parent, which we want to exclude
            should_exclude = False
            if node_id in parent_map:
                parent_node = parent_map[node_id]
                if parent_node:
                    # Exclude if parent is a '반론'
                    if parent_node.type == "반론":
                        should_exclude = True
                    
                    # Exclude if sibling transitions > 2
                    if count_sibling_transitions(parent_node.id, parent_map, sibling_map) > 2:
                        should_exclude = True
                    
            # Include node if it's not excluded and has been updated
            if not should_exclude and node_id in previous_tree and node.updated_at != previous_tree[node_id].updated_at:
                new_nodes.append(node)

    return new_nodes
