import asyncio
from typing import List, Dict, Any
from copy import deepcopy
from models import TreeNode
from services.tree_utils import get_all_nodes, find_new_nodes
from services.openai_service import generate_review, rank_reviews


class ReviewService:
    def __init__(self):
        self.previous_tree = None
        self.unselected_reviews = []
    
    def reset_state(self):
        """Reset the service state by clearing previous tree and unselected reviews."""
        self.previous_tree = None
        self.unselected_reviews = []
    
    async def process_review_request(self, tree: TreeNode, review_num: int) -> List[Dict[str, Any]]:
        """Process a review request and return ranked reviews."""
        print("=== Review Service: Processing request ===")
        print(f"Tree root ID: {tree.id}")
        print(f"Tree root type: {tree.type}")
        print(f"Review num: {review_num}")
        print(f"Number of child nodes: {len(tree.child)}")
        print(f"Number of sibling nodes: {len(tree.sibling)}")
        
        # Get current tree state as dictionary
        current_tree_dict = get_all_nodes(tree)
        print("Previous tree exists:", self.previous_tree is not None)
        
        # Determine which nodes to review
        new_nodes = self._determine_nodes_to_review(current_tree_dict, tree)
        print(f"New nodes to review: {len(new_nodes)}")
        
        # Generate reviews for new nodes in parallel
        reviews = await self._generate_reviews_for_nodes(new_nodes, tree)
        print(f"Generated reviews: {len(reviews)}")
        
        # Combine newly generated reviews with previously unselected reviews
        combined_reviews = reviews + self.unselected_reviews
        print(f"Combined reviews (new + unselected): {len(combined_reviews)}")
        
        # Rank the reviews and get the top ones
        ranked_reviews = await rank_reviews(combined_reviews, tree, review_num)
        print(f"Ranked reviews: {len(ranked_reviews)}")
        
        # Update unselected reviews for future use
        self._update_unselected_reviews(combined_reviews, ranked_reviews)
        
        # Store the current tree for future comparison
        self.previous_tree = deepcopy(tree)
        
        return ranked_reviews
    
    def _determine_nodes_to_review(self, current_tree_dict: Dict[str, TreeNode], tree: TreeNode) -> List[TreeNode]:
        """Determine which nodes need to be reviewed."""
        if self.previous_tree:
            # Get previous tree state
            previous_tree_dict = get_all_nodes(self.previous_tree)
            
            # Find new nodes
            new_nodes = find_new_nodes(current_tree_dict, previous_tree_dict)
            
            if not new_nodes:
                # If no new nodes, review root level nodes as fallback
                new_nodes = [tree] + tree.child
        else:
            # First time seeing any tree, filter for nodes of type '근거' or '답변'
            new_nodes = []
            for node_id, node in current_tree_dict.items():
                if node.type in ["근거", "답변"]:
                    new_nodes.append(node)
        
        return new_nodes
    
    async def _generate_reviews_for_nodes(self, nodes: List[TreeNode], tree: TreeNode) -> List[Dict[str, Any]]:
        """Generate reviews for the given nodes in parallel."""
        review_tasks = [generate_review(node, tree) for node in nodes]
        reviews = await asyncio.gather(*review_tasks)
        
        # Flatten the list of reviews if any are lists themselves
        flattened_reviews = []
        for review in reviews:
            if isinstance(review, list):
                flattened_reviews.extend(review)
            else:
                flattened_reviews.append(review)
        
        return flattened_reviews
    
    def _update_unselected_reviews(self, combined_reviews: List[Dict[str, Any]], ranked_reviews: List[Dict[str, Any]]):
        """Update the list of unselected reviews for future use."""
        if len(combined_reviews) > len(ranked_reviews):
            # Find reviews that weren't selected
            self.unselected_reviews = []
            
            for review in combined_reviews:
                review_id = review.get("parent", "")
                review_content = review.get("tree", {}).get("content", "")
                
                # Check if this review isn't in the selected list
                # or if it's a different review for the same parent node
                is_duplicate = False
                for selected_review in ranked_reviews:
                    if (selected_review.get("parent", "") == review_id and 
                        selected_review.get("tree", {}).get("content", "") == review_content):
                        is_duplicate = True
                        break
                
                if not is_duplicate and review not in ranked_reviews:
                    self.unselected_reviews.append(review)
            
            print(f"Stored {len(self.unselected_reviews)} unselected reviews for future use")
