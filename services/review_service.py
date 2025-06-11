import asyncio
import os
import json
from typing import List, Dict, Any, Optional, Tuple
from copy import deepcopy
from models import TreeNode
from services.tree_utils import get_all_nodes, find_new_nodes
from services.openai_service import generate_review, rank_reviews


class ReviewService:
    def __init__(self):
        # In-memory cache of previous_tree and unselected_reviews per student and assignment
        self.state_cache = {}
        
        # Ensure data directory exists
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        os.makedirs(self.data_dir, exist_ok=True)
    
    def _get_file_path(self, student_id: str, assignment_id: str) -> str:
        """Get the file path for a student's assignment data."""
        return os.path.join(self.data_dir, f"{student_id}_{assignment_id}.json")
    
    def _get_state(self, student_id: str, assignment_id: str) -> Tuple[Optional[TreeNode], List[Dict[str, Any]]]:
        """Get the state for a student's assignment, either from cache or from file."""
        cache_key = f"{student_id}_{assignment_id}"
        
        # Check if state is in cache
        if cache_key in self.state_cache:
            return self.state_cache[cache_key]["previous_tree"], self.state_cache[cache_key]["unselected_reviews"]
        
        # Try to load from file
        file_path = self._get_file_path(student_id, assignment_id)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                previous_tree = None
                if data.get("previous_tree"):
                    previous_tree = TreeNode.parse_obj(data["previous_tree"])
                
                unselected_reviews = data.get("unselected_reviews", [])
                
                # Update cache
                self.state_cache[cache_key] = {
                    "previous_tree": previous_tree,
                    "unselected_reviews": unselected_reviews
                }
                
                return previous_tree, unselected_reviews
            except Exception as e:
                print(f"Error loading state from file: {e}")
        
        # Return default empty state if no data found
        return None, []
    
    def _save_state(self, student_id: str, assignment_id: str, previous_tree: Optional[TreeNode], unselected_reviews: List[Dict[str, Any]]):
        """Save the state for a student's assignment to both cache and file."""
        cache_key = f"{student_id}_{assignment_id}"
        
        # Update cache
        self.state_cache[cache_key] = {
            "previous_tree": previous_tree,
            "unselected_reviews": unselected_reviews
        }
        
        # Save to file
        file_path = self._get_file_path(student_id, assignment_id)
        try:
            data = {
                "previous_tree": previous_tree.dict() if previous_tree else None,
                "unselected_reviews": unselected_reviews
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving state to file: {e}")
    
    def reset_state(self, student_id: str, assignment_id: str):
        """Reset the service state for a specific student and assignment."""
        cache_key = f"{student_id}_{assignment_id}"
        
        # Clear from cache
        if cache_key in self.state_cache:
            del self.state_cache[cache_key]
        
        # Remove file if it exists
        file_path = self._get_file_path(student_id, assignment_id)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error removing state file: {e}")
    
    def reset_all(self):
        """Reset all stored states."""
        # Clear cache
        self.state_cache = {}
        
        # Remove all data files
        if os.path.exists(self.data_dir):
            for filename in os.listdir(self.data_dir):
                if filename.endswith(".json"):
                    try:
                        os.remove(os.path.join(self.data_dir, filename))
                    except Exception as e:
                        print(f"Error removing state file {filename}: {e}")
    
    async def process_review_request(self, tree: TreeNode, review_num: int, student_id: str, assignment_id: str) -> List[Dict[str, Any]]:
        """Process a review request and return ranked reviews."""
        print("=== Review Service: Processing request ===")
        print(f"Tree root ID: {tree.id}")
        print(f"Tree root type: {tree.type}")
        print(f"Review num: {review_num}")
        print(f"Number of child nodes: {len(tree.child)}")
        print(f"Number of sibling nodes: {len(tree.sibling)}")
        print(f"Student ID: {student_id}")
        print(f"Assignment ID: {assignment_id}")
        
        # Get current state for this student and assignment
        previous_tree, unselected_reviews = self._get_state(student_id, assignment_id)
        
        # Get current tree state as dictionary
        current_tree_dict = get_all_nodes(tree)
        print("Previous tree exists:", previous_tree is not None)
        
        # Determine which nodes to review
        new_nodes = self._determine_nodes_to_review(current_tree_dict, tree, previous_tree)
        print(f"New nodes to review: {len(new_nodes)}")
        
        # Generate reviews for new nodes if available
        reviews = []
        if new_nodes:
            reviews = await self._generate_reviews_for_nodes(new_nodes, tree)
            print(f"Generated reviews: {len(reviews)}")
        # If there are no new nodes and no unselected reviews, raise an exception
        elif not unselected_reviews:
            raise ValueError("No '근거' type nodes found to review and no unselected reviews available. Please add or update nodes of type '근거'.")
        else:
            print(f"No new nodes to review, using {len(unselected_reviews)} unselected reviews")
        
        # Combine newly generated reviews with previously unselected reviews
        combined_reviews = reviews + unselected_reviews
        print(f"Combined reviews (new + unselected): {len(combined_reviews)}")
        
        # Rank the reviews and get the top ones
        ranked_reviews = await rank_reviews(combined_reviews, tree, review_num)
        print(f"Ranked reviews: {len(ranked_reviews)}")
        
        # If we have multiple ranked reviews, use Cerebras to select the best one and place it first
        if len(ranked_reviews) > 1:
            from services.openai_service import select_best_overall_review
            best_review = await select_best_overall_review(ranked_reviews, tree)
            print(f"Selected best overall review of type: {best_review.get('tree', {}).get('type', 'unknown')}")
            
            # Put the best review at the top of the list
            if best_review in ranked_reviews:
                ranked_reviews.remove(best_review)
                ranked_reviews.insert(0, best_review)
        
        # Update unselected reviews for future use
        new_unselected_reviews = self._calculate_unselected_reviews(combined_reviews, ranked_reviews)
        
        # Store the current tree and unselected reviews for future comparison
        self._save_state(student_id, assignment_id, deepcopy(tree), new_unselected_reviews)
        
        return ranked_reviews
    
    def _determine_nodes_to_review(self, current_tree_dict: Dict[str, TreeNode], tree: TreeNode, previous_tree: Optional[TreeNode]) -> List[TreeNode]:
        """Determine which nodes need to be reviewed."""
        if previous_tree:
            # Get previous tree state
            previous_tree_dict = get_all_nodes(previous_tree)
            
            # Find new nodes
            new_nodes = find_new_nodes(current_tree_dict, previous_tree_dict)
            
            # No fallback - if no new nodes, return empty list
            # An empty list will cause the review generation process to fail gracefully
        else:
            # First time seeing any tree, filter for nodes of type '근거' or '답변'
            new_nodes = []
            for node_id, node in current_tree_dict.items():
                if node.type in ["근거", "답변"]:
                    new_nodes.append(node)
        
        return new_nodes
    
    async def _generate_reviews_for_nodes(self, nodes: List[TreeNode], tree: TreeNode) -> List[Dict[str, Any]]:
        """Generate reviews for the given nodes in parallel using the new workflow.
        
        Each node will go through the following steps:
        1. Generate search query with GPT-4.1-mini
        2. Get search results from Perplexity API
        3. Generate reviews with different personas in parallel
        4. Select the best review for each evidence using Cerebras API
        """
        print("Generating reviews with new workflow for multiple nodes")
        review_tasks = [generate_review(node, tree) for node in nodes]
        reviews = await asyncio.gather(*review_tasks)
        
        # Flatten the list of reviews if any are lists themselves
        flattened_reviews = []
        for review in reviews:
            if isinstance(review, list):
                flattened_reviews.extend(review)
            else:
                flattened_reviews.append(review)
        
        print(f"Generated {len(flattened_reviews)} reviews for {len(nodes)} nodes using new workflow")
        return flattened_reviews
    
    def _calculate_unselected_reviews(self, combined_reviews: List[Dict[str, Any]], ranked_reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate the list of unselected reviews for future use."""
        unselected = []
        
        if len(combined_reviews) > len(ranked_reviews):
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
                    unselected.append(review)
            
            print(f"Stored {len(unselected)} unselected reviews for future use")
        
        return unselected
