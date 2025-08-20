"""
View Selection Algorithms for Close-up-GS
Implementation of anchor view and to-be-updated view selection algorithms
Paper Section 4.3.1-4.3.2, Algorithm 1 (Appendix A)
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
import math
from dataclasses import dataclass

@dataclass
class ViewInfo:
    """Container for view information"""
    pose: torch.Tensor  # 4x4 camera-to-world matrix
    camera_center: torch.Tensor  # 3D camera center
    viewing_direction: torch.Tensor  # 3D viewing direction
    index: int  # Original view index
    coverage_score: Optional[float] = None
    similarity_score: Optional[float] = None

class ViewSelector:
    """
    Implements anchor view and to-be-updated view selection algorithms
    Based on paper Section 4.3.1-4.3.2 and Algorithm 1 (Appendix A)
    """
    
    def __init__(self, 
                 image_width: int = 512,
                 image_height: int = 512,
                 focal_length: float = 500.0,
                 distance_discount_beta: float = 0.8):
        """
        Initialize view selector
        
        Args:
            image_width: Image width for coverage calculation
            image_height: Image height for coverage calculation  
            focal_length: Camera focal length for projection
            distance_discount_beta: Distance discount factor β (Appendix C)
        """
        self.image_width = image_width
        self.image_height = image_height
        self.focal_length = focal_length
        self.beta = distance_discount_beta
        
        # 步骤4: 设置全局设备
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # 步骤6: 全局常量处理
        self.up_vector = torch.tensor([0., 1., 0.], device=self.device)
        self.beta_tensor = torch.tensor(self.beta, device=self.device)
        
    def select_anchors(self, 
                      known_views: List[torch.Tensor],
                      frontier_views: List[torch.Tensor], 
                      p_target: torch.Tensor,
                      k: int = 5,
                      max_iterations: int = 1000) -> Tuple[List[int], Dict]:
        """
        Select anchor views using greedy optimization
        Paper Section 4.3.1, Algorithm 1 (Appendix A)
        
        Optimizes: max s^T w - s^T E s
        where:
        - w_i: coverage score (pixel overlap, center-weighted)
        - E: similarity matrix
        - s: selection binary vector
        
        Args:
            known_views: List of known view poses (4x4 matrices)
            frontier_views: List of frontier view poses (4x4 matrices)
            p_target: Object center position (3D point)
            k: Number of anchors to select
            max_iterations: Maximum greedy iterations
            
        Returns:
            Tuple of (selected_indices, selection_info)
        """
        # 步骤4: 设备统一
        if self.device.type == 'cuda':
            pass
        
        # 确保p_target在正确设备上
        p_target = p_target.to(self.device)
        
        print(f"Selecting {k} anchor views from {len(known_views)} known views")
        
        # Convert poses to ViewInfo objects
        view_infos = self._create_view_infos(known_views, p_target)
        frontier_infos = self._create_view_infos(frontier_views, p_target)
        
        # Calculate coverage scores w_i (pixel overlap, center-weighted)
        coverage_scores = self._calculate_coverage_scores(view_infos, frontier_infos, p_target)
        
        # Calculate similarity matrix E
        similarity_matrix = self._calculate_similarity_matrix(view_infos)
        
        # Greedy optimization: max s^T w - s^T E s
        selected_indices = self._greedy_optimization(
            coverage_scores, similarity_matrix, k, max_iterations
        )
        
        # Prepare selection info
        selection_info = {
            'coverage_scores': coverage_scores,
            'similarity_matrix': similarity_matrix,
            'selected_views': [view_infos[i] for i in selected_indices],
            'objective_value': self._compute_objective_value(
                selected_indices, coverage_scores, similarity_matrix
            )
        }
        
        print(f"Selected anchor views: {selected_indices}")
        print(f"Objective value: {selection_info['objective_value']:.4f}")
        
        return selected_indices, selection_info
    
    def select_to_be_updated(self,
                           M_samples: List[torch.Tensor],
                           anchors: List[torch.Tensor], 
                           frontiers: List[torch.Tensor],
                           p_target: torch.Tensor,
                           k: int = 3) -> Tuple[List[int], Dict]:
        """
        Select views to be updated using similar optimization strategy
        Paper Section 4.3.2
        
        Args:
            M_samples: Candidate views for updating
            anchors: Selected anchor views
            frontiers: Frontier views
            p_target: Object center position
            k: Number of views to select for updating
            
        Returns:
            Tuple of (selected_indices, selection_info)
        """
        # 步骤3: 确保p_target在CUDA设备上
        if self.device.type == 'cuda':
            pass
        
        # 确保p_target在正确设备上
        p_target = p_target.to(self.device)
        
        print(f"Selecting {k} views to be updated from {len(M_samples)} candidates")
        
        # Convert to ViewInfo objects
        sample_infos = self._create_view_infos(M_samples, p_target)
        anchor_infos = self._create_view_infos(anchors, p_target)
        frontier_infos = self._create_view_infos(frontiers, p_target)
        
        # Calculate coverage with respect to frontier views
        coverage_scores = self._calculate_coverage_scores(sample_infos, frontier_infos, p_target)
        
        # Calculate similarity matrix for samples
        similarity_matrix = self._calculate_similarity_matrix(sample_infos)
        
        # Apply distance discount β (Appendix C)
        coverage_scores = self._apply_distance_discount(
            coverage_scores, sample_infos, p_target
        )
        
        # Greedy optimization
        selected_indices = self._greedy_optimization(
            coverage_scores, similarity_matrix, k, 1000
        )
        
        selection_info = {
            'coverage_scores': coverage_scores,
            'similarity_matrix': similarity_matrix,
            'selected_views': [sample_infos[i] for i in selected_indices],
            'objective_value': self._compute_objective_value(
                selected_indices, coverage_scores, similarity_matrix
            )
        }
        
        print(f"Selected update views: {selected_indices}")
        print(f"Objective value: {selection_info['objective_value']:.4f}")
        
        return selected_indices, selection_info
    
    def _create_view_infos(self, poses: List[torch.Tensor], p_target: torch.Tensor) -> List[ViewInfo]:
        """Convert pose matrices to ViewInfo objects"""
        view_infos = []
        for i, pose in enumerate(poses):
            # 步骤4: 确保pose在正确设备上
            pose = pose.to(self.device)
            
            # Extract camera center (translation part)
            camera_center = pose[:3, 3]
            
            # Extract viewing direction (negative z-axis in camera coordinates)
            viewing_direction = -pose[:3, 2]  # Camera looks along -z
            viewing_direction = viewing_direction / torch.norm(viewing_direction)
            
            view_info = ViewInfo(
                pose=pose,
                camera_center=camera_center,
                viewing_direction=viewing_direction,
                index=i
            )
            view_infos.append(view_info)
            
        return view_infos
    
    def _calculate_coverage_scores(self, 
                                 view_infos: List[ViewInfo],
                                 target_views: List[ViewInfo], 
                                 p_target: torch.Tensor) -> torch.Tensor:
        """
        Calculate coverage scores w_i (pixel overlap, center-weighted)
        Paper Section 4.3.1
        """
        n_views = len(view_infos)
        coverage_scores = torch.zeros(n_views)
        
        for i, view in enumerate(view_infos):
            total_coverage = 0.0
            
            for target_view in target_views:
                # Calculate pixel overlap between view and target_view
                overlap = self._calculate_pixel_overlap(view, target_view, p_target)
                
                # Center-weighted coverage
                center_weight = self._calculate_center_weight(view, p_target)
                weighted_coverage = overlap * center_weight
                
                total_coverage += weighted_coverage
            
            coverage_scores[i] = total_coverage
            
        # Normalize coverage scores
        if coverage_scores.sum() > 0:
            coverage_scores = coverage_scores / coverage_scores.sum()
            
        return coverage_scores
    
    def _calculate_pixel_overlap(self, 
                               view1: ViewInfo, 
                               view2: ViewInfo, 
                               p_target: torch.Tensor) -> float:
        """
        Calculate pixel overlap between two views
        Based on viewing angle and distance to target
        """
        # 步骤1: 诊断tensor设备
        print(f"DEBUG: view1.camera_center device: {view1.camera_center.device}")
        print(f"DEBUG: view2.camera_center device: {view2.camera_center.device}")
        print(f"DEBUG: p_target device: {p_target.device}")
        
        # 步骤2: 移动p_target到CUDA设备
        target_device = view1.camera_center.device
        p_target = p_target.to(target_device)
        print(f"DEBUG: p_target moved to device: {p_target.device}")
        
        # Calculate viewing angle difference
        cos_angle = torch.dot(view1.viewing_direction, view2.viewing_direction)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        angle_diff = torch.acos(cos_angle)
        
        # Calculate distance to target
        dist1 = torch.norm(view1.camera_center - p_target)
        dist2 = torch.norm(view2.camera_center - p_target)
        
        # Overlap decreases with angle difference and distance difference
        angle_factor = torch.exp(-angle_diff / (math.pi / 4))  # Decay over 45 degrees
        dist_factor = torch.exp(-torch.abs(dist1 - dist2) / dist1.max())
        
        overlap = float(angle_factor * dist_factor)
        return overlap
    
    def _calculate_center_weight(self, view: ViewInfo, p_target: torch.Tensor) -> float:
        """
        Calculate center-weighted factor for coverage
        Higher weight when target is near image center
        """
        # 步骤1: 诊断tensor设备
        print(f"DEBUG: view.viewing_direction device: {view.viewing_direction.device}")
        print(f"DEBUG: self.up_vector device: {self.up_vector.device}")
        
        # 步骤2: 确保设备一致
        target_device = view.viewing_direction.device
        up_vector = self.up_vector.to(target_device)
        p_target = p_target.to(target_device)
        print(f"DEBUG: up_vector moved to device: {up_vector.device}")
        
        # Project target to image plane
        relative_pos = p_target - view.camera_center
        
        # Transform to camera coordinates (simplified)
        cam_z = torch.dot(relative_pos, view.viewing_direction)
        if cam_z <= 0:
            return 0.0  # Behind camera
        
        # Project to image plane
        cam_x = torch.dot(relative_pos, torch.cross(view.viewing_direction, up_vector))
        cam_y = torch.dot(relative_pos, torch.cross(torch.cross(view.viewing_direction, up_vector), view.viewing_direction))
        
        pixel_x = (cam_x * self.focal_length / cam_z) + self.image_width / 2
        pixel_y = (cam_y * self.focal_length / cam_z) + self.image_height / 2
        
        # Calculate distance from center
        center_x, center_y = self.image_width / 2, self.image_height / 2
        dist_from_center = torch.sqrt((pixel_x - center_x)**2 + (pixel_y - center_y)**2)
        max_dist = torch.sqrt(torch.tensor((self.image_width/2)**2 + (self.image_height/2)**2))
        
        # Weight decreases with distance from center
        center_weight = torch.exp(-dist_from_center / (max_dist / 2))
        return float(center_weight)
    
    def _calculate_similarity_matrix(self, view_infos: List[ViewInfo]) -> torch.Tensor:
        """
        Calculate similarity matrix E
        Paper Section 4.3.1
        """
        n_views = len(view_infos)
        similarity_matrix = torch.zeros(n_views, n_views)
        
        for i in range(n_views):
            for j in range(n_views):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # Calculate similarity based on viewing direction and position
                    view_i, view_j = view_infos[i], view_infos[j]
                    
                    # Viewing direction similarity
                    dir_similarity = torch.dot(view_i.viewing_direction, view_j.viewing_direction)
                    dir_similarity = (dir_similarity + 1) / 2  # Map [-1,1] to [0,1]
                    
                    # Position similarity (inverse distance)
                    pos_distance = torch.norm(view_i.camera_center - view_j.camera_center)
                    pos_similarity = torch.exp(-pos_distance)
                    
                    # Combined similarity
                    similarity = 0.7 * dir_similarity + 0.3 * pos_similarity
                    similarity_matrix[i, j] = similarity
                    
        return similarity_matrix
    
    def _apply_distance_discount(self, 
                               coverage_scores: torch.Tensor,
                               view_infos: List[ViewInfo], 
                               p_target: torch.Tensor) -> torch.Tensor:
        """
        Apply distance discount β (Appendix C)
        """
        # 步骤6: 全局tensor处理
        if self.device.type == 'cuda':
            pass
        
        # 步骤1: 诊断tensor设备
        if len(view_infos) > 0:
            print(f"DEBUG: view.camera_center device: {view_infos[0].camera_center.device}")
            print(f"DEBUG: p_target device: {p_target.device}")
            
            # 步骤2: 移动p_target到CUDA设备
            target_device = view_infos[0].camera_center.device
            p_target = p_target.to(target_device)
            print(f"DEBUG: p_target moved to device: {p_target.device}")
        
        discounted_scores = coverage_scores.clone()
        print(f"DEBUG: discounted_scores device: {discounted_scores.device}")
        
        # 使用类常量beta_tensor
        beta_tensor = self.beta_tensor.to(discounted_scores.device)
        
        for i, view in enumerate(view_infos):
            distance = torch.norm(view.camera_center - p_target)
            print(f"DEBUG: distance device: {distance.device}")
            
            # 确保distance在正确设备上
            distance = distance.to(discounted_scores.device)
            
            # 步骤2: 使用正确设备的discount_factor
            # Apply exponential distance discount
            discount_factor = torch.pow(beta_tensor, distance)
            print(f"DEBUG: discount_factor device: {discount_factor.device if hasattr(discount_factor, 'device') else 'scalar or cpu'}")
            print(f"DEBUG: discounted_scores[{i}] device: {discounted_scores[i].device}")
            
            # 确保discount_factor在正确设备上
            discount_factor = discount_factor.to(discounted_scores.device)
            discounted_scores[i] = discounted_scores[i] * discount_factor
            
        return discounted_scores
    
    def _greedy_optimization(self, 
                           coverage_scores: torch.Tensor,
                           similarity_matrix: torch.Tensor, 
                           k: int,
                           max_iterations: int) -> List[int]:
        """
        Greedy optimization of: max s^T w - s^T E s
        Paper Algorithm 1 (Appendix A)
        """
        n_views = len(coverage_scores)
        selected = set()
        
        for iteration in range(min(k, max_iterations)):
            best_gain = -float('inf')
            best_candidate = -1
            
            for candidate in range(n_views):
                if candidate in selected:
                    continue
                    
                # Calculate gain from adding this candidate
                gain = self._calculate_selection_gain(
                    candidate, selected, coverage_scores, similarity_matrix
                )
                
                if gain > best_gain:
                    best_gain = gain
                    best_candidate = candidate
            
            if best_candidate != -1:
                selected.add(best_candidate)
                print(f"  Iteration {iteration+1}: Selected view {best_candidate}, gain: {best_gain:.4f}")
            else:
                break
                
        return list(selected)
    
    def _calculate_selection_gain(self,
                                candidate: int,
                                current_selection: set,
                                coverage_scores: torch.Tensor, 
                                similarity_matrix: torch.Tensor) -> float:
        """
        Calculate gain from adding candidate to current selection
        """
        # Coverage gain
        coverage_gain = coverage_scores[candidate]
        
        # Similarity penalty with already selected views
        similarity_penalty = 0.0
        for selected_idx in current_selection:
            similarity_penalty += similarity_matrix[candidate, selected_idx]
        
        # Self-similarity penalty
        similarity_penalty += similarity_matrix[candidate, candidate]
        
        # Total gain: coverage - similarity penalty
        total_gain = coverage_gain - similarity_penalty
        return float(total_gain)
    
    def _compute_objective_value(self,
                               selected_indices: List[int],
                               coverage_scores: torch.Tensor,
                               similarity_matrix: torch.Tensor) -> float:
        """
        Compute final objective value: s^T w - s^T E s
        """
        if not selected_indices:
            return 0.0
            
        # Create selection vector
        n_views = len(coverage_scores)
        s = torch.zeros(n_views)
        for idx in selected_indices:
            s[idx] = 1.0
        
        # Compute s^T w
        coverage_term = torch.dot(s, coverage_scores)
        
        # Compute s^T E s
        similarity_term = torch.dot(s, torch.matmul(similarity_matrix, s))
        
        objective_value = coverage_term - similarity_term
        return float(objective_value)

def create_test_poses(n_views: int, radius: float = 3.0, p_target: torch.Tensor = None) -> List[torch.Tensor]:
    """
    Create test camera poses in a circle around target
    """
    if p_target is None:
        p_target = torch.tensor([0., 0., 0.])
    
    poses = []
    for i in range(n_views):
        angle = 2 * math.pi * i / n_views
        
        # Camera position in circle
        cam_x = radius * math.cos(angle)
        cam_z = radius * math.sin(angle)
        cam_y = 0.5  # Slightly above
        
        camera_center = torch.tensor([cam_x, cam_y, cam_z])
        
        # Look at target
        forward = p_target - camera_center
        forward = forward / torch.norm(forward)
        
        up = torch.tensor([0., 1., 0.])
        right = torch.cross(forward, up)
        right = right / torch.norm(right)
        up = torch.cross(right, forward)
        
        # Create 4x4 pose matrix (camera-to-world)
        pose = torch.eye(4)
        pose[:3, 0] = right
        pose[:3, 1] = up
        pose[:3, 2] = -forward  # Camera looks along -z
        pose[:3, 3] = camera_center
        
        poses.append(pose)
    
    return poses
